# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import json
import os
import tarfile

import numpy as np
import yaml
from pyquaternion import Quaternion
from scipy.interpolate import interp1d

from .camera_model import FThetaCamera
from .rig import get_rig_transform


def get_egopose_interp(egoposes):
    """We move the egoposes to the coordinate frame defined by the
    first egopose. Then we return an interpolator for the ego pose
    in that frame.
    """
    quats = [Quaternion(x=row["qx"], y=row["qy"], z=row["qz"], w=row["qw"]) for row in egoposes]
    xyzs = np.array([[row["x"], row["y"], row["z"]] for row in egoposes])

    # choose the first timestep to define the global coordinate frame
    TIX = 0
    qbase = quats[TIX].inverse
    qbase_rot = qbase.rotation_matrix
    xyzbase = -qbase_rot @ xyzs[TIX]
    cliptbase = egoposes[TIX]["t"]

    # transform to the base frame
    quats = [qbase * quat for quat in quats]
    xyzs = (qbase_rot @ xyzs.T + xyzbase.reshape((3, 1))).T

    # now adjust the timestamps to the base timestamp
    clipts = np.array([(row["t"] - cliptbase) * 1e-6 for row in egoposes])

    # ready for the interpolator
    states = [{"quat": quat, "xyz": xyz, "t": t} for quat, xyz, t in zip(quats, xyzs, clipts)]
    interp = get_6dof_interpolator(states)

    return interp, cliptbase


def get_6dof_interpolator(states):
    """states is a list of timestamped 6 DOF poses structured like
    [
        {
            quat: Quaternion,
            xyz: [x,y,z],
            t: t,
            dims: [length,width,height], (optional)
        },
    ]
    Before feeding to the interpolator, we sort by time, normalize the quaternions,
    canonicalize the quaternion directions, and cast to float32.
    The interpolator returns (qw, qx, qy, qz, x, y, z)
    """
    # sort by time
    states = sorted(states, key=lambda x: x["t"])

    # normalize quaternions
    # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L847
    quats = [state["quat"] for state in states]
    [quat._fast_normalise() for quat in quats]
    quats = steady_quaternion_direction(quats)

    interpvals = np.array(
        [
            [
                quat.w,
                quat.x,
                quat.y,
                quat.z,
                state["xyz"][0],
                state["xyz"][1],
                state["xyz"][2],
            ]
            for quat, state in zip(quats, states)
        ],
        dtype=np.float32,
    )

    # also interpolate bbox dimensions if available
    if "dims" in states[0]:
        interpvals = np.concatenate(
            (
                interpvals,
                np.array([state["dims"] for state in states], dtype=np.float32),
            ),
            1,
        )

    tvals = np.array([state["t"] for state in states], dtype=np.float32)
    # make sure t parameter is strictly increasing
    assert np.all(0 < tvals[1:] - tvals[:-1]), tvals

    interp = {
        "interp": interp1d(
            tvals,
            interpvals,
            kind="linear",
            axis=0,
            copy=False,
            bounds_error=True,
            assume_sorted=True,
        ),
        "tmin": tvals[0],
        "tmax": tvals[-1],
    }

    return interp


def steady_quaternion_direction(quats):
    """negate any of the quaternions as necessary
    to keep the orientation consistent
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L847
    """
    newquats = [quats[0]]
    for q1 in quats[1:]:
        q0 = newquats[-1]
        dot = np.dot(q0.q, q1.q)
        if dot >= 0.0:
            newquats.append(q1)
        else:
            newquats.append(-q1)
    return newquats


def get_obstacle_interp(og_obstacles, egopose_lerp, cliptbase):
    tvals, obstacles = local_time_filter(og_obstacles, egopose_lerp["tmin"], egopose_lerp["tmax"], cliptbase)

    # evaluate ego pose at the obstacle timestamps
    egoposes = egopose_lerp["interp"](tvals)

    # move obstacles into those coordinate frames
    obsquats, obsxyzs = convert_full_obstacle_state(obstacles, egoposes)

    # track id -> 6 DOF states
    # TODO it seems sometimes tracks have no detections for ~1.5 seconds,
    # linear interpolation is probably bad at those time scales.
    processed = {}
    for row, tval, obsquat, obsxyz in zip(obstacles, tvals, obsquats, obsxyzs):
        if not row["id"] in processed:
            processed[row["id"]] = []
        processed[row["id"]].append(
            {
                "quat": obsquat,
                "xyz": obsxyz,
                "t": tval,
                "dims": [row["le"], row["wi"], row["he"]],
            }
        )

    # filter any tracks with fewer than 2 observations since otherwise we can't interp
    processed = {k: v for k, v in processed.items() if 1 < len(v)}

    # interpolators
    interps = {k: get_6dof_interpolator(v) for k, v in processed.items()}

    return interps


def local_time_filter(obstacles, tmin, tmax, cliptbase):
    """Evaluate the time for each of the detections relative to the base time,
    and remove any detections from outside the time window when
    we have the ego pose
    """
    # convert obstacle timestamps to the clip time frame
    tvals = np.array([(row["t"] - cliptbase) * 1e-6 for row in obstacles], dtype=np.float32)

    # filter out any obstacles outside of the range when we have ego pose
    obsmask = np.logical_and(tmin <= tvals, tvals <= tmax)
    tvals = tvals[obsmask]
    obstacles = [row for rowi, row in enumerate(obstacles) if obsmask[rowi]]

    return tvals, obstacles


def convert_full_obstacle_state(obstacles, egoposes):
    # catch the case where there are no detections quickly
    if len(obstacles) == 0:
        return [], np.empty((0, 3), dtype=np.float32)

    # Nx3 ego xyzs and N list of ego quaternions
    egoquats = []
    egoxyzs = []
    for qw, qx, qy, qz, x, y, z in egoposes:
        egoquats.append(Quaternion(w=qw, x=qx, y=qy, z=qz))
        egoxyzs.append(np.array([x, y, z]))
    egorots = np.stack([quat.rotation_matrix for quat in egoquats]).astype(np.float32)
    egoxyzs = np.stack(egoxyzs)

    obsquats = []
    obsxyzs = []
    for rowi, row in enumerate(obstacles):
        obsxyzs.append([row["x"], row["y"], row["z"]])
        # only yaw reported in the detections
        obsquats.append(Quaternion(axis=[0, 0, 1], angle=row["theta"]))
    obsxyzs = np.array(obsxyzs, dtype=np.float32)

    # transform
    obsquats = [egoquat * obsquat for egoquat, obsquat in zip(egoquats, obsquats)]
    obsxyzs = np.squeeze(egorots @ obsxyzs[:, :, np.newaxis], axis=2) + egoxyzs

    return obsquats, obsxyzs


def pose_to_corn(qw, qx, qy, qz, x, y, z, le, wi, he):
    quat = Quaternion(w=qw, x=qx, y=qy, z=qz)

    le2 = le / 2
    wi2 = wi / 2
    he2 = he / 2

    corn = np.array(
        [
            [-le2, -wi2, -he2],
            [le2, -wi2, -he2],
            [le2, wi2, -he2],
            [-le2, wi2, -he2],
            [-le2, -wi2, he2],
            [le2, -wi2, he2],
            [le2, wi2, he2],
            [-le2, wi2, he2],
        ]
    )

    corn = corn @ quat.rotation_matrix.T + np.array([x, y, z])

    return corn


def get_calibration_tfs(calibration, lidarkey, camkeys):
    tfs = {}
    tfs["lidar2rig"] = get_rig_transform(calibration[lidarkey], rig2sensor=False)

    for camkey in camkeys:
        tfs[f"rig2{camkey}"] = get_rig_transform(calibration[camkey], rig2sensor=True)
        tfs[f"ftheta{camkey}"] = FThetaCamera.from_dict(calibration[camkey])

    # make sure didn't overwrite anything
    assert len(tfs) == 1 + 2 * len(camkeys), len(tfs)

    return tfs


def get_clip_to_tar(datapath):
    """Read the clip_to_tar.json metadata for the folder datapath. Returns
    a dictionary
        id -> (tar file location, key for this clip in that tar file)
    """
    # clipid->tar stored at clip_to_tar.json
    fpath = os.path.join(datapath, "clip_to_tar.json")

    # print("reading clip_to_tar.json from", fpath)
    with open(fpath, "r") as reader:
        info = json.load(reader)

    # id -> (tar file location, key for this clip in that tar file)
    parsed_info = {key2id(key): (os.path.join(datapath, val), key) for key, val in info.items()}

    # shouldn't lose any clips assuming the id is unique
    assert len(parsed_info) == len(info), f"{len(parsed_info)} {len(info)}"

    return parsed_info


def key2id(key):
    """The clip ids are saved as e.g. '6c896d92-ad90-11ed-ac93-00044bf65f70_2-clip396904'
    but the "_2" is not the same across maglev jobs. So we remove that part to
    get the clip id.
    """
    sessionid, s1 = key.split("_")
    _, clipid = s1.split("-")
    return (sessionid, clipid)


def parse_obstacle_data(info, check_sensor):
    """Based on https://tegra-sw-opengrok.nvidia.com/source/xref/ndas-main/tools/avmf/tools/scripts/sql_gt_to_rclog/av_proto_makers.py#183

    "info" is a list of detections. We collect the detections for each track id
    to get a dictionary track id -> [detection1, detection2,...] with detections
    sorted by time.

    We also parse the sensor name corresponding to the coordinate frame the boxes
    are represented in.
    """
    processed = []
    duplicate_tracker = {}
    for row in info["labels_data"]:
        row = {k: v for k, v in zip(info["labels_keys"], row)}
        label = json.loads(row["label_data"])

        # skip these
        if "emptyLabel" in label and label["emptyLabel"]:
            continue

        atts = {att["name"]: att for att in label["shape3d"]["attributes"]}
        # make sure no duplicates in the attribute names
        assert len(atts) == len(label["shape3d"]["attributes"])

        id = int(float(atts["gt_trackline_id"]["text"]))
        # skip these
        if id is None or id < 0:
            continue

        # TODO seems we sometimes (~1% of clips) have duplicates. This
        # fixes it by taking only one detection per track per lidar spin.
        moment_id = (id, row["frame_number"])
        if moment_id in duplicate_tracker:
            continue
        duplicate_tracker[moment_id] = True

        processed.append(
            {
                "x": label["shape3d"]["cuboid3d"]["center"]["x"],
                "y": label["shape3d"]["cuboid3d"]["center"]["y"],
                "z": label["shape3d"]["cuboid3d"]["center"]["z"],
                "theta": np.arctan2(atts["direction"]["vec2"]["y"], atts["direction"]["vec2"]["x"]),
                "le": label["shape3d"]["cuboid3d"]["dimensions"]["x"],
                "wi": label["shape3d"]["cuboid3d"]["dimensions"]["y"],
                "he": label["shape3d"]["cuboid3d"]["dimensions"]["z"],
                "npct": atts["label_name"]["enum"],
                "t": int(float(atts["timestamp"]["text"])),
                "id": id,
            }
        )

        # make sure sensor coordinate frame is the same as ego pose
        assert row["sensor_name"] == check_sensor, f'{row["sensor_name"]} {check_sensor}'

    return processed


def parse_egopose_data(egopose_info):
    # TODO verify that all timestamps are on the same clock as the detections
    # and videos
    parsed = []
    sensor_names = []
    for row in egopose_info:
        assert row["coordinate_frame"] == row["sensor_name"], f"{row['coordinate_frame']} {row['sensor_name']}"
        sensor_names.append(row["coordinate_frame"])
        parsed.append(
            {
                "x": row["x"],
                "y": row["y"],
                "z": row["z"],
                "qx": row["qx"],
                "qy": row["qy"],
                "qz": row["qz"],
                "qw": row["qw"],
                "t": row["timestamp"],
            }
        )
    # sort chronologically
    parsed = sorted(parsed, key=lambda x: x["t"])

    # make sure all poses are in the same sensor frame
    assert len(set(sensor_names)) == 1, set(sensor_names)
    sensor_name = sensor_names[0]

    return parsed, sensor_name


def get_video_parameters(datapaths, camkeys):
    """Load the pre-processing config file for the dataset.
    Currently we assert that all cameras use the same parameters,
    we could loosen that in the future.
    """
    collect = {"crop_h": [], "crop_w": [], "size_h": [], "size_w": [], "fps": []}
    for datapath in datapaths:
        f = os.path.join(datapath, "generator_config.yaml")

        # print("reading", f)
        with open(f, "r") as reader:
            conf = yaml.safe_load(reader)

        for camkey in camkeys:
            for key in collect:
                collect[key].append(conf["data"][camkey.replace(".mp4", "")][key])
    # make sure they're all the same
    for key in collect:
        assert len(set(collect[key])) == 1, collect[key]

    # select the first one since they're all the same
    out = {k: v[0] for k, v in collect.items()}

    return out


def adjust_calib_name(name):
    """The sensor names in the calibration file are e.g. camera:cross:left:120fov.
    We adjust them to match the names in the wds tar files, eg camera_cross_left_120fov.mp4
    """
    # the sensor kind (eg radar/camera/lidar etc)
    kind = name.split(":")[0]

    if kind == "camera":
        return f"{name.replace(':', '_')}.mp4"
    return name.replace(":", "_")


def parse_calibration_data(calibration):
    # sensor calibration dictionary
    rig_info = {adjust_calib_name(sensor["name"]): sensor for sensor in calibration["rig"]["sensors"]}
    # make sure no overwriting when adjusting the sensor name
    assert len(rig_info) == len(calibration["rig"]["sensors"])

    # TODO seems we can get adjustment of rig coordinate frame to ego bbox?
    # from calibration['rig']['vehicle']['value']['body']['boundingBoxPosition']

    egoparams = {
        "le": calibration["rig"]["vehicle"]["value"]["body"]["length"],
        "wi": calibration["rig"]["vehicle"]["value"]["body"]["width"],
        "he": calibration["rig"]["vehicle"]["value"]["body"]["height"],
        "platform": calibration["rig"]["properties"]["platform_name"],
    }

    return rig_info, egoparams


def extract_obstacle_and_epomotion_from_tar(sample_key, camera_keys, clip2tar, load_obstacle=True):
    """Extract egomotions from tar, used by AV1.1"""

    metadata = {}
    try:
        cid = key2id(sample_key)
        if not cid in clip2tar:
            return metadata

        tarf, tarkey = clip2tar[cid]
        tarreader = tarfile.open(tarf)
        for camera_name in camera_keys:
            camera_metadata = {}
            camera = camera_name.replace(".mp4", "")
            egopose_file = tarreader.extractfile(f"{tarkey}.{camera}_egomotion.json")
            egopose_info = json.loads(egopose_file.read())
            egopose_parsed, sensor_name = parse_egopose_data(egopose_info)
            camera_metadata["egopose"] = egopose_parsed
            camera_metadata["sensor_name"] = sensor_name

            # obstacle data
            if load_obstacle:
                obstacle_file = tarreader.extractfile(f"{tarkey}.obstacle_3d.npz")
                obstacle_info = np.load(obstacle_file, allow_pickle=True)
                obstacle_parsed = parse_obstacle_data(obstacle_info, check_sensor=sensor_name)
                camera_metadata["obstacles"] = obstacle_parsed

            metadata[camera_name] = camera_metadata
    except Exception as error:
        print(f"Failed to extract egomotion metadata for {sample_key}: {error}")
        return None
    return metadata


def extract_calibration_from_tar(sample_key, clip2tar):
    """Extract calibrations from tar, used by AV1.1"""

    metadata = {}
    try:
        cid = key2id(sample_key)
        if not cid in clip2tar:
            return metadata

        tarf, tarkey = clip2tar[cid]
        tarreader = tarfile.open(tarf)
        calibration_file = tarreader.extractfile(f"{tarkey}.rig.json")
        calibration_info = json.loads(calibration_file.read())
        rig_info, egoparams = parse_calibration_data(calibration_info)
        metadata["rig_info"] = rig_info
        metadata["egoparams"] = egoparams
    except Exception as error:
        print(f"Failed to extract calibration for {sample_key}: {error}")
        return None
    return metadata
