# Copyright (c) 2019 NVIDIA Corporation
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core import *

from torchvision import transforms
import torch
import os
from os.path import exists
import glob
import json
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance
from math import acos
from math import sqrt
from math import pi
import colorsys


class AddRandomContrast(object):
    """
    Apply some random contrast from PIL
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):
        contrast = ImageEnhance.Contrast(im)
        im = contrast.enhance(np.random.normal(1, self.sigma))
        return im


class AddRandomBrightness(object):
    """
    Apply some random brightness from PIL
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):
        bright = ImageEnhance.Brightness(im)
        im = bright.enhance(np.random.normal(1, self.sigma))
        return im


class AddNoise(object):
    """
    Given mean: (R, G, B) and std: (R, G, B),
      will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        # t = torch.FloatTensor(tensor.size()).uniform_(self.min,self.max)
        t = torch.FloatTensor(tensor.size()).normal_(0, self.std)

        t = tensor.add(t)
        t = torch.clamp(t, -1, 1)  # this is expansive
        return t


def crop(img, i, j, h, w):
    """
    Crop the given PIL.Image.

    Args:
       img (PIL.Image): Image to be cropped.
       i: Upper pixel coordinate.
       j: Left pixel coordinate.
       h: Height of the cropped image.
       w: Width of the cropped image.
    Returns:
       PIL.Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))


"""
Some simple vector math functions to find the angle between two points, used by
affinity fields.
"""


def length(v):
    return sqrt(v[0]**2+v[1]**2)


def dot_product(v, w):
    return v[0]*w[0]+v[1]*w[1]


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm


def determinant(v, w):
    return v[0]*w[1]-v[1]*w[0]


def inner_angle(v, w):
    cosx = dot_product(v, w)/(length(v)*length(w))
    rad = acos(cosx)  # in radians
    return rad*180/pi  # returns degrees


def py_ang(A, B=(1, 0)):
    inner = inner_angle(A, B)
    det = determinant(A, B)
    # this is a property of the det. If the det < 0 then B is clockwise of A
    if det < 0:
        return inner
    else:  # if the det > 0 then A is immediately clockwise of B
        return 360-inner


def GenerateMapAffinity(img, nb_vertex, pointsInterest,
                        objects_centroid, scale):
    """
    Function to create the affinity maps, e.g., vector maps pointing toward the
    object center.

    Args:
       img: PIL image
       nb_vertex: (int) number of points
       pointsInterest: list of points
       objects_centroid: (x,y) centroids for the obects
       scale: (float) by how much you need to scale down the image
    return:
       return a list of tensors for each point except centroid point
    """

    # Apply the downscale right now, so the vectors are correct.
    img_affinity = Image.new(
        img.mode, (int(img.size[0]/scale), int(img.size[1]/scale)), "black")
    # Create the empty tensors
    totensor = transforms.Compose([transforms.ToTensor()])

    affinities = []
    for i_points in range(nb_vertex):
        affinities.append(torch.zeros(
            2, int(img.size[1]/scale), int(img.size[0]/scale)))

    for i_pointsImage in range(len(pointsInterest)):
        pointsImage = pointsInterest[i_pointsImage]
        center = objects_centroid[i_pointsImage]
        for i_points in range(nb_vertex):
            affinity_pair, img_affinity = getAfinityCenter(
                int(img.size[0]/scale),
                int(img.size[1]/scale),
                tuple((np.array(pointsImage[i_points])/scale).tolist()),
                tuple((np.array(center)/scale).tolist()),
                img_affinity=img_affinity, radius=1)

            affinities[i_points] = (affinities[i_points] + affinity_pair)/2

            # Normalizing
            v = affinities[i_points].numpy()
            xvec = v[0]
            yvec = v[1]

            norms = np.sqrt(xvec * xvec + yvec * yvec)
            nonzero = norms > 0

            xvec[nonzero] /= norms[nonzero]
            yvec[nonzero] /= norms[nonzero]

            affinities[i_points] = torch.from_numpy(
                np.concatenate([[xvec], [yvec]]))

    affinities = torch.cat(affinities, 0)
    return affinities


def getAfinityCenter(width, height, point, center,
                     radius=7, img_affinity=None):
    """
    Function to create the affinity maps, e.g., vector maps pointing
    toward the object center.

    Args:
       width: image wight
       height: image height
       point: (x,y)
       center: (x,y)
       radius: pixel radius
       img_affinity: tensor to add to
    return:
       return a tensor
    """

    tensor = torch.zeros(2, height, width).float()

    # Create the canvas for the afinity output
    imgAffinity = Image.new("RGB", (width, height), "black")
    totensor = transforms.Compose([transforms.ToTensor()])

    draw = ImageDraw.Draw(imgAffinity)
    r1 = radius
    p = point
    draw.ellipse((p[0]-r1, p[1]-r1, p[0]+r1, p[1]+r1), (255, 255, 255))

    del draw

    # Compute the array to add the afinity
    array = (np.array(imgAffinity)/255)[:, :, 0]

    angle_vector = np.array(center) - np.array(point)
    angle_vector = normalize(angle_vector)
    affinity = np.concatenate(
        [[array*angle_vector[0]], [array*angle_vector[1]]])

    # print (tensor)
    if img_affinity is not None:
        # Find the angle vector
        # print (angle_vector)
        if length(angle_vector) > 0:
            angle = py_ang(angle_vector)
        else:
            angle = 0
        # print(angle)
        c = np.array(colorsys.hsv_to_rgb(angle/360, 1, 1)) * 255
        draw = ImageDraw.Draw(img_affinity)
        draw.ellipse((p[0]-r1, p[1]-r1, p[0]+r1, p[1]+r1),
                     fill=(int(c[0]), int(c[1]), int(c[2])))
        del draw
    re = torch.from_numpy(affinity).float() + tensor
    return re, img_affinity


def CreateBeliefMap(img, pointsBelief, nbpoints, sigma=16):
    """
    Args:
     img: image
     pointsBelief: list of points in the form of
        [nb object, nb points, 2 (x,y)]
     nbpoints: (int) number of points, DOPE uses 8 points here
     sigma: (int) size of the belief map point
    Return:
     return an array of PIL black and white images representing the belief maps
    """
    beliefsImg = []
    sigma = int(sigma)
    for numb_point in range(nbpoints):
        array = np.zeros(img.size)
        out = np.zeros(img.size)

        for point in pointsBelief:
            p = point[numb_point]
            w = int(sigma*2)
            if p[0]-w >= 0 and p[0]+w < img.size[0] and p[1]-w >= 0 \
                    and p[1]+w < img.size[1]:
                for i in range(int(p[0])-w, int(p[0])+w):
                    for j in range(int(p[1])-w, int(p[1])+w):
                        array[i, j] = \
                            np.exp(-(((i - p[0]) ** 2 +
                                      (j - p[1])**2)/(2*(sigma**2))))

        stack = np.stack([array, array, array], axis=0).transpose(2, 1, 0)
        imgBelief = Image.new(img.mode, img.size, "black")
        beliefsImg.append(Image.fromarray((stack*255).astype('uint8')))
    return beliefsImg


class DopeDataLayer(DataLayerNM):
    """This class wraps the FAT data set into the NeuralModule API."""

    @staticmethod
    def create_ports(input_size=(512, 256)):
        input_ports = {}
        output_ports = {
            "image": NeuralType({0: AxisType(BatchTag),
                                 1: AxisType(ChannelTag),
                                 2: AxisType(HeightTag, input_size[1]),
                                 3: AxisType(WidthTag, input_size[0])}),
            "affinity_label": NeuralType({0: AxisType(BatchTag)}),
            "belief_label": NeuralType({0: AxisType(BatchTag)}),
        }
        return input_ports, output_ports

    def __init__(
            self, *,
            batch_size,
            root,
            train=True,
            shuffle=True,
            num_threads=1,
            noise=0.1,
            sigma=16,
            object="soup",
            datasize=None,
            **kwargs
    ):
        self._input_size = (512, 256)
        create_port_args = {"input_size": self._input_size}
        DataLayerNM.__init__(self, create_port_args=create_port_args, **kwargs)

        self._batch_size = batch_size
        self._root = root
        self._train = train
        self._shuffle = shuffle
        self._num_threads = num_threads

        # DOPE specific parameters
        self._noise = noise
        self._sigma = sigma
        self._contrast = 0.2
        self._brightness = 0.2
        self._object = object
        self._datasize = datasize
        self._normal_imgs = [0.59, 0.25]

        self._imagesize = 400
        self._transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # self._root = "/root/src/ngc/11767/single/004_sugar_box_16k/"
        self._dataset = MultipleVertexJson(
            root=self._root,
            objectsofinterest=self._object,
            keep_orientation=True,
            noise=self._noise,
            sigma=self._sigma,
            data_size=self._datasize,
            # save a visual batch and quit, this is for debugging purposes
            save=False,
            transform=transforms.Compose([AddRandomContrast(self._contrast),
                                          AddRandomBrightness(
                self._brightness),
                transforms.Scale(self._imagesize)]),
            normal=self._normal_imgs,
            target_transform=transforms.Compose(
                [transforms.Scale(self._imagesize//8)])
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None

##################################################
# UTILS CODE FOR LOADING THE DATA
##################################################


def default_loader(path):
    return Image.open(path).convert('RGB')


def loadjson(path, objectsofinterest, img):
    """
    Loads the data from a json file.
    If there are no objects of interest, then load all the objects.
    """

    with open(path) as data_file:
        data = json.load(data_file)

    pointsBelief = []
    boxes = []
    points_keypoints_3d = []
    points_keypoints_2d = []
    pointsBoxes = []
    poses = []
    centroids = []

    translations = []
    rotations = []
    points = []

    for i_line in range(len(data['objects'])):
        info = data['objects'][i_line]
        if objectsofinterest is not None and \
           objectsofinterest not in info['class'].lower():
            continue

        box = info['bounding_box']
        boxToAdd = []

        boxToAdd.append(float(box['top_left'][0]))
        boxToAdd.append(float(box['top_left'][1]))
        boxToAdd.append(float(box["bottom_right"][0]))
        boxToAdd.append(float(box['bottom_right'][1]))
        boxes.append(boxToAdd)

        boxpoint = [(boxToAdd[0], boxToAdd[1]), (boxToAdd[0], boxToAdd[3]),
                    (boxToAdd[2], boxToAdd[1]), (boxToAdd[2], boxToAdd[3])]

        pointsBoxes.append(boxpoint)

        # 3dbbox with belief maps
        points3d = []

        pointdata = info['projected_cuboid']
        for p in pointdata:
            points3d.append((p[0], p[1]))

        # Get the centroids
        pcenter = info['projected_cuboid_centroid']

        points3d.append((pcenter[0], pcenter[1]))
        pointsBelief.append(points3d)
        points.append(points3d + [(pcenter[0], pcenter[1])])
        centroids.append((pcenter[0], pcenter[1]))

        # load translations
        location = info['location']
        translations.append([location[0], location[1], location[2]])

        # quaternion
        rot = info["quaternion_xyzw"]
        rotations.append(rot)

    return {
        "pointsBelief": pointsBelief,
        "rotations": rotations,
        "translations": translations,
        "centroids": centroids,
        "points": points,
        "keypoints_2d": points_keypoints_2d,
        "keypoints_3d": points_keypoints_3d,
    }


def loadimages(root):
    """
    Find all the images in the path and folders, return them in imgs.
    """
    imgs = []

    def add_json_files(path,):
        IMAGE_TYPE = "jpg"
        for imgpath in glob.glob(path+"/*.jpg"):
            if exists(imgpath) and exists(imgpath.replace('jpg', "json")):
                imgs.append((imgpath, imgpath.replace(path, "")
                             .replace("/", ""),
                             imgpath.replace('jpg', "json")))

    def explore(path):
        if not os.path.isdir(path):
            return

        folders = [os.path.join(path, o) for o in os.listdir(path)
                   if os.path.isdir(os.path.join(path, o))]
        if len(folders) > 0:
            for path_entry in folders:
                explore(path_entry)
        else:
            add_json_files(path)

    explore(root)
    return imgs


class MultipleVertexJson(torch.utils.data.Dataset):
    """
    Dataloader for the data generated by
    NDDS (https://github.com/NVIDIA/Dataset_Synthesizer).
    This is the same data as the data used in FAT.
    """

    def __init__(self,
                 root=None,
                 transform=None,
                 nb_vertex=8,
                 keep_orientation=True,
                 normal=None, test=False,
                 target_transform=None,
                 loader=default_loader,
                 objectsofinterest="",
                 img_size=400,
                 save=False,
                 noise=2,
                 data_size=None,
                 sigma=16,
                 random_translation=(25.0, 25.0),
                 random_rotation=15.0):
        ###################
        self.save = save
        self.objectsofinterest = objectsofinterest
        self.img_size = img_size
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.imgs = []
        self.test = test
        self.normal = normal
        self.keep_orientation = keep_orientation
        self.save = save
        self.noise = noise
        self.data_size = data_size
        self.sigma = sigma
        self.random_translation = random_translation
        self.random_rotation = random_rotation

        def load_data(path):
            """Recursively load the data.
            This is useful to load all of the FAT dataset."""
            imgs = loadimages(path)

            # Check all the folders in path
            for name in os.listdir(str(path)):
                imgs += loadimages(path + "/"+name)
            return imgs

        self.imgs = load_data(root)

        # Shuffle the data, this is useful when we want to use a subset.
        np.random.shuffle(self.imgs)

    def __len__(self):
        # When limiting the number of data
        if self.data_size is not None:
            return int(self.data_size)
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Depending on how the data loader is configured, this will return the
        debug info with the cuboid drawn on it, this happens when self.save is
        set to true.  Otherwise, during training this function returns the
        belief maps and affinity fields and image as tensors.
        """

        path, name, txt = self.imgs[index]
        img = self.loader(path)
        img_size = img.size
        img_size = (400, 400)

        loader = loadjson

        data = loader(txt, self.objectsofinterest, img)

        pointsBelief = data['pointsBelief']
        objects_centroid = data['centroids']
        points_all = data['points']
        points_keypoints = data['keypoints_2d']
        translations = torch.from_numpy(np.array(data['translations'])).float()
        rotations = torch.from_numpy(np.array(data['rotations'])).float()

        if len(points_all) == 0:
            points_all = torch.zeros(1)

        # self.save == true assumes there is only one object instance
        # in the scene
        if translations.size()[0] > 1:
            translations = translations[0].unsqueeze(0)
            rotations = rotations[0].unsqueeze(0)

        # If there are no objects, still need to return similar shape array
        if len(translations) == 0:
            translations = torch.zeros(1, 3).float()
            rotations = torch.zeros(1, 4).float()

        # Camera intrinsics
        path_cam = path.replace(name, '_camera_settings.json')
        with open(path_cam) as data_file:
            data = json.load(data_file)

        # Assumes one camera
        cam = data['camera_settings'][0]['intrinsic_settings']

        matrix_camera = np.zeros((3, 3))
        matrix_camera[0, 0] = cam['fx']
        matrix_camera[1, 1] = cam['fy']
        matrix_camera[0, 2] = cam['cx']
        matrix_camera[1, 2] = cam['cy']
        matrix_camera[2, 2] = 1

        # Load the cuboid sizes
        path_set = path.replace(name, '_object_settings.json')
        with open(path_set) as data_file:
            data = json.load(data_file)

        cuboid = torch.zeros(1)

        if self.objectsofinterest is None:
            cuboid = np.array(data['exported_objects'][0]['cuboid_dimensions'])
        else:
            for info in data["exported_objects"]:
                if self.objectsofinterest in info['class']:
                    cuboid = np.array(info['cuboid_dimensions'])

        img_original = img.copy()

        def Reproject(points, tm, rm):
            """
            Reprojection of points when rotating the image
            """
            proj_cuboid = np.array(points)

            rmat = np.identity(3)
            rmat[0:2] = rm
            tmat = np.identity(3)
            tmat[0:2] = tm

            new_cuboid = np.matmul(
                rmat, np.vstack((proj_cuboid.T, np.ones(len(points)))))
            new_cuboid = np.matmul(tmat, new_cuboid)
            new_cuboid = new_cuboid[0:2].T

            return new_cuboid

        # Random image manipulation, rotation and translation with zero padding
        dx = round(np.random.normal(0, 2) * float(self.random_translation[0]))
        dy = round(np.random.normal(0, 2) * float(self.random_translation[1]))
        angle = round(np.random.normal(0, 1) * float(self.random_rotation))

        tm = np.float32([[1, 0, dx], [0, 1, dy]])
        rm = cv2.getRotationMatrix2D((img.size[0]/2, img.size[1]/2), angle, 1)

        for i_objects in range(len(pointsBelief)):
            points = pointsBelief[i_objects]
            new_cuboid = Reproject(points, tm, rm)
            pointsBelief[i_objects] = new_cuboid.tolist()
            objects_centroid[i_objects] = tuple(new_cuboid.tolist()[-1])
            pointsBelief[i_objects] = list(map(tuple, pointsBelief[i_objects]))

        for i_objects in range(len(points_keypoints)):
            points = points_keypoints[i_objects]
            new_cuboid = Reproject(points, tm, rm)
            points_keypoints[i_objects] = new_cuboid.tolist()
            points_keypoints[i_objects] = list(
                map(tuple, points_keypoints[i_objects]))

        image_r = cv2.warpAffine(np.array(img), rm, img.size)
        result = cv2.warpAffine(image_r, tm, img.size)
        img = Image.fromarray(result)

        # Note:  All point coordinates are in the image space, eg, pixel value.
        # This is used when we do saving --- helpful for debugging
        if self.save or self.test:
            # Use the save to debug the data
            if self.test:
                draw = ImageDraw.Draw(img_original)
            else:
                draw = ImageDraw.Draw(img)

            # PIL drawing functions, here for sharing draw
            def DrawKeypoints(points):
                for key in points:
                    DrawDot(key, (12, 115, 170), 7)

            def DrawLine(point1, point2, lineColor, lineWidth):
                if point1 is not None and point2 is not None:
                    draw.line([point1, point2],
                              fill=lineColor, width=lineWidth)

            def DrawDot(point, pointColor, pointRadius):
                if point is not None:
                    xy = [point[0]-pointRadius, point[1]-pointRadius,
                          point[0]+pointRadius, point[1]+pointRadius]
                    draw.ellipse(xy, fill=pointColor, outline=pointColor)

            def DrawCube(points, which_color=0, color=None):
                """Draw cube with a thick solid line
                across the front top edge."""
                lineWidthForDrawing = 2
                lineColor1 = (255, 215, 0)  # yellow-ish
                lineColor3 = (45, 195, 35)  # green-ish
                if which_color == 3:
                    lineColor = lineColor3
                else:
                    lineColor = lineColor1

                if color is not None:
                    lineColor = color

                # draw front
                # lineWidthForDrawing)
                DrawLine(points[0], points[1], lineColor, 8)
                DrawLine(points[1], points[2], lineColor, lineWidthForDrawing)
                DrawLine(points[3], points[2], lineColor, lineWidthForDrawing)
                DrawLine(points[3], points[0], lineColor, lineWidthForDrawing)

                # draw back
                DrawLine(points[4], points[5], lineColor, lineWidthForDrawing)
                DrawLine(points[6], points[5], lineColor, lineWidthForDrawing)
                DrawLine(points[6], points[7], lineColor, lineWidthForDrawing)
                DrawLine(points[4], points[7], lineColor, lineWidthForDrawing)

                # draw sides
                DrawLine(points[0], points[4], lineColor, lineWidthForDrawing)
                DrawLine(points[7], points[3], lineColor, lineWidthForDrawing)
                DrawLine(points[5], points[1], lineColor, lineWidthForDrawing)
                DrawLine(points[2], points[6], lineColor, lineWidthForDrawing)

                # draw dots
                DrawDot(points[0], pointColor=(255, 255, 255), pointRadius=3)
                DrawDot(points[1], pointColor=(0, 0, 0), pointRadius=3)

            # Draw all the found objects.
            for points_belief_objects in pointsBelief:
                DrawCube(points_belief_objects)
            for keypoint in points_keypoints:
                DrawKeypoints(keypoint)

            img = self.transform(img)

            return {
                "img": img,
                "translations": translations,
                "rot_quaternions": rotations,
                'pointsBelief': np.array(points_all[0]),
                'matrix_camera': matrix_camera,
                'img_original': np.array(img_original),
                'cuboid': cuboid,
                'file_name': name,
            }

        # Create the belief map
        beliefsImg = CreateBeliefMap(
            img,
            pointsBelief=pointsBelief,
            nbpoints=9,
            sigma=self.sigma)

        # Create the image maps for belief
        transform = transforms.Compose([transforms.Resize(min(img_size))])
        totensor = transforms.Compose([transforms.ToTensor()])

        for j in range(len(beliefsImg)):
            beliefsImg[j] = self.target_transform(beliefsImg[j])
            # beliefsImg[j].save('{}.png'.format(j))
            beliefsImg[j] = totensor(beliefsImg[j])

        beliefs = torch.zeros(
            (len(beliefsImg), beliefsImg[0].size(1), beliefsImg[0].size(2)))
        for j in range(len(beliefsImg)):
            beliefs[j] = beliefsImg[j][0]

        # Create affinity maps
        scale = 8
        if min(img.size) / 8.0 != min(img_size)/8.0:
            # print (scale)
            scale = min(img.size)/(min(img_size)/8.0)

        affinities = GenerateMapAffinity(
            img, 8, pointsBelief, objects_centroid, scale)
        img = self.transform(img)

        # Transform the images for training input
        w_crop = np.random.randint(0, img.size[0] - img_size[0]+1)
        h_crop = np.random.randint(0, img.size[1] - img_size[1]+1)
        totensor = transforms.Compose([transforms.ToTensor()])

        if self.normal is not None:
            normalize = transforms.Compose([transforms.Normalize
                                            ((self.normal[0], self.normal[0],
                                              self.normal[0]),
                                             (self.normal[1], self.normal[1],
                                              self.normal[1])),
                                            AddNoise(self.noise)])
        else:
            normalize = transforms.Compose([AddNoise(0.0001)])

        img = crop(img, h_crop, w_crop, img_size[1], img_size[0])
        img = totensor(img)

        img = normalize(img)

        w_crop = int(w_crop/8)
        h_crop = int(h_crop/8)

        affinities = affinities[:, h_crop:h_crop +
                                int(img_size[1]/8),
                                w_crop:w_crop+int(img_size[0]/8)]
        beliefs = beliefs[:, h_crop:h_crop +
                          int(img_size[1]/8), w_crop:w_crop+int(img_size[0]/8)]

        if affinities.size()[1] == 49 and not self.test:
            affinities = torch.cat([affinities, torch.zeros(16, 1, 50)], dim=1)

        if affinities.size()[2] == 49 and not self.test:
            affinities = torch.cat([affinities, torch.zeros(16, 50, 1)], dim=2)

        # return {
        #  'img':img,
        #  "affinities":affinities,
        #  'beliefs':beliefs,
        # }
        return (img, affinities, beliefs)
