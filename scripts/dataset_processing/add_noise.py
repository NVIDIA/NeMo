import subprocess
import argparse
import random
import os
import datetime
import numpy as np
import soundfile as sf
import copy
import multiprocessing, os
import json
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.preprocessing.perturb import NoisePerturbation
from pytorch_lightning import seed_everything


num_cores = multiprocessing.cpu_count() - 5
print(f"Detected {num_cores} CPU cores")
num_cores=1
rng= None


def get_out_dir_name(out_dir, input_name, noise_name, snr):
    return os.path.join(out_dir, input_name, noise_name +  "_" + str(snr) + "db")

def create_manifest(input_manifest, noise_manifest, snrs, out_path):
    os.makedirs(os.path.join(out_path, "manifests"), exist_ok=True)
    for snr in snrs:
        out_dir = get_out_dir_name(
                out_path,
                os.path.splitext(os.path.basename(input_manifest))[0],
                os.path.splitext(os.path.basename(noise_manifest))[0],
                snr)
        out_mfst = os.path.join(os.path.join(out_path, "manifests"), os.path.splitext(os.path.basename(input_manifest))[0] + "_" + os.path.splitext(os.path.basename(noise_manifest))[0] + "_" + str(snr) + "db" + ".json")
        with open(input_manifest,"r") as inf, open(out_mfst, "w") as outf:
            for line in inf:
                row = json.loads(line.strip())
                row['audio_filepath'] = os.path.join(out_dir, os.path.basename(row['audio_filepath']))
                outf.write(json.dumps(row) + "\n")

def process_row(row):
    audio_file = row['audio_filepath']
        #print(audio_file)
    data_orig=AudioSegment.from_file(audio_file, target_sr=16000, offset=0)
    for snr in row['snrs']:
        min_snr_db = snr
        max_snr_db = snr
        att_factor = 0.8
        perturber = NoisePerturbation(manifest_path=row['noise_manifest'], min_snr_db=min_snr_db,
                                        max_snr_db=max_snr_db, rng=rng)

        
        out_dir = get_out_dir_name(row['out_dir'],
                os.path.splitext(os.path.basename(row['input_manifest']))[0],
                os.path.splitext(os.path.basename(row['noise_manifest']))[0], snr)
        os.makedirs(out_dir, exist_ok=True)
        out_f = os.path.join(out_dir, os.path.basename(audio_file))
        if os.path.exists(out_f):
           continue
        data = copy.deepcopy(data_orig)           
        perturber.perturb(data)
            
        max_level = np.max(np.abs(data.samples))
        
        norm_factor = att_factor/max_level
        new_samples = norm_factor * data.samples 
        sf.write(out_f, new_samples.transpose(), 16000)
    #break
def add_noise(infile, snrs, noise_manifest, out_dir):
    allrows=[]

    with open(infile,"r") as inf:
        for line in inf:
            row = json.loads(line.strip())
            row['snrs'] = snrs
            row['out_dir'] = out_dir
            row['noise_manifest'] = noise_manifest
            row['input_manifest'] = infile
            allrows.append(row)
    print(allrows[0])

    print(f"nfiles: {len(allrows)}")

    pool = multiprocessing.Pool(num_cores)
    pool.map(process_row, allrows)
    pool.close()
    print('Done!')
    
# args: input_manifest, noise_manifest, snrs
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest", type=str, required=True, help="clean test set",
    )
    parser.add_argument("--noise_manifest", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--out_dir", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--snrs", type=int, nargs="+", default=[0,10,20,30])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    global rng
    rng = random.Random(args.seed)
    
    add_noise(args.input_manifest, args.snrs, args.noise_manifest, args.out_dir)
    create_manifest(args.input_manifest, args.noise_manifest, args.snrs, args.out_dir)

if __name__ == '__main__':
    main()
