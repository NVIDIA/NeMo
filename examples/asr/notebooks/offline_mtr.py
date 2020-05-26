import os
import sys
import glob
import json
import pickle
import wrapt
import inspect
import librosa
import re

# import argparse
import array
import math
import numpy as np
import random
import wave

from sklearn.model_selection import train_test_split
sr = 16000
duration_stride = 1.0


# classification for freesound
background_classes = [
    "Air brake",
    "Static",
    "Acoustic environment",
    "Distortion",
    "Tape hiss",
    "Hubbub",
    "Vibration",
    "Cacophony",
    "Throbbing",
    "Reverberation",
    "Inside, public space",
    "Inside, small room",
    "Echo",
    "Outside, rural",
    "Outside, natural",
    "Outside, urban",
    "Outside, manmade",
    "Car",
    "Bus",
    "Traffic noise",
    "Roadway noise",
    "Truck",
    "Emergency vehicle",
    "Motorcycle",
    "Aircraft engine",
    "Aircraft",
    "Helicopter",
    "Bicycle",
    "Skateboard",
    "Subway, metro, underground",
    "Railroad car",
    "Train wagon",
    "Train",
    "Sailboat",
    "Rowboat",
    "Ship",
]

speech_classes = [
    "Male speech",
    "Female speech",
    "Speech synthesizer",
    "Babbling",
    "Conversation",
    "Child speech",
    "Narration",
    "Laughter",
    "Yawn",
    "Whispering",
    "Whimper",
    "Baby cry",
    "Sigh",
    "Groan",
    "Humming",
    "Male singing",
    "Female singing",
    "Child singing",
    "Children shouting",
]

def normalize_str(x: str):
    x = x.replace(' ', '_')
    return x

# Normalize classes
background_classes = {normalize_str(s):idx for idx, s in enumerate(background_classes)}
speech_classes = {normalize_str(s):idx for idx, s in enumerate(speech_classes)}


# google speech command v2
URL_v2 = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource

    Returns:

    """
    if not os.path.exists(destination):
        logging.info(f"{destination} does not exist. Downloading ...")
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        logging.info(f"Downloaded {destination}.")
    else:
        logging.info(f"Destination {destination} exists. Skipping.")
    return destination


def __extract_all_files(filepath: str, data_root: str, data_dir: str):
    if not os.path.exists(data_dir):
        extract_file(filepath, data_dir)
    else:
        logging.info(f'Skipping extracting. Data already there {data_dir}')


def extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info('Not extracting. Maybe already there?')


def split_dataset_class(data_dir):
    speech_num = 0
    background_num = 0
    all_num = 0
    
    files = sorted(glob.glob(data_dir + '/*/*.wav'))

    all_path = os.path.join(data_dir, 'all.txt')
    background_path = os.path.join(data_dir, 'background.txt')
    speech_path = os.path.join(data_dir, 'speech.txt')
    
    paths = [all_path, background_path, speech_path]
    for i in paths:
        if os.path.exists(i):
            print(f'File {i} exists. Overwrite it!')
            os.remove(i)
    
    with open(all_path, 'a') as allfile:
        for filepath in files:
            head, filename = os.path.split(filepath)
            _, clsname = os.path.split(head)
            clsname = normalize_str(clsname)
            if clsname in background_classes:
#                 label = 'background'
                background_num += 1
                with open(background_path, 'a') as bfile:
                    bfile.write(filepath.split(data_dir)[1]) 
                    bfile.write('\n')
            elif clsname in speech_classes:
#                 label = 'speech'
                speech_num += 1
                with open(speech_path, 'a') as sfile:
                    sfile.write(filepath.split(data_dir)[1]) 
                    sfile.write('\n')
            else:
                raise ValueError(f'Label {clsname} doesnt belong to either backgound or speech class sets !')
            all_num += 1
            allfile.write(filepath.split(data_dir)[1]) 
            allfile.write('\n')
    print(f'=== {all_num} total samples! {speech_num} speech samples, {background_num} background samples! ===')     
        
#         all_command_files = glob.glob(os.path.join(sc_data_folder, '*/*wav'))
        
#         for entry in all_command_files:
#             pattern = re.compile(r"(.+\/)?(\w+)\/([^_]+)_.+wav")
#             r = re.match(pattern, entry)
#             if r:
#                 label, uid = r.group(2), r.group(3)
#                 if label == '_background_noise_':
#                     continue
#                 allfile.write(filepath.split(data_dir)[1]) 
#             allfile.write('\n')
    print("Finished split and write to file by class !")
    
def split_train_val_test(data_dir, to_split_file, test_size, val_size):
    file_type = to_split_file.split('.txt')[0] 
    with open(os.path.join(data_dir, to_split_file), 'r') as allfile:
        X = allfile.read().splitlines() 
   
    X_train, X_test = train_test_split(X, test_size = test_size, random_state=1)
    val_size_tmp = val_size / (1-test_size)
    X_train, X_val = train_test_split(X_train, test_size = val_size_tmp, random_state=1)
    
    with open(os.path.join(data_dir, file_type + "_training_list.txt"), "w") as outfile:
        outfile.write("\n".join(X_train))
    with open(os.path.join(data_dir, file_type + "_testing_list.txt") , "w") as outfile:
        outfile.write("\n".join(X_test))
    with open(os.path.join(data_dir, file_type + "_validation_list.txt"), "w") as outfile:
        outfile.write("\n".join(X_val))

    print(f'Overall: {len(all_file_list)}, Train: {len(X_train)}, Validatoin: {len(X_val)}, Test: {len(X_test)}')
    print(f"Finished split train, val and test for {to_split_file}. Write to files !")
    
    
def write_manifest(data_dir, out_dir, files, manifest_name, duration_stride=1.0, duration_max=None, filter_long=False, duration_limit=10.0):
    seg_num = 0
    skip_num = 0
    if duration_max is None:
        duration_max = 1e9
    
    if not os.path.exists(out_dir):
        print(f'outdir {out_dir} does not exist. Creat directory.')
        os.mkdir(out_dir)
    
    output_path = os.path.join(out_dir, manifest_name + '.json')
    with open(output_path, 'w') as fout:
        for file in files:
            clsname, filename = file.split('/')

            clsname = normalize_str(clsname)
            label = None

            filepath = os.path.join(data_dir, file)
            
            if clsname in background_classes:
                label = 'background'
#             elif clsname in speech_classes:
#                 label = 'speech_freesound'
            else:
                label = 'speech'
            
            try:
                x, _sr = librosa.load(filepath, sr=sr)
                duration = librosa.get_duration(x, sr=sr)
                
            except Exception:
                print(f"\n>>>>>>>>> WARNING: Librosa failed to load file {filepath}. Skipping this file !\n")
                continue
                
            if filter_long and duration > duration_limit:
                print(f"Skipping sound sample {filepath}, exceeds duration limit of {duration_limit}")
                skip_num += 1
                continue
            
            offsets = []
            durations = []

            if duration > duration_max:
                current_offset = 0.0

                while current_offset < duration:
                    difference = duration - current_offset
                    segment_duration = min(duration_max, difference)

                    offsets.append(current_offset)
                    durations.append(segment_duration)

                    current_offset += duration_stride

            else:
                offsets.append(0.0)
                durations.append(duration)


            for duration, offset in zip(durations, offsets):
                metadata = {
                    'audio_filepath': filepath,
                    'duration': duration,
                    'label': label,
                    'text': '_',  # for compatibility with ASRAudioText
                    'offset': offset,
                }
                
                json.dump(metadata, fout)
                fout.write('\n')
                fout.flush()
                
                seg_num += 1
#             print(f"Wrote {len(durations)} segments for filename {filename} with label {label}")
            
    print(f"=== Finished preparing manifest ! Skip {skip_num} samples ===")
    print(f'=== Writing {seg_num} to {output_path} ===' )
    return skip_num, seg_num


def load_list_write_manifest(data_dir, out_dir, filename, prefix, duration_stride = 1.0, duration_max = 1.0):
    
    file_path = os.path.join(data_dir + filename)
    with open(os.path.join(data_dir, file_path), 'r') as allfile:
        files = allfile.read().splitlines() 

    skip_num, seg_num = write_manifest(data_dir, out_dir, files, 
                                       prefix + '_' + filename.split('_list.txt')[0] + '_manifest',
                                       duration_stride, duration_max, filter_long=True, duration_limit=100.0)   
    return skip_num, seg_num

def combine_test_set(manifest_to_combine, fout_path):
    num = 0
    list_to_combine = manifest_to_combine.split(",")
    with open(fout_path, 'a') as fout:
        for filepath in list_to_combine:
            print(filepath)
            for line in open(filepath, 'r'):
                data = json.loads(line)
                json.dump(data, fout)
                fout.write('\n')
                fout.flush()
                num += 1
    return num

def get_clean_max_json(data_dir , out_dir, sc_data_json, max_limit, prefix):

    data = []
    sc_seg = 0
    filepath = os.path.join(data_dir, sc_data_json)
    for line in open(filepath, 'r'):
        data.append(json.loads(line))    
    fout_path = os.path.join(out_dir, prefix + "_" + sc_data_json)

    for i in data:
        if sc_seg <= max_limit:
            with open(fout_path, 'a') as fout:
                sc_seg += 1
                if 'label' not in i:
                    i['label'] = 'commands'
                
                if 'command' in i:
                    del(i['command'])
                json.dump(i, fout)
                fout.write('\n')
                fout.flush()
        else:
            break
    print(f'Get {sc_seg}/{max_limit} speech command to {fout_path} from speech commands')
    
    
def process_google_speech_train(data_dir):

    # TODO filter out _background_noise_
    files = sorted(glob.glob(data_dir + '/*/*.wav'))
    short_files = [i.split(data_dir)[1] for i in files]
    
    with open(os.path.join(data_dir, 'testing_list.txt'), 'r') as allfile:
        testing_list = allfile.read().splitlines() 
    
    with open(os.path.join(data_dir, 'validation_list.txt'), 'r') as allfile:
        validation_list = allfile.read().splitlines()  

    exist_set = set(testing_list).copy()
    exist_set.update(set(validation_list))
    
    
    training_list = [i for i in short_files if i not in exist_set]
    
    with open(os.path.join(data_dir,  "training_list.txt"), "w") as outfile:
        outfile.write("\n".join(training_list))

    print(f'Overall: {len(files)}, Train: {len(training_list)}, Validatoin: {len(validation_list)}, Test: {len(testing_list)}')
    
    
def infer_single_by_manifest(filepath, out_dir, manifest_name, duration_stride, step):
    
    if not os.path.exists(out_dir):
        print(f'outdir {out_dir} does not exist. Creat directory.')
        os.mkdir(out_dir)
    
    output_path = os.path.join(out_dir, manifest_name + '.json')
    with open(output_path, 'w') as fout:       
        try:
            x, _sr = librosa.load(filepath, sr=sr)
            duration = librosa.get_duration(x, sr=sr)

        except Exception:
            print(f"\n>>>>>>>>> WARNING: Librosa failed to load file {filepath}. Skipping this file !\n")


        offsets = []
        durations = []

        if duration < duration_stride:
            offsets.append(0.0)
            durations.append(duration)
        else:
            
            current_offset = 0.0
            while current_offset < duration:
                difference = duration - current_offset
                segment_duration = min(duration_stride, difference)
                offsets.append(current_offset)
                durations.append(segment_duration)
                current_offset += step
                
        seg_num = 0
        for duration, offset in zip(durations, offsets):
            metadata = {
                'audio_filepath': filepath,
                'duration': duration,
                'label': 'commands', ###
                'text': '_',  # for compatibility with ASRAudioText
                'offset': offset,
            }

            json.dump(metadata, fout)
            fout.write('\n')
            fout.flush()
            seg_num+= 1
#             print(f"Wrote {len(durations)} segments for filename {filename} with label {label}")

    print(f'=== Writing {seg_num} to {output_path} ===' )
    
    


def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def make_noisy_speech(clean_speech_file, noise_file, 
                      clean_speech_dir, noise_dir,  out_dir, 
                      min_snr_level = 1, max_snr_level = 20,
                      snr = None):


    subdir, filename = clean_speech_file.split('/')
    out_subdir = os.path.join(out_dir, subdir)

    
    clean_speech_file = os.path.join(clean_speech_dir, clean_speech_file)
    noise_file = os.path.join(noise_dir, noise_file)
    
    
    if not snr:
        snr = random.randint(min_snr_level, max_snr_level)
        
    output_mixed_filename = filename.split('.wav')[0] + '_n' + str(snr) + '.wav'
    output_mixed_file = os.path.join(out_subdir, output_mixed_filename)
    
    clean_y, clean_sr = librosa.load(clean_speech_file)
    noise_y, noise_sr = librosa.load(noise_file)
    
   
    
    # if noise is too short, we concatenate it self (several times) to match with clean audio.
    while len(noise_y) < 2 * len(clean_y):
        print(f'Duplicate noise for {noise_file}')
        noise_y = np.concatenate((noise_y, noise_y), axis=0)

    start = random.randint(0, len(noise_y)-len(clean_y))
    divided_noise_y = noise_y[start: start + len(clean_y)]
    
    clean_rms = np.sqrt(np.mean(np.square(clean_y), axis=-1))
    noise_rms = np.sqrt(np.mean(np.square(noise_y), axis=-1))
    
#     noise_rms = librosa.feature.rms(y = divided_noise_y)
    

    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr) 
    adjusted_noise_y = divided_noise_y * (adjusted_noise_rms / noise_rms) 
    mixed_y = (clean_y + adjusted_noise_y)
    
    librosa.output.write_wav(output_mixed_file, mixed_y, clean_sr)
    
    return output_mixed_file, snr

def main():
    parser = argparse.ArgumentParser(description='Google Speech Command Data download')
    parser.add_argument("--data_root", required=True, default=None, type=str)
    parser.add_argument('--data_version', required=True, default=1, type=int, choices=[1, 2])
    parser.add_argument('--rebalance', required=False, action='store_true')
    parser.add_argument('--log', required=False, action='store_true')
    parser.set_defaults(log=False, rebalance=False)
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=logging.DEBUG)

    data_root = args.data_root
    data_set = "google_speech_recognition_v{0}".format(args.data_version)
    data_folder = os.path.join(data_root, data_set)

    logging.info(f"Working on: {data_set}")

    if args.data_version == 1:
        URL = URL_v1
    else:
        URL = URL_v2

    # Download and extract
    if not os.path.exists(data_folder):
        file_path = os.path.join(data_root, data_set + ".tar.bz2")
        logging.info(f"Getting {data_set}")
        __maybe_download_file(file_path, URL)
        logging.info(f"Extracting {data_set}")
        __extract_all_files(file_path, data_root, data_folder)

    logging.info(f"Processing {data_set}")
    __process_data(data_folder, data_folder, rebalance=args.rebalance)
    logging.info('Done!')