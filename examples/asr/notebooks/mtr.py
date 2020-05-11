import time

from multiprocessing import Pool
import os
from process_vad_data import * 


clean_speech_file_list_path = '/home/fjia/data/google_dataset_v2/google_speech_recognition_v2/training_list.txt'
noise_file_list_path = '/home/fjia/data/freesound_resampled/background_training_list.txt'
clean_data_dir = '/home/fjia/data/google_dataset_v2/google_speech_recognition_v2/'
noise_dir = '/home/fjia/data/freesound_resampled/'
out_dir = '/home/fjia/data/google_dataset_v2/google_speech_recognition_v2_noisy/'

if not os.path.exists(out_dir):
    print('Create output dir for noisy output!')
    os.mkdir(out_dir)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

subdir = get_immediate_subdirectories(clean_data_dir)

for i in subdir:
    if not os.path.exists(os.path.join(out_dir, i)):
        os.mkdir(os.path.join(out_dir, i))
        
        

with open(clean_speech_file_list_path) as f:
    clean_speech_file_list = f.read().splitlines() 




with open(noise_file_list_path) as f:
    noise_file_list = f.read().splitlines() 
    

def process_one_file(clean_speech_file):
    noise_file = random.choice(noise_file_list)
     ## [TODO]Adjust SNR 0-30 from google paper
    output_mixed_file, snr = make_noisy_speech(clean_speech_file, noise_file, 
                                               clean_data_dir, noise_dir, out_dir, 0, 30, None) 
    output_mixed_file_shortpath =  output_mixed_file.split(out_dir)[1]
    
#     with open('/home/fjia/data/google_dataset_v2/google_speech_recognition_v2_noisy/training_list.txt', 'a') as f:
#         f.write(output_mixed_file_shortpath) 
#         f.write('\n')
     
    return output_mixed_file_shortpath


p = Pool(processes=18)
data = p.map(process_one_file, clean_speech_file_list)

p.close()
print(data)
# np.save('output', data)


with open('/home/fjia/data/google_dataset_v2/google_speech_recognition_v2_noisy/training_list.txt', 'a') as f:
    for i in data:
        f.write(i) 
        f.write('\n')
