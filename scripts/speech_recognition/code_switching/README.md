# Scripts for creation of synthetic code-switched data from monolingual sources
Follow the 2 steps listed below in order - 


1. Create the (intermediate) manifest file using `code_switching_manifest_creation.py`.  It's usage is as follows:

    `python code_switching_manifest_creation.py --manifest_language1 <absolute path of Language 1's manifest file> --manifest_language2 <absolute path of Language 2's manifest file> --manifest_save_path <absolute path to save the created manifest> --id_language1 <language code for language 1 (e.g. en)> --id_language2 <language code for language 2 (e.g. es)> --max_sample_duration_sec <maximum duration of generated sample in seconds> --min_sample_duration_sec <maximum duration of generated sample in seconds> --dataset_size_required_hrs <size of generated synthetic dataset required in hrs>`

2. Create the syntheic audio data and the corresponding manifest file using `code_switching_audio_data_creation.py` It's usage is as follows: 

    `python code_switching_audio_data_creation.py --manifest_path <absolute path to intermediate CS manifest generated in step 1> --audio_save_folder_path <absolute path to directory where you want to save the snythesized audios> --manifest_save_path <absolute path to save the created manifest> --audio_normalized_amplitude <sclaed normalized amplitude desired> --sample_beginning_pause_msec <pause to be added to the beginning of the generated sample in milli seconds> --sample_joining_pause_msec <pause to be added between segments while joining, in milli seconds> --sample_end_pause_msec <pause to be added to the end of the generated sample in milli seconds> `
