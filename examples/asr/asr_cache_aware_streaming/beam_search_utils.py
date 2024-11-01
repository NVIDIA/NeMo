from tqdm import tqdm
from typing import Dict, List
from pydiardecode import build_diardecoder
import numpy as np
import copy 
import os
import json 
import concurrent.futures
import kenlm

__INFO_TAG__ = "[BeamSearchUtil INFO]"

class SpeakerTaggingBeamSearchDecoder:
    def __init__(self, loaded_kenlm_model: kenlm, cfg: dict):
        self.realigning_lm_params = cfg
        self.realigning_lm = self._load_realigning_LM(loaded_kenlm_model=loaded_kenlm_model)
        self._SPLITSYM = "@"

    def _load_realigning_LM(self, loaded_kenlm_model: kenlm):
        """
        Load ARPA language model for realigning speaker labels for words.
        """
        diar_decoder = build_diardecoder(
            loaded_kenlm_model=loaded_kenlm_model,
            kenlm_model_path=self.realigning_lm_params['arpa_language_model'], 
            alpha=self.realigning_lm_params['alpha'], 
            beta=self.realigning_lm_params['beta'],
            word_window=self.realigning_lm_params['word_window'],
            use_ngram=self.realigning_lm_params['use_ngram'],
        )
        return diar_decoder

    def realign_words_with_lm(self, word_dict_seq_list: List[Dict[str, float]], speaker_count: int = None, port_num=None) -> List[Dict[str, float]]:
        if speaker_count is None:
            spk_list = []
            for k, line_dict in enumerate(word_dict_seq_list):
                _, spk_label = line_dict['word'], line_dict['speaker']
                spk_list.append(spk_label)
        else:
            spk_list = [ f"speaker_{k}" for k in range(speaker_count)]

        realigned_list = self.realigning_lm.decode_beams(beam_width=self.realigning_lm_params['beam_width'],
                                                         speaker_list=sorted(list(set(spk_list))), 
                                                         beam_prune_logp=self.realigning_lm_params.beam_prune_logp,
                                                         word_dict_seq_list=word_dict_seq_list,
                                                         port_num=port_num)
        return realigned_list

    def beam_search_diarization(
        self,
        trans_info_dict: Dict[str, Dict[str, list]],
        port_num: List[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Match the diarization result with the ASR output.
        The words and the timestamps for the corresponding words are matched in a for loop.

        Args:

        Returns:
            trans_info_dict (dict):
                Dictionary containing word timestamps, speaker labels and words from all sessions.
                Each session is indexed by a unique ID.
        """
        for uniq_id, session_dict in tqdm(trans_info_dict.items(), total=len(trans_info_dict), disable=True):
            # print(f"{__INFO_TAG__} Processing session {uniq_id}")
            word_dict_seq_list = session_dict['words']
            output_beams = self.realign_words_with_lm(word_dict_seq_list=word_dict_seq_list, speaker_count=session_dict['speaker_count'], port_num=port_num)
            word_dict_seq_list = output_beams[0][2]
            trans_info_dict[uniq_id]['words'] = word_dict_seq_list
        return trans_info_dict

    def beam_search_diarization_single(
        self,
        # trans_info_dict: Dict[str, Dict[str, list]],
        word_dict_seq_list: List[Dict[str, float]],
        speaker_count: int,
        port_num: List[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Match the diarization result with the ASR output.
        The words and the timestamps for the corresponding words are matched in a for loop.

        Args:

        Returns:
            trans_info_dict (dict):
                Dictionary containing word timestamps, speaker labels and words from all sessions.
                Each session is indexed by a unique ID.
        """
        # print(f"{__INFO_TAG__} Processing session {uniq_id}")
        # word_dict_seq_list = trans_info_dict['words']
        output_beams = self.realign_words_with_lm(word_dict_seq_list=word_dict_seq_list, speaker_count=speaker_count, port_num=port_num)
        word_dict_seq_list = output_beams[0][2]
        return word_dict_seq_list
        # trans_info_dict['words'] = word_dict_seq_list
        # return trans_info_dict

    def merge_div_inputs(self, div_trans_info_dict, org_trans_info_dict, win_len=250, word_window=16, limit_max_spks=8):
        """
        Merge the outputs of parallel processing.
        """
        uniq_id_list = list(org_trans_info_dict.keys())
        sub_div_dict = {}
        for seq_id in div_trans_info_dict.keys():
            div_info = seq_id.split(self._SPLITSYM)
            uniq_id, sub_idx, total_count = div_info[0], int(div_info[1]), int(div_info[2])
            if uniq_id not in sub_div_dict:
                sub_div_dict[uniq_id] = [None] * total_count
            sub_div_dict[uniq_id][sub_idx] = div_trans_info_dict[seq_id]['words']

        processed_trans_info_dict = {}    
        for uniq_id in uniq_id_list:
            processed_trans_info_dict[uniq_id] = {'words': []}

            if uniq_id in sub_div_dict:
                for k, div_words in enumerate(sub_div_dict[uniq_id]):
                    if k == 0:
                        div_words = div_words[:win_len]
                    else:
                        div_words = div_words[word_window:]
                    processed_trans_info_dict[uniq_id]['words'].extend(div_words)

                org_trans_info_dict[uniq_id]['words'] = processed_trans_info_dict[uniq_id]['words']
            else:
                processed_trans_info_dict[uniq_id]['words'] = org_trans_info_dict[uniq_id]['words']
        return processed_trans_info_dict
        # return org_trans_info_dict
    
    def divide_chunks(self, trans_info_dict, win_len, word_window, limit_max_spks, port):
        """
        Divide word sequence into chunks of length `win_len` for parallel processing.    

        Args:
            trans_info_dict (_type_): _description_
            diar_logits (_type_): _description_
            win_len (int, optional): _description_. Defaults to 250.
        """
        if len(port) > 1:
            num_workers = len(port) 
        else:
            num_workers = 25
        div_trans_info_dict = {}
        for uniq_id in trans_info_dict.keys():
            
            uniq_trans = trans_info_dict[uniq_id]
            if 'status' in uniq_trans:
                del uniq_trans['status']
            if 'transcription' in uniq_trans:
                del uniq_trans['transcription']
            if 'sentences' in uniq_trans:
                del uniq_trans['sentences']
            word_seq = uniq_trans['words']
            num_spks = len(set([x['speaker'] for x in word_seq]))
            if num_spks > limit_max_spks:
                continue

            div_word_seq = [] 
            if win_len is None:
                win_len = int(np.ceil(len(word_seq)/num_workers))
            n_chunks = int(np.ceil(len(word_seq)/win_len))
            
            for k in range(n_chunks):
                div_word_seq.append(word_seq[max(k*win_len - word_window, 0):(k+1)*win_len])
            
            total_count = len(div_word_seq)
            for k, w_seq in enumerate(div_word_seq):
                seq_id = uniq_id + f"{self._SPLITSYM}{k}{self._SPLITSYM}{total_count}"
                div_trans_info_dict[seq_id] = dict(uniq_trans)
                div_trans_info_dict[seq_id]['words'] = w_seq
        return div_trans_info_dict

def run_mp_beam_search_decoding(
    speaker_beam_search_decoder, 
    loaded_kenlm_model, 
    div_trans_info_dict, 
    org_trans_info_dict, 
    div_mp, 
    win_len, 
    word_window, 
    limit_max_spks, 
    port=None, 
    use_ngram=False
    ):
    if len(port) > 1:
        port = [int(p) for p in port]
    if use_ngram:
        port = [None]
        num_workers = 24
    else:
        num_workers = len(port)
    uniq_id_list = sorted(list(div_trans_info_dict.keys() ))
    tp = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    count = 0
    print(f"{__INFO_TAG__} Number of unique chunks to process: {len(uniq_id_list)}")
    for uniq_id in uniq_id_list:
        print(f"{__INFO_TAG__} Running beam search decoding for {uniq_id}...")
        if port is not None:
            port_num = port[count % len(port)]    
        else:
            port_num = None
        count += 1
        uniq_trans_info_dict = {uniq_id: div_trans_info_dict[uniq_id]}
        futures.append(tp.submit(speaker_beam_search_decoder.beam_search_diarization, uniq_trans_info_dict, port_num=port_num))

    pbar = tqdm(total=len(uniq_id_list), desc="Running beam search decoding", unit="files")
    count = 0
    output_trans_info_dict = {}
    for done_future in concurrent.futures.as_completed(futures):
        count += 1
        pbar.update()
        output_trans_info_dict.update(done_future.result())
    pbar.close() 
    tp.shutdown()
    if div_mp:
        output_trans_info_dict = speaker_beam_search_decoder.merge_div_inputs(div_trans_info_dict=output_trans_info_dict, 
                                                                              org_trans_info_dict=org_trans_info_dict, 
                                                                              win_len=win_len, 
                                                                              word_window=word_window, 
                                                                              limit_max_spks=limit_max_spks)
    return output_trans_info_dict

def count_num_of_spks(json_trans_list):
    spk_set = set()
    for sentence_dict in json_trans_list:
        spk_set.add(sentence_dict['speaker'])
    speaker_map = { spk_str: idx for idx, spk_str in enumerate(spk_set)}
    return speaker_map

def add_placeholder_speaker_softmax(json_trans_list, peak_prob=0.94 ,max_spks=4): 
    nemo_json_dict = {}
    word_dict_seq_list = []
    if peak_prob > 1 or peak_prob < 0:
        raise ValueError(f"peak_prob must be between 0 and 1 but got {peak_prob}")
    speaker_map = count_num_of_spks(json_trans_list)
    base_array = np.ones(max_spks) * (1 - peak_prob)/(max_spks-1)
    stt_sec, end_sec = None, None
    for sentence_dict in json_trans_list:
        word_list = sentence_dict['words'].split()
        speaker = sentence_dict['speaker']
        for word in word_list:
            speaker_softmax = copy.deepcopy(base_array)
            speaker_softmax[speaker_map[speaker]] = peak_prob
            word_dict_seq_list.append({'word': word, 
                                    'start_time': stt_sec, 
                                    'end_time': end_sec, 
                                    'speaker': speaker_map[speaker], 
                                    'speaker_softmax': speaker_softmax}
                                    )
    nemo_json_dict.update({'words': word_dict_seq_list, 
                           'status': "success", 
                           'sentences': json_trans_list, 
                           'speaker_count': len(speaker_map), 
                           'transcription': None}
                        )
    return nemo_json_dict

def convert_nemo_json_to_seglst(trans_info_dict):
    seglst_seq_list = []
    seg_lst_dict, spk_wise_trans_sessions = {}, {}
    for uniq_id in trans_info_dict.keys():
        spk_wise_trans_sessions[uniq_id] = {}
        seglst_seq_list = []
        word_seq_list = trans_info_dict[uniq_id]['words']
        prev_speaker, sentence = None, ''
        for widx, word_dict in enumerate(word_seq_list):
            curr_speaker = word_dict['speaker']

            # For making speaker wise transcriptions
            word = word_dict['word']
            if curr_speaker not in spk_wise_trans_sessions[uniq_id]:
                spk_wise_trans_sessions[uniq_id][curr_speaker] = word
            elif curr_speaker in spk_wise_trans_sessions[uniq_id]:
                spk_wise_trans_sessions[uniq_id][curr_speaker] = f"{spk_wise_trans_sessions[uniq_id][curr_speaker]} {word_dict['word']}"

            # For making segment wise transcriptions
            if curr_speaker!= prev_speaker and prev_speaker is not None:
                seglst_seq_list.append({'session_id': uniq_id, 
                                        'words': sentence.strip(), 
                                        'start_time': 0.0,
                                        'end_time': 0.0,
                                        'speaker': prev_speaker, 
                })
                sentence = word_dict['word']
            else:
                sentence = f"{sentence} {word_dict['word']}"
            prev_speaker = curr_speaker

        # For the last word:
        # (1) If there is no speaker change, add the existing sentence and exit the loop
        # (2) If there is a speaker change, add the last word and exit the loop
        if widx == len(word_seq_list) - 1:
            seglst_seq_list.append({'session_id': uniq_id, 
                                    'words': sentence.strip(), 
                                    'start_time': 0.0,
                                    'end_time': 0.0,
                                    'speaker': curr_speaker, 
            })
        seg_lst_dict[uniq_id] = seglst_seq_list
    return seg_lst_dict

def load_input_jsons(input_error_src_list_path, ext_str=".seglst.json", peak_prob=0.94, max_spks=4):
    trans_info_dict = {}
    json_filepath_list = open(input_error_src_list_path).readlines()
    for json_path in json_filepath_list:
        json_path = json_path.strip()
        uniq_id = os.path.split(json_path)[-1].split(ext_str)[0]
        if os.path.exists(json_path):
            with open(json_path, "r") as file:
                json_trans = json.load(file)
        else:
            raise FileNotFoundError(f"{json_path} does not exist. Aborting.")
        nemo_json_dict = add_placeholder_speaker_softmax(json_trans, peak_prob=peak_prob, max_spks=max_spks)
        trans_info_dict[uniq_id] = nemo_json_dict
    return trans_info_dict

def load_reference_jsons(reference_seglst_list_path,  ext_str=".seglst.json"):
    reference_info_dict = {}
    json_filepath_list = open(reference_seglst_list_path).readlines()
    for json_path in json_filepath_list:
        json_path = json_path.strip()
        uniq_id = os.path.split(json_path)[-1].split(ext_str)[0]
        if os.path.exists(json_path):
            with open(json_path, "r") as file:
                json_trans = json.load(file)
        else:
            raise FileNotFoundError(f"{json_path} does not exist. Aborting.")
        json_trans_uniq_id = []
        for sentence_dict in json_trans:
            sentence_dict['session_id'] = uniq_id
            json_trans_uniq_id.append(sentence_dict)
        reference_info_dict[uniq_id] = json_trans_uniq_id
    return reference_info_dict 

def write_seglst_jsons(
    seg_lst_sessions_dict: dict, 
    input_error_src_list_path: str, 
    diar_out_path: str, 
    ext_str: str, 
    write_individual_seglst_jsons=True
    ):
    """
    Writes the segment list (seglst) JSON files to the output directory.

    Parameters:
        seg_lst_sessions_dict (dict): A dictionary containing session IDs as keys and their corresponding segment lists as values.
        input_error_src_list_path (str): The path to the input error source list file.
        diar_out_path (str): The path to the output directory where the seglst JSON files will be written.
        type_string (str): A string representing the type of the seglst JSON files (e.g., 'hyp' for hypothesis or 'ef' for reference).
        write_individual_seglst_jsons (bool, optional): A flag indicating whether to write individual seglst JSON files for each session. Defaults to True.

    Returns:
        None
    """
    total_infer_list = []
    total_output_filename = os.path.split(input_error_src_list_path)[-1].replace(".list", "")
    for session_id, seg_lst_list in seg_lst_sessions_dict.items():
        total_infer_list.extend(seg_lst_list)
        if write_individual_seglst_jsons:
            print(f"{__INFO_TAG__} Writing {diar_out_path}/{session_id}.seglst.json")
            with open(f'{diar_out_path}/{session_id}.seglst.json', 'w') as file:
                json.dump(seg_lst_list, file, indent=4)  # indent=4 for pretty printing

    print(f"{__INFO_TAG__} Writing {diar_out_path}/{session_id}.seglst.json")
    total_output_filename = total_output_filename.replace("src", ext_str).replace("ref", ext_str)
    write_fn = f"{diar_out_path}/{total_output_filename}.seglst.json"
    if os.path.exists(write_fn):
        print(f"{__INFO_TAG__} {write_fn} already exists. Deleting it.")
        os.remove(write_fn)
    with open(write_fn, 'w') as file:
        json.dump(total_infer_list, file, indent=4)  # indent=4 for pretty printing
