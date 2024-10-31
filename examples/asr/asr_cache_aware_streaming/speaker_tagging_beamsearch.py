import hydra
from typing import List, Optional
from dataclasses import dataclass, field
import kenlm
from beam_search_utils import (
    SpeakerTaggingBeamSearchDecoder,
    load_input_jsons,
    load_reference_jsons,
    run_mp_beam_search_decodin,
    convert_nemo_json_to_seglst,
)
from hydra.core.config_store import ConfigStore
from hyper_optim import (
    optuna_hyper_optim, 
    evaluate,
    evaluate_diff,
)


@dataclass
class RealigningLanguageModelParameters:
    # Beam search parameters
    batch_size: int = 32
    use_mp: bool = True
    input_error_src_list_path: Optional[str] = None
    groundtruth_ref_list_path: Optional[str] = None
    arpa_language_model: Optional[str] = None
    word_window: int = 32
    port: List[int] = field(default_factory=list)
    parallel_chunk_word_len: int = 250
    use_ngram: bool = True
    peak_prob: float = 0.95
    limit_max_spks: int = 2
    alpha: float = 0.5
    beta: float = 0.05
    beam_width: int = 16
    out_dir: Optional[str] = None

    # Optuna parameters
    hyper_params_optim: bool = False
    optuna_n_trials: int = 200
    workspace_dir: Optional[str]  = None
    asrdiar_file_name: Optional[str]  = None
    storage: Optional[str] = "sqlite:///optuna-speaker-beam-search.db"
    optuna_study_name: Optional[str] = "speaker_beam_search"
    output_log_file: Optional[str] = None
    temp_out_dir: Optional[str] = None

cs = ConfigStore.instance()
cs.store(name="config", node=RealigningLanguageModelParameters)

@hydra.main(config_name="config", version_base="1.1")
def main(cfg: RealigningLanguageModelParameters) -> None:
    __INFO_TAG__ = "[INFO]"
    trans_info_dict = load_input_jsons(input_error_src_list_path=cfg.input_error_src_list_path, peak_prob=float(cfg.peak_prob))
    reference_info_dict  = load_reference_jsons(reference_seglst_list_path=cfg.groundtruth_ref_list_path)
    source_info_dict = load_reference_jsons(reference_seglst_list_path=cfg.input_error_src_list_path)

    # Load ARPA language model in advance 
    loaded_kenlm_model = kenlm.Model(cfg.arpa_language_model)
    speaker_beam_search_decoder = SpeakerTaggingBeamSearchDecoder(loaded_kenlm_model=loaded_kenlm_model, cfg=cfg)
    
    if cfg.hyper_params_optim:
        print(f"{__INFO_TAG__} Optimizing hyper-parameters...")
        cfg = optuna_hyper_optim(cfg=cfg,
                                speaker_beam_search_decoder=speaker_beam_search_decoder,
                                loaded_kenlm_model=loaded_kenlm_model,
                                org_trans_info_dict=trans_info_dict,
                                source_info_dict=source_info_dict,
                                reference_info_dict=reference_info_dict, 
                                ) 
        
        __INFO_TAG__ = f"{__INFO_TAG__} Optimized hyper-parameters - "
    else:
        div_trans_info_dict = speaker_beam_search_decoder.divide_chunks(trans_info_dict=trans_info_dict, 
                                                                        win_len=cfg.parallel_chunk_word_len, 
                                                                        word_window=cfg.word_window,
                                                                        limit_max_spks=cfg.limit_max_spks,
                                                                        port=cfg.port,)
        result_trans_info_dict = run_mp_beam_search_decoding(speaker_beam_search_decoder, 
                                                loaded_kenlm_model=loaded_kenlm_model,
                                                div_trans_info_dict=div_trans_info_dict, 
                                                org_trans_info_dict=trans_info_dict, 
                                                div_mp=True,
                                                win_len=cfg.parallel_chunk_word_len,
                                                word_window=cfg.word_window,
                                                limit_max_spks=cfg.limit_max_spks,
                                                port=cfg.port,
                                                use_ngram=cfg.use_ngram,
                                                )
        hypothesis_sessions_dict = convert_nemo_json_to_seglst(result_trans_info_dict) 

        evaluate(cfg, 
                 cfg.out_dir, 
                 cfg.asrdiar_file_name, 
                 source_info_dict, 
                 hypothesis_sessions_dict, 
                 reference_info_dict
                 )
        
        evaluate_diff(cfg, 
                 cfg.out_dir, 
                 cfg.asrdiar_file_name, 
                 source_info_dict, 
                 hypothesis_sessions_dict, 
                 reference_info_dict
                 )

        print(f"{__INFO_TAG__} Parameters used: \
                \n ALPHA: {cfg.alpha} \
                \n BETA: {cfg.beta} \
                \n BEAM WIDTH: {cfg.beam_width} \
                \n Word Window: {cfg.word_window} \
                \n Use Ngram: {cfg.use_ngram} \
                \n Chunk Word Len: {cfg.parallel_chunk_word_len} \
                \n SpeakerLM Model: {cfg.arpa_language_model}")

if __name__ == '__main__':
    main()