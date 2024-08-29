from lhotse.recipes.librilight import _parse_utterance, _prepare_subset
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.cut import CutSet, Cut
from lhotse.utils import Pathlike, split_sequence
from lhotse.serialization import load_manifest_lazy_or_eager, load_manifest

import os
import shutil
import logging
from tqdm import tqdm
from functools import partial
from typing import Optional
from pathlib import Path
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed

def change_prefix(cut, old_prefix, new_prefix):
    old_path = cut.recording.sources[0].source
    new_path = old_path.replace(old_prefix, new_prefix)
    cut.recording.sources[0].source = new_path

    if ',' in cut.id:
        print(cut.id)

    # else:
    #     os.makedirs(output_path, exist_ok=True)

    return cut


def save_text_and_audio(cut: Cut, storage_path, t_id, **kwargs):
    spk_id = cut.id.split('/')[1]
    storage_subdir = f"{storage_path}/{spk_id}"
    os.makedirs(storage_subdir, exist_ok=True)

    text = cut.supervisions[0].custom["texts"][t_id]
    f_id = ','.join(cut.id.split('/'))
    with open(f"{storage_subdir}/{f_id}.lab", 'w') as f:
        f.write(text)

    if not os.path.exists(f"{storage_subdir}/{f_id}.wav"):
        cut.save_audio(
            storage_path=f"{storage_subdir}/{f_id}.wav",
            **kwargs
        )

    return cut

def save_texts_and_audios(
        cuts: CutSet,
        storage_path: Pathlike,
        t_id: int = 0,
        progress_bar: bool = True,
        num_jobs: Optional[int] = None,
        executor: Optional[Executor] = None,
        shuffle_on_split: bool = True,
        **kwargs,
    ):
    """ modify CutSet.save_audios() into save_texts(CutSet)
    Args:
        t_id: type of transcript.
                0: original transcript, cases and punctuations
                1: ASR predicted, upper_case without punctuations
    """
    from cytoolz import identity

    from lhotse.manipulation import combine

    # Pre-conditions and args setup
    progress = (
        identity  # does nothing, unless we overwrite it with an actual prog bar
    )
    if num_jobs is None:
        num_jobs = 1
    if num_jobs == 1 and executor is not None:
        logging.warning(
            "Executor argument was passed but num_jobs set to 1: "
            "we will ignore the executor and use non-parallel execution."
        )
        executor = None
        
    if executor is None and num_jobs == 1:
        if progress_bar:
            progress = partial(
                tqdm, desc="Storing audio and transcripts", total=len(cuts)
            )
        return CutSet.from_cuts(
            progress(
                save_text_and_audio(cut=cut, storage_path=storage_path, t_id=t_id, **kwargs)
                for cut in cuts
            )
        )

    # Parallel execution: prepare the CutSet splits
    cut_sets = cuts.split(num_jobs, shuffle=shuffle_on_split)

    # Initialize the default executor if None was given
    if executor is None:
        import multiprocessing

        # The `is_caching_enabled()` state gets transfered to
        # the spawned sub-processes implictly (checked).
        executor = ProcessPoolExecutor(
            max_workers=num_jobs,
            mp_context=multiprocessing.get_context("spawn"),
        )

    # Submit the chunked tasks to parallel workers.
    # Each worker runs the non-parallel version of this function inside.
    futures = [
        executor.submit(
            save_texts_and_audios,
            cs,
            storage_path=storage_path,
            t_id=t_id,
            # Disable individual workers progress bars for readability
            progress_bar=(i==0),
            **kwargs
        )
        for i, cs in enumerate(cut_sets)
    ]

    if progress_bar:
        progress = partial(
            tqdm,
            desc="Storing audio and transcripts (chunks progress)",
            total=len(futures),
            position=1,
        )

    cuts = combine(progress(f.result() for f in futures))
    return cuts


from textgrid import TextGrid, IntervalTier, PointTier, Interval, Point

def cut_to_interval_tier(cut, t_id=0):
    cut_st = cut.start
    cut_ed = cut_st + cut.duration
    assert len(cut.supervisions) == 1

    sup = cut.supervisions[0]
    sup_st = cut_st + sup.start
    sup_ed = sup_st + sup.duration
    assert cut_ed >= sup_ed

    cut_id = cut.id
    spk_id = cut.id.split('/')[1]
    # TODO: change name into cut.id, pre-pad fixed-length spk_id for MFA
    tier = IntervalTier(
        name=cut_id,
        minTime=cut_st,
        maxTime=cut_ed,
    )
    tier.add(
        minTime=sup_st,
        maxTime=sup_ed,
        mark=sup.custom["texts"][t_id],
    )
    return tier


def save_rec_audio_and_textgrid(
        rec_id,
        cuts,
        recs,
        rec_to_cuts,
        storage_path,
        t_id=0
    ):
    rec, cut_ids = recs[rec_id], rec_to_cuts[rec_id]
    rec_cuts = cuts.subset(cut_ids=cut_ids)
    tg = TextGrid(name=rec.id, minTime=0, maxTime=rec.duration)
    for rec_cut in rec_cuts:
        tier = cut_to_interval_tier(cut=rec_cut, t_id=t_id)
        tg.append(tier)

    cut_id = rec_id
    spk_id = rec_id.split('/')[1]
    # TODO: change name into cut.id, pre-pad fixed-length spk_id for MFA
    f_id = ','.join(rec.id.split('/'))
    tg.write(f"{storage_path}/{spk_id:0>6},{f_id}.TextGrid")

    audio_src = rec.sources[0].source
    audio_dst = f"{storage_path}/{spk_id:0>6},{f_id}.flac"
    os.symlink(audio_src, audio_dst)
    # audio_dst = f"{storage_path}/{f_id}.wav"
    # os.system(f"ffmpeg -i {audio_src} {audio_dst}")

    return tg

def save_recs_audios_and_textgrids(
        rec_ids,
        cuts,
        recs,
        rec_to_cuts,
        storage_path,
        t_id=0,
        progress_bar: bool = True,
        num_jobs: Optional[int] = None,
        executor: Optional[Executor] = None,
        shuffle_on_split: bool = True,
        **kwargs
    ):
    """ modify CutSet.save_audios() into save_texts(CutSet)
    Args:
        t_id: type of transcript.
                0: original transcript, cases and punctuations
                1: ASR predicted, upper_case without punctuations
    """
    from cytoolz import identity

    from lhotse.manipulation import combine

    # Pre-conditions and args setup
    progress = (
        identity  # does nothing, unless we overwrite it with an actual prog bar
    )
    if num_jobs is None:
        num_jobs = 1
    if num_jobs == 1 and executor is not None:
        logging.warning(
            "Executor argument was passed but num_jobs set to 1: "
            "we will ignore the executor and use non-parallel execution."
        )
        executor = None
        
    if executor is None and num_jobs == 1:
        if progress_bar:
            progress = partial(
                tqdm, desc="Storing textgrids", total=len(rec_ids)
            )
        return list(progress(
            save_rec_audio_and_textgrid(rec_id=rec_id, cuts=cuts, recs=recs, rec_to_cuts=rec_to_cuts, storage_path=storage_path, t_id=t_id)
            for rec_id in rec_ids
        ))


    # Parallel execution: prepare the CutSet splits
    rec_sets = split_sequence(rec_ids, num_splits=num_jobs, shuffle=shuffle_on_split)

    # Initialize the default executor if None was given
    if executor is None:
        import multiprocessing

        # The `is_caching_enabled()` state gets transfered to
        # the spawned sub-processes implictly (checked).
        executor = ProcessPoolExecutor(
            max_workers=num_jobs,
            mp_context=multiprocessing.get_context("spawn"),
        )

    # Submit the chunked tasks to parallel workers.
    # Each worker runs the non-parallel version of this function inside.
    futures = [
        executor.submit(
            save_recs_audios_and_textgrids,
            r_ids,
            cuts=cuts,
            recs=recs,
            rec_to_cuts=rec_to_cuts,
            storage_path=storage_path,
            t_id=t_id,
            # Disable individual workers progress bars for readability
            progress_bar=(i==0),
            **kwargs
        )
        for i, r_ids in enumerate(rec_sets)
    ]

    if progress_bar:
        progress = partial(
            tqdm,
            desc="Storing textgrids (chunks progress)",
            total=len(futures),
        )

    tgs = combine(progress(f.result() for f in futures))
    return tgs


def save_audios_and_textgrids(
        cuts,
        storage_path,
        **kwargs
    ):
    recs = {}
    rec_to_cuts = {}
    for cut in cuts:
        rec = cut.recording
        rec_id = rec.id
        if rec_id not in rec_to_cuts:
            assert len(rec.sources) == 1
            recs[rec_id] = rec
            rec_to_cuts[rec_id] = [cut.id]
        else:
            assert len(rec.sources) == 1
            assert recs[rec_id].sources[0].source == rec.sources[0].source
            assert recs[rec_id].duration == rec.duration
            rec_to_cuts[rec_id].append(cut.id)

    os.makedirs(storage_path, exist_ok=True)

    return save_recs_audios_and_textgrids(
        rec_to_cuts.keys(),
        cuts=cuts,
        recs=recs,
        rec_to_cuts=rec_to_cuts,
        storage_path=storage_path,
        **kwargs
    )


def get_subset_audio(subset, libriheavy_dir, old_prefix, librilight_dir, eval_dir):
    cuts: CutSet = load_manifest_lazy_or_eager(f"{libriheavy_dir}/libriheavy_cuts_{subset}.jsonl.gz", CutSet)
    cuts = cuts.filter(lambda c: ',' not in c.id)
    cuts = cuts.map(partial(change_prefix, old_prefix=old_prefix, new_prefix=librilight_dir))
    cuts = cuts.to_eager()

    for cut in tqdm(cuts):
        rec = cut.recording
        audio_src: Pathlike = rec.sources[0].source
        audio_dst: Pathlike = audio_src.replace(librilight_dir, eval_dir)
        os.makedirs(audio_dst.rsplit('/', 1)[0], exist_ok=True)
        shutil.copy2(audio_src, audio_dst)


if __name__ == "__main__":
    old_prefix = "download/librilight"
    librilight_dir = "/datasets/LibriLight"
    libriheavy_dir = "data/LibriHeavy"
    corpus_dir = f"/datasets/LibriLight_aligned/raw_data_cuts"

    copy_eval_audio = True
    parse_mfa_dir = True

    # collect used eval audios, in future case of not fully downloading LibriLight dataset subsets
    if copy_eval_audio:
        subsets = ["dev", "test_clean", "test_other"]
        for subset in subsets:
            get_subset_audio(subset=subset, libriheavy_dir=libriheavy_dir, old_prefix=old_prefix, librilight_dir=librilight_dir, eval_dir="/datasets/LibriLight_aligned/raw_eval_data")


    if parse_mfa_dir:
        # subsets = ["small"]
        subsets = ["medium"]
        # subsets = ["large", "test_clean_large", "tesst_other_large"]

        for subset in subsets:
            # can not lazily split with progress bar
            cuts: CutSet = load_manifest_lazy_or_eager(f"{libriheavy_dir}/libriheavy_cuts_{subset}.jsonl.gz", CutSet)
            cuts = cuts.filter(lambda c: ',' not in c.id)
            cuts = cuts.map(partial(change_prefix, old_prefix=old_prefix, new_prefix=librilight_dir))

            storage_path=f"{corpus_dir}/{subset}"
            cuts = cuts.to_eager()
            save_texts_and_audios(cuts=cuts, storage_path=storage_path, num_jobs=32)