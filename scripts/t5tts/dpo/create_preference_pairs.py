import argparse
import os
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_manifest", type=str, default="/Data/testing_240subset.json")
parser.add_argument("--generated_audio_dir", type=str, default="/Data/Experiments/NewT5TTSDPO/Debug/Generations/T5TTS/version_0/audios")
parser.add_argument("--group_size", type=int, default=4)
parser.add_argument("--cer_threshold", type=float, default=0.01)
parser.add_argument("--val_size", type=int, default=64)
args = parser.parse_args()


def read_records(manifest_path):
    with open(manifest_path, 'r') as f:
        lines = f.readlines()
        records = []
        for line in lines:
            records.append(json.loads(line.strip()))
    return records

def write_records(fp, records):
    with open(fp, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print("Wrote {} records to: {}".format(len(records), fp))

def find_audio_files(directory):
    audio_files = []
    unique_ranks = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                rank_num = int(file.split("Rank")[1].split("_")[0])
                unique_ranks[rank_num] = True
                audio_num = int(file.split(".wav")[0].split("_")[-1])
                audio_files.append( (rank_num, audio_num, os.path.join(root, file)) )
    ranked_audio_files = []
    for af in audio_files:
        rank, num, path = af
        audio_num = num * len(unique_ranks) + rank
        ranked_audio_files.append((audio_num, path))
    ranked_audio_files = sorted(ranked_audio_files, key=lambda x: x[0])
    ranked_audio_files = [x[1] for x in ranked_audio_files]
    ranked_codec_files = [f.replace(".wav", "_codes.pt") for f in ranked_audio_files]
    metric_files = [f.replace(".wav", "_metrics.json") for f in ranked_audio_files]

    return ranked_audio_files, ranked_codec_files, metric_files

def create_chosen_rejected_records(records, group_size=6):
    assert len(records) % group_size == 0
    num_groups = len(records) // group_size
    best_records = []
    worst_records = []
    num_skipped = 0
    for gidx in range(num_groups):
        gsi = gidx * group_size
        gei = (gidx + 1) * group_size
        group = records[gsi:gei]
        
        cer_sim_indices = []
        for sidx, record in enumerate(group):
            if record['pred_transcript'] == "<INVALID>":
                print(f"Skipping group starting at index {gsi} due to invalid entries.")
                num_skipped += len(group)
                continue
            cer_sim_indices.append((record['cer_gts'], -record['pred_context_similarity'], sidx))
        cer_sim_indices = sorted(cer_sim_indices)
        best_record = group[cer_sim_indices[0][2]]
        worst_record = group[cer_sim_indices[-1][2]]
        best_record['reward'] = 1
        if worst_record['pred_context_similarity'] > best_record['pred_context_similarity']:
            reward_delta = (worst_record['cer_gts'] - best_record['cer_gts'])
        else:
            reward_delta = (worst_record['cer_gts'] - best_record['cer_gts']) + (best_record['pred_context_similarity'] - worst_record['pred_context_similarity'])
        
        if not (reward_delta > 0):
            # Make sure reward delta is not negative
            print("Warning reward_delta is not positive", reward_delta)
            print(best_record, worst_record)
        
        reward_delta = max(0.001, reward_delta)
        worst_record['reward'] = 1.0 - reward_delta
        best_records.append(best_record)
        worst_records.append(worst_record)
    print(f"Skipped {num_skipped} records due to invalid entries in associated groups.")    
    return best_records, worst_records

def filter_best_and_worst_records(best_records, worst_records, cer_threshold=0.02):
    ridx = 0
    filtered_best_records = []
    filtered_worst_records = []
    best_cer_avg = 0.0
    worst_cer_avg = 0.0
    skipped_records = 0
    while ridx < len(best_records):
        # print(ridx, len(best_records))
        best_record = best_records[ridx]
        if best_record['cer_gts'] < cer_threshold:
            worst_record = worst_records[ridx]
            if (worst_record['duration'] > 19.0 or best_record['duration'] > 19.0) or (worst_record['duration'] < 1.5 or best_record['duration'] < 1.5):
                skipped_records += 1
                print("Skipping record with answer duration > 20.0", ridx, skipped_records)
                ridx += 1
                continue
            assert best_record['cer_gts'] <= worst_record['cer_gts']
            if worst_record['cer_gts'] == best_record['cer_gts']:
                assert worst_record['pred_context_similarity'] <= best_record['pred_context_similarity']
            
            filtered_best_records.append(best_record)
            filtered_worst_records.append(worst_record)
            best_cer_avg += best_record['cer_gts']
            worst_cer_avg += worst_record['cer_gts']
        ridx += 1
    
    best_cer_avg /= len(filtered_best_records)
    worst_cer_avg /= len(filtered_worst_records)
    print(f"Best CER avg: {best_cer_avg}, Worst CER avg: {worst_cer_avg}")
    return filtered_best_records, filtered_worst_records

records = read_records(args.input_manifest)
audio_files, codec_files, metric_files = find_audio_files(args.generated_audio_dir)
assert len(records) <= len(audio_files), "Mismatch between number of records and number of generated audio files" # For multi-node, it generates more audio than records

for idx, record in tqdm(enumerate(records)):
    record['audio_filepath'] = audio_files[idx]
    record['target_audio_codes_path'] = codec_files[idx]
    with open(metric_files[idx], 'r') as f:
        metrics = json.load(f)
        record['duration'] = metrics['duration']
        record['cer_gts'] = metrics['cer_gt']
        record['wer_gts'] = metrics['wer_gt']
        record['pred_context_similarity'] = metrics['spk_similarity']
        record['pred_transcript'] = metrics['pred_transcript']
        record['gt_transcript'] = metrics['gt_transcript']

out_manifest_dir = args.generated_audio_dir.replace("/audios", "/manifests")
if not os.path.exists(out_manifest_dir):
    os.makedirs(out_manifest_dir)

out_manifest = os.path.join(out_manifest_dir, "manifest_with_metrics.json")
write_records(out_manifest, records)

group_size = args.group_size
cer_threshold = args.cer_threshold
val_size = args.val_size

all_best_records, all_worst_records = create_chosen_rejected_records(records, group_size)
print("Len all_best_records: ", len(all_best_records))
print("Len all_worst_records: ", len(all_worst_records))
best_records, worst_records = filter_best_and_worst_records(all_best_records, all_worst_records, args.cer_threshold)
print("Len filtered best_records: ", len(best_records))
print("Len filtered worst_records: ", len(worst_records))

ridx = 0
final_records = []
while ridx + 1 < len(best_records):
    best_record1 = best_records[ridx]
    best_record2 = best_records[ridx+1]
    worst_record1 = worst_records[ridx]
    worst_record2 = worst_records[ridx+1]
    assert best_record1['reward'] == 1
    assert best_record2['reward'] == 1
    assert worst_record1['reward'] < 1
    assert worst_record2['reward'] < 1
    final_records.append(best_record1)
    final_records.append(worst_record1)
    final_records.append(best_record2)
    final_records.append(worst_record2)
    ridx += 2

final_records_val = final_records[:val_size]
final_records_train = final_records[val_size:]

train_manifest = os.path.join(out_manifest_dir, "dpo_train_manifest.json")
val_manifest = os.path.join(out_manifest_dir, "dpo_val_manifest.json")

write_records(train_manifest, final_records_train)
write_records(val_manifest, final_records_val)
