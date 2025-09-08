import json
import os,io
import webdataset as wds

from tqdm import tqdm
from PIL import Image

llava_pretrain_dir = '/datasets'

# Paths to the dataset files
json_file = os.path.join(llava_pretrain_dir, 'PATHTOJSON.json')
output = os.path.join(llava_pretrain_dir, 'wds')

if not os.path.exists(output):
    os.mkdir(output)

# Load data
with open(json_file, 'r') as f:
    data = json.load(f)

with wds.ShardWriter(os.path.join(output, 'pretrain-%d.tar'), maxcount=10000) as shard_writer:
    for entry in tqdm(data):
        img_path = os.path.join(llava_pretrain_dir, entry["image"])
        if not os.path.exists(img_path):
            continue

        # 이미지 → jpg 바이트
        with Image.open(img_path) as im:
            # image_data = im
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG")
            image_data = buf.getvalue()
        sample = {
            "__key__": str(entry['id']),
            "jpg": image_data,
            "json": json.dumps(entry['conversations']).encode("utf-8"),
        }
        shard_writer.write(sample)

print(f"Dataset successfully converted to wds")

