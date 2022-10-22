import json
import random
from argparse import ArgumentParser
from os.path import join

from tqdm.auto import tqdm

parser = ArgumentParser(description="Prepare input for GIZA")
parser.add_argument("--input_manifest", required=True, type=str, help='Path to manifest file')
parser.add_argument("--output_name", type=str, required=True, help="Output file")
parser.add_argument("--out_dir", type=str, required=True, help="Path to output folder")
parser.add_argument("--giza_dir", type=str, required=True, help="Path to folder with GIZA++ binaries")
parser.add_argument("--mckls_binary", type=str, required=True, help="Path to mckls binary")

args = parser.parse_args()


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


with open(join(args.out_dir, "run.sh"), "w") as out:
    out.write("GIZA_PATH=\"" + args.giza_dir + "\"\n")
    out.write("MKCLS=\"" + args.mckls_binary + "\"\n")
    out.write("\n")
    out.write("${GIZA_PATH}/plain2snt.out src dst\n")
    out.write("${MKCLS} -m2 -psrc -c4 -Vsrc.classes opt >& mkcls1.log\n")
    out.write("${MKCLS} -m2 -pdst -c4 -Vdst.classes opt >& mkcls2.log\n")
    out.write("${GIZA_PATH}/snt2cooc.out src.vcb dst.vcb src_dst.snt > src_dst.cooc\n")
    out.write(
        "${GIZA_PATH}/GIZA++ -S src.vcb -T dst.vcb -C src_dst.snt -coocurrencefile src_dst.cooc -p0 0.98 -o GIZA++ >& GIZA++.log\n"
    )
    out.write("##reverse direction\n")
    out.write("${GIZA_PATH}/snt2cooc.out dst.vcb src.vcb dst_src.snt > dst_src.cooc\n")
    out.write(
        "${GIZA_PATH}/GIZA++ -S dst.vcb -T src.vcb -C dst_src.snt -coocurrencefile dst_src.cooc -p0 0.98 -o GIZA++reverse >& GIZA++reverse.log\n"
    )

test_data = read_manifest(args.input_manifest)

# extract just the text corpus from the manifest
ref_text = [data['text'] for data in test_data]
pred_text = [data['pred_text'] for data in test_data]

size = len(ref_text)

assert size == len(pred_text)

with open(args.output_name, "w", encoding="utf-8") as out:
    for i in range(size):
        ref = ref_text[i].replace(" ", "_")
        hyp = pred_text[i].replace(" ", "_")
        if hyp == "":
            continue
        ref_letters = list(ref)
        hyp_letters = list(hyp)

        out.write(" ".join(hyp_letters) + "\t" + " ".join(ref_letters) + "\n")
