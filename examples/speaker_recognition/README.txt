To get embeddings from a pretrained checkpoint use the following structure:

for voxceleb trial files you can download 

voxceleb1_test_v2.txt from http://www.openslr.org/resources/49/voxceleb1_test_v2.txt and
veri_test2.txt from http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt 

Generate manifest files for voxceleb test files (refer $NeMo/scipts/scp_to_manifest.py for help) 

name=CKPT_NAME
python ./spkr_get_emb.py --batch_size=128 --exp_name=$name \
	--num_epochs=40 --model_config='./configs/<config_name>' \
	--eval_datasets '<manifest_path>/test_manifest.json' \
	--checkpoint_dir='./myExps/checkpoints/' --work_dir='./myExps'

Above cmd line call would create embeddings in work_dir/<embeddings>/<manifest_file>.npy and corresponding labels in npy format based on checkpoint name present in checkpoint_dir with supporting config file from configs directory. 

To get EER for voxceleb trial files run:

python voxceleb_eval.py --trial_file=<trial_file> --emb=<npy_embeddings> --manifest='manifest of trail file>'

voxceleb EER values:
Quartznet:
EER trial_files/veri_test2.txt 2.62%
EER trial_files/voxceleb1_test_v2.txt 2.76%

JASPER:
EER trial_files/veri_test2.txt 2.33%
EER trial_files/voxceleb1_test_v2.txt 2.52%

