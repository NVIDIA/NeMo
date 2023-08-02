echo "unset all SLURM_, PMI_, PMIX_ Variables"
set -x
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done
set +x


mpirun --allow-run-as-root --oversubscribe --np 1 python tests/export/test_nemo_refitting.py \
    --existing_test_models True --min_gpus 1 --model_name LLAMA2-7B-base --tp_size 1 --pp_size 1

mpirun --allow-run-as-root --oversubscribe --np 4 python tests/export/test_nemo_refitting.py \
    --existing_test_models True --min_gpus 4 --model_name LLAMA2-7B-base --tp_size 1 --pp_size 4

mpirun --allow-run-as-root --oversubscribe --np 4 python tests/export/test_nemo_refitting.py \
    --existing_test_models True --min_gpus 4 --model_name LLAMA2-7B-base --tp_size 4 --pp_size 1

mpirun --allow-run-as-root --oversubscribe --np 4 python tests/export/test_nemo_refitting.py \
    --existing_test_models True --min_gpus 4 --model_name LLAMA2-7B-base --tp_size 2 --pp_size 2
