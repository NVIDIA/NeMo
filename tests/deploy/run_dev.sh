echo "unset all SLURM_, PMI_, PMIX_ Variables"
set -x
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done
set +x

py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_LLAMA2_7B_base_1gpu

# py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_LLAMA2_70B_base_2gpu
# py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_NV_GPT_8B_Base_4k_1gpu
# py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_NV_GPT_8B_QA_4k_1gpu
# py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_NV_GPT_8B_Chat_4k_SFT_1gpu
# py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_NV_GPT_8B_Chat_4k_RLHF_1gpu

# py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_LLAMA2_13B_base_1gpu
