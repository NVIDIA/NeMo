PARTITION=${1:-"GH_DVT_CG1"}
# PARTITION=GH_EVT_CG1

srun \
-p $PARTITION \
-A gracehopper \
-N 1 \
--exclusive \
--ntasks-per-node=1 \
--container-image /home/scratch.guyueh_sw/2023su/cpu_offload/nanz+ampere-bringup+llm_ranger+gitlab_8718727-devel-arm64.sqsh \
--container-mounts="/home/scratch.guyueh_sw:/home/scratch.guyueh_sw" \
-t 01:00:0 --pty bash