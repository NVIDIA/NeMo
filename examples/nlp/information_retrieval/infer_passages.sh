for chunk_id in {0..9}
do
  python -m torch.distributed.launch --nproc_per_node=8 dpr_infer.py --data_dir /infer --batch_size 1024 --restore_path=/result/dpr_reranking/checkpoints/PassageBERT-STEP-$1.pt --data_for_eval passages --work_dir /result/dpr_reranking --chunk_id ${chunk_id}
done
