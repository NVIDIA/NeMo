python tests/collections/vlm/test_llava_next_train.py \
--devices=1 \
--max-steps=5 \
--experiment-dir=/tmp/nemo2_llava_next_results/${{ github.run_id }}
