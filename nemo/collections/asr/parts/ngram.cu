using int64_t = long long int;

extern "C" __global__
void compute_scores_and_states(
        int64_t batch_size,
        int64_t vocab_size,
        int64_t *states,
        int64_t *new_states,
        float *out_scores,
        int64_t start_state,
        int64_t max_order,
        int64_t *backoff_to_states,
        float *backoff_weights,
        int64_t *state_start_arcs,
        int64_t *state_end_arcs,
        int64_t *to_states,
        int64_t *ilabels,
        float *arcs_weights) {
    int64_t batch_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_i >= batch_size) {
        return;
    }

    int64_t label_expand_factor = (vocab_size + blockDim.y - 1) / blockDim.y;
    assert(blockIdx.y == 0);
    int64_t label_block_i_start = threadIdx.y * label_expand_factor;

    bool done = false;
    int64_t state = states[batch_i];

    for (int i = 0; i < max_order; ++i) {
        // TODO: number of iterations - optimize?
        __syncthreads();
        for (int64_t label_block_i = label_block_i_start;
             label_block_i < min(label_block_i_start + label_expand_factor, vocab_size);
             ++label_block_i) {
            int64_t arc_index = state_start_arcs[state] + label_block_i;
            int64_t label_i = ilabels[arc_index];
            bool label_i_valid = label_block_i < vocab_size && arc_index < state_end_arcs[state];

            if (!done && label_i_valid && new_states[batch_i * vocab_size + label_i] == -1) {
                new_states[batch_i * vocab_size + label_i] = to_states[arc_index];
                out_scores[batch_i * vocab_size + label_i] += arcs_weights[arc_index];
            }
        }
        __syncthreads();
        done |= state == start_state;
        if (!done) {
            for (int64_t label_block_i = label_block_i_start;
                 label_block_i < min(label_block_i_start + label_expand_factor, vocab_size);
                 ++label_block_i) {
                if (new_states[batch_i * vocab_size + label_block_i] == -1) {
                    out_scores[batch_i * vocab_size + label_block_i] += backoff_weights[state];
                }
            }
            state = backoff_to_states[state];
        }
    }
}
