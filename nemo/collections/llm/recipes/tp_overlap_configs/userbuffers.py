# Model configs: H100/70B/TP8/MBS1/SeqLen8K
userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192 = {
    'qkv_dgrad': {'method': 'bulk', 'num_sm': 4, 'cga_size': 2, 'set_sm_margin': 0},
    'qkv_wgrad': {'method': 'bulk', 'num_sm': 24, 'cga_size': 2, 'set_sm_margin': 0},
    'fc1_dgrad': {'method': 'bulk', 'num_sm': 2, 'cga_size': 2, 'set_sm_margin': 0},
    'fc1_wgrad': {'method': 'bulk', 'num_sm': 4, 'cga_size': 2, 'set_sm_margin': 0},
    'qkv_fprop': {'method': 'ring_exchange', 'aggregate': 0},
    'proj_dgrad': {'method': 'ring_exchange', 'aggregate': 0},
    'fc1_fprop': {'method': 'ring_exchange', 'aggregate': 0},
    'fc2_dgrad': {'method': 'ring_exchange', 'aggregate': 0},
    'proj_fprop': {'method': 'pipeline', 'num_sm': 24, 'cga_size': 2, 'num_splits': 4, 'set_sm_margin': 1},
    'fc2_fprop': {'method': 'pipeline', 'num_sm': 16, 'cga_size': 2, 'num_splits': 4, 'set_sm_margin': 1},
}

# Model configs: H100/70B/TP8/MBS1/SeqLen8K/FP8
userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192 = {
    'qkv_dgrad': {'method': 'bulk', 'num_sm': 4, 'cga_size': 2, 'set_sm_margin': 0},
    'qkv_wgrad': {'method': 'bulk', 'num_sm': 24, 'cga_size': 2, 'set_sm_margin': 0},
    'fc1_dgrad': {'method': 'bulk', 'num_sm': 2, 'cga_size': 2, 'set_sm_margin': 0},
    'fc1_wgrad': {'method': 'bulk', 'num_sm': 4, 'cga_size': 2, 'set_sm_margin': 0},
    'qkv_fprop': {'method': 'ring_exchange', 'aggregate': 0},
    'proj_dgrad': {'method': 'ring_exchange', 'aggregate': 0},
    'fc1_fprop': {'method': 'ring_exchange', 'aggregate': 0},
    'fc2_dgrad': {'method': 'ring_exchange', 'aggregate': 0},
    'proj_fprop': {
        'method': 'pipeline',
        'num_sm': 24,
        'cga_size': 2,
        'num_splits': 4,
        'set_sm_margin': 1,
        'fp8_buf': 1,
    },
    'fc2_fprop': {
        'method': 'pipeline',
        'num_sm': 16,
        'cga_size': 2,
        'num_splits': 4,
        'set_sm_margin': 1,
        'fp8_buf': 1,
    },
}
