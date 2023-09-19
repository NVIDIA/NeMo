import argparse

def calculate_peak_memory(args):
    # model memory
    hidden_size = args.hidden_size
    if args.ffn_hidden_size == None:
        ffn_hidden_size = hidden_size * 4
    else:
        ffn_hidden_size = args.ffn_hidden_size
    model_parameter_count = (
        (
            (hidden_size * hidden_size * 3) + 
            (hidden_size * hidden_size) + 
            (hidden_size * ffn_hidden_size) +
            (ffn_hidden_size * hidden_size) 
        ) * args.num_layers
        + 
        (hidden_size * args.vocab_size)
    )
    
    weight_width, master_weight_width, grad_width, optim_state_width = 2, 4, 4, 8
    model_mem_size = model_parameter_count * (weight_width + master_weight_width + grad_width + optim_state_width)

    # activation memory
    assert args.activations_checkpoint_num_layers >= 0 and args.activations_checkpoint_num_layers <= args.num_layers
    assert args.cpu_offloading_num_layers >= 0 and args.cpu_offloading_num_layers <= args.num_layers

    assert not ((args.activations_checkpoint_num_layers > 0) and (args.cpu_offloading_num_layers > 0)), \
    "Currently memory modeling for both checkpointing and offloading is not supported."

    S, B = args.seq_length, args.micro_batch_size
    activation_elements_output_layer = S*B*hidden_size + S*B*args.vocab_size
    activation_elements_per_layer = (
        S*B*hidden_size +       # input ln
        S*B*hidden_size +       # qkv
        S*B*hidden_size*4 +     # self_attention
        S*B*hidden_size +       # attn_out
        S*B*hidden_size // 2 +  # for dropout
        S*B*hidden_size +       # pre mlp ln
        S*B*hidden_size +       # ffn1
        S*B*ffn_hidden_size +   # gelu
        S*B*ffn_hidden_size +   # ffn2
        S*B*hidden_size // 2    # for dropout
    )
    activation_elements_per_layer_with_opt = activation_elements_per_layer
    num_opt_layers = 0
    num_layers_with_no_activations = 0
    
    if args.cpu_offloading_num_layers > 0:
        if args.cpu_offloading_method == "group_async":
            num_opt_layers = max(0, args.cpu_offloading_num_layers - 2)
        else:
            num_opt_layers = args.cpu_offloading_num_layers
        num_layers_with_no_activations = 0
        regions = args.cpu_offloading_region.split(',')
        if 'transformer_layer' in regions:
            activation_elements_per_layer_with_opt = 0
        else:
            activation_elements_per_layer_with_opt = activation_elements_per_layer
            if 'ln' in regions:
                activation_elements_per_layer_with_opt -= 2 * S*B*hidden_size
            if 'ffn_act' in regions:
                activation_elements_per_layer_with_opt -= S*B*ffn_hidden_size
            if 'bias_dropout_add' in regions:
                activation_elements_per_layer_with_opt -= S*B*hidden_size
            if 'attn_fn' in regions:
                activation_elements_per_layer_with_opt -= S*B*hidden_size*4
            if 'qkv_proj' in regions:
                activation_elements_per_layer_with_opt -= S*B*hidden_size
            if 'out_proj' in regions:
                activation_elements_per_layer_with_opt -= S*B*hidden_size
            if 'ffn1' in regions:
                activation_elements_per_layer_with_opt -= S*B*hidden_size
            if 'ffn2' in regions:
                activation_elements_per_layer_with_opt -= S*B*ffn_hidden_size
    elif args.activations_checkpoint_num_layers >= 0:
        if args.activations_checkpoint_method == "block":
            num_opt_layers = args.activations_checkpoint_num_layers
            num_layers_with_no_activations = 0
        elif args.activations_checkpoint_method == "uniform":
            num_opt_layers = args.num_layers // args.activations_checkpoint_num_layers
            num_layers_with_no_activations = args.num_layers - num_opt_layers
        if args.activations_checkpoint_granularity == 'full':
            activation_elements_per_layer_with_opt = S*B*hidden_size
        else:
            activation_elements_per_layer_with_opt = activation_elements_per_layer

    activation_width = 2
    activation_mem_size = (
        activation_elements_per_layer * (args.num_layers - num_opt_layers - num_layers_with_no_activations) + 
        activation_elements_per_layer_with_opt * num_opt_layers + 
        activation_elements_output_layer
    ) * activation_width

    return ((model_mem_size + activation_mem_size) // 1024 // 1024, 
            model_mem_size // 1024 // 1024, 
            activation_mem_size // 1024 // 1024)

def main():
    parser = argparse.ArgumentParser()
    # default is a gpt-5B config
    # assume the settings of precision=bf16, megatron_amp_O2=true, use_flash_attention=true
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=4096)
    parser.add_argument('--ffn_hidden_size', type=int, default=None)
    parser.add_argument('--num_attention_heads', type=int, default=32)
    parser.add_argument('--seq_length', type=int, default=4096)
    parser.add_argument('--vocab_size', type=int, default=50304)
    parser.add_argument('--micro_batch_size', type=int, default=4)
    
    # activation checkpointing options
    parser.add_argument('--activations_checkpoint_granularity', default='full', choices=['selective', 'full'])
    parser.add_argument('--activations_checkpoint_method', default='block', choices=['block', 'uniform'])
    parser.add_argument('--activations_checkpoint_num_layers', type=int, default=0)

    # activation offloading options
    parser.add_argument('--cpu_offloading_num_layers', type=int, default=0)
    parser.add_argument('--cpu_offloading_region', default="transformer_layer")
    parser.add_argument('--cpu_offloading_method', default='group_sync', choices=['group_sync', 'group_async'])

    args = parser.parse_args()

    results = calculate_peak_memory(args)
    peak_memory_MiB, model_and_optimizer_MiB, activation_memory_MiB = results
    print(f"Peak memory {peak_memory_MiB} MiB")
    print(f"Model + optimizer memory {model_and_optimizer_MiB} MiB")
    print(f"Activation memory {activation_memory_MiB} MiB")


if __name__ == "__main__":
    main()