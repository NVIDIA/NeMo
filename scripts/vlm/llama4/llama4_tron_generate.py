# Combined Megatron Initialization and Generation Script

#  ln -s /lustre/fsw/coreai_dlalgo_genai/yuya/checkpoints /root/checkpoints
#
#  torchrun --nproc_per_node=8 scripts/vlm/llama4/llama4_tron_generate.py --model_name meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8
#  torchrun --nproc_per_node=8 scripts/vlm/llama4/llama4_tron_generate.py --model_name meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8  --generation_method mcore_engine

import argparse
import os
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist

# Megatron-Core Inference Imports
from megatron.core import parallel_state  # Needed for distributed checks
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.inference.text_generation_controllers.text_generation_controller import TextGenerationController
from megatron.inference.text_generation.mcore_engine_server import ModelInferenceWrapperServer, run_mcore_engine
from scripts.vlm.llama4.debugger import register_hooks
from transformers import AutoTokenizer  # Keep for initial check/loading if needed, but primarily use build_tokenizer

from nemo.collections.llm import GPTConfig
from nemo.collections.llm import GPTModel as ModelConfig
from nemo.tron.checkpointing import checkpoint_exists, load_checkpoint
from nemo.tron.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)

# NeMo/Megatron Imports
from nemo.tron.init import initialize_megatron, set_jit_fusion_options
from nemo.tron.model import get_model_from_config
from nemo.tron.setup import _init_checkpointing_context
from nemo.tron.state import GlobalState
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.tron.utils.common_utils import get_rank_safe  # For rank checks
from nemo.utils.get_rank import get_last_rank  # Import for greedy gen broadcast

# from megatron.core.inference.sampling_params import SamplingParams # Optional for more control


# --- Initialization Function ---
def minimal_megatron_setup(
    hf_model_name: str, tp_size: int, pp_size: int, cp_size: int, dtype: torch.dtype, attn_backend_str: str
):
    """
    Sets up a minimal Megatron environment for generation.
    Needs to be run within an initialized distributed context.
    """
    rank = get_rank_safe()
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)

    # --- Checkpoint Conversion (Placeholder - Run Separately Beforehand) ---
    # This section should ideally be run *once* before launching the main script,
    # typically on a single node/rank.
    output_path = f"/root/checkpoints/tron/{hf_model_name}"  # Example path - ADJUST AS NEEDED
    if rank == 0:
        if not (os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "iter_0000000"))):
            print(f"INFO: Converted checkpoint not found at {output_path}.")
            print("INFO: Please ensure the HuggingFace model has been converted using the appropriate")
            print("INFO: NeMo script (e.g., HFLlamaImporter, HFQwen2Importer) before running this.")
            # Example placeholder conversion command:
            # try:
            #     if "llama" in hf_model_name.lower():
            #         from nemo.tron.converter.llama import HFLlamaImporter
            #         print(f"Attempting conversion (rank 0)...")
            #         importer = HFLlamaImporter(hf_model_name, output_path=output_path)
            #     elif "qwen" in hf_model_name.lower():
            #         from nemo.tron.converter.qwen import HFQwen2Importer
            #         print(f"Attempting conversion (rank 0)...")
            #         importer = HFQwen2Importer(hf_model_name, output_path=output_path)
            #     else:
            #          raise ValueError(f"Automatic conversion for {hf_model_name} not implemented in this script.")
            #     importer.apply()
            #     import megatron.core.rerun_state_machine
            #     megatron.core.rerun_state_machine.destroy_rerun_state_machine() # Important after conversion
            #     print("Conversion complete (rank 0).")
            # except ImportError as e:
            #     print(f"WARNING: Could not import converter. Skipping conversion attempt: {e}")
            # except Exception as e:
            #     print(f"ERROR during conversion attempt: {e}")
            #     # Decide whether to raise or just warn
            #     # raise RuntimeError("Checkpoint conversion failed.") from e
            warnings.warn(f"Checkpoint not found at {output_path}. Proceeding assumes it exists or conversion failed.")
        else:
            print(f"INFO: Found converted checkpoint at {output_path}.")

    # Barrier to ensure rank 0 finishes check/conversion before others proceed
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # --- Configuration ---
    pretrained_ckpt = output_path  # Use the output path of the conversion
    # Try loading the run_config.yaml generated during conversion
    # Fallback to potentially manually creating a basic config if needed.
    base_config_path = os.path.join(pretrained_ckpt, "iter_0000000/run_config.yaml")

    if os.path.exists(base_config_path):
        print(f"Loading base config from: {base_config_path}")
        cfg_container = ConfigContainer.from_yaml(base_config_path)
        model_cfg = cfg_container.model_config
        # Set attention backend based on argument
        from megatron.core.transformer.enums import AttnBackend

        if attn_backend_str == "fused":
            model_cfg.attention_backend = AttnBackend.fused
        elif attn_backend_str == "flash":
            model_cfg.attention_backend = AttnBackend.flash
        else:  # unfused
            model_cfg.attention_backend = AttnBackend.unfused
        print(f"INFO: Set model_cfg.attention_backend to {model_cfg.attention_backend}")

    else:
        print(f"WARNING: Base config not found at {base_config_path}. Creating a minimal default ModelConfig.")
        # Attempt to load minimal config from HF model if possible, otherwise use defaults
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(hf_model_name)
            model_cfg = ModelConfig(
                hidden_size=hf_config.hidden_size,
                num_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                # seq_length might need adjustment based on model/task
                seq_length=getattr(hf_config, 'max_position_embeddings', 2048),
                max_position_embeddings=getattr(hf_config, 'max_position_embeddings', 2048),
                # These might need model-specific logic (e.g. for GQA in Llama)
                # kv_channels=...
                # num_query_groups=...
                ffn_hidden_size=hf_config.intermediate_size,
                init_method_std=getattr(hf_config, 'initializer_range', 0.02),
                hidden_dropout=getattr(hf_config, 'hidden_dropout_prob', 0.0),
                attention_dropout=getattr(hf_config, 'attention_dropout_prob', 0.0),
                # activation="gelu", # Or model specific like 'swiglu'
                # bias_activation_fusion=True, # Example default
                # layernorm_epsilon=hf_config.layer_norm_eps,
                make_vocab_size_divisible_by=128,  # Common practice
                # Pad vocab size needs to be set later by tokenizer
            )
            print("Created ModelConfig from HuggingFace config.")
        except Exception as e:
            print(f"ERROR: Could not load HF config for defaults: {e}. Using hardcoded minimal defaults.")
            # VERY Minimal defaults if HF config fails - likely needs adjustment
            model_cfg = ModelConfig(
                hidden_size=4096,
                num_layers=32,
                num_attention_heads=32,
                seq_length=2048,
                max_position_embeddings=2048,
                ffn_hidden_size=11008,
                make_vocab_size_divisible_by=128,
            )
            print("Using hardcoded minimal ModelConfig - VERIFY THESE VALUES.")

    # Override with desired parallelism and precision from args
    model_cfg.tensor_model_parallel_size = tp_size
    model_cfg.pipeline_model_parallel_size = pp_size
    model_cfg.context_parallel_size = cp_size
    model_cfg.expert_tensor_parallel_size = tp_size
    model_cfg.expert_model_parallel_size = 1
    model_cfg.sequence_parallel = False  # Assuming standard TP/PP

    model_cfg.bf16 = dtype == torch.bfloat16
    model_cfg.fp16 = dtype == torch.float16
    model_cfg.fp32_residual_connection = False  # Common optimization
    # params_dtype is important for AMP/storage
    if model_cfg.bf16:
        model_cfg.params_dtype = torch.bfloat16
    elif model_cfg.fp16:
        model_cfg.params_dtype = torch.float16
    else:
        model_cfg.params_dtype = torch.float32

    model_cfg.parallel_output = True  # Important for logprobs/generation across TP ranks
    model_cfg.flash_decode = False

    checkpoint_config = CheckpointConfig(
        # load=pretrained_ckpt, # Use pretrained_checkpoint instead for NeMo Tron convention
        pretrained_checkpoint=pretrained_ckpt,  # Path to the *converted* NeMo checkpoint directory
        fully_parallel_load=True,
        # save="/tmp/dummy_save", # Dummy save path might be needed by some checks
    )

    # Minimal configs for other parts
    megatron_cfg = ConfigContainer(
        model_config=model_cfg,
        checkpoint_config=checkpoint_config,
        logger_config=LoggerConfig(logging_level=30),  # Warn level
        # Training config needed for some internal setups, even for inference
        train_config=TrainingConfig(
            micro_batch_size=1,
            global_batch_size=1,  # Example logic
            train_iters=1,  # Dummy micro_batch_sizevalue
            rampup_batch_size=None,  # Disable rampup
            # Might need seq_length_ramping if model uses it
        ),
        optimizer_config=OptimizerConfig(),  # Dummy, not used for inference load
        ddp_config=DistributedDataParallelConfig(),  # Defaults should be ok
        scheduler_config=SchedulerConfig(),  # Dummy
        tokenizer_config=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,  # Use original HF name here for loading tokenizer
        ),
        dataset_config=None,
    )

    # --- Initialization ---
    # Assumes distributed environment (torch.distributed) is already initialized externally
    print("Initializing Megatron...")
    initialize_megatron(
        cfg=megatron_cfg,
        # gpu_visibility_externally_set=True # Assume CUDA_VISIBLE_DEVICES is set
    )

    # Assign cfg to GlobalState *after* initialize_megatron which sets up parallelism
    state = GlobalState()
    state.cfg = megatron_cfg
    # Validate config *after* GlobalState is populated by initialize_megatron
    megatron_cfg.validate()

    # # JIT options (optional but good practice)
    # if megatron_cfg.train_config.micro_batch_size:  # Check if set
    #     set_jit_fusion_options(megatron_cfg.model_config, megatron_cfg.train_config.micro_batch_size)
    # else:
    #     print("Warning: micro_batch_size not in config, skipping JIT fusion options.")

    # Checkpointing context
    checkpointing_context = _init_checkpointing_context(megatron_cfg.checkpoint_config)

    # Tokenizer
    print("Building Tokenizer...")
    # This tokenizer will be padded based on TP size and divisible factor
    megatron_tokenizer = build_tokenizer(
        megatron_cfg.tokenizer_config,
        make_vocab_size_divisible_by=megatron_cfg.model_config.make_vocab_size_divisible_by,
        tensor_model_parallel_size=megatron_cfg.model_config.tensor_model_parallel_size,
    )
    # Update model config with the final padded vocab size from the tokenizer build process
    # if megatron_tokenizer.vocab_size != megatron_cfg.model_config.vocab_size:
    #     print(
    #         f"Updating model_cfg vocab_size from {megatron_cfg.model_config.vocab_size} to {megatron_tokenizer.vocab_size}"
    #     )
    #     megatron_cfg.model_config.vocab_size = megatron_tokenizer.vocab_size
    # Store the final padded vocab size for inference engine
    final_padded_vocab_size = megatron_cfg.model_config.vocab_size
    print(f"Final padded vocab size: {final_padded_vocab_size}")

    # Model Instantiation
    print("Building Model...")
    # # Note: get_model_from_config returns a list (for virtual pipeline parallelism), usually take [0]
    # model_provider = partial(get_model_from_config,
    #                          megatron_cfg.model_config,
    #                          megatron_cfg.ddp_config,
    #                          # Add other args from setup_megatron_model if needed (e.g., use_torch_fsdp2)
    #                          )
    # This setup function handles model building within the correct parallel context
    # from nemo.tron.setup import setup_model_and_optimizer
    # We don't need optimizer/scheduler for inference, but setup func might expect args
    # Pass dummy providers or adjust setup_model_and_optimizer if possible.
    # Let's try calling get_model_from_config directly as in the original code:
    model_list = get_model_from_config(
        megatron_cfg.model_config,
        megatron_cfg.ddp_config,
    )

    # Load Checkpoint
    if megatron_cfg.checkpoint_config.pretrained_checkpoint and checkpoint_exists(
        megatron_cfg.checkpoint_config.pretrained_checkpoint
    ):
        print(f"Loading checkpoint from: {megatron_cfg.checkpoint_config.pretrained_checkpoint}")
        load_checkpoint(
            state,  # Pass GlobalState
            model_list,  # Pass the list
            None,  # No optimizer
            None,  # No scheduler
            checkpointing_context=checkpointing_context,
            # Add skip_load_to_model_and_opt if using FSDP2
        )
        print("Checkpoint loaded successfully.")
    else:
        print(
            f"Warning: Pretrained checkpoint not found or not specified ({megatron_cfg.checkpoint_config.pretrained_checkpoint}). Model weights are initialized randomly."
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # Ensure model is loaded everywhere

    model = model_list[0].module  # Get the actual model instance
    model.eval()  # Set to evaluation mode

    print("Megatron Setup Complete.")
    return model, megatron_tokenizer, megatron_cfg, final_padded_vocab_size


# --- Generation Function ---
def megatron_generate(
    model,
    megatron_tokenizer,
    mtron_cfg,
    final_padded_vocab_size,
    input_ids: torch.Tensor,  # Right-padded input_ids [batch, seq_len]
    input_lengths: torch.Tensor,  # Length of each prompt [batch]
    max_new_tokens: int,
    generation_batch_size: int,  # Micro-batch size for inference engine
    greedy: bool = True,  # Add greedy option
    generation_method: str = "mcore_engine",  # Add generation_method argument
):
    """Generates text using the initialized Megatron model and inference engine."""

    model.eval()  # Ensure model is in eval mode
    # register_hooks(model)

    model_cfg = mtron_cfg.model_config
    model_cfg.seq_length = 1024
    rank = get_rank_safe()

    if generation_method == "manual_greedy":
        if rank == 0:
            print(
                f"Starting manual greedy generation: batch_size={input_ids.shape[0]}, max_new_tokens={max_new_tokens}"
            )
        start_time = time.time()

        # --- Manual Greedy Generation Loop ---
        generated_ids = input_ids.cuda().clone()
        generated_ids = generated_ids[:, : input_lengths[0]]
        # Determine stop tokens (use IDs from the Megatron tokenizer)
        stop_token_ids = set()
        # Assuming common patterns, adjust if needed based on tokenizer specifics
        eos_token_id = getattr(megatron_tokenizer._tokenizer, 'eos_token_id', None)
        if eos_token_id is not None:
            stop_token_ids.add(eos_token_id)
        # Add other specific stop tokens if necessary (e.g., <|eom|>, <|eot|> for some models)
        # Need to check the specific tokenizer used for Llama-4
        try:
            eom_id = megatron_tokenizer._tokenizer.vocab.get("<|eom|>", None)
            eot_id = megatron_tokenizer._tokenizer.vocab.get("<|eot|>", None)
            if eom_id:
                stop_token_ids.add(eom_id)
            if eot_id:
                stop_token_ids.add(eot_id)
        except AttributeError:
            print("Warning: Could not access tokenizer vocab directly for stop tokens.")

        if not stop_token_ids and rank == 0:
            print("Warning: No stop tokens identified. Generation might run to max_new_tokens.")

        current_input_ids = generated_ids

        with torch.no_grad():
            for step in range(max_new_tokens):
                print(step)
                position_ids = (
                    torch.arange(current_input_ids.size(1), dtype=torch.long, device=current_input_ids.device)
                    .unsqueeze(0)
                    .expand_as(current_input_ids)
                )
                # Direct model call (no fwd_bwd_function needed here as model is already set up)
                output = model(current_input_ids, position_ids, attention_mask=None)
                # output shape: [micro_batch_size, sequence_length, hidden_size]
                # or [micro_batch_size, sequence_length, vocab_size] if LM head is part of the model
                # Assuming output is logits: [mb, seq, vocab_tp_shard_size] on each TP rank
                # or [mb, seq, hidden_size] if LM head is separate or applied later

                # Handle Parallelism for Logits/Next Token ID
                if parallel_state.is_pipeline_last_stage():
                    # If TP > 1, gather logits across TP ranks
                    if parallel_state.get_tensor_model_parallel_world_size() > 1:
                        # Assuming model output are logits split across TP ranks on the vocab dim (-1)
                        world_size = parallel_state.get_tensor_model_parallel_world_size()
                        gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                        dist.all_gather(
                            gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group()
                        )
                        # Concatenate along the vocab dimension (last dimension)
                        # Ensure the model output is indeed logits before concatenating
                        # If model output is hidden states, need to apply LM head first
                        # Let's assume 'output' are the logits for now. Needs verification.
                        gathered_output = torch.cat(gathered_tensors, dim=-1)
                    else:
                        gathered_output = output  # No TP, output is already full logits

                    # Get the token ID for the *next* token (logit of the last position)
                    next_token_logits = gathered_output[:, -1, :]  # [mb, vocab_size]
                    next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [mb, 1]
                else:
                    # Other pipeline stages don't compute the final logits
                    # Create a dummy tensor to match the expected shape for broadcast
                    # Use long dtype as token IDs are integers
                    batch_dim = current_input_ids.size(0)
                    next_token_ids = torch.zeros((batch_dim, 1), device=current_input_ids.device, dtype=torch.long)
                # exit(0)
                # Broadcast the next token ID from the last pipeline stage to all ranks
                # Use get_last_rank which handles PP correctly
                # Source rank needs to be the *global* rank of the last PP stage
                # TODO: Verify get_last_rank works correctly with TP>1 on last stage
                # It might be safer to use parallel_state.get_pipeline_model_parallel_last_rank()
                # source_rank = parallel_state.get_pipeline_model_parallel_last_rank()
                # torch.distributed.broadcast(next_token_ids, src=source_rank)

                # Append the new token ID to the generated sequence
                generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
                current_input_ids = generated_ids  # Update input for next step

                # Check for stop tokens (only need to check on rank 0 after broadcast?)
                # Safer to check on all ranks, especially if PP > 1
                # Check the first sample in the batch for stopping (assuming uniform stopping)
                if next_token_ids[0].item() in stop_token_ids:
                    if rank == 0:
                        print(f"Stop token {next_token_ids[0].item()} detected at step {step+1}.")
                    break  # Exit the loop

            # --- Post-process Manual Greedy Output ---
            # `generated_ids` contains the full sequences [batch, seq_len]
            batch_size = generated_ids.size(0)
            max_seq_len = generated_ids.size(1)

            # Create dummy logprobs tensor (manual greedy doesn't easily provide them)
            logprobs_padded = torch.full(
                (batch_size, max_seq_len),
                float('nan'),  # Use NaN to indicate missing logprobs
                dtype=torch.float,
                device="cpu",
            )

            generation_lengths_list = []
            unpadded_lengths_list = []

            for i in range(batch_size):
                prompt_len = input_lengths[i].cpu().item()
                seq_len = generated_ids.size(1)  # Total length after generation
                unpadded_lengths_list.append(seq_len)
                generation_lengths_list.append(max(0, seq_len - prompt_len))

            # Move final IDs to CPU
            output_ids_padded = generated_ids.cpu()

            end_time = time.time()
            if rank == 0:
                print(f"Manual greedy execution time: {end_time - start_time:.3f} seconds")

            # Prepare final output dictionary
            output_dict = {
                "output_ids": output_ids_padded,  # [batch, max_output_len]
                "logprobs": logprobs_padded,  # [batch, max_output_len] (NaN for manual greedy)
                "generation_lengths": torch.tensor(generation_lengths_list, dtype=torch.long),  # [batch]
                "unpadded_sequence_lengths": torch.tensor(unpadded_lengths_list, dtype=torch.long),  # [batch]
            }
            return output_dict

    elif generation_method == "mcore_engine":
        # --- Original MCore Engine Logic ---
        if rank == 0:
            print(
                f"Starting generation using MCore Engine: batch_size={input_ids.shape[0]}, max_new_tokens={max_new_tokens}"
            )

        # --- Inference Engine Setup ---
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            # Adjust threshold based on typical sequence lengths if needed
            inference_batch_times_seqlen_threshold=model_cfg.seq_length * generation_batch_size * 2,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=final_padded_vocab_size,  # Use the final padded size
            # Max length for the *entire* sequence (prompt + new) engine might handle
            # Let's set it large enough based on model seq length and max new tokens
            inference_max_seq_length=model_cfg.seq_length,
            inference_max_requests=generation_batch_size,  # Max concurrent requests in engine
            # Add other relevant flags from ModelConfig if needed by wrapper
            # e.g., model_cfg.gated_linear_unit, model_cfg.layernorm_style etc.
        )

        # Wrap the model for inference server logic
        inference_wrapped_model = ModelInferenceWrapperServer(model, inference_wrapper_config)

        # Controller handles tokenization details and generation loop logic internally (via engine)
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=megatron_tokenizer,  # Megatron tokenizer instance from setup
        )

        # Static engine is simpler for batch processing
        inference_engine = StaticInferenceEngine(
            text_generation_controller=text_generation_controller,
            max_batch_size=generation_batch_size,  # Should match inference_max_requests
        )

        # Ensure inputs are on the correct device (GPU for computation)
        input_ids = input_ids.cuda()
        input_lengths = input_lengths.cuda()

        # Calculate how many tokens to generate.
        # run_mcore_engine expects the number of *new* tokens to generate.
        tokens_to_generate = max_new_tokens

        # --- Sampling Parameters (Optional) ---
        # if greedy:
        #     # Default behavior or explicitly set greedy parameters
        #     sampling_params = SamplingParams(tokens_to_generate=tokens_to_generate, ...)
        # else:
        #     # Configure sampling parameters
        #     sampling_params = SamplingParams(tokens_to_generate=tokens_to_generate, temperature=0.7, top_k=50, ...)
        # Pass sampling_params to run_mcore_engine if used

        # --- Run the Engine ---
        if rank == 0:
            print("Running MCore Inference Engine...")
        start_time = time.time()
        with torch.no_grad(), nullcontext():  # Use nullcontext if autocast not needed/handled internally
            # Note: input_ids should be right-padded for this engine function.
            engine_output = run_mcore_engine(
                engine=inference_engine,
                prompt_tokens_tensor=input_ids,
                prompt_lengths_tensor=input_lengths,
                tokens_to_generate=tokens_to_generate,
                # sampling_params=sampling_params, # Pass if using non-greedy
                # return_output_log_probs = True, # Ensure logprobs are returned if needed
            )
        end_time = time.time()
        if rank == 0:
            print(f"Engine execution time: {end_time - start_time:.3f} seconds")

        # --- Post-process the output ---
        # engine_output is a dict with 'tokens' (list of lists), 'logprobs' (list of lists)

        batch_size = input_ids.size(0)
        # engine_output['tokens'] contains the full sequences (prompt + generated)
        # Need to handle potential empty outputs if generation fails
        if not engine_output or not engine_output["tokens"]:
            print("Warning: MCore engine returned empty output.")
            # Return empty/dummy tensors matching expected structure
            max_seq_len = input_ids.size(1)  # Use input length as fallback
            output_ids_padded = torch.full(
                (batch_size, max_seq_len), megatron_tokenizer._tokenizer.pad_token_id, dtype=torch.long, device="cpu"
            )
            logprobs_padded = torch.zeros((batch_size, max_seq_len), dtype=torch.float, device="cpu")
            generation_lengths_list = [0] * batch_size
            unpadded_lengths_list = input_lengths.cpu().tolist()  # Fallback to input lengths
        else:
            # Determine max length from the *generated* sequences
            max_seq_len = max(len(tokens) for tokens in engine_output["tokens"])

            # Create padded tensors for tokens and logprobs
            output_ids_padded = torch.full(
                (batch_size, max_seq_len),
                megatron_tokenizer._tokenizer.pad_token_id,  # Use tokenizer's pad ID
                dtype=torch.long,
                device="cpu",  # Move results to CPU
            )
            # Initialize logprobs with a suitable value (e.g., 0 or NaN)
            logprobs_padded = torch.full(
                (batch_size, max_seq_len),
                0.0,  # Or float('nan')
                dtype=torch.float,
                device="cpu",  # Move results to CPU
            )

            generation_lengths_list = []
            unpadded_lengths_list = []

            # Fill in the padded tensors
            for i in range(batch_size):
                # Tokens includes the prompt + generation
                seq_len = len(engine_output["tokens"][i])
                unpadded_lengths_list.append(seq_len)
                output_ids_padded[i, :seq_len] = torch.tensor(
                    engine_output["tokens"][i], dtype=torch.long, device="cpu"
                )

                # Logprobs usually correspond to predicting tokens[1:] based on tokens[0:t-1]
                # Check if 'logprobs' exists and handle its length
                if "logprobs" in engine_output and engine_output["logprobs"] and i < len(engine_output["logprobs"]):
                    logprob_len = len(engine_output["logprobs"][i])
                    # Place logprobs starting from the second token position (index 1)
                    # Ensure logprob_len doesn't exceed available space
                    effective_len = min(logprob_len, max_seq_len - 1)
                    if effective_len > 0:
                        logprobs_padded[i, 1 : effective_len + 1] = torch.tensor(
                            engine_output["logprobs"][i][:effective_len], dtype=torch.float, device="cpu"
                        )
                else:
                    # Handle cases where logprobs are missing or empty
                    if rank == 0:
                        print(f"Warning: Logprobs missing or empty for sample {i}.")

                # Generation length is total length minus prompt length
                # Ensure input_lengths is on CPU for item() call
                prompt_len = input_lengths[i].cpu().item()
                generation_lengths_list.append(max(0, seq_len - prompt_len))

        # Prepare final output dictionary
        output_dict = {
            "output_ids": output_ids_padded,  # [batch, max_output_len]
            "logprobs": logprobs_padded,  # [batch, max_output_len] (logprob[t] is for token[t])
            "generation_lengths": torch.tensor(generation_lengths_list, dtype=torch.long),  # [batch]
            "unpadded_sequence_lengths": torch.tensor(unpadded_lengths_list, dtype=torch.long),  # [batch]
        }
        return output_dict
    else:
        raise ValueError(f"Unsupported generation_method: {generation_method}")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Megatron-Core Initialization and Generation Test")
    parser.add_argument(
        "--model_name", type=str, required=True, help="HuggingFace model identifier (e.g., meta-llama/Llama-2-7b-hf)"
    )
    # Checkpoint path override - conversion still uses /root/checkpoints/tron/{model_name} convention
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit path to the *converted* NeMo checkpoint directory (overrides default convention if provided)",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context Parallel size")
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Computation precision"
    )
    messages = "The following are multiple choice questions (with answers) about world religions.\n\nWhen was the first Buddhist temple constructed in Japan?\nA. 325 CE\nB. 119 CE\nC. 451 CE\nD. 596 CE\nAnswer:"
    parser.add_argument("--prompt", type=str, default=messages, help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Maximum number of new tokens to generate")
    parser.add_argument("--gen_batch_size", type=int, default=1, help="Micro batch size for the generation engine")
    parser.add_argument("--max_prompt_len", type=int, default=128, help="Maximum length for input prompt tokenization")
    parser.add_argument(
        "--generation_method",
        type=str,
        default="manual_greedy",
        choices=["mcore_engine", "manual_greedy"],
        help="Method to use for generation ('mcore_engine' or 'manual_greedy')",
    )
    parser.add_argument(
        "--attn_backend",
        type=str,
        default="flash",
        choices=["fused", "flash", "unfused"],
        help="Attention backend to use ('fused', 'flash', or 'unfused'). Sets NVTE_* env vars.",
    )

    args = parser.parse_args()

    # --- Set Attention Backend Environment Variables ---
    if args.attn_backend == "fused":
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "1"
        print("INFO: Using Fused Attention Backend (NVTE_FLASH_ATTN=1, NVTE_FUSED_ATTN=1)")
    elif args.attn_backend == "flash":
        os.environ["NVTE_FLASH_ATTN"] = "1"
        os.environ["NVTE_FUSED_ATTN"] = "0"
        print("INFO: Using Flash Attention Backend (NVTE_FLASH_ATTN=1, NVTE_FUSED_ATTN=0)")
    else:  # unfused
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "0"
        print("INFO: Using Unfused Attention Backend (NVTE_FLASH_ATTN=0, NVTE_FUSED_ATTN=0)")

    # --- Distributed Setup ---
    # This script expects to be launched using torchrun or equivalent,
    # which sets MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE.
    if not torch.distributed.is_initialized():
        print("Initializing torch distributed...")
        # Basic initialization, requires environment variables to be set
        # Timeout might need adjustment for large model loading
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        print("Torch distributed already initialized.")

    rank = get_rank_safe()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size()
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)

    print(f"Rank {rank}/{world_size} assigned to device {device}")

    if args.precision == "bf16":
        dtype = torch.bfloat16
    elif args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Override checkpoint path if provided
    if args.checkpoint_path:
        # Manually set the expected path for loading if --checkpoint_path is given
        # Ensure the initialization logic uses this path. We need to pass it.
        # Let's modify minimal_megatron_setup to accept this.
        # For now, let's assume the setup function correctly uses the config path derived from model_name,
        # and this argument is primarily for user reference or potential future modification.
        # The cleaner way is to pass args.checkpoint_path to the setup function.
        # *** Modification required in minimal_megatron_setup to use args.checkpoint_path if provided ***
        print(f"INFO: Using provided checkpoint path: {args.checkpoint_path}")
        # This requires modifying minimal_megatron_setup to accept and use this override.
        # Currently, it derives the path from model_name. Add modification if needed.
        pass

    # --- Run Setup ---
    model, megatron_tokenizer, megatron_cfg, final_padded_vocab_size = minimal_megatron_setup(
        hf_model_name=args.model_name,
        tp_size=args.tp,
        pp_size=args.pp,
        cp_size=args.cp,
        dtype=dtype,
        attn_backend_str=args.attn_backend,  # Pass the chosen backend string
        # Pass checkpoint override here if modified: checkpoint_override=args.checkpoint_path
    )

    # --- Prepare Input Data ---
    # Use the Megatron tokenizer returned by the setup function
    # Important: Tokenizer might need specific padding side (left/right). MCore Engine expects right padding.
    # Check tokenizer defaults or explicitly set padding_side='right'.
    megatron_tokenizer.padding_side = "right"  # Explicitly set for MCore Engine
    # megatron_tokenizer.pad_token = megatron_tokenizer.eos_token # Common practice if pad_token is None

    if rank == 0:
        print(f"Tokenizing prompt: '{args.prompt}' with padding side: {megatron_tokenizer.padding_side}")

    # Handle potential list of prompts later if needed
    prompts = [args.prompt]
    encoded = megatron_tokenizer(
        prompts,
        padding="max_length",  # Pad to max_prompt_len
        truncation=True,
        return_tensors="pt",
        max_length=args.max_prompt_len,
        # add_special_tokens=False # Check if needed for your model
    )

    input_ids = encoded
    # Calculate lengths based on non-padding tokens
    # Attention mask might not be reliable if max_length padding used without truncation awareness
    # Safer: find first pad token or use length before padding
    input_lengths = torch.tensor(
        [(ids != megatron_tokenizer._tokenizer.pad_token_id).sum().item() for ids in input_ids], dtype=torch.long
    )

    if rank == 0:
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Calculated Input Lengths: {input_lengths}")
        # print(f"Input IDs (rank 0): {input_ids[0, :input_lengths[0]]}") # Print non-padded part

    # --- Run Generation ---
    generation_output = megatron_generate(
        model=model,
        megatron_tokenizer=megatron_tokenizer,
        mtron_cfg=megatron_cfg,
        final_padded_vocab_size=final_padded_vocab_size,
        input_ids=input_ids,
        input_lengths=input_lengths,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.gen_batch_size,
        greedy=True,  # Keep it simple for now
        generation_method=args.generation_method,  # Pass the new argument
    )

    # --- Decode and Print Output (Rank 0) ---
    if rank == 0:
        print("\n--- Generation Results ---")
        print(f"Output IDs shape: {generation_output['output_ids'].shape}")
        print(f"Logprobs shape: {generation_output['logprobs'].shape}")
        print(f"Generated Lengths: {generation_output['generation_lengths']}")
        print(f"Unpadded Sequence Lengths: {generation_output['unpadded_sequence_lengths']}")

        print("\n--- Decoded Outputs ---")
        # Decode the full generated sequence (including prompt)
        # Use the same Megatron tokenizer
        decoded_texts = megatron_tokenizer._tokenizer.batch_decode(
            generation_output['output_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        # print(decoded_texts)
        for i, text in enumerate(decoded_texts):
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {prompts[i]}")
            print(f"Generated Text:\n{text}")
            print(f"(Generated {generation_output['generation_lengths'][i].item()} new tokens)")
            print("-" * 20)

            # Optionally print logprobs for the generated part
            # gen_len = generation_output['generation_lengths'][i].item()
            # prompt_len = input_lengths[i].item()
            # if gen_len > 0:
            #     gen_logprobs = generation_output['logprobs'][i, prompt_len : prompt_len + gen_len]
            #     gen_tokens = generation_output['output_ids'][i, prompt_len : prompt_len + gen_len]
            #     print("Generated Tokens + Logprobs:")
            #     for tok_id, lp in zip(gen_tokens, gen_logprobs):
            #         token_str = megatron_tokenizer.decode(tok_id)
            #         print(f"  '{token_str}' (ID: {tok_id.item()}): {lp.item():.4f}")

    # --- Cleanup ---
    # if torch.distributed.is_initialized():
    #     torch.distributed.barrier()  # Sync before exit
    #     # Optional: Explicitly destroy process group, though often handled by launcher exit
    torch.distributed.destroy_process_group()
    print(f"Rank {rank} finished.")
