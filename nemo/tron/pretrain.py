from typing import Callable

from nemo.tron import fault_tolerance
from nemo.tron.checkpointing import save_checkpoint
from nemo.tron.config import ConfigContainer
from nemo.tron.data.dataset import train_valid_test_datasets_provider
from nemo.tron.eval import evaluate_and_print_results
from nemo.tron.setup import setup
from nemo.tron.state import GlobalState
from nemo.tron.train import train
from nemo.tron.utils.async_utils import maybe_finalize_async_save
from nemo.tron.utils.common_utils import barrier_and_log, print_rank_0


def pretrain(
    config: ConfigContainer,
    forward_step_func: Callable,
    train_valid_test_datasets_provider=train_valid_test_datasets_provider,
):
    (
        model,
        optimizer,
        scheduler,
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
        checkpointing_context,
    ) = setup(config, train_valid_test_datasets_provider)

    state = GlobalState()
    if not config.megatron_lm_config.skip_train:
        print_rank_0("training ...")

        iteration = 0
        if state.train_state.do_train and config.megatron_lm_config.train_iters > 0:
            train(
                forward_step_func=forward_step_func,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_data_iterator=train_data_iterator,
                valid_data_iterator=valid_data_iterator,
                process_non_loss_data_func=None,
                global_state=state,
                checkpointing_context=checkpointing_context,
                non_loss_data_func=None,
            )

        barrier_and_log("after training is done")

        if (
            config.checkpoint_config.save
            and iteration != 0
            and iteration % config.checkpoint_config.save_interval != 0
        ):
            save_checkpoint(
                state=state,
                model=model,
                optimizer=optimizer,
                opt_param_scheduler=scheduler,
                num_floating_point_operations_so_far=state.train_state.floating_point_operations_so_far,
                cfg=state.cfg,
                checkpointing_context=checkpointing_context,
                train_data_iterator=train_data_iterator,
                preprocess_common_state_dict_fn=None,
            )
    else:
        print_rank_0("skipping training (--skip-train is on) ...")

        iteration = config.megatron_lm_config.train_iters

    if state.train_state.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            state=state,
            prefix=prefix,
            forward_step_func=forward_step_func,
            model=model,
            data_iterator=valid_data_iterator,
            process_non_loss_data_func=None,
            config=state.cfg.model_config,
            verbose=True,
        )

    if state.train_state.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            state=state,
            prefix=prefix,
            forward_step_func=forward_step_func,
            model=model,
            data_iterator=test_data_iterator,
            process_non_loss_data_func=None,
            config=state.cfg.model_config,
            verbose=True,
        )

    if state.wandb_logger:
        state.wandb_logger.finish()

    fault_tolerance.on_checkpointing_start(global_state=state)
    maybe_finalize_async_save(ckpt_cfg=config.checkpoint_config, blocking=True, terminate=True)
    fault_tolerance.on_checkpointing_end(is_async_finalization=True, global_state=state)

    fault_tolerance.shutdown(global_state=state)
