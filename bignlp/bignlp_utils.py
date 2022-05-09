import hydra
import omegaconf


def convert_to_cli(cfg):
    result = ""
    if cfg.get("data_config") is not None:
        result += f"data_preparation={cfg['data_config']} "
    if cfg.get("training_config") is not None:
        result += f"training={cfg['training_config']} "
    if cfg.get("finetuning_config") is not None:
        result += f"finetuning={cfg['finetuning_config']} "
    if cfg.get("evaluation_config") is not None:
        result += f"evaluation={cfg['evaluation_config']} "
    if cfg.get("conversion_config") is not None:
        result += f"conversion={cfg['conversion_config']} "

    for k, v in cfg.items():
        if k in ["dgxa100_gpu2core", "dgxa100_gpu2mem"]:
            continue

        if isinstance(v, omegaconf.dictconfig.DictConfig):
            output = convert_to_cli(v).split(" ")
            result += " ".join([f"{k}.{x}" for x in output if x != ""]) + " "
        elif isinstance(v, omegaconf.listconfig.ListConfig):
            if k == "data_prefix":
                if v is None:
                    v = "null"
                else:
                    v = [x for x in v]  # Needed because of lazy omegaconf interpolation.
            result += f"{k}={str(v).replace(' ', '')} "
        elif isinstance(v, str) and "{" in v:
            continue
        elif k in ["splits_string", "file_numbers", "languages"]:
            result += f"{k}=\\'{v}\\' "
        elif k == "checkpoint_name":
            v = v.replace("=", "\=")
            result += f"{k}='{v}' "
        else:
            result += f"{k}={convert_to_null(v)} "
    return result


def convert_to_null(val):
    if val is None:
        return "null"
    return val


def fake_submit(*args, **kwargs):
    print(args, kwargs)
    fake_id = 123456
    return str(fake_id).encode()


def add_container_mounts(container_mounts):
    mounts_str = ""
    if container_mounts is not None:
        assert isinstance(
            container_mounts, omegaconf.listconfig.ListConfig
        ), "container_mounts must be a list."
        for mount in container_mounts:
            if mount is not None and isinstance(mount, str):
                mounts_str += f",{mount}" if ":" in mount else f",{mount}:{mount}"
    return mounts_str


def valid_node_counts(gbs, mbs, tp, pp, gpus_per_node=8, max_node_count=200):
    """Returns all the possible node counts to use for a given config of
    GBS, MBS, TP, PP, and gpus_per_node. The maximum number of nodes can
    be limited by using the max_node_count parameter.

    Parameters:
    gbs: int, Global Batch Size.
    mbs: int, Micro Batch Size.
    tp: int, Tensor Model Parallelism.
    pp: int, Pipeline Model Parallelism.
    gpus_per_node: int, number of GPUs per node.
    max_node_count: int, numbers of nodes larger than this number will
        not be added to the list.

    Returns:
    valid_nodes: list, all the valid node counts.
    """
    try:
        highest = int(gbs * pp * tp / (gpus_per_node * mbs))
        valid_nodes = []
        for nodes in range(
            max(1, int(tp * pp / gpus_per_node)), min(highest + 1, max_node_count + 1)
        ):
            if (
                gbs % (mbs * nodes * gpus_per_node / (tp * pp)) == 0
                and (nodes * gpus_per_node) % (tp * pp) == 0
            ):
                valid_nodes.append(nodes)
        return valid_nodes
    except:
        print("Invalid arguments passed.")
