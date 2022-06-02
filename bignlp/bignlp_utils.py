import hydra
import omegaconf


def convert_to_cli(cfg, root=True):
    result = []
    if cfg.get("data_config") is not None:
        result.append(f"data_preparation={cfg['data_config']}")
    if cfg.get("training_config") is not None:
        result.append(f"training={cfg['training_config']}")
    if cfg.get("finetuning_config") is not None:
        result.append(f"finetuning={cfg['finetuning_config']}")
    if cfg.get("prompt_learning_config") is not None:
        result.append(f"prompt_learning={cfg['prompt_learning_config']}")
    if cfg.get("evaluation_config") is not None:
        result.append(f"evaluation={cfg['evaluation_config']}")
    if cfg.get("conversion_config") is not None:
        result.append(f"conversion={cfg['conversion_config']}")

    for k, v in cfg.items():
        if k in ["dgxa100_gpu2core", "dgxa100_gpu2mem", "container", "ci_test"]:
            continue
        if k == 'task_templates':
            def dict2str(d):
                output = ','.join([f'{str(key)}:"{str(val)}"'
                                   if key == "prompt_template" else f'{str(key)}:{str(val)}'
                                   for key, val in d.items()])
                return f'{{{output}}}'
            v = f"[{','.join([dict2str(d) for d in v])}]"
            result.append(f"{k}='{v}'")
            continue

        if isinstance(v, omegaconf.dictconfig.DictConfig):
            output = convert_to_cli(v, False)
            result.extend([f"{k}.{x}" for x in output if x != ""])
        elif isinstance(v, omegaconf.listconfig.ListConfig):
            if k == "data_prefix" or "_ds" in k:
                if v is None:
                    v = "null"
                else:
                    v = [x for x in v]  # Needed because of lazy omegaconf interpolation.
            result.append(f"{k}={str(v).replace(' ', '')}")
        elif isinstance(v, str) and "{" in v:
            continue
        elif k in ["splits_string", "file_numbers", "languages"]:
            result.append(f"{k}=\\'{v}\\'")
        elif k == "checkpoint_name":
            v = v.replace("=", "\=")
            result.append(f"{k}=\'{v}\'")
        else:
            result.append(f"{k}={convert_to_null(v)}")

    return " \\\n  ".join(result) if root else result

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


def create_slurm_file(
    new_script_path,
    slurm_cmd,
    job_name,
    flags="",
    dependency=None,
    time="04:00:00",
    exclusive=True,
    mem=0,
    overcommit=True,
    nodes=1,
    ntasks_per_node=8,
    gpus_per_task=None,
    gpus_per_node=None,
    partition="batch",
    account=None,
):
    """
    Creates a slurm file to launch a training job.
    """
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --nodes={nodes}\n")
        f.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        if gpus_per_task is not None:
            f.writelines(f"#SBATCH --gpus-per-task={gpus_per_task}\n")
        if gpus_per_node is not None:
            f.writelines(f"#SBATCH --gpus-per-node={gpus_per_node}\n")
        if dependency is not None:
            if dependency != "singleton":
                dependency = f"afterany:{dependency}"
            f.writelines(f"#SBATCH --dependency={dependency}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        if account is not None:
            f.writelines(f"#SBATCH -A {account}\n")
        f.writelines(f"#SBATCH --job-name={job_name}\n")
        f.writelines(f"#SBATCH --mem={mem}\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        if overcommit:
            f.writelines("#SBATCH --overcommit\n")
        f.writelines(f"#SBATCH --time={time}\n\n")
        f.writelines(f'srun {flags} sh -c "{slurm_cmd}"\n\n')
        f.writelines("set +x\n")


def create_bcp_file(
    bcp_cmd,
    num_nodes,
    log_file,
    new_script_path,
    env_exports=None,
):
    with open(new_script_path, "w") as f:
        if env_exports is not None:
            env_cmd = f"--env {env_exports}"
        f.writelines(f'bcprun -n {num_nodes} {env_cmd} -c "{bcp_cmd}" >> {log_file} 2>&1 \n')
        f.writelines("\n")
        f.writelines("set +x \n")
    os.chmod(new_script_path, 0o755)