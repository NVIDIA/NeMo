import nemo.collections.multimodal.models.controlnet.uniformer.mmseg as mmseg
from nemo.collections.multimodal.models.controlnet.uniformer.mmcv.utils import collect_env as collect_base_env
from nemo.collections.multimodal.models.controlnet.uniformer.mmcv.utils import get_git_hash


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
