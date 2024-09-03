# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools
import os


def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename), encoding='utf-8') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]


def setup_export():
    install_requires = req_file("requirements_infer_nim.txt")

    setuptools.setup(
        name="NeMo Export",
        # Versions should comply with PEP440.  For a discussion on single-sourcing
        # the version across setup.py and the project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        version=24.08,
        description="NeMo Export - a module to export nemo checkpoints to TensorRT-LLM",
        long_description="NeMo Export - a module to export nemo checkpoints to TensorRT-LLM",
        # Author details
        author="NVIDIA",
        license='Apache2',
        packages=[
            "nemo",
            "nemo.export",
            "nemo.export.trt_llm",
            "nemo.export.vllm",
            "nemo.export.multimodal",
            "nemo.export.quantize",
            "nemo.export.trt_llm.converter",
            "nemo.export.trt_llm.nemo_ckpt_loader",
            "nemo.export.trt_llm.qnemo",
            "nemo.deploy",
        ],
        install_requires=install_requires,
        # Add in any packaged data.
        include_package_data=True,
        exclude=['tools', 'tests'],
        package_data={'': ['*.tsv', '*.txt', '*.far', '*.fst', '*.cpp', 'Makefile']},
        zip_safe=False,
    )


if __name__ == '__main__':
    setup_export()