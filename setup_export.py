# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools


def setup_export():
    
    setuptools.setup(
        name="NeMo Export",
        # Versions should comply with PEP440.  For a discussion on single-sourcing
        # the version across setup.py and the project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        version=1,
        description="NeMo Export - a module to export nemo checkpoints to TensorRT-LLM",
        long_description="NeMo Export - a module to export nemo checkpoints to TensorRT-LLM",
        # Author details
        author="NVIDIA",
        license='Apache2',
        packages=[
            "nemo", 
            "nemo.export", 
            "nemo.export.trt_llm", 
            "nemo.export.trt_llm.decoder",
            "nemo.export.trt_llm.nemo",
            "nemo.deploy",
        ],
        # Add in any packaged data.
        include_package_data=True,
        exclude=['tools', 'tests'],
        package_data={'': ['*.tsv', '*.txt', '*.far', '*.fst', '*.cpp', 'Makefile']},
        zip_safe=False,
    )


if __name__ == '__main__':
    setup_export()
