# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import pytest

from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core import NeuralGraph, OperationMode


@pytest.mark.usefixtures("neural_factory")
class TestNeuralGraphImportExport:
    """
        Class testing Neural Graph configuration import/export.
    """

    @pytest.mark.unit
    def test_graph_simple_import_export(self, tmpdir):
        """
            Tests whether the Neural Module can instantiate a simple module by loading a configuration file.

            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=100, batch_size=1, name="tgio1_dl")
        tn = TaylorNet(dim=4, name="tgio1_tn")
        loss = MSELoss(name="tgio1_loss")

        # Create the graph.
        with NeuralGraph(operation_mode=OperationMode.training) as g1:
            x, t = dl()
            p = tn(x=x)
            _ = loss(predictions=p, target=t)

        # Serialize graph
        serialized_g1 = g1.serialize()

        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.mkdir("export").join("simple_graph.yml"))

        # Export graph to file.
        g1.export_to_config(tmp_file_name)

        # Create the second graph - import!
        g2 = NeuralGraph.import_from_config(tmp_file_name, reuse_existing_modules=True)
        serialized_g2 = g2.serialize()

        # Must be the same.
        assert serialized_g1 == serialized_g2
