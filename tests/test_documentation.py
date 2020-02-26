#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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

from sphinx.application import Sphinx

from tests.common_setup import NeMoUnitTest


class DocTest(NeMoUnitTest):
    source_dir = u'docs/sources/source/'
    config_dir = u'docs/sources/source/'
    output_dir = u'docs/sources/source/test_build'
    doctree_dir = u'docs/sources/source/test_build/doctrees'

    all_files = True

    def test_html_documentation(self):
        """ Tests whether the HTML documentation can be build properly. """
        app = Sphinx(
            self.source_dir,
            self.config_dir,
            self.output_dir,
            self.doctree_dir,
            buildername='html',
            warningiserror=True,
        )
        app.build(force_all=self.all_files)
