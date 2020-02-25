#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from sphinx.application import Sphinx


class DocTest(unittest.TestCase):
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
