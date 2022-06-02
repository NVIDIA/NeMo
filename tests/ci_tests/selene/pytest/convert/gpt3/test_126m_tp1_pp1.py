import os
import json
from pathlib import Path

import pytest


CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")

class TestCIGPT126m:

    def test_ci_convert_gpt3_126m_tp1_pp1(self):
        pass

