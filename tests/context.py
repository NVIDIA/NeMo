# Copyright (c) 2019 NVIDIA Corporation
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../nemo')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../collections/')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../collections/nemo_asr')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../collections/nemo_nlp')))

import nemo
import nemo_asr
import nemo_nlp
