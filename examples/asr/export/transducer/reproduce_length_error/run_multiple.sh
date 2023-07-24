#!/bin/bash

set -euo pipefail

for i in $(seq 1 20); do
    python run_subgraph.py
done
