#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
