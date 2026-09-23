#!/usr/bin/env bash
# Full pipeline. Run from the project root: bash scripts/run_all.sh
set -e
echo "=== Stage 1: classifier ===" && python -m src.train_classifier
echo "=== Stage 2: attacks ===" && python -m src.generate_attacks
echo "=== Stage 3: detector ===" && python -m src.train_detector
echo "=== Stage 4: adaptive attack ===" && python -m src.adaptive_attack
echo "=== Stage 5: evaluation ===" && python -m src.evaluate
echo "Done. Launch the demo with: streamlit run app/streamlit_app.py"