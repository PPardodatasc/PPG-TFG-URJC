#!/bin/bash

source /home/ppardog/Escritorio/tfg/PPG-TFG-URJC/.venv/bin/activate

# get project root from .env or use current directory
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT/src}"

echo "======================================================"
echo "STARTING HYPERPARAMETER OPTIMIZATION FOR IMPUTATION MODELS"
echo "======================================================"

# echo -e "\n>>> STARTING SAITS EXPERIMENTS <<<"
# python src/predictions/main.py \
#     --model SAITS \
#     --search_space src/predictions/configs_nni/saits_search_space.json \
#     --trials 22

echo -e "\n>>> STARTING CSDI EXPERIMENTS <<<"
python src/predictions/main.py \
    --model CSDI \
    --search_space src/predictions/configs_nni/csdi_search_space.json \
    --trials 22

echo "======================================================"
echo " ALL IMPUTATION EXPERIMENTS COMPLETED."
echo "======================================================"