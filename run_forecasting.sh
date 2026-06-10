#!/bin/bash

# get project root from .env or use current directory
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT/src}"

echo "======================================================"
echo "STARTING HYPERPARAMETER OPTIMIZATION FOR FORECASTING"
echo "TASK: $TASK_TYPE | HORIZON: $PRED_STEPS steps"
echo "======================================================"

echo -e "\n>>> STARTING DLinear EXPERIMENTS (BASELINE) <<<"
python src/predictions/main.py \
    --model DLinear \
    --search_space src/predictions/configs/dlinear_search_space.json \
    --trials 22

# echo -e "\n>>> STARTING TimesNet EXPERIMENTS <<<"
# python src/predictions/main.py \
#     --model TimesNet \
#     --search_space src/predictions/configs/timesnet_search_space.json \
#     --trials 10

# echo -e "\n>>> STARTING Transformer EXPERIMENTS <<<"
# python src/predictions/main.py \
#     --model Transformer \
#     --search_space src/predictions/configs/transformer_search_space.json \
#     --trials 10

echo "======================================================"
echo " ALL FORECASTING EXPERIMENTS COMPLETED."
echo "======================================================"