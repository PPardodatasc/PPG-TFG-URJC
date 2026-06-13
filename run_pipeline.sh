#!/bin/bash

LOG_DIR="logs_experiments"
LOG_FILE="${LOG_DIR}/pipeline_metrics_summary.txt"

mkdir -p "$LOG_DIR"

echo "========================================" > "$LOG_FILE"
echo " RESULTADOS PIPELINE: LIMPIEZA + PREDICCIÓN" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Imputation and forecasting best models (SAITS -> MICN 24h)
echo -e "\nEjecutando Prueba 1/4: SAITS -> MICN_w192_ps96"
python src/predictions/evaluate_pipeline.py --imputer SAITS --forecaster MICN_w192_ps96 --pred_steps 96

# SAITS -> DLinear 24h
echo -e "\nEjecutando Prueba 2/4: SAITS -> DLinear_w192_ps96"
python src/predictions/evaluate_pipeline.py --imputer SAITS --forecaster DLinear_w192_ps96 --pred_steps 96

# SAITS -> DLinear 48h
echo -e "\nEjecutando Prueba 3/4: SAITS -> DLinear_w384_ps192"
python src/predictions/evaluate_pipeline.py --imputer SAITS --forecaster DLinear_w384_ps192 --pred_steps 192

# SAITS -> Transformer 24h
echo -e "\nEjecutando Prueba 4/4: SAITS -> Transformer_w192_ps96"
python src/predictions/evaluate_pipeline.py --imputer SAITS --forecaster Transformer_w192_ps96 --pred_steps 96


echo -e "\nTODOS LOS EXPERIMENTOS HAN CONCLUIDO."
echo "Resultados limpios guardados en: $LOG_FILE"