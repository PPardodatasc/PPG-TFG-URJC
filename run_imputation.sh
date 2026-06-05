#!/bin/bash
cd "$(dirname "$0")" # 

echo "======================================================"
echo "LAUNCHING EXPERIMENT 1: SAITS Imputation"
echo "======================================================"
nnictl create --config src/predictions/configs_nni/saits_imputation.yml --port 8080 --foreground

echo "======================================================"
echo "SAITS imputation DONE"
echo "======================================================"
read -p "Go to http://localhost:8080 and check the results. Press [ENTER] to continue with the CSDI imputation..." # Pause script: requires user confirmation


echo "Iniciando limpieza de puertos..."
nnictl stop --all

echo "======================================================"
echo "LAUNCHING EXPERIMENT 2: CSDI Imputation"
echo "======================================================"
nnictl create --config src/predictions/configs_nni/csdi_imputation.yml --port 8080 --foreground

echo "======================================================"
echo "CSDI imputation DONE. Check the results at http://localhost:8080"
echo "======================================================"