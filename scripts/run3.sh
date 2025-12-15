#!/usr/bin/env bash
set -e

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 nohup python ./experiments/A_no_time/train.py       > logs/A_baseline.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python ./experiments/Aprime_time_scalar/train.py > logs/Aprime.log      2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python ./experiments/Adblprime_time_pe/train.py  > logs/Adblprime.log   2>&1 &

echo "Launched 3 runs:"
echo "  GPU0 -> A_baseline      (logs/A_baseline.log)"
echo "  GPU1 -> Aprime_time_scalar (logs/Aprime.log)"
echo "  GPU2 -> Adblprime_time_pe  (logs/Adblprime.log)"
