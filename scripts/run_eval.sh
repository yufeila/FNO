#!/usr/bin/env bash
PY=/workspace/env/yyf/miniconda3/envs/d2l/bin/python
ROOT=/workspace/code/yyf/Proj/neuraloperator/TimeDimension_add

$PY $ROOT/experiments/compare/results/eval_time_analysis.py \
  --model_baseline "$ROOT/experiments/A_no_time/runs/baseline_modes1_32_modes2_128_epoch_10/model_ssp_tl_baseline.pth" \
  --model_scalar   "$ROOT/experiments/Aprime_time_scalar/runs/modes1_32_modes2_128_epoch_10/model_ssp_tl_time_scalar.pth" \
  --model_pe       "$ROOT/experiments/Adblprime_time_pe/runs/modes1_32_modes2_128_epoch_10/model_ssp_tl_time_pe.pth"
