#!/bin/bash
export TRANSVIDEOBENCH_ROOT=/inspire/ssd/project/traffic-congestion-management/public/bench-v1

# 64frame
python run.py \
  --data TransVideoBench_MCQ_64frame \
  --model Keye-VL-1_5-8B \
  --verbose

# 1fps
python run.py \
  --data TransVideoBench_MCQ_1fps \
  --model Keye-VL-1_5-8B \
  --verbose

# raw (使用模型默认采样策略)
python run.py \
  --data TransVideoBench_MCQ_raw \
  --model Keye-VL-1_5-8B \
  --verbose
