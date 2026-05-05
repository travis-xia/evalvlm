#!/bin/bash
export TRANSVIDEOBENCH_ROOT=/inspire/ssd/project/traffic-congestion-management/public/bench-v1


# python run.py \
#   --data TransVideoBench_MCQ_raw \
#   --model InternVideo2_5_Chat_8B \
#   --verbose



python run.py \
  --data TransVideoBench_MV_raw \
  --model InternVideo2_5_Chat_8B \
  --verbose
