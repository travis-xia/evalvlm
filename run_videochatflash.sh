#!/bin/bash
export TRANSVIDEOBENCH_ROOT=/inspire/ssd/project/traffic-congestion-management/public/bench-v1

# python run.py \
#   --data TransVideoBench_MCQ_raw \
#   --model VideoChat_Flash_Qwen2_7B_res448 \
#   --verbose

# python run.py \
#   --data TransVideoBench_TG_raw \
#   --model VideoChat_Flash_Qwen2_7B_res448 \
#   --verbose

python run.py \
  --data TransVideoBench_MV_raw \
  --model VideoChat_Flash_Qwen2_7B_res448 \
  --verbose
