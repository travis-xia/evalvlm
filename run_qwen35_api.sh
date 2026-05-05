export TRANSVIDEOBENCH_ROOT=/inspire/ssd/project/traffic-congestion-management/public/bench-v1

# vllm serve "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-27B" \
#   --served-model-name "qwen3.5-27b" \
#   --port 12345 \
#   --tensor-parallel-size 8 \
#   --max-model-len 262144 \
#   --reasoning-parser qwen3


# python run.py \
#   --data TransVideoBench_MCQ_64frame \
#   --model qwen3.5-27b \
#   --base-url http://127.0.0.1:12345/v1 \
#   --temperature 0 \
#   --timeout 1800 \
#   --api-nproc 8 \
#   --verbose




# vllm serve "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-35B-A3B" \
#   --served-model-name "qwen3.5-35bA3b" \
#   --port 12346 \
#   --tensor-parallel-size 8 \
#   --max-model-len 262144 \
#   --reasoning-parser qwen3


# python run.py \
#   --data TransVideoBench_MCQ_64frame \
#   --model qwen3.5-35bA3b \
#   --base-url http://127.0.0.1:12346/v1 \
#   --temperature 0 \
#   --timeout 1800 \
#   --api-nproc 8 \
#   --verbose



# ========== 通过 DashScope 官方 API 调用 qwen3.5-plus ==========
# export DASHSCOPE_API_KEY=sk-xxx
# 思考: DASHSCOPE_ENABLE_THINKING=1 或 =0；不设则沿用 config
# 原生整段视频: 必须加 --video-llm；不加则数据集按多帧图片拼 prompt

export DASHSCOPE_API_KEY=sk-xxx

DASHSCOPE_ENABLE_THINKING=1 \
python run.py \
  --data TransVideoBench_MCQ_64frame \
  --model Qwen3.5-Plus-DashScope \
  --timeout 1800 \
  --api-nproc 4 \
  --verbose

# 与服务共用磁盘且可走本地路径时： export VLMEVAL_LOCAL_MEDIA=1 再加 --local-media
