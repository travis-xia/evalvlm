export TRANSVIDEOBENCH_ROOT=/inspire/ssd/project/traffic-congestion-management/public/bench-v1

# vllm serve "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-27B" \
#   --served-model-name "qwen3.5-27b" \
#   --port 12345 \
#   --tensor-parallel-size 8 \
#   --max-model-len 262144 \
#   --reasoning-parser qwen3



# VLMEVAL_QWEN_MAX_PIXELS=$((1280 * 28 * 28)) \
# python run.py \
#   --data TransVideoBench_MV_64frame \
#   --model qwen3.5-27b \
#   --base-url http://127.0.0.1:12345/v1 \
#   --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
#   --temperature 0 \
#   --timeout 1800 \
#   --reuse \
#   --api-nproc 8 \
#   --verbose

# VLMEVAL_QWEN_MAX_PIXELS=$((1280 * 28 * 28)) \
# python run.py \
#   --data TransVideoBench_MV_64frame \
#   --model qwen3.5-27b \
#   --base-url http://127.0.0.1:12345/v1 \
#   --custom-prompt qwen3 \
#   --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
#   --temperature 0 \
#   --timeout 1800 \
#   --reuse \
#   --api-nproc 8 \
#   --verbose



# 关闭「思考」输出：请求里带上 chat_template_kwargs（与 config 里 InstructMode_api 一致）。
# 也可在较新 vLLM 上为服务端设默认：--default-chat-template-kwargs '{"enable_thinking": false}'
# --reasoning-parser qwen3 只在模型仍会产出 reasoning 时用于拆分字段；要禁用思考本身靠 enable_thinking。
# vllm serve "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-35B-A3B" \
#   --served-model-name "qwen3.5-35bA3b" \
#   --port 12346 \
#   --tensor-parallel-size 8 \
#   --max-model-len 262144 \
#   --reasoning-parser qwen3


# 降视觉 token：需同时 --custom-prompt qwen3；像素上限用环境变量（整数，与 HF Qwen-VL 的 max_pixels 含义一致）。
# 默认常见上限约 12845056（= 16384*28*28）。可先试 1280*28*28=1003520，仍超限再降到 512*28*28 等。
# export VLMEVAL_QWEN_MAX_PIXELS=$((1280 * 28 * 28))


# VLMEVAL_QWEN_MAX_PIXELS=$((1280 * 28 * 28)) \
# python run.py \
#   --data TransVideoBench_TG_64frame \
#   --model qwen3.5-35bA3b \
#   --base-url http://127.0.0.1:12346/v1 \
#   --custom-prompt qwen3 \
#   --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
#   --temperature 0 \
#   --timeout 1800 \
#   --api-nproc 8 \
#   --verbose


# vllm serve "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen3.5-9B"   --served-model-name "qwen3.5-9b"   --port 12347   --tensor-parallel-size 8   --max-model-len 262144   --reasoning-parser qwen3 

# VLMEVAL_QWEN_MAX_PIXELS=$((1280 * 28 * 28)) \
# python run.py \
#   --data TransVideoBench_MV_64frame \
#   --model qwen3.5-9b \
#   --base-url http://127.0.0.1:12347/v1 \
#   --custom-prompt qwen3 \
#   --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
#   --temperature 0 \
#   --timeout 1800 \
#   --api-nproc 8 \
#   --verbose


# ========== 通过 DashScope 官方 API 调用 qwen3.5-plus ==========
# export DASHSCOPE_API_KEY=sk-xxx
# 思考: DASHSCOPE_ENABLE_THINKING=1 或 =0；不设则沿用 config
# 原生整段视频: 必须加 --video-llm；不加则数据集按多帧图片拼 prompt

# export DASHSCOPE_API_KEY=sk-xxx

# DASHSCOPE_ENABLE_THINKING=1 \
# python run.py \
#   --data TransVideoBench_MCQ_64frame \
#   --model Qwen3.5-Plus-DashScope \
#   --timeout 1800 \
#   --api-nproc 4 \
#   --verbose

# export DASHSCOPE_API_KEY=sk-f55
VLMEVAL_QWEN_MAX_PIXELS=$((1280 * 28 * 28)) \
DASHSCOPE_ENABLE_THINKING=0 \
python run.py \
  --data TransVideoBench_MV_64frame \
  --model Qwen3.5-Plus-DashScope \
  --timeout 1800 \
  --api-nproc 8 \
  --verbose



# 与服务共用磁盘且可走本地路径时： export VLMEVAL_LOCAL_MEDIA=1 再加 --local-media






