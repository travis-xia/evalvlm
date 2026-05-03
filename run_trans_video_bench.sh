export TRANSVIDEOBENCH_ROOT=/inspire/ssd/project/traffic-congestion-management/public/bench-v1
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python run.py \
  --data TransVideoBench_1fps \
  --model llava_video_qwen2_7b_inspire_auto \
  --verbose



# 这个库同时支持api（例如有些模型这个库还不支持，我们也可以本地下载用vllm起服务后暴露为api模式）和本地起的model
# api起model方式可以看一下说明文档，然后写到这个脚本里
# 和我们bench最相关的部分 vlmeval/dataset/transvideobench.py； 数据评测采样配置 vlmeval/dataset/video_dataset_config.py的 line308; 模型配置 vlmeval/config.py （配置模型本地路径支持vllm）
# 即使不需要用本VLMEvalKit直接跑，也可以遵照transvideobench.py单独写一下，需要我们形式统一
# /inspire/hdd/project/traffic-congestion-management/public/bench/trans_data_example中有数据基本格式；
# 跑完后的结果：
# {
#     "overall|count": 10,
#     "overall|overall": 51.365,
#     "overall|mcq_count": 5,
#     "overall|mcq_acc": 60.0,
#     "overall|tg_count": 5,
#     "overall|tg_mIoU": 42.729,
#     "overall|tg_R@0.3": 60.0,
#     "overall|tg_R@0.5": 40.0,
#     "overall|tg_R@0.7": 40.0,
#     "static_dynamic|\u52a8\u6001|count": 9,
#     "static_dynamic|\u52a8\u6001|overall": 48.525,
#     "static_dynamic|\u52a8\u6001|mcq_count": 5,
#     "static_dynamic|\u52a8\u6001|mcq_acc": 60.0,
#     "static_dynamic|\u52a8\u6001|tg_count": 4,
#     "static_dynamic|\u52a8\u6001|tg_mIoU": 34.181,
#     "static_dynamic|\u52a8\u6001|tg_R@0.3": 50.0,
#     "static_dynamic|\u52a8\u6001|tg_R@0.5": 25.0,
#     "static_dynamic|\u52a8\u6001|tg_R@0.7": 25.0,
#     "category|\u5355\u4e00\u8f66\u8f86\u884c\u4e3a|count": 3,
#     "category|\u5355\u4e00\u8f66\u8f86\u884c\u4e3a|overall": 41.07,
#     "category|\u5355\u4e00\u8f66\u8f86\u884c\u4e3a|mcq_count": 0,
#     "category|\u5355\u4e00\u8f66\u8f86\u884c\u4e3a|mcq_acc": NaN,
#     "category|\u5355\u4e00\u8f66\u8f86\u884c\u4e3a|tg_count": 3,
#     "category|\u5355\u4e00\u8f66\u8f86\u884c\u4e3a|tg_mIoU": 41.07,
#     "category|\u5355\u4e00\u8f66\u8f86\u884c\u4e3a|tg_R@0.3": 66.667,
#     "category|\u5355\u4e00\u8f66\u8f86\u884c\u4e3a|tg_R@0.5": 33.333,
#     "category|\u5355\u4e00\u8f66\u8f86\u884c\u4e3a|tg_R@0.7": 33.333,
#     "category|\u666e\u901a\u6444\u50cf\u5934\u89c6\u89d2\uff08\u7ba1\u7406\u8005\u53c2\u4e0e\u8005\uff09|count": 1,
#     "category|\u666e\u901a\u6444\u50cf\u5934\u89c6\u89d2\uff08\u7ba1\u7406\u8005\u53c2\u4e0e\u8005\uff09|overall": 13.514,
#     "category|\u666e\u901a\u6444\u50cf\u5934\u89c6\u89d2\uff08\u7ba1\u7406\u8005\u53c2\u4e0e\u8005\uff09|mcq_count": 0,
#     "category|\u666e\u901a\u6444\u50cf\u5934\u89c6\u89d2\uff08\u7ba1\u7406\u8005\u53c2\u4e0e\u8005\uff09|mcq_acc": NaN,
#     "category|\u666e\u901a\u6444\u50cf\u5934\u89c6\u89d2\uff08\u7ba1\u7406\u8005\u53c2\u4e0e\u8005\uff09|tg_count": 1,
#     "category|\u666e\u901a\u6444\u50cf\u5934\u89c6\u89d2\uff08\u7ba1\u7406\u8005\u53c2\u4e0e\u8005\uff09|tg_mIoU": 13.514,
#     "category|\u666e\u901a\u6444\u50cf\u5934\u89c6\u89d2\uff08\u7ba1\u7406\u8005\u53c2\u4e0e\u8005\uff09|tg_R@0.3": 0.0,
#     "category|\u666e\u901a\u6444\u50cf\u5934\u89c6\u89d2\uff08\u7ba1\u7406\u8005\u53c2\u4e0e\u8005\uff09|tg_R@0.5": 0.0,
#     "category|\u666e\u901a\u6444\u50cf\u5934\u89c6\u89d2\uff08\u7ba1\u7406\u8005\u53c2\u4e0e\u8005\uff09|tg_R@0.7": 0.0,
#     "category|\u8f66\u8f86\u95f4\u4ea4\u4e92|count": 5,
#     "category|\u8f66\u8f86\u95f4\u4ea4\u4e92|overall": 60.0,
#     "category|\u8f66\u8f86\u95f4\u4ea4\u4e92|mcq_count": 5,
#     "category|\u8f66\u8f86\u95f4\u4ea4\u4e92|mcq_acc": 60.0,
#     "category|\u8f66\u8f86\u95f4\u4ea4\u4e92|tg_count": 0,
#     "category|\u8f66\u8f86\u95f4\u4ea4\u4e92|tg_mIoU": NaN,
#     "category|\u8f66\u8f86\u95f4\u4ea4\u4e92|tg_R@0.3": NaN,
#     "category|\u8f66\u8f86\u95f4\u4ea4\u4e92|tg_R@0.5": NaN,
#     "category|\u8f66\u8f86\u95f4\u4ea4\u4e92|tg_R@0.7": NaN,
#     "duration_bucket|0-30s|count": 7,
#     "duration_bucket|0-30s|overall": 55.887,
#     "duration_bucket|0-30s|mcq_count": 5,
#     "duration_bucket|0-30s|mcq_acc": 60.0,
#     "duration_bucket|0-30s|tg_count": 2,
#     "duration_bucket|0-30s|tg_mIoU": 45.604,
#     "duration_bucket|0-30s|tg_R@0.3": 50.0,
#     "duration_bucket|0-30s|tg_R@0.5": 50.0,
#     "duration_bucket|0-30s|tg_R@0.7": 50.0,
#     "duration_bucket|30s\u4ee5\u4e0a|count": 3,
#     "duration_bucket|30s\u4ee5\u4e0a|overall": 40.812,
#     "duration_bucket|30s\u4ee5\u4e0a|mcq_count": 0,
#     "duration_bucket|30s\u4ee5\u4e0a|mcq_acc": NaN,
#     "duration_bucket|30s\u4ee5\u4e0a|tg_count": 3,
#     "duration_bucket|30s\u4ee5\u4e0a|tg_mIoU": 40.812,
#     "duration_bucket|30s\u4ee5\u4e0a|tg_R@0.3": 66.667,
#     "duration_bucket|30s\u4ee5\u4e0a|tg_R@0.5": 33.333,
#     "duration_bucket|30s\u4ee5\u4e0a|tg_R@0.7": 33.333
# }
# [2026-04-29 13:29:17] INFO - vlmeval.__main__: Run Summary Report:
# [2026-04-29 13:29:17] INFO - vlmeval.__main__: 
# benchmark             infer_fail_rate    judge_fail_rate    primary_metric      primary_metric_value  skip_reason    eval_error
# --------------------  -----------------  -----------------  ----------------  ----------------------  -------------  ------------
# TransVideoBench_1fps  0.00% (0/10)       0.00% (0/10)       Overall Score                      51.37  -              -
