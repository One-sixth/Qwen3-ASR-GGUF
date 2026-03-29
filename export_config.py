
from pathlib import Path
model_home = Path('~/.cache/modelscope/hub/models/Qwen').expanduser()


# [源模型路径] 官方下载好的 SafeTensors 模型文件夹
# ASR_MODEL_DIR =  model_home / 'Qwen3-ASR-1.7B'
# ALIGNER_MODEL_DIR =  model_home / 'Qwen3-ForcedAligner-0.6B'
ASR_MODEL_DIR =  r'E:\Software\LLM\Qwen3-ASR-1.7B'
# ASR_MODEL_DIR =  r'E:\Software\LLM\Qwen3-ASR-0.6B'
ALIGNER_MODEL_DIR =  r'E:\Software\LLM\Qwen3-ForcedAligner-0.6B'

# [导出目标路径] 转换后的 ONNX, GGUF 和权重汇总目录
EXPORT_DIR = r'./model'

# 量化输出的大小，默认是 q4_k
# QUANTIZE_TYPE = "q4_k"
QUANTIZE_TYPE = "q8_0"
# QUANTIZE_TYPE = "f16"
# ENC_QUANTIZE_TYPE = "int4"
ENC_QUANTIZE_TYPE = "int8"
# ENC_QUANTIZE_TYPE = "fp16"
