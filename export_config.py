'''
[模型导出配置]
'''

from pathlib import Path

# [源模型路径] 官方默认下载的 SafeTensors 模型文件夹
# model_home = Path('~/.cache/modelscope/hub/models/Qwen').expanduser()
# ASR_MODEL_DIR =  model_home / 'Qwen3-ASR-0.6B'
# ASR_MODEL_DIR =  model_home / 'Qwen3-ASR-1.7B'
# ALIGNER_MODEL_DIR =  model_home / 'Qwen3-ForcedAligner-0.6B'

# 本地权重路径设定，如果放在其他地方。
ASR_MODEL_DIR =  r'E:\Software\LLM\Qwen3-ASR-1.7B'
# ASR_MODEL_DIR =  r'E:\Software\LLM\Qwen3-ASR-0.6B'
ALIGNER_MODEL_DIR =  r'E:\Software\LLM\Qwen3-ForcedAligner-0.6B'

# [导出目标路径] 转换后的 ONNX, GGUF 和权重汇总目录
EXPORT_DIR = r'./model'

# LLM层 量化输出类型，默认是 q4_k
# LLM_QUANTIZE_TYPE = "q4_k"
# LLM_QUANTIZE_TYPE = "q6_k"
LLM_QUANTIZE_TYPE = "q8_0"
# LLM_QUANTIZE_TYPE = "f16"

# 编码层 量化类型，默认是 int8
# ENC_QUANTIZE_TYPE = "int4"
ENC_QUANTIZE_TYPE = "int8"
# ENC_QUANTIZE_TYPE = "fp16"


assert LLM_QUANTIZE_TYPE in ('f16', 'q8_0', 'q6_k', 'q4_k'), "LLM_QUANTIZE_TYPE 必须是 f16, q8_0, q6_k, q4_k 中的一个"
assert ENC_QUANTIZE_TYPE in ('fp16', 'int8', 'int4'), "ENC_QUANTIZE_TYPE 必须是 fp16, int8, int4 中的一个"
