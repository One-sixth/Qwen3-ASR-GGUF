"""
FunASR-GGUF: 混合 ASR 推理引擎

使用 ONNX Runtime (encoder/CTC) + llama.cpp (GGUF decoder) 进行语音识别

API 兼容 sherpa-onnx，可直接替换使用。
"""

import sys
import os
from loguru import logger

# 获取项目根目录 (适配打包环境)
if getattr(sys, 'frozen', False):
    # 打包环境：sys.executable 位于 dist/Project/ 根目录
    ROOT_DIR = os.path.dirname(sys.executable)
else:
    # 源码环境：__file__ 位于 root/qwen_asr_gguf/__init__.py
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

default_log_file = os.path.join(ROOT_DIR, "logs", "qwen_asr_{time}.log")

def setup_logging(level: str = "INFO", log_file: str = default_log_file):
    """
    配置全局日志

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件名模板

    Returns:
        配置好的 logger 实例
    """
    # 清除已有处理器
    logger.remove()
    
    # 配置文件输出，设置1个月自动删除
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 添加文件处理器，设置1个月自动删除
    logger.add(
        log_file,
        rotation="1 month",  # 每月轮换
        retention="1 month",  # 保留1个月
        compression="zip",  # 压缩旧日志
        encoding="utf-8",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}"
    )

    return logger


# 初始化默认日志配置（默认 INFO 级别）
try:
    from .. import logger
except:
    logger = setup_logging(level="INFO")

# 导出logger供其他模块使用
__all__ = ["logger", "setup_logging", "ROOT_DIR"]
