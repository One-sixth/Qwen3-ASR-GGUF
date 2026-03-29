# coding=utf-8
import os
import sys
import re
import time
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference import QwenASREngine, itn, load_audio, ASREngineConfig, AlignerConfig
from qwen_asr_gguf.inference import exporters
from export_config import QUANTIZE_TYPE, ENC_QUANTIZE_TYPE, EXPORT_DIR

def main():

    audio_path = "test_audio/asr_zh.aac"
    context = ""

    # 配置引擎
    config = ASREngineConfig(
        encoder_frontend_fn=f"qwen3_asr_encoder_frontend.{ENC_QUANTIZE_TYPE}.onnx",
        encoder_backend_fn=f"qwen3_asr_encoder_backend.{ENC_QUANTIZE_TYPE}.onnx",
        llm_fn = f"qwen3_asr_llm.{QUANTIZE_TYPE}.gguf",
        model_dir=EXPORT_DIR,
        onnx_provider = 'DML',
        llm_use_gpu = True,
        enable_aligner = True,
        align_config = AlignerConfig(
            encoder_frontend_fn=f"qwen3_aligner_encoder_frontend.{ENC_QUANTIZE_TYPE}.onnx",
            encoder_backend_fn=f"qwen3_aligner_encoder_backend.{ENC_QUANTIZE_TYPE}.onnx",
            onnx_provider='DML',
            llm_use_gpu=True,
            model_dir=EXPORT_DIR,
            llm_fn=f"qwen3_aligner_llm.{QUANTIZE_TYPE}.gguf"
        )
    )

    # 初始化引擎
    t0 = time.time()
    engine = QwenASREngine(config=config)
    print(f"--- [QwenASR] 引擎初始化耗时: {time.time() - t0:.2f} 秒 ---")

    # 执行转录
    res = engine.transcribe(
        audio_file=audio_path,
        context=context,
        # language="Chinese",
        language="",
        start_second=0,
        duration=40
    )


    # 导出文本（每行一句）
    txt_path = str(Path(audio_path).with_suffix('.txt'))
    exporters.export_to_txt(txt_path, res)

    # 导出 SRT（仅当有对齐时间戳时，才会有内容输出）
    srt_path = str(Path(audio_path).with_suffix('.srt'))
    exporters.export_to_srt(srt_path, res)

    # 导出 JSON（仅当有对齐时间戳时，才会有内容输出）
    json_path = str(Path(audio_path).with_suffix('.json'))
    exporters.export_to_json(json_path, res)

    # 对齐预览 (仅当有结果时)
    if res.alignment:
        print("\n" + "="*15 + " 对齐结果预览 (前10个) " + "="*15)
        for it in res.alignment.items[:10]:
            print(f"{it.text:<10} | {it.start_time:7.3f}s | {it.end_time:7.3f}s")
        print("="*52)

    # 优雅退出
    engine.shutdown()

if __name__ == "__main__":
    main()
