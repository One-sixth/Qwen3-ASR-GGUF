# coding=utf-8
import os, sys, time
from pathlib import Path
from typing import List, Optional

# 获取项目根目录 (适配打包环境)
if getattr(sys, 'frozen', False):
    # 打包环境：sys.executable 位于 dist/Project/ 根目录
    PROJ_DIR = Path(sys.executable).parent
else:
    # 源码环境
    PROJ_DIR = Path(__file__).parent

# ---------------------------------------------------------------
# 设置默认的错误处理策略为 'ignore'，避免在控制台输出emoji时，编码转换失败，然后出现奇怪的乱码和报错
# 该奇怪的问题出现在 pyinstaller 打包后的可执行文件中
sys.stdin.reconfigure(encoding='utf8', errors='ignore')
sys.stdout.reconfigure(encoding='utf8', errors='ignore')
sys.stderr.reconfigure(encoding='utf8', errors='ignore')
# ---------------------------------------------------------------

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig, AlignerConfig, exporters

# 用来做命令行参数的默认值
from export_config import LLM_QUANTIZE_TYPE, ENC_QUANTIZE_TYPE

app = typer.Typer(help="Qwen3-ASR GGUF 命令行转录工具", add_completion=False)
console = Console()


def get_model_filenames(precision: str, is_aligner: bool = False):
    """根据精度返回对应的模型文件名"""
    prefix = "qwen3_aligner" if is_aligner else "qwen3_asr"
    return {
        "frontend": f"{prefix}_encoder_frontend.{precision}.onnx",
        "backend": f"{prefix}_encoder_backend.{precision}.onnx"
    }


def get_llm_filenames(precision: str, is_aligner: bool = False):
    """根据精度返回对应的模型文件名"""
    prefix = "qwen3_aligner" if is_aligner else "qwen3_asr"
    return f"{prefix}_llm.{precision}.gguf"


def check_model_files(config: ASREngineConfig):
    """检查模型文件完整性"""
    missing_files = []

    # ASR 核心文件
    asr_llm = Path(config.model_dir) / config.llm_fn
    asr_frontend = Path(config.model_dir) / config.encoder_frontend_fn
    asr_backend = Path(config.model_dir) / config.encoder_backend_fn

    for f in [asr_llm, asr_frontend, asr_backend]:
        if not f.exists():
            missing_files.append(str(f))

    # Aligner 文件
    if config.enable_aligner and config.align_config:
        align_llm = Path(config.align_config.model_dir) / config.align_config.llm_fn
        align_frontend = Path(config.align_config.model_dir) / config.align_config.encoder_frontend_fn
        align_backend = Path(config.align_config.model_dir) / config.align_config.encoder_backend_fn

        for f in [align_llm, align_frontend, align_backend]:
            if not f.exists():
                missing_files.append(str(f))

    if missing_files:
        console.print("\n[bold red]错误：找不到以下所需模型文件：[/bold red]")
        for f in missing_files:
            console.print(f"  - {f}")
        console.print("\n[bold yellow]请到以下链接下载模型文件，并解压到 model 目录：[/bold yellow]")
        console.print("[blue]https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/tag/models[/blue]\n")
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    files: List[Path] = typer.Argument(..., help="要转录的音频文件列表"),
    out_dir: Path|None = typer.Option(None, "--output-dir", "-o", help="输出目录，不设定时，默认输出到源文件所在目录", rich_help_panel="输入输出"),

    # 组 1: 模型与硬件
    model_dir: str = typer.Option(str(PROJ_DIR / "model"), "--model-dir", "-m", help="模型权重根目录", rich_help_panel="模型配置"),
    enc_precision: str = typer.Option(ENC_QUANTIZE_TYPE, "--enc-prec", help="使用的编码器精度: 可选：fp16 int8 int4", rich_help_panel="模型配置"),
    llm_precision: str = typer.Option(LLM_QUANTIZE_TYPE, "--llm-prec", help="使用的LLM模型精度，可选：f16 q8_0 q6_k q4_k", rich_help_panel="模型配置"),
    timestamp: bool = typer.Option(True, "--timestamp/--no-ts", help="是否开启时间戳引擎", rich_help_panel="模型配置"),
    onnx_provider: str = typer.Option("DML", "--provider", "-p", help="ONNX 执行后端: CPU, CUDA, DML, TRT", rich_help_panel="模型配置"),
    llm_use_gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="LLM 是否使用 GPU 加速", rich_help_panel="模型配置"),
    use_vulkan: bool = typer.Option(True, "--vulkan/--no-vulkan", help="是否开启 Vulkan 加速 (设置 GGML_VULKAN=1)", rich_help_panel="模型配置"),
    n_ctx: int = typer.Option(2048, "--n-ctx", help="LLM 上下文窗口大小", rich_help_panel="模型配置"),

    # 组 2: 转录逻辑
    language: Optional[str] = typer.Option(None, "--language", "-l", help="强制指定语种 (例: Chinese, English)", rich_help_panel="转录设置"),
    context: str = typer.Option("", "--context", "-ctx", help="上下文提示词 (Prompt)", rich_help_panel="转录设置"),
    temperature: float = typer.Option(0.6, "--temperature", help="采样温度", rich_help_panel="转录设置"),

    seek_start: float = typer.Option(0.0, "--seek-start", "-ss", help="音频开始位置 (秒)", rich_help_panel="音频切片"),
    duration: Optional[float] = typer.Option(None, "--duration", "-t", help="处理音频的时长 (秒)", rich_help_panel="音频切片"),

    # 组 2.1：额外的解码配置
    top_k: int = typer.Option(20, "--top-k", help="仅保留概率最高的top_k个token，其余token概率置零，再重新归一化", rich_help_panel="额外解码设定"),
    top_p: float = typer.Option(0.95, "--top-p", help="Nucleus Sampling，从概率最高的token开始累加，直到累计概率超过top_p，然后仅在此'核心'集合中采样", rich_help_panel="额外解码设定"),
    min_p: float = typer.Option(0.05, "--min-p", help="相对阈值过滤。移除所有概率低于 max_prob * min_p 的 token（max_prob 是当前最高概率 token 的概率）", rich_help_panel="额外解码设定"),
    repeat_penalty: float = typer.Option(1.0, "--repeat-penalty", help="若 token 已出现在最近的 penalty_last_n 序列中，其logit除以 repeat_penalty", rich_help_panel="额外解码设定"),
    frequency_penalty: float = typer.Option(0.02, "--frequency-penalty", help="token 在序列中出现的次数越多，惩罚就越多。logit=logit-frequency_penalty*count", rich_help_panel="额外解码设定"),
    presence_penalty: float = typer.Option(0.0, "--presence-penalty", help="当token在序列中出现过一次就施加一个固定惩罚，该惩罚固定的，不会随次数增加而增加。logit=logit-presence_penalty*occurred（occurred为0或1）", rich_help_panel="额外解码设定"),
    penalty_last_n: int = typer.Option(20, "--penalty-last-n", help="只考虑最近生成的 penalty_last_n 个 token 来计算惩罚", rich_help_panel="额外解码设定"),

    # 组 3: 音频裁剪与性能
    chunk_size: float = typer.Option(40.0, "--chunk-size", "-c", help="分段识别时长 (秒)", rich_help_panel="流式配置"),
    memory_num: int = typer.Option(1, "--memory-num", help="记忆的历史片段数量", rich_help_panel="流式配置"),

    # 组 4: 其他
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="是否打印详细日志", rich_help_panel="其他选项"),

    # 组 5: 可以关闭部分输出
    no_json: bool = typer.Option(False, "--no-json", "-njson", help="不输出json文件", rich_help_panel="输出控制"),
    no_srt: bool = typer.Option(False, "--no-srt", "-nsrt", help="不输出srt文件", rich_help_panel="输出控制"),
    no_txt: bool = typer.Option(False, "--no-txt", "-ntxt", help="不输出txt文件", rich_help_panel="输出控制"),

):
    """
    使用 Qwen3-ASR GGUF 模型对音频进行高精度转录。
    """

    # 1. 环境准备
    if not use_vulkan:
        os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan

    # 2. 构造配置
    asr_files = get_model_filenames(enc_precision, is_aligner=False)
    align_files = get_model_filenames(enc_precision, is_aligner=True)

    asr_gguf = get_llm_filenames(llm_precision, is_aligner=False)
    align_gguf = get_llm_filenames(llm_precision, is_aligner=True)

    align_config = None
    if timestamp:
        align_config = AlignerConfig(
            llm_fn=align_gguf,
            model_dir=model_dir,
            onnx_provider=onnx_provider,
            llm_use_gpu=llm_use_gpu,
            encoder_frontend_fn=align_files["frontend"],
            encoder_backend_fn=align_files["backend"],
            n_ctx=n_ctx
        )

    config = ASREngineConfig(
        llm_fn=asr_gguf,
        model_dir=model_dir,
        onnx_provider=onnx_provider,
        llm_use_gpu=llm_use_gpu,
        encoder_frontend_fn=asr_files["frontend"],
        encoder_backend_fn=asr_files["backend"],
        n_ctx=n_ctx,
        chunk_size=chunk_size,
        memory_num=memory_num,
        enable_aligner=timestamp,
        align_config=align_config,
        verbose=verbose
    )

    decode_other_kwargs = {
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "penalty_last_n": penalty_last_n,
    }

    # 3. 打印配置面板
    config_table = Table(show_header=False, box=None)
    config_table.add_row("输出目录", f"[green]{out_dir or '源文件所在目录'}[/green]")
    config_table.add_row("模型目录", f"[green]{model_dir}[/green]")
    config_table.add_row("编码器精度", f"[cyan]{enc_precision}[/cyan]")
    config_table.add_row("解码器精度", f"[cyan]{llm_precision}[/cyan]")
    config_table.add_row("加速设备", f"ONNX:{onnx_provider} | LLM-GPU:{'[green]ON[/green]' if llm_use_gpu else '[red]OFF[/red]'} | Vulkan:{'[green]ON[/green]' if use_vulkan else '[red]OFF[/red]'}")
    config_table.add_row("时间戳对齐", f"{'[green]启用[/green]' if timestamp else '[red]禁用[/red]'}")
    config_table.add_row("语言设定", f"{language or '自动识别'}")

    console.print(Panel(config_table, title="[bold cyan]Qwen3-ASR 配置选项[/bold cyan]", expand=False))

    # 4. 检查模型文件是否存在
    check_model_files(config)

    # 5. 初始化引擎
    with console.status("[bold yellow]正在初始化引擎，请稍候...[/bold yellow]") as status:
        try:
            t0 = time.time()
            engine = QwenASREngine(config=config, decode_other_kwargs=decode_other_kwargs)
            init_duration = time.time() - t0
            console.print(f"--- [QwenASR] 引擎初始化耗时: {init_duration:.2f} 秒 ---")
        except Exception as e:
            console.print(f"[bold red]引擎初始化失败:[/bold red]\n{e}")
            console.print(f"[bold yellow]建议解决方案：[/bold yellow]")
            console.print(f"  1. 尝试使用 CPU 后端: 使用 [cyan]--provider CPU --no-gpu[/cyan]")
            console.print(f"  2. 尝试关闭 Vulkan 加速: 使用 [cyan]--no-vulkan[/cyan]")
            console.print(f"  3. 如果问题仍然存在，请在 GitHub 提交 Issue 并附带 [cyan]{PROJ_DIR}\\logs\\latest.log[/cyan] 日志文件。")
            raise typer.Exit(code=1)

    # 6. 循环处理文件
    try:
        for audio_path in files:
            if not audio_path.exists():
                console.print(f"[yellow]跳过不存在的文件: {audio_path}[/yellow]")
                continue

            console.print(f"\n[bold blue]开始处理:[/bold blue] {audio_path.name}\n")

            # 检查输出文件冲突
            if out_dir is None:
                base_out = audio_path.with_suffix("")
            else:
                base_out = out_dir / audio_path.stem

            base_out.parent.mkdir(parents=True, exist_ok=True)

            txt_out = f"{base_out}.txt"

            res = engine.transcribe(
                audio_file=str(audio_path),
                language=language,
                context=context,
                start_second=seek_start,
                duration=duration,
                temperature=temperature,
            )

            # 7. 导出结果
            if not no_txt:
                exporters.export_to_txt(txt_out, res)

            if timestamp and res.alignment:
                srt_out = f"{base_out}.srt"
                if not no_srt:
                    exporters.export_to_srt(srt_out, res)

                json_out = f"{base_out}.json"
                if not no_json:
                    exporters.export_to_json(json_out, res)

    finally:
        engine.shutdown()
        console.print("\n[bold green]所有任务已完成。[/bold green]")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    app()
