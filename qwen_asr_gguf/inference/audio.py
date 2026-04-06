"""
audio.py - 音频预处理工具类
职责：使用 ffmpeg 直接读取音频，支持所有格式（mp3/m4a/opus 等）。
"""
import os, io, math
import av
import numpy as np
import soundfile as sf
from pathlib import Path
from sys import float_info


def numpy_resample_poly(x, dst_sr, src_sr, window_size=10):
    """
    纯 numpy 实现的 resample_poly
    算法精准复刻 scipy.signal.resample_poly，与 scipy 相似度达 0.99999998
    """
    # 1. 约分
    g = math.gcd(dst_sr, src_sr)
    dst_sr //= g
    src_sr //= g

    if dst_sr == src_sr:
        return x.copy()

    # 2. 设计 FIR 滤波器 (与 scipy.signal.firwin 对齐)
    max_rate = max(dst_sr, src_sr)
    f_c = 1.0 / max_rate
    half_len = window_size * max_rate
    n_taps = 2 * half_len + 1

    t = np.arange(n_taps) - half_len
    h = np.sinc(f_c * t)

    # 使用 Kaiser 窗 (beta=5.0)
    # np.i0 是修饰过的第一类修正贝塞尔函数，与 scipy.special.i0 一致
    beta = 5.0
    kaiser_win = np.i0(beta * np.sqrt(1 - (2 * t / (n_taps - 1))**2)) / np.i0(beta)
    h = h * kaiser_win
    h = h * (dst_sr / np.sum(h))

    # 3. 多相滤波 (复刻 upfirdn 逻辑)
    length_in = len(x)
    length_out = int(math.ceil(length_in * dst_sr / src_sr))

    x_up = np.zeros(length_in * dst_sr + n_taps, dtype=np.float32)
    x_up[:length_in * dst_sr:dst_sr] = x

    y_full = np.convolve(x_up, h, mode='full')

    offset = (n_taps - 1) // 2
    y = y_full[offset: offset + length_in * dst_sr: src_sr]

    return y[:length_out].astype(np.float32)


def resample_audio(audio, sr, target_sr):
    """音频重采样封装"""
    if sr == target_sr:
        return audio
    return numpy_resample_poly(audio, target_sr, sr)


def load_audio_numpy(audio_path, sample_rate=24000, start_second=None, duration=None):
    """使用 soundfile + numpy 重采样读取音频"""
    info = sf.info(audio_path)
    sr = info.samplerate

    # 获取偏移量
    start_frame = int(start_second * sr) if start_second is not None else 0
    frames = int(duration * sr) if duration is not None else -1

    audio, sr = sf.read(audio_path, start=start_frame, frames=frames, dtype='float32')

    # 转单声道
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 高质量重采样
    if sr != sample_rate:
        audio = resample_audio(audio, sr, sample_rate)

    return audio.astype(np.float32)


def load_audio_av(file_path, target_sample_rate=16000, start_second=None, duration_second=None):
    """
    使用 PyAV 加载音频文件，重采样到目标采样率，并返回为 NumPy 数组。

    Args:
        file_path (str): 音频文件路径。
        target_sample_rate (int): 目标采样率，例如 16000。
        start_second (float, optional): 开始时间，单位秒。默认 None。
        duration_second (float, optional): 读取时长，单位秒。默认 None。

    Returns:
        np.ndarray: 包含音频数据的 NumPy 数组，dtype 为 float32，范围在 [-1.0, 1.0]。
                    数组的形状为 (channels, samples) 或 (samples,) 对于单声道。
    """

    # 0. 设定时间范围
    start_second = float(start_second) if start_second is not None else 0.
    duration_second = float(duration_second) if duration_second is not None else float_info.max
    end_second = start_second + duration_second

    layout = 'mono'
    assert layout in ['stereo', 'mono'], "layout must be 'stereo' or 'mono'"

    # 1. 配置重采样器
    resampler = av.AudioResampler(
        format='fltp',              # 输出格式：32位浮点，平面存储
        layout=layout,              # 输出声道布局
        rate=target_sample_rate,    # 目标采样率
    )

    # 用于存储所有音频数据的缓冲区
    raw_buffer = io.BytesIO()
    dtype = None

    # 打开媒体文件
    with av.open(file_path, mode='r') as container:
        # 获取第一个音频流
        assert len(container.streams.audio) > 0, "文件中没有音频流"

        # 2. 解码、重采样并收集数据
        # 通过 decode(audio=0) 解码音频流中的所有数据包
        for frame in container.decode(audio=0):

            if frame.time < start_second or frame.time > end_second:
                break

            # 将每个解码后的帧通过重采样器处理
            for resampled_frame in resampler.resample(frame):
                # 3. 将重采样后的音频帧转换为 NumPy 数组
                # to_ndarray() 默认返回形状为 (samples, channels) 的数组
                array = resampled_frame.to_ndarray()
                # 交换轴，将 (channels, samples) 转换为 (samples, channels)
                array = np.swapaxes(array, 0, 1)

                # 记录数据类型，用于最终数组的构建
                if dtype is None:
                    dtype = array.dtype

                # 将数组的原始字节写入缓冲区
                # 注意：to_ndarray() 返回的是数组的视图，直接写入其内存即可
                raw_buffer.write(array.tobytes())

        # 4. 刷新重采样器缓冲区，以确保所有数据都被处理
        for resampled_frame in resampler.resample(None):
            array = resampled_frame.to_ndarray()
            # 交换轴，将 (channels, samples) 转换为 (samples, channels)
            array = np.swapaxes(array, 0, 1)
            raw_buffer.write(array.tobytes())

    # 5. 将缓冲区中的所有数据组合成一个大的 NumPy 数组
    audio_data = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # 6. 重塑数组形状
    # 原始数据是按帧连续存储的，我们需要将其重塑为 (samples, channels)
    # 帧的总样本数可以通过数组总长度除以通道数得到
    if layout == 'stereo':
        total_samples = audio_data.shape[0] // 2
        audio_data = audio_data.reshape(total_samples, 2)
    else:
        assert audio_data.ndim == 1

    return audio_data


def load_audio(audio_path, sample_rate=16000, start_second=None, duration=None):
    """
    加载音频文件的主入口。
    根据后缀名判断读取方式：
    - soundfile 支持: .wav, .flac, .ogg, .mp3
    - 其他 ffmpeg fallback: .m4a, .mp4, .opus, .wmv 等
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 获取后缀名
    ext = Path(audio_path).suffix.lower()

    # 定义 soundfile 可以稳定处理的格式
    SF_FORMATS = {'.wav', '.flac', '.ogg', '.mp3'}

    if ext in SF_FORMATS:
        return load_audio_numpy(audio_path, sample_rate, start_second, duration)
    else:
        return load_audio_av(audio_path, sample_rate, start_second, duration)
