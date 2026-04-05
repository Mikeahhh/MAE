"""
音频混合工具
用于生成训练和测试数据

功能：
1. 混合无人机噪音 + 环境噪音 → 训练集（正常样本）
2. 混合无人机噪音 + 环境噪音 + 人声 → 测试集（异常样本，不同SNR）
3. 音频重采样、归一化
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import argparse


def load_audio(file_path: str, sr: int = 48000) -> Tuple[np.ndarray, int]:
    """
    加载音频文件

    Args:
        file_path: 音频文件路径
        sr: 目标采样率

    Returns:
        audio: 音频数据 (n_samples,)
        sr: 采样率
    """
    # Use soundfile first (handles more WAV codecs than audioread)
    try:
        audio, orig_sr = sf.read(file_path, dtype="float32")
    except Exception:
        audio, orig_sr = librosa.load(file_path, sr=None, mono=True)

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # 重采样到目标采样率
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    return audio, sr


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    归一化音频到目标dB

    Args:
        audio: 输入音频
        target_db: 目标dB值

    Returns:
        normalized_audio: 归一化后的音频
    """
    # 计算当前RMS
    rms = np.sqrt(np.mean(audio ** 2))

    # 避免除零
    if rms < 1e-10:
        return audio

    # 计算当前dB
    current_db = 20 * np.log10(rms)

    # 计算增益
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)

    # 应用增益
    normalized = audio * gain_linear

    # 防止削波
    max_val = np.abs(normalized).max()
    if max_val > 0.99:
        normalized = normalized * 0.99 / max_val

    return normalized


def mix_audio_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    按指定SNR混合信号和噪音

    Args:
        signal: 信号（如人声）
        noise: 噪音（如无人机+环境噪音）
        snr_db: 信噪比（dB）

    Returns:
        mixed: 混合后的音频
    """
    # 确保长度一致
    min_len = min(len(signal), len(noise))
    signal = signal[:min_len]
    noise = noise[:min_len]

    # 计算RMS
    signal_rms = np.sqrt(np.mean(signal ** 2))
    noise_rms = np.sqrt(np.mean(noise ** 2))

    # 避免除零
    if noise_rms < 1e-10:
        return signal + noise

    # 计算所需的噪音增益
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise_gain = target_noise_rms / noise_rms

    # 混合
    mixed = signal + noise * noise_gain

    # 归一化防止削波
    max_val = np.abs(mixed).max()
    if max_val > 0.99:
        mixed = mixed * 0.99 / max_val

    return mixed


def mix_physics_attenuation(
    voice: np.ndarray,
    background: np.ndarray,
    attenuation_linear: float,
) -> tuple:
    """
    Physics-based audio mixing using propagation model attenuation.

    Whiteboard formula: attenuated_voice = voice * attenuation_linear
    where attenuation_linear ≈ 1/d (waveform domain).

    Args:
        voice: clean voice waveform
        background: drone + ambient background waveform
        attenuation_linear: waveform attenuation factor from PropagationModel

    Returns:
        (mixed, snr_db): mixed audio and the resulting physical SNR in dB
    """
    min_len = min(len(voice), len(background))
    voice = voice[:min_len]
    background = background[:min_len]

    voice_attenuated = voice * attenuation_linear
    mixed = voice_attenuated + background

    # Compute resulting SNR
    voice_power = np.sum(voice_attenuated ** 2)
    bg_power = np.sum(background ** 2)
    if bg_power < 1e-20:
        snr_db = float('inf')
    else:
        snr_db = 10.0 * np.log10(voice_power / bg_power)

    # Prevent clipping
    max_val = np.abs(mixed).max()
    if max_val > 0.99:
        mixed = mixed * 0.99 / max_val

    return mixed, snr_db


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    填充或裁剪音频到目标长度

    Args:
        audio: 输入音频
        target_length: 目标长度（采样点数）

    Returns:
        processed_audio: 处理后的音频
    """
    current_length = len(audio)

    if current_length < target_length:
        # 填充
        pad_length = target_length - current_length
        audio = np.pad(audio, (0, pad_length), mode='constant')
    elif current_length > target_length:
        # 裁剪（随机起始位置）
        start = np.random.randint(0, current_length - target_length + 1)
        audio = audio[start:start + target_length]

    return audio


def generate_training_sample(
    drone_noise_path: str,
    ambient_noise_path: str,
    output_path: str,
    sr: int = 48000,
    duration: float = 1.0
):
    """
    生成训练样本（无人机噪音 + 环境噪音）

    Args:
        drone_noise_path: 无人机噪音文件路径
        ambient_noise_path: 环境噪音文件路径
        output_path: 输出文件路径
        sr: 采样率
        duration: 持续时间（秒）
    """
    target_length = int(sr * duration)

    # 加载音频
    drone_audio, _ = load_audio(drone_noise_path, sr)
    ambient_audio, _ = load_audio(ambient_noise_path, sr)

    # 调整长度
    drone_audio = pad_or_trim(drone_audio, target_length)
    ambient_audio = pad_or_trim(ambient_audio, target_length)

    # 归一化
    drone_audio = normalize_audio(drone_audio, target_db=-20.0)
    ambient_audio = normalize_audio(ambient_audio, target_db=-25.0)

    # 混合（环境噪音稍弱）
    mixed = drone_audio + ambient_audio * 0.5

    # 归一化防止削波
    max_val = np.abs(mixed).max()
    if max_val > 0.99:
        mixed = mixed * 0.99 / max_val

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, mixed, sr)

    print(f"Generated training sample: {output_path}")


def generate_test_sample_normal(
    drone_noise_path: str,
    ambient_noise_path: str,
    output_path: str,
    sr: int = 48000,
    duration: float = 1.0
):
    """
    生成测试样本（正常，无异常信号）
    与训练样本相同，但用于测试
    """
    generate_training_sample(
        drone_noise_path,
        ambient_noise_path,
        output_path,
        sr,
        duration
    )


def generate_test_sample_anomaly(
    drone_noise_path: str,
    ambient_noise_path: str,
    human_voice_path: str,
    output_path: str,
    snr_db: float,
    sr: int = 48000,
    duration: float = 1.0
):
    """
    生成测试样本（异常，含人声）

    Args:
        drone_noise_path: 无人机噪音文件路径
        ambient_noise_path: 环境噪音文件路径
        human_voice_path: 人声文件路径
        output_path: 输出文件路径
        snr_db: 人声的信噪比（相对于背景噪音）
        sr: 采样率
        duration: 持续时间（秒）
    """
    target_length = int(sr * duration)

    # 加载音频
    drone_audio, _ = load_audio(drone_noise_path, sr)
    ambient_audio, _ = load_audio(ambient_noise_path, sr)
    human_audio, _ = load_audio(human_voice_path, sr)

    # 调整长度
    drone_audio = pad_or_trim(drone_audio, target_length)
    ambient_audio = pad_or_trim(ambient_audio, target_length)
    human_audio = pad_or_trim(human_audio, target_length)

    # 归一化
    drone_audio = normalize_audio(drone_audio, target_db=-20.0)
    ambient_audio = normalize_audio(ambient_audio, target_db=-25.0)
    human_audio = normalize_audio(human_audio, target_db=-15.0)

    # 先混合背景噪音
    background = drone_audio + ambient_audio * 0.5

    # 按SNR混合人声
    mixed = mix_audio_snr(human_audio, background, snr_db)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, mixed, sr)

    print(f"Generated anomaly sample (SNR={snr_db}dB): {output_path}")


def batch_generate_training_data(
    drone_noise_dir: str,
    ambient_noise_dir: str,
    output_dir: str,
    scenario: str,
    n_samples: int = 1000,
    sr: int = 48000
):
    """
    批量生成训练数据

    Args:
        drone_noise_dir: 无人机噪音目录
        ambient_noise_dir: 环境噪音目录
        output_dir: 输出目录
        scenario: 场景名称（desert/forest/ocean）
        n_samples: 生成样本数量
        sr: 采样率
    """
    # 获取文件列表
    drone_files = list(Path(drone_noise_dir).glob("*.wav"))
    ambient_files = list(Path(ambient_noise_dir).glob("*.wav"))

    if len(drone_files) == 0:
        print(f"Error: No drone noise files found in {drone_noise_dir}")
        return

    if len(ambient_files) == 0:
        print(f"Error: No ambient noise files found in {ambient_noise_dir}")
        return

    print(f"Found {len(drone_files)} drone noise files")
    print(f"Found {len(ambient_files)} ambient noise files")
    print(f"Generating {n_samples} training samples for {scenario}...")

    # 生成样本
    for i in range(n_samples):
        # 随机选择文件
        drone_file = np.random.choice(drone_files)
        ambient_file = np.random.choice(ambient_files)

        # 输出路径
        output_path = os.path.join(
            output_dir,
            f"{scenario}_train_normal_{i:04d}.wav"
        )

        # 生成
        try:
            generate_training_sample(
                str(drone_file),
                str(ambient_file),
                output_path,
                sr=sr
            )
        except Exception as e:
            print(f"Error generating sample {i}: {e}")

    print(f"Done! Generated {n_samples} samples in {output_dir}")


def batch_generate_test_data(
    drone_noise_dir: str,
    ambient_noise_dir: str,
    human_voice_dir: str,
    output_dir: str,
    scenario: str,
    n_normal: int = 100,
    n_anomaly_per_snr: int = 15,
    snr_values: list = [-10, -5, 0, 5, 10, 15, 20],
    sr: int = 48000
):
    """
    批量生成测试数据

    Args:
        drone_noise_dir: 无人机噪音目录
        ambient_noise_dir: 环境噪音目录
        human_voice_dir: 人声目录
        output_dir: 输出目录
        scenario: 场景名称
        n_normal: 正常样本数量
        n_anomaly_per_snr: 每个SNR的异常样本数量
        snr_values: SNR值列表
        sr: 采样率
    """
    # 获取文件列表
    drone_files = list(Path(drone_noise_dir).glob("*.wav"))
    ambient_files = list(Path(ambient_noise_dir).glob("*.wav"))
    human_files = list(Path(human_voice_dir).glob("*.wav"))

    print(f"Generating test data for {scenario}...")

    # 生成正常样本
    print(f"Generating {n_normal} normal samples...")
    normal_dir = os.path.join(output_dir, "normal")
    for i in range(n_normal):
        drone_file = np.random.choice(drone_files)
        ambient_file = np.random.choice(ambient_files)

        output_path = os.path.join(
            normal_dir,
            f"{scenario}_test_normal_{i:04d}.wav"
        )

        try:
            generate_test_sample_normal(
                str(drone_file),
                str(ambient_file),
                output_path,
                sr=sr
            )
        except Exception as e:
            print(f"Error: {e}")

    # 生成异常样本（不同SNR）
    print(f"Generating anomaly samples for SNR values: {snr_values}...")
    anomaly_dir = os.path.join(output_dir, "anomaly")

    for snr in snr_values:
        print(f"  SNR = {snr} dB...")
        for i in range(n_anomaly_per_snr):
            drone_file = np.random.choice(drone_files)
            ambient_file = np.random.choice(ambient_files)
            human_file = np.random.choice(human_files)

            output_path = os.path.join(
                anomaly_dir,
                f"{scenario}_test_anomaly_snr_{snr:+d}_{i:04d}.wav"
            )

            try:
                generate_test_sample_anomaly(
                    str(drone_file),
                    str(ambient_file),
                    str(human_file),
                    output_path,
                    snr_db=snr,
                    sr=sr
                )
            except Exception as e:
                print(f"Error: {e}")

    print(f"Done! Test data saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training and test data")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Generation mode: train or test")
    parser.add_argument("--scenario", type=str, required=True,
                        help="Scenario name (e.g., desert, forest, ocean)")
    parser.add_argument("--drone_dir", type=str, required=True,
                        help="Directory containing drone noise files")
    parser.add_argument("--ambient_dir", type=str, required=True,
                        help="Directory containing ambient noise files")
    parser.add_argument("--human_dir", type=str, default=None,
                        help="Directory containing human voice files (for test mode)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples to generate (for train mode)")
    parser.add_argument("--sr", type=int, default=48000,
                        help="Sample rate")

    args = parser.parse_args()

    if args.mode == "train":
        batch_generate_training_data(
            drone_noise_dir=args.drone_dir,
            ambient_noise_dir=args.ambient_dir,
            output_dir=args.output_dir,
            scenario=args.scenario,
            n_samples=args.n_samples,
            sr=args.sr
        )
    elif args.mode == "test":
        if args.human_dir is None:
            print("Error: --human_dir is required for test mode")
            exit(1)

        batch_generate_test_data(
            drone_noise_dir=args.drone_dir,
            ambient_noise_dir=args.ambient_dir,
            human_voice_dir=args.human_dir,
            output_dir=args.output_dir,
            scenario=args.scenario,
            sr=args.sr
        )
