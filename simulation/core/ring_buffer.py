"""
Retroactive DMA ring buffer module (Paper Section II.C).

Implements zero-loss retroactive trigger mechanism:
  - In State 0 (DSP low-power), DMA continuously writes to ring buffer
  - On hardware interrupt, system reads back (dt + d_wake) seconds of history
  - Guarantees direct-path wavefront is not lost (99.7% signal integrity)

Paper formula (8):
    n_ret = (dt + d_wake) * Fs
    dt     = 0.5 s   (retroactive margin, tolerates MAE detection latency)
    d_wake = 0.3 s   (hardware State 0 -> State 1 wake latency)
    Fs     = 48000 Hz
    => n_ret = 38400 samples = 800 ms history depth
"""

import numpy as np
from typing import Tuple


class RetroactiveRingBuffer:
    """
    Multi-channel retroactive ring buffer.

    DMA writes continuously in State 0; on trigger, reads back historical audio.
    """

    def __init__(
        self,
        n_mics: int = 9,
        fs: int = 48000,
        delta_t: float = 0.5,
        delta_wake: float = 0.3
    ):
        """
        Args:
            n_mics:     Number of microphones (Paper: M=9)
            fs:         Sample rate [Hz] (Paper: 48 kHz)
            delta_t:    Retroactive margin [s] (Paper Table I: 500 ms)
            delta_wake: Hardware wake latency [s] (Paper Table I: 300 ms)
        """
        self.n_mics = n_mics
        self.fs = fs
        self.delta_t = delta_t
        self.delta_wake = delta_wake

        # Paper formula (8): n_ret = (dt + d_wake) * Fs
        self.n_ret = int((delta_t + delta_wake) * fs)  # = 38400 samples
        self.buffer_size = self.n_ret * 2               # = 76800 samples

        self.buffer = np.zeros((n_mics, self.buffer_size), dtype=np.float32)
        self.write_ptr = 0
        self.total_written = 0

    def write(self, mic_signals_chunk: np.ndarray) -> None:
        """Write multi-channel audio chunk to circular buffer (DMA simulation)."""
        chunk_size = mic_signals_chunk.shape[1]
        assert chunk_size <= self.buffer_size, (
            f"Chunk size ({chunk_size}) exceeds buffer capacity ({self.buffer_size}). "
            f"Ring buffer would silently overwrite unread data."
        )
        remaining_space = self.buffer_size - self.write_ptr

        if chunk_size <= remaining_space:
            self.buffer[:, self.write_ptr:self.write_ptr + chunk_size] = mic_signals_chunk
        else:
            self.buffer[:, self.write_ptr:] = mic_signals_chunk[:, :remaining_space]
            overflow = chunk_size - remaining_space
            self.buffer[:, :overflow] = mic_signals_chunk[:, remaining_space:]

        self.write_ptr = (self.write_ptr + chunk_size) % self.buffer_size
        self.total_written += chunk_size

    def read_retroactive(self) -> Tuple[np.ndarray, bool]:
        """
        Read n_ret historical samples (called on trigger).

        Returns:
            (retroactive_audio, is_valid)
            retroactive_audio: (n_mics, n_ret) historical audio
            is_valid: True if buffer has enough history
        """
        if self.total_written < self.n_ret:
            is_valid = False
            available = self.total_written
            result = np.zeros((self.n_mics, self.n_ret), dtype=np.float32)
            if available > 0:
                start_ptr = (self.write_ptr - available) % self.buffer_size
                result[:, self.n_ret - available:] = self._read_circular(start_ptr, available)
            return result, is_valid

        start_ptr = (self.write_ptr - self.n_ret) % self.buffer_size
        retroactive_audio = self._read_circular(start_ptr, self.n_ret)
        return retroactive_audio, True

    def read_latest(self, n_samples: int) -> np.ndarray:
        """Read the most recent n_samples."""
        n_samples = min(n_samples, self.total_written, self.buffer_size)
        start_ptr = (self.write_ptr - n_samples) % self.buffer_size
        return self._read_circular(start_ptr, n_samples)

    def _read_circular(self, start_ptr: int, n_samples: int) -> np.ndarray:
        """Read contiguous data from circular buffer."""
        end_ptr = start_ptr + n_samples
        if end_ptr <= self.buffer_size:
            return self.buffer[:, start_ptr:end_ptr].copy()
        else:
            part1 = self.buffer[:, start_ptr:]
            part2 = self.buffer[:, :end_ptr - self.buffer_size]
            return np.concatenate([part1, part2], axis=1)

    def get_buffer_delay_seconds(self) -> float:
        return min(self.total_written, self.buffer_size) / self.fs

    def get_retroactive_delay_seconds(self) -> float:
        return self.delta_t + self.delta_wake

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.write_ptr = 0
        self.total_written = 0

    def is_warmed_up(self) -> bool:
        return self.total_written >= self.n_ret

    def __repr__(self) -> str:
        return (
            f"RetroactiveRingBuffer("
            f"n_mics={self.n_mics}, fs={self.fs}, "
            f"dt={self.delta_t}s, d_wake={self.delta_wake}s, "
            f"n_ret={self.n_ret} [{self.n_ret / self.fs * 1000:.0f} ms], "
            f"warmed_up={self.is_warmed_up()})"
        )
