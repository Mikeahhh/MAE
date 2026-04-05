"""
Shared SNR formatting and parsing utilities.

All scripts use these functions to ensure consistent SNR tag formatting
across directory names, filenames, JSON keys, and CLI arguments.

Backward compatible: integer SNR values produce the same tags as the old
`f"snr_{snr:+d}dB"` format (e.g. "snr_+5dB", "snr_-10dB").
"""
from __future__ import annotations


def format_snr_tag(snr_db: float) -> str:
    """Format SNR value as a directory/key tag.

    Examples:
        format_snr_tag(-10)  -> "snr_-10dB"
        format_snr_tag(5)    -> "snr_+5dB"
        format_snr_tag(-2.5) -> "snr_-2.5dB"
        format_snr_tag(0)    -> "snr_+0dB"
    """
    if snr_db == int(snr_db):
        return f"snr_{int(snr_db):+d}dB"
    return f"snr_{snr_db:+g}dB"


def format_snr_dir(snr_db: float) -> str:
    """Format SNR value as a directory name (same as tag)."""
    return format_snr_tag(snr_db)


def parse_snr_tag(tag: str) -> float:
    """Parse an SNR tag string back to a float value.

    Examples:
        parse_snr_tag("snr_-10dB")  -> -10.0
        parse_snr_tag("snr_+5dB")   ->  5.0
        parse_snr_tag("snr_-2.5dB") -> -2.5
    """
    s = tag.replace("snr_", "").replace("dB", "")
    return float(s)


def generate_fine_snr_values(
    start: float = -15, stop: float = 20, step: float = 0.5,
) -> list[float]:
    """Generate fine-grained SNR values from start to stop (inclusive).

    Default: -15.0 to +20.0 in 0.5 dB steps = 71 values.
    """
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 1) for i in range(n)]


COARSE_SNR: list[float] = [-15, -10, -5, 0, 5, 10, 15, 20]
FINE_SNR: list[float] = generate_fine_snr_values()  # 71 values


def format_height_tag(height_m: float) -> str:
    """Format flight height as a directory tag.

    Examples:
        format_height_tag(5)   -> "h_05m"
        format_height_tag(50)  -> "h_50m"
        format_height_tag(7.5) -> "h_7.5m"
    """
    if height_m == int(height_m):
        return f"h_{int(height_m):02d}m"
    return f"h_{height_m}m"


def parse_height_tag(tag: str) -> float:
    """Parse a height tag string back to a float value.

    Examples:
        parse_height_tag("h_05m")  -> 5.0
        parse_height_tag("h_50m")  -> 50.0
        parse_height_tag("h_7.5m") -> 7.5
    """
    s = tag.replace("h_", "").replace("m", "")
    return float(s)
