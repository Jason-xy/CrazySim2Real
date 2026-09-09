"""Deterministic synthetic runs, not drone recordings or flight validation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .protocol import CHANNELS


def make_run(path, seed=0, duration_s=40, fs=100,
             gain=1.0, tau_s=0.04, delay_s=0.01, noise_std=0.002, channels=None,
             second_order=False, damping=0.5, natural_frequency_hz=5.0):
    """Generate a known discrete response to independent broadband commands."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Synthetic output directory must be new or empty")
    channels = set(CHANNELS if channels is None else channels)
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * fs)) / fs
    data = {"t_s": t}
    if second_order:
        wn = natural_frequency_hz * 2 * np.pi
        b, a, _ = signal.cont2discrete(([gain * wn ** 2], [1, 2 * damping * wn, wn ** 2]), 1 / fs)
        b = b.ravel()
    else:
        alpha = np.exp(-1 / (fs * tau_s))
        b, a = [gain * (1 - alpha)], [1, -alpha]
    delay = int(round(delay_s * fs))
    for key, (r_column, y_column) in CHANNELS.items():
        if key not in channels:
            continue
        excitation = signal.lfilter([0.4], [1, -0.6], rng.normal(size=len(t)))
        excitation *= 10 if key.startswith("angle.") else 50
        response = signal.lfilter(b, a, excitation)
        if delay:
            response = np.r_[np.zeros(delay), response[:-delay]]
        response += noise_std * rng.normal(size=len(t))
        data[r_column], data[y_column] = excitation, response
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path / "samples.csv", index=False)
    return path


def generate_demo(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Demo output must be new or empty")
    manifest = {"reference_group": "real", "baseline_group": "sim_before", "groups": {}}
    for i, (name, gain, tau, delay) in enumerate((
        ("real", 1.0, 0.04, 0.01),
        ("sim_before", 0.7, 0.09, 0.04),
        ("sim_after", 0.98, 0.043, 0.01),
    )):
        manifest["groups"][name] = []
        for repeat in range(3):
            run_id = f"{name}-{repeat}"
            make_run(output / run_id, seed=i * 100 + repeat, gain=gain, tau_s=tau, delay_s=delay)
            manifest["groups"][name].append(run_id)
    (output / "comparison.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output / "comparison.json"


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic examples without any flight connection")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(generate_demo(args.out))


if __name__ == "__main__":
    main()
