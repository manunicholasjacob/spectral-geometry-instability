#!/usr/bin/env python3
"""Sweep over top-k eigenspace dimensions."""

from run_robustness_sweep import run_robustness_sweep

if __name__ == "__main__":
    run_robustness_sweep("configs/universe_sector_etfs.yaml", "top_k")
