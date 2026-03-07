# Results Summary

## Project: Spectral Geometry Instability (SGI)

**Authors**: Manu Nicholas Jacob, Ronit Ghai  
**Last Updated**: [Auto-generated on experiment run]

---

## Executive Summary

This document summarizes the key findings from the SGI research project.

### Core Hypothesis

> Large rotations of the dominant covariance eigenspace indicate structural market instability, degradation of diversification assumptions, and elevated fragility.

---

## Key Results

### 1. Signal Characteristics

- **SGI Definition**: Principal-angle-based measure of eigenspace rotation
- **Typical Range**: [To be filled after experiments]
- **Crisis Behavior**: [To be filled after experiments]

### 2. Predictive Evidence

| Target | Baseline R² | With SGI R² | Incremental |
|--------|-------------|-------------|-------------|
| Forward Drawdown (20d) | TBD | TBD | TBD |
| Forward Volatility (20d) | TBD | TBD | TBD |
| Forward Correlation (20d) | TBD | TBD | TBD |

### 3. Portfolio Performance

| Strategy | CAGR | Sharpe | Max DD | Turnover |
|----------|------|--------|--------|----------|
| Equal Weight | TBD | TBD | TBD | TBD |
| Min Variance | TBD | TBD | TBD | TBD |
| SGI-Conditioned | TBD | TBD | TBD | TBD |

### 4. Robustness

- **Window Sensitivity**: [To be filled]
- **Top-k Sensitivity**: [To be filled]
- **Estimator Sensitivity**: [To be filled]
- **Universe Generality**: [To be filled]

---

## Reviewer Defense Points

### "This is just volatility in disguise"
- Correlation between SGI and realized vol: [TBD]
- Incremental R² after controlling for vol: [TBD]

### "This is just average correlation"
- Correlation between SGI and avg correlation: [TBD]
- Incremental R² after controlling for correlation: [TBD]

### "This is just absorption ratio repackaged"
- Correlation between SGI and absorption ratio: [TBD]
- Incremental R² after controlling for AR: [TBD]

### "This only works in one crisis"
- GFC 2008 results: [TBD]
- COVID 2020 results: [TBD]
- Inflation 2022 results: [TBD]

### "This disappears with transaction costs"
- Net Sharpe at 0 bps: [TBD]
- Net Sharpe at 10 bps: [TBD]
- Net Sharpe at 25 bps: [TBD]

---

## Current Status

- [x] Repository scaffold complete
- [x] Core modules implemented
- [x] Config system working
- [ ] Full experiments run
- [ ] Results tables populated
- [ ] Paper draft started

---

## Next Steps

1. Run full signal pipeline on all universes
2. Complete predictive tests
3. Run portfolio backtests with cost sweeps
4. Generate publication-ready figures
5. Write methods section
6. Write results section

---

## Notes

[Add experiment notes and observations here]
