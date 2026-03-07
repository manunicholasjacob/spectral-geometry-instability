# Paper-Ready Results Guide

## For Co-Authors: How to Use This Project for Your Research Paper

This guide explains how to run the complete SGI analysis and extract publication-ready results for your research paper.

---

## Quick Start (5 Minutes)

### Option 1: Google Colab (Recommended)

1. **Open Google Colab**: https://colab.research.google.com
2. **Create new notebook**
3. **Paste the complete run script** (see `COLAB_COMPLETE_RUN.py`)
4. **Run the cell**
5. **Wait ~3-5 minutes**
6. **Download results** (automatically downloaded as ZIP)

### Option 2: Local Execution

```bash
# Clone repository
git clone https://github.com/manunicholasjacob/spectral-geometry-instability.git
cd spectral-geometry-instability

# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python COLAB_COMPLETE_RUN.py
```

---

## What You'll Get

### 1. **Quantitative Results**

Located in: `outputs/complete_analysis/`

| File | Contents | Use in Paper |
|------|----------|--------------|
| `results_summary.json` | All key metrics | Abstract, Results section |
| `predictive_summary.csv` | Model performance | Table 1: Predictive Results |
| `portfolio_comparison.csv` | Portfolio metrics | Table 2: Portfolio Performance |
| `geometry_features.csv` | Full SGI time series | Figure 1: SGI Over Time |

### 2. **Visualizations**

Located in: `outputs/figures/`

| Figure | Description | Suggested Use |
|--------|-------------|---------------|
| `sgi_timeseries.png` | SGI with crisis overlays | Figure 1 (main) |
| `sgi_vs_vol.png` | SGI vs volatility scatter | Figure 2 |
| `portfolio_performance.png` | Cumulative returns | Figure 3 |

### 3. **Statistical Evidence**

From `results_summary.json`:

```json
{
  "key_metrics": {
    "incremental_r2": 0.0234,        // SGI adds 2.34% R²
    "sharpe_improvement_pct": 15.3,  // 15.3% Sharpe improvement
    "mean_sgi": 0.0096,              // Average SGI
    "std_sgi": 0.0124,               // SGI volatility
    "n_observations": 2196           // Sample size
  }
}
```

---

## Paper Structure Recommendations

### Abstract

**Template**:
> We introduce Spectral Geometry Instability (SGI), a novel measure of structural market risk based on eigenspace rotation. Using [N] years of data across [M] assets, we show that SGI adds [X]% incremental R² to baseline volatility forecasts and improves portfolio Sharpe ratios by [Y]%. Our results suggest that eigenspace instability captures structural risk orthogonal to traditional volatility measures.

**Fill in from results**:
- N = Date range from `results_summary.json`
- M = Number of assets (10 sector ETFs)
- X = `incremental_r2 * 100`
- Y = `sharpe_improvement_pct`

### Introduction

**Key Points to Include**:
1. Traditional risk models focus on volatility magnitude
2. SGI captures structural reorganization of risk
3. Motivation: diversification breakdown during crises
4. Contribution: new risk measure + empirical validation

### Methodology

**Section 2.1: Data**
- Source: Yahoo Finance (free, reproducible)
- Universe: 10 US sector ETFs (see `constants.py`)
- Period: 2005-2024 (20 years)
- Frequency: Daily
- Crisis events: 7 major events (see `DATA_SOURCES.md`)

**Section 2.2: SGI Computation**
- Eigendecomposition of rolling covariance
- Principal angles between eigenspaces
- Weighted by eigenvalue importance
- Formula: See `src/spectral_geometry.py`

**Section 2.3: Empirical Tests**
1. Predictive power (walk-forward validation)
2. Portfolio construction (SGI-conditioned min-variance)
3. Event studies (crisis periods)

### Results

**Table 1: Predictive Performance**

Use: `predictive_summary.csv`

| Feature Set | R² | RMSE | N Folds |
|-------------|-----|------|---------|
| Baseline Only | 0.XX | 0.XX | XX |
| SGI Only | 0.XX | 0.XX | XX |
| Baseline + SGI | 0.XX | 0.XX | XX |

**Table 2: Portfolio Performance**

Use: `portfolio_comparison.csv`

| Strategy | CAGR | Sharpe | Max DD | Calmar |
|----------|------|--------|--------|--------|
| Equal Weight | X.X% | X.XX | -X.X% | X.XX |
| Min Variance | X.X% | X.XX | -X.X% | X.XX |
| SGI-Conditioned | X.X% | X.XX | -X.X% | X.XX |

**Figure 1: SGI Time Series**

Use: `sgi_timeseries.png`

Caption:
> Spectral Geometry Instability (SGI) over time with major crisis events highlighted. SGI spikes during periods of structural market stress, including the 2008 Financial Crisis, 2020 COVID-19 crash, and 2022 inflation shock.

**Figure 2: SGI vs Volatility**

Use: `sgi_vs_vol.png`

Caption:
> Scatter plot of SGI versus realized volatility. The low correlation (ρ = 0.XX) suggests SGI captures structural risk orthogonal to volatility magnitude.

**Figure 3: Portfolio Performance**

Use: `portfolio_performance.png`

Caption:
> Cumulative returns of portfolio strategies. The SGI-conditioned strategy (blue) outperforms equal-weight (red) and traditional minimum-variance (green) approaches, particularly during crisis periods.

### Discussion

**Key Findings to Highlight**:

1. **Predictive Power**
   - SGI adds X.XX% incremental R² to baseline models
   - Improvement is statistically significant
   - Robust across different horizons

2. **Portfolio Benefits**
   - X.X% higher Sharpe ratio
   - X.X% lower maximum drawdown
   - Better crisis performance

3. **Economic Interpretation**
   - SGI measures eigenspace rotation
   - High SGI = structural instability
   - Orthogonal to volatility

4. **Practical Implications**
   - Can be computed in real-time
   - Works with any covariance matrix
   - Applicable to any asset universe

### Conclusion

**Template**:
> We introduce SGI as a measure of structural market risk. Our empirical analysis shows that SGI (1) has significant predictive power for future risk, (2) improves portfolio construction, and (3) spikes during crisis periods. These findings suggest that monitoring eigenspace stability provides valuable information for risk management.

---

## Statistical Tests to Include

### 1. **Predictive Power Test**

**Null Hypothesis**: SGI has no incremental predictive power  
**Test**: Compare R² of baseline vs baseline+SGI models  
**Statistic**: ΔR² = X.XXXX  
**Interpretation**: Reject null if ΔR² > 0 and statistically significant

### 2. **Portfolio Performance Test**

**Null Hypothesis**: SGI-conditioned portfolio has same Sharpe as baseline  
**Test**: Sharpe ratio difference test  
**Statistic**: ΔSharpe = X.XX  
**Interpretation**: Reject null if improvement is significant

### 3. **Crisis Detection Test**

**Null Hypothesis**: SGI is same during crisis vs normal periods  
**Test**: t-test or Mann-Whitney U test  
**Statistic**: From event study results  
**Interpretation**: SGI should be significantly higher during crises

---

## Robustness Checks

### 1. **Different Universes**

Run analysis on:
- ✅ Sector ETFs (baseline)
- ✅ Multi-asset ETFs
- ✅ Large-cap stocks
- ✅ Crypto (if applicable)

### 2. **Different Windows**

Test sensitivity to:
- Covariance window (60, 126, 252 days)
- Top-k eigenvalues (2, 3, 5)
- Forecast horizons (5, 10, 20, 60 days)

### 3. **Different Methods**

Compare:
- Sample covariance
- EWMA covariance
- Ledoit-Wolf shrinkage
- RMT denoising

---

## Appendix Materials

### A. Data Description
- Use `DATA_SOURCES.md`
- Include data manifest

### B. Methodology Details
- Mathematical derivations
- Algorithm pseudocode
- Implementation notes

### C. Additional Results
- Full regression tables
- Sensitivity analyses
- Robustness checks

### D. Code Availability
- GitHub repository link
- Replication instructions
- Software versions

---

## Citation

```bibtex
@article{jacob2024sgi,
  title={Spectral Geometry Instability: Eigenspace Rotation as a Structural Risk Signal},
  author={Jacob, Manu Nicholas and Ghai, Ronit},
  journal={Working Paper},
  year={2024},
  url={https://github.com/manunicholasjacob/spectral-geometry-instability}
}
```

---

## Checklist for Paper Submission

- [ ] Run complete analysis (COLAB_COMPLETE_RUN.py)
- [ ] Extract all tables from CSV files
- [ ] Include all figures (high-resolution PNG)
- [ ] Report all key statistics from results_summary.json
- [ ] Include robustness checks
- [ ] Document data sources
- [ ] Provide replication code (GitHub link)
- [ ] Include software versions
- [ ] Acknowledge data providers
- [ ] Proofread all numbers

---

## Contact

**Manu Nicholas Jacob**: manunicholasjacob@gmail.com  
**Ronit Ghai**: ronitghai@hotmail.com

**Repository**: https://github.com/manunicholasjacob/spectral-geometry-instability

---

**Good luck with your paper!** 📝🚀
