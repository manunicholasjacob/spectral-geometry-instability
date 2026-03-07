"""
Random Matrix Theory (RMT) Denoising module for the SGI project.
Implements Marcenko-Pastur based eigenvalue filtering for covariance matrices.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize_scalar

from .constants import EPSILON


def marcenko_pastur_pdf(x: np.ndarray, q: float, sigma: float = 1.0) -> np.ndarray:
    """
    Compute Marcenko-Pastur probability density function.
    
    Args:
        x: Eigenvalue points
        q: Ratio T/N (observations/assets)
        sigma: Variance parameter
        
    Returns:
        PDF values at x
    """
    lambda_plus = sigma**2 * (1 + np.sqrt(1/q))**2
    lambda_minus = sigma**2 * (1 - np.sqrt(1/q))**2
    
    pdf = np.zeros_like(x)
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    
    if np.any(mask):
        pdf[mask] = (q / (2 * np.pi * sigma**2)) * \
                    np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / x[mask]
    
    return pdf


def compute_mp_bounds(q: float, sigma: float = 1.0) -> Tuple[float, float]:
    """
    Compute Marcenko-Pastur eigenvalue bounds.
    
    Args:
        q: Ratio T/N
        sigma: Variance parameter
        
    Returns:
        Tuple of (lambda_minus, lambda_plus)
    """
    lambda_plus = sigma**2 * (1 + np.sqrt(1/q))**2
    lambda_minus = sigma**2 * (1 - np.sqrt(1/q))**2
    
    return lambda_minus, lambda_plus


def fit_mp_sigma(eigenvalues: np.ndarray, q: float) -> float:
    """
    Fit the sigma parameter of Marcenko-Pastur distribution.
    
    Args:
        eigenvalues: Observed eigenvalues
        q: Ratio T/N
        
    Returns:
        Fitted sigma value
    """
    # Use median eigenvalue as initial estimate
    initial_sigma = np.sqrt(np.median(eigenvalues))
    
    def objective(sigma):
        lambda_minus, lambda_plus = compute_mp_bounds(q, sigma)
        # Count eigenvalues within MP bounds
        in_bounds = np.sum((eigenvalues >= lambda_minus) & (eigenvalues <= lambda_plus))
        # We want most eigenvalues to be within bounds
        return -in_bounds
    
    result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
    
    return result.x if result.success else initial_sigma


def identify_signal_eigenvalues(
    eigenvalues: np.ndarray,
    q: float,
    sigma: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Identify signal vs noise eigenvalues using RMT.
    
    Args:
        eigenvalues: Eigenvalues sorted descending
        q: Ratio T/N
        sigma: Variance parameter (fitted if None)
        
    Returns:
        Tuple of (signal_eigenvalues, noise_eigenvalues, lambda_plus)
    """
    if sigma is None:
        sigma = fit_mp_sigma(eigenvalues, q)
    
    lambda_minus, lambda_plus = compute_mp_bounds(q, sigma)
    
    # Eigenvalues above lambda_plus are signal
    signal_mask = eigenvalues > lambda_plus
    
    signal_eigenvalues = eigenvalues[signal_mask]
    noise_eigenvalues = eigenvalues[~signal_mask]
    
    return signal_eigenvalues, noise_eigenvalues, lambda_plus


def denoise_covariance_rmt(
    cov: np.ndarray,
    n_observations: int,
    method: str = "constant_residual"
) -> np.ndarray:
    """
    Denoise covariance matrix using RMT.
    
    Args:
        cov: Covariance matrix
        n_observations: Number of observations used to estimate covariance
        method: Denoising method
            - "constant_residual": Replace noise eigenvalues with average
            - "shrink_noise": Shrink noise eigenvalues toward average
            - "targeted_shrinkage": Apply targeted shrinkage based on eigenvalue
            
    Returns:
        Denoised covariance matrix
    """
    n_assets = cov.shape[0]
    q = n_observations / n_assets
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Identify signal vs noise
    signal_eig, noise_eig, lambda_plus = identify_signal_eigenvalues(eigenvalues, q)
    n_signal = len(signal_eig)
    
    # Create denoised eigenvalues
    denoised_eigenvalues = eigenvalues.copy()
    
    if method == "constant_residual":
        # Replace noise eigenvalues with their average
        if len(noise_eig) > 0:
            noise_avg = np.mean(noise_eig)
            denoised_eigenvalues[n_signal:] = noise_avg
            
    elif method == "shrink_noise":
        # Shrink noise eigenvalues toward average
        if len(noise_eig) > 0:
            noise_avg = np.mean(noise_eig)
            alpha = 0.5  # Shrinkage intensity
            denoised_eigenvalues[n_signal:] = (1 - alpha) * noise_eig + alpha * noise_avg
            
    elif method == "targeted_shrinkage":
        # Apply targeted shrinkage based on eigenvalue position
        for i in range(n_signal, len(eigenvalues)):
            # Shrink more for smaller eigenvalues
            shrink_factor = (i - n_signal + 1) / (len(eigenvalues) - n_signal)
            target = np.mean(eigenvalues[n_signal:])
            denoised_eigenvalues[i] = (1 - shrink_factor) * eigenvalues[i] + shrink_factor * target
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure positive eigenvalues
    denoised_eigenvalues = np.maximum(denoised_eigenvalues, EPSILON)
    
    # Reconstruct covariance
    denoised_cov = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T
    
    # Ensure symmetry
    denoised_cov = (denoised_cov + denoised_cov.T) / 2
    
    return denoised_cov


def compute_effective_rank(eigenvalues: np.ndarray) -> float:
    """
    Compute effective rank (participation ratio) of eigenvalue spectrum.
    
    Args:
        eigenvalues: Eigenvalues
        
    Returns:
        Effective rank
    """
    eigenvalues = np.abs(eigenvalues)
    total = np.sum(eigenvalues)
    
    if total <= EPSILON:
        return 0.0
    
    normalized = eigenvalues / total
    
    # Participation ratio
    return 1.0 / np.sum(normalized ** 2)


def compute_rmt_diagnostics(
    cov: np.ndarray,
    n_observations: int
) -> Dict:
    """
    Compute RMT-based diagnostics for covariance matrix.
    
    Args:
        cov: Covariance matrix
        n_observations: Number of observations
        
    Returns:
        Dictionary of diagnostics
    """
    n_assets = cov.shape[0]
    q = n_observations / n_assets
    
    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # MP bounds
    sigma = fit_mp_sigma(eigenvalues, q)
    lambda_minus, lambda_plus = compute_mp_bounds(q, sigma)
    
    # Signal identification
    signal_eig, noise_eig, _ = identify_signal_eigenvalues(eigenvalues, q, sigma)
    
    diagnostics = {
        "n_assets": n_assets,
        "n_observations": n_observations,
        "q_ratio": q,
        "fitted_sigma": float(sigma),
        "mp_lambda_minus": float(lambda_minus),
        "mp_lambda_plus": float(lambda_plus),
        "n_signal_eigenvalues": len(signal_eig),
        "n_noise_eigenvalues": len(noise_eig),
        "signal_variance_fraction": float(np.sum(signal_eig) / np.sum(eigenvalues)),
        "effective_rank": float(compute_effective_rank(eigenvalues)),
        "largest_eigenvalue": float(eigenvalues[0]),
        "smallest_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[0] / (eigenvalues[-1] + EPSILON)),
    }
    
    return diagnostics


def rolling_rmt_denoised_covariance(
    returns: pd.DataFrame,
    window: int,
    method: str = "constant_residual"
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Compute rolling RMT-denoised covariance matrices.
    
    Args:
        returns: Returns DataFrame
        window: Rolling window size
        method: Denoising method
        
    Returns:
        Dictionary mapping dates to denoised covariance matrices
    """
    cov_matrices = {}
    dates = returns.index
    n_obs = len(dates)
    
    for i in range(window, n_obs + 1):
        start_idx = i - window
        end_idx = i
        
        window_returns = returns.iloc[start_idx:end_idx]
        
        # Skip if insufficient data
        if len(window_returns) < window // 2:
            continue
        
        # Compute sample covariance
        sample_cov = window_returns.cov().values
        
        # Denoise
        try:
            denoised_cov = denoise_covariance_rmt(
                sample_cov, len(window_returns), method
            )
            end_date = dates[end_idx - 1]
            cov_matrices[end_date] = denoised_cov
        except Exception:
            continue
    
    return cov_matrices
