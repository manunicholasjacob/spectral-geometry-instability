# Bug Fix Log

## Critical Bugs Fixed

### Bug #1: Variable Shadowing in `spectral_geometry.py` (CRITICAL)

**Date**: 2024  
**Severity**: Critical - Prevents all experiments from running  
**Status**: ✅ FIXED

#### Problem
```python
TypeError: 'bool' object is not callable
```

**Root Cause**: Variable name collision in `compute_rolling_geometry_features()` function.

The function had a parameter named `compute_eigengap` (boolean), but inside the function it tried to call `compute_eigengap()` as a function. This created a shadowing issue where the parameter overwrote the function reference.

**Location**: `src/spectral_geometry.py:276`

```python
# BEFORE (BROKEN)
def compute_rolling_geometry_features(
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    top_k: int,
    compute_eigengap: bool = True  # ← Parameter shadows function name
) -> pd.DataFrame:
    ...
    if compute_eigengap:  # ← This is the boolean parameter
        record["eigengap"] = compute_eigengap(eigenvalues, top_k)  # ← Tries to call boolean!
```

**Fix**: Renamed parameter from `compute_eigengap` to `include_eigengap`

```python
# AFTER (FIXED)
def compute_rolling_geometry_features(
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    top_k: int,
    include_eigengap: bool = True  # ← Renamed parameter
) -> pd.DataFrame:
    ...
    if include_eigengap:  # ← Check boolean parameter
        record["eigengap"] = compute_eigengap(eigenvalues, top_k)  # ← Call function
```

**Commit**: `fbb4551`  
**Files Changed**: `src/spectral_geometry.py`

#### Impact
- **Before**: All experiments failed immediately with TypeError
- **After**: Full pipeline runs successfully

#### Testing
Run `test_basic_workflow.py` to verify the fix:
```bash
python test_basic_workflow.py
```

---

## Google Colab Compatibility Issues Fixed

### Issue #1: Missing Dependencies
**Status**: ✅ FIXED

Added to `requirements.txt`:
- `seaborn>=0.12.0` (for plotting)
- `pandas-datareader>=0.10.0` (for factor data)

### Issue #2: No Setup Automation
**Status**: ✅ FIXED

Created `colab_setup.py` for one-command setup in Google Colab.

### Issue #3: Circular Import Issues
**Status**: ✅ FIXED

Removed automatic imports from `src/__init__.py` to avoid circular dependencies.

### Issue #4: No Testing Framework
**Status**: ✅ FIXED

Created:
- `test_imports.py` - Validates all module imports
- `test_basic_workflow.py` - End-to-end workflow test
- `COLAB_QUICKSTART.md` - Comprehensive Colab guide

---

## Lessons Learned

1. **Avoid shadowing built-in or module function names** with parameters
2. **Always test in Google Colab** before claiming Colab compatibility
3. **Create minimal test cases** to catch issues early
4. **Document all dependencies** explicitly in requirements.txt

---

## How to Report Bugs

If you encounter issues:

1. Run `python test_imports.py` to check imports
2. Run `python test_basic_workflow.py` to test basic functionality
3. Check the error traceback carefully
4. Open an issue on [GitHub](https://github.com/manunicholasjacob/spectral-geometry-instability/issues)

---

**Last Updated**: 2024  
**Version**: 0.2.1
