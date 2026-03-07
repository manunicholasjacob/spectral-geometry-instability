@echo off
REM Push Spectral Geometry Instability project to GitHub
REM Repository: https://github.com/manunicholasjacob/spectral-geometry-instability

echo ============================================
echo Pushing SGI Project to GitHub
echo ============================================

cd /d "%~dp0"

REM Check if git is initialized
if not exist ".git" (
    echo Initializing git repository...
    git init
    git remote add origin https://github.com/manunicholasjacob/spectral-geometry-instability.git
) else (
    echo Git repository already initialized.
)

REM Stage all files
echo.
echo Staging files...
git add .

REM Show status
echo.
echo Current status:
git status --short

REM Commit
echo.
echo Committing changes...
git commit -m "SGI research pipeline: full implementation with predictive models, portfolio backtests, and event studies"

REM Set main branch
git branch -M main

REM Push to GitHub
echo.
echo Pushing to GitHub...
git push -u origin main

REM Create version tag
echo.
echo Creating version tag v0.1...
git tag v0.1
git push origin v0.1

echo.
echo ============================================
echo Push complete!
echo Repository: https://github.com/manunicholasjacob/spectral-geometry-instability
echo ============================================

pause
