#!/usr/bin/env bash
set -euo pipefail

#------------------------------------------
# CONFIG
#------------------------------------------
PKGNAME="openabc"                         # your package name
TESTENV="/tmp/${PKGNAME}_pkgtest"         # isolated test environment
VENV=".venv"                              

#------------------------------------------
# CLEAN OLD ENVIRONMENTS
#------------------------------------------
echo "==> Removing old .venv..."
rm -rf "$VENV"

echo "==> Removing old test env: $TESTENV"
rm -rf "$TESTENV"

echo "==> Removing previous build artifacts..."
rm -rf dist build "${PKGNAME}.egg-info" MANIFEST

#------------------------------------------
# CREATE DEV BUILD ENV
#------------------------------------------
echo "==> Creating new build venv..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"

echo "==> Upgrading pip + build tools..."
pip install --upgrade pip build wheel setuptools

#------------------------------------------
# BUILD DISTRIBUTION ARTIFACTS
#------------------------------------------
echo "==> Building wheel + sdist..."
python -m build

echo "==> Build complete. Contents of dist/:"
ls -l dist

deactivate

#------------------------------------------
# CREATE FRESH TEST ENV
#------------------------------------------
echo "==> Creating clean test environment at $TESTENV"
python3 -m venv "$TESTENV"
source "$TESTENV/bin/activate"

echo "==> Installing wheel exactly as a user would..."
pip install --upgrade pip
pip install dist/*.whl

#------------------------------------------
# INSTALL JUPYTER + REGISTER KERNEL
#------------------------------------------
echo "==> Installing JupyterLab + kernel..."
pip install ipykernel jupyterlab

python -m ipykernel install --user \
    --name "${PKGNAME}_test" \
    --display-name "Python (${PKGNAME}_test)"

#------------------------------------------
# SMOKE TEST THE INSTALL
#------------------------------------------
echo "==> Running smoke tests..."
python - <<EOF
import sys
import ${PKGNAME}
print("Python executable:", sys.executable)
print("${PKGNAME} version:", ${PKGNAME}.__version__)
EOF

echo "==> Success! Launching JupyterLab using the test environment..."
echo ""
echo "IMPORTANT: In Jupyter select kernel:  Python (${PKGNAME}_test)"
echo ""

python -m jupyterlab


