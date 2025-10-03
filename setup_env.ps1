# setup_env.ps1
# Create Python virtual environment and install dependencies

Write-Host "Start creating virtual environment .venv ..."

# 1) Create virtual environment
python -m venv .venv

# 2) Activate virtual environment
Write-Host "Activate virtual environment..."
.\.venv\Scripts\Activate.ps1

# 3) Upgrade pip
Write-Host "Upgrade pip..."
pip install --upgrade pip

# 4) Install dependencies
Write-Host "Install dependencies (torch, torchvision, matplotlib, numpy, pillow, scikit-learn)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pillow scikit-learn

# 5) Export requirements.txt
Write-Host "Export requirements.txt..."
pip freeze > requirements.txt

# 6) Finish
Write-Host "Environment setup completed. Use  .\.venv\Scripts\Activate.ps1  to activate the environment."
