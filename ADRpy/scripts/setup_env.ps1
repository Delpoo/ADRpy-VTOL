$ErrorActionPreference = "Stop"

Write-Host "Creando entorno .venv..."
python -m venv .venv

Write-Host "Activando entorno..."
. .venv\Scripts\Activate.ps1

Write-Host "Actualizando pip..."
python -m pip install --upgrade pip

Write-Host "Instalando requirements..."
pip install -r requirements.txt

Write-Host "Registrando kernel de Jupyter..."
python -m ipykernel install --user --name "ADRpy" --display-name "Python (ADRpy)"

Write-Host "Listo. Selecciona el kernel 'Python (ADRpy)' en Jupyter."
