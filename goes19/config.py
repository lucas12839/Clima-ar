from pathlib import Path

# GOES-19
SATELLITE = "G19"
BUCKET = "noaa-goes19"
AWS_REGION = "us-east-1"

# ClimaAR
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "goes19"

ABI_DIR = DATA_DIR / "abi"
GLM_DIR = DATA_DIR / "glm"
CLOUDTOP_DIR = DATA_DIR / "cloudtop"
DMW_DIR = DATA_DIR / "dmw"

# Ubicación piloto
LATITUDE = -38.72
LONGITUDE = -62.27

# Zona temporal de datos
MAX_DATA_AGE_MINUTES = 30
