from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

HORIZON = 24                 
TEST_FRACTION = 0.15        
CAPACITY_PERCENTILE = 0.95   
RANDOM_STATE = 42