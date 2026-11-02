from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

HORIZON = 24                 
TEST_FRACTION = 0.15        
CAPACITY_PERCENTILE = 0.95   
RANDOM_STATE = 42

SYNTHETIC_DEMAND_CSV = DATA_DIR / "demand_synthetic.csv"
FAULT_REPORTS_CSV = DATA_DIR / "fault_reports_synthetic.csv"
INTENTS_JSON = DATA_DIR / "intents.json"