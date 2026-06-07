RANDOM_SEED = 42
FS = 25600

RAW_DATA_DIR = "data"
PROCESSED_DIR = "processed"
RESULTS_DIR = "results"

CONDITION_MAP = {
    "35Hz12kN": {"condition_id": "C1", "speed_rpm": 2100.0, "load_kn": 12.0},
    "37.5Hz11kN": {"condition_id": "C2", "speed_rpm": 2250.0, "load_kn": 11.0},
    "40Hz10kN": {"condition_id": "C3", "speed_rpm": 2400.0, "load_kn": 10.0},
}

METADATA_COLUMNS = [
    "bearing_id",
    "condition_id",
    "speed_rpm",
    "load_kn",
    "file_path",
    "file_index",
    "time_index",
    "failure_time",
    "rul",
    "normalized_rul",
]

ACTIVE_MODEL_ORDER = [
    "Ridge",
    "LSTM",
    "TCN",
    "Transformer",
    "latent_ode",
]
