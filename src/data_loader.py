import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset CSV dari path tertentu.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Dataset loaded: {filepath}, shape={df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Gagal load dataset: {e}")
        return pd.DataFrame()

