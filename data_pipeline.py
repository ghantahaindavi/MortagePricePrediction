import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df

if __name__ == "__main__":
    data = load_data('../data/raw/mortgage_data.csv')
