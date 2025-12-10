import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "Cleaned")


def load_all_csv():
    dfs = {}
    csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        var_name = f"{file_name}_raw"
        try:
            df = pd.read_csv(file)

            # convert date col
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

            dfs[var_name] = df
            print(f"Loaded: {var_name}")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    return dfs

def show(lst):
    for name, file in lst.items():
        print(f"\n--- Tổng quan về: {name} ---")
        print(file.info())
    
if __name__ == "__main__":
    lst = load_all_csv()
    show(lst)
