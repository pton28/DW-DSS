import pandas as pd

data_path = '/Dataset/Raw/GOOG.csv'

df = pd.read_csv(data_path)
print("Dữ liệu ban đầu:")
print(df.describe())