import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "Raw")

# ----------------------------
# FORMAT TIME AXIS (ĐẸP & TỰ ĐỘNG)
# ----------------------------
def format_xaxis(ax):
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.setp(ax.get_xticklabels(), rotation=30)
    plt.tight_layout()


# ----------------------------
# LOAD ALL CSV FILES
# ----------------------------
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


# ----------------------------
# VISUALIZATION FUNCTIONS
# ----------------------------

def plot_price_volume(df):
    date_col = df.columns[0]
    df = df.sort_values(date_col)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df[date_col], df['close'], label='Close Price')
    ax1.set_ylabel("Price")

    ax2 = ax1.twinx()
    ax2.bar(df[date_col], df['volume'], alpha=0.25, label='Volume', color='gray')
    ax2.set_ylabel("Volume")
    plt.legend()
    plt.title("Price & Volume Over Time")

    format_xaxis(ax1)
    plt.show()


def plot_revenue_growth(df):
    date_col = df.columns[0]
    df = df.sort_values(date_col)

    plt.figure(figsize=(10, 5))
    plt.plot(df[date_col], df['totalRevenue'], marker='o')
    plt.title("Revenue Growth Over Time")
    plt.grid(True)

    ax = plt.gca()
    format_xaxis(ax)
    plt.show()


def plot_cost_structure_avg(df):
    # Tính trung bình các nhóm chi phí
    avg_cogs = df['costOfRevenue'].mean()
    avg_opex = df['operatingExpenses'].mean()
    avg_interest = df['interestExpense'].mean()

    labels = ["COGS (Avg)", "Operating Expense (Avg)", "Interest Expense (Avg)"]
    values = [avg_cogs, avg_opex, avg_interest]

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title("Average Cost Structure")
    plt.show()


def plot_cash_flow(df):
    date_col = df.columns[0]
    df = df.sort_values(date_col)

    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df['operatingCashflow'], label="Operating CF")
    plt.plot(df[date_col], df['cashflowFromInvestment'], label="Investing CF")
    plt.plot(df[date_col], df['cashflowFromFinancing'], label="Financing CF")
    plt.legend()
    plt.title("Cash Flow Components Over Time")
    plt.grid(True)

    ax = plt.gca()
    format_xaxis(ax)
    plt.show()


def plot_total_assets(df):
    date_col = df.columns[0]
    df = df.sort_values(date_col)

    plt.figure(figsize=(10, 5))
    plt.plot(df[date_col], df['totalAssets'], marker='o')
    plt.title("Total Assets Over Time")
    plt.grid(True)

    ax = plt.gca()
    format_xaxis(ax)
    plt.show()


def plot_correlation_heatmap(df, name="Dataset"):
    # Lọc dữ liệu số
    df_num = df.select_dtypes(include=['float64', 'int64'])

    if df_num.shape[1] < 2:
        print(f"Không đủ dữ liệu số để tạo correlation heatmap cho {name}")
        return

    plt.figure(figsize=(10, 8))
    corr = df_num.corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Heatmap – {name}")
    plt.tight_layout()
    plt.show()


# ----------------------------
# AUTO RUN & PLOT
# ----------------------------
def run_all_and_plot():
    dfs = load_all_csv()

    print("\n=== TỰ ĐỘNG VẼ TẤT CẢ BIỂU ĐỒ ===")

    for name, df in dfs.items():
        print(f"\n--- ĐANG XỬ LÝ: {name} ---")

        cols = df.columns.tolist()
        # googl_daily_prices_raw
        if 'close' in cols and 'volume' in cols:
            print("Vẽ price-volume...")
            plot_price_volume(df)

        # googl_income_statement_raw
        if 'totalRevenue' in cols:
            print("Vẽ revenue growth...")
            plot_revenue_growth(df)

        # googl_income_statement_raw
        if all(c in cols for c in ['costOfRevenue', 'operatingExpenses', 'interestExpense']):
            print("Vẽ cost structure...")
            plot_cost_structure_avg(df)

        # googl_cash_flow_statement_raw
        if all(c in cols for c in ['operatingCashflow', 'cashflowFromInvestment', 'cashflowFromFinancing']):
            print("Vẽ cash flow...")
            plot_cash_flow(df)
        
        # googl_balance_sheet_raw
        if 'totalAssets' in cols:
            print("Vẽ total assets...")
            plot_total_assets(df)

        print("Vẽ correlation heatmap...")
        plot_correlation_heatmap(df, name)


    print("\n=== HOÀN TẤT AUTO PLOT ===")
    return dfs


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    run_all_and_plot()
