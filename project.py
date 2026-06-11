import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set stylish theme for visualizations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def create_raw_dataset():
    """Generates a mock raw dataset with duplicates, missing values, and outliers."""
    np.random.seed(42)
    n_rows = 200

    data = {
        "Transaction_ID": [f"TXN_{1000 + i}" for i in range(n_rows)],
        "Age": np.random.choice(
            [23, 34, 45, 51, np.nan, 29, 38, 999, -5],
            size=n_rows,
            p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.02, 0.02, 0.01],
        ),
        "Income": np.random.choice(
            [30000, 45000, 60000, 85000, 120000, np.nan, 500000],
            size=n_rows,
            p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05],
        ),
        "Category": np.random.choice(
            ["Electronics", "Clothing", "Home", "Books", None],
            size=n_rows,
            p=[0.3, 0.3, 0.2, 0.15, 0.05],
        ),
        "Sales": np.random.uniform(10, 500, size=n_rows),
    }

    df = pd.DataFrame(data)

    # Inject explicit duplicates
    df = pd.concat([df, df.iloc[10:20]], ignore_index=True)
    df.to_csv("raw_sales_data.csv", index=False)
    print("✓ Success: 'raw_sales_data.csv' generated with intentional anomalies.")


def clean_data(file_path):
    """Loads, cleans, and processes the raw dataset."""
    print("\n--- Starting Data Cleaning Phase ---")
    df = pd.read_csv(file_path)
    print(f"Initial Shape: {df.shape}")

    # 1. Handle Duplicates
    duplicate_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"-> Removed {duplicate_count} duplicate rows.")

    # 2. Address Structural Outliers / Invalid Entries in Age
    # Replace unrealistic ages (negative or over 110) with NaN to handle systematically
    df.loc[(df["Age"] < 0) | (df["Age"] > 110), "Age"] = np.nan

    # 3. Handle Missing Values
    # Impute numerical columns with median (robust to extreme outliers)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Impute categorical column with mode
    df["Category"] = df["Category"].fillna(df["Category"].mode()[0])
    print("-> Treated missing values and structural outliers.")

    # 4. Handle Statistical Outliers (Income column using IQR method)
    Q1 = df["Income"].quantile(0.25)
    Q3 = df["Income"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap outliers to the upper/lower bounds instead of losing data points
    df["Income"] = np.where(df["Income"] > upper_bound, upper_bound, df["Income"])
    df["Income"] = np.where(df["Income"] < lower_bound, lower_bound, df["Income"])
    print("-> Handled Income outliers using IQR bounding.")

    # Validate cleaning
    print(f"Final Shape: {df.shape}")
    print(f"Remaining Missing Values:\n{df.isnull().sum()}")

    df.to_csv("cleaned_sales_data.csv", index=False)
    print("✓ Success: Cleaned data saved to 'cleaned_sales_data.csv'.")
    return df


def generate_visualizations(df):
    """Creates a 3-part reporting dashboard for insights storytelling."""
    print("\n--- Generating Visualizations ---")
    os.makedirs("plots", exist_ok=True)

    # Visualization 1: Distribution of Total Sales by Category (Bar Plot)
    plt.figure()
    category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    sns.barplot(x=category_sales.index, y=category_sales.values, palette="Blues_r")
    plt.title("Total Revenue Contribution by Category", fontsize=14, pad=15)
    plt.xlabel("Product Category", fontsize=12)
    plt.ylabel("Total Sales ($)", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/1_sales_by_category.png")
    plt.close()

    # Visualization 2: Correlation Analysis (Income vs Sales colored by Age)
    plt.figure()
    scatter = plt.scatter(
        df["Income"],
        df["Sales"],
        c=df["Age"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="w",
        s=80,
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Customer Age", fontsize=11)
    plt.title("Customer Dynamics: Income vs. Sales Spend", fontsize=14, pad=15)
    plt.xlabel("Adjusted Annual Income ($)", fontsize=12)
    plt.ylabel("Transaction Purchase Amount ($)", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/2_income_vs_sales.png")
    plt.close()

    # Visualization 3: Distribution Density of Sales (Box & Kernel Density Plot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(ax=axes[0], x=df["Sales"], color="#4c72b0")
    axes[0].set_title("Sales Metric Spread (Boxplot)", fontsize=12)

    sns.kdeplot(ax=axes[1], x=df["Sales"], fill=True, color="#55a868")
    axes[1].set_title("Sales Value Density Trend (KDE)", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/3_sales_distribution.png")
    plt.close()

    print("✓ Success: Dashboard components exported to the './plots' directory.")


if __name__ == "__main__":
    # Pipeline execution
    create_raw_dataset()
    cleaned_df = clean_data("raw_sales_data.csv")
    generate_visualizations(cleaned_df)
    print("\n[Pipeline Complete] Everything is ready for submission!")
