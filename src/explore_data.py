import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find '{file_path}'. Check your path directory structure.")
    return pd.read_csv(file_path)

def plot_correlation_matrix(df, plot_dir):
    numeric_cols = ['sq_meters', 'bedrooms', 'has_garage', 'price']
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Housing Price Correlation Factors - Zamora")
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_dir, "price_correlation_matrix.png"))
    plt.close()

def plot_price_vs_size(df, plot_dir):
    plt.figure(figsize=(10, 6))
    valid_data = df[(df['sq_meters'] > 0) & (df['price'] < df['price'].quantile(0.99))] 
    
    sns.regplot(
        data=valid_data, x='sq_meters', y='price', 
        scatter_kws={'alpha': 0.5, 'color': '#2b7bba'}, line_kws={'color': '#e74c3c'}
    )
    plt.title("Property Price vs. Size in Zamora")
    plt.xlabel("Square Meters (m²)")
    plt.ylabel("Price (€)")
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_dir, "price_vs_size_scatter.png"))
    plt.close()

def plot_price_by_bedrooms(df, plot_dir):
    plt.figure(figsize=(9, 6))
    common_beds = df[df['bedrooms'].between(1, 5)]
    
    sns.boxplot(data=common_beds, x='bedrooms', y='price', palette='Blues', hue='bedrooms', legend=False)
    plt.title("Price Distribution by Number of Bedrooms")
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_dir, "price_by_bedrooms_box.png"))
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Setup paths relative to the project root
    input_file = os.path.join(base_dir, "data", "processed", "zamora_houses_clean.csv")
    plot_dir = os.path.join(base_dir, "outputs", "plots")
    
    # Safely create the outputs/plots folders if they don't exist yet
    os.makedirs(plot_dir, exist_ok=True)
    
    print("🔄 Loading clean dataset...")
    df = load_data(input_file)
    
    print("📊 Generating and saving plots to outputs/plots/...")
    plot_correlation_matrix(df, plot_dir)
    plot_price_vs_size(df, plot_dir)
    plot_price_by_bedrooms(df, plot_dir)
    
    print("✅ Done! All charts saved successfully.")

if __name__ == "__main__":
    main()