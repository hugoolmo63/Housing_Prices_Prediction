import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import contextily as cx

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir) if os.path.basename(base_dir) == "src" else base_dir
    
    input_file = os.path.join(project_root, "data", "processed", "zamora_houses_clean.csv")
    plot_dir = os.path.join(project_root, "outputs", "plots")
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(plot_dir, exist_ok=True)
    
    df = pd.read_csv(input_file)
    
    # 1. GEOCORDS LOOKUP: Create latitude & longitude if they aren't in the CSV file
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("📍 GPS columns missing in CSV. Fetching coordinates for unique locations...")
        geocoder = Nominatim(user_agent="zamora_housing_app_v2")
        geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1)
        
        unique_locs = pd.DataFrame(df['location'].unique(), columns=['location'])
        unique_locs['search_query'] = unique_locs['location'] + ", Zamora, Spain"
        
        print(f"Looking up {len(unique_locs)} unique areas (This takes a few minutes)...")
        unique_locs['geo_data'] = unique_locs['search_query'].apply(geocode)
        
        unique_locs['latitude'] = unique_locs['geo_data'].apply(lambda x: x.latitude if x else None)
        unique_locs['longitude'] = unique_locs['geo_data'].apply(lambda x: x.longitude if x else None)
        
        coord_map = unique_locs.set_index('location')[['latitude', 'longitude']]
        df = df.join(coord_map, on='location')
        
        # Save coordinates back to your CSV so you never have to wait for this lookup again!
        df.to_csv(input_file, index=False)
        print("💾 Coordinates cached back to your processed CSV file.")
    
    # Clean out any missing values for mapping
    df_map = df.dropna(subset=['latitude', 'longitude', 'price']).copy()
    
    if df_map.empty:
        print("❌ Error: No valid coordinates found to map.")
        return

    # Cap outliers for clean visualization scaling
    price_cap = df_map['price'].quantile(0.95)
    df_map_filtered = df_map[df_map['price'] <= price_cap].copy()

    # ==========================================================
    # FIXED ASSET 1: INTERACTIVE HTML MAP (No more 403 Blocks)
    # ==========================================================
    print("🔥 Building interactive Folium HTML map using CartoDB tiles...")
    zamora_map = folium.Map(location=[41.5063, -5.7446], zoom_start=14, tiles="CartoDB positron")
    
    heat_data = df_map_filtered[['latitude', 'longitude', 'price']].values.tolist()
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(zamora_map)
    
    html_output = os.path.join(output_dir, "zamora_market_heatmap.html")
    zamora_map.save(html_output)
    print(f"✅ Interactive map saved to: {html_output}")

    # ==========================================================
    # FIXED ASSET 2: STATIC HEATMAP WITH MAP BACKGROUND
    # ==========================================================
    print("🎨 Generating static price heatmap PNG with real map background...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(
        x=df_map_filtered['longitude'], 
        y=df_map_filtered['latitude'], 
        c=df_map_filtered['price'], 
        cmap='YlOrRd', 
        s=90, 
        alpha=0.75, 
        edgecolors='black', 
        linewidth=0.6,
        zorder=2
    )
    
    # Render map images underneath the dataset coordinate dots
    cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron)
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Property Price (€)', fontsize=12, labelpad=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}€"))
    
    ax.set_title("Geographic Real Estate Price Heatmap (Zamora)", fontsize=14, pad=15)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    png_output = os.path.join(plot_dir, "zamora_price_heatmap.png")
    plt.savefig(png_output, dpi=300)
    plt.close()
    print(f"✅ Static heatmap image saved to: {png_output}")

if __name__ == "__main__":
    main()