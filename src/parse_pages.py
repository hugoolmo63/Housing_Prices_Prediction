from bs4 import BeautifulSoup
import pandas as pd
import re
import os

def parse_and_clean_idealista():
    all_properties = []
    folder_path = "data/raw"
    
    print(f"Reading active HTML files inside: {folder_path}...")
    
    # Loop through pages 1 to 19
    for i in range(1, 20):
        possible_paths = [
            os.path.join(folder_path, f"page{i}"), 
            os.path.join(folder_path, f"page{i}.html")
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
                
        if not file_path:
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        soup = BeautifulSoup(html_content, "html.parser")
        cards = soup.find_all("article", class_=lambda x: x and "item" in x)
        
        for card in cards:
            try:
                link_tag = card.find("a", class_="item-link")
                if not link_tag:
                    continue
                    
                title = link_tag.text.strip()
                
                price_tag = card.find("span", class_="item-price")
                price_text = price_tag.text.strip() if price_tag else "0"
                clean_price = price_text.replace("€", "").replace(".", "").strip()
                
                details = card.find_all("span", class_="item-detail")
                details_text = " | ".join([d.text.strip() for d in details])
                
                if clean_price.isdigit() and int(clean_price) > 0:
                    all_properties.append({
                        "raw_title": title,
                        "price": int(clean_price),
                        "raw_details": details_text
                    })
            except Exception:
                continue

    df = pd.DataFrame(all_properties)
    
    if df.empty:
        print("Error: No data found inside the files.")
        return

    print(f"\nExtracted {len(df)} raw rows. Beginning feature engineering process...")

    # --- 1. Split raw_title into Property Type and Location ---
    # We split on ' en ' (case-insensitive) but only on the first occurrence
    df[['property_type', 'location']] = df['raw_title'].str.split(r'\s+[eE]n\s+', n=1, expand=True)
    # Fallback if text structural parsing missed ' en '
    df['property_type'] = df['property_type'].fillna(df['raw_title'])
    df['location'] = df['location'].fillna("Zamora Provincia")

    # --- 2. Extract Square Meters (Numeric) ---
    # Matches a number preceding " m²"
    df['sq_meters'] = df['raw_details'].str.extract(r'(\d+)\s*m²')
    df['sq_meters'] = pd.to_numeric(df['sq_meters']).fillna(0).astype(int)

    # --- 3. Extract Bedrooms (Numeric) ---
    # Matches a number preceding " hab" (hab. / habitaciones)
    df['bedrooms'] = df['raw_details'].str.extract(r'(\d+)\s*hab')
    df['bedrooms'] = pd.to_numeric(df['bedrooms']).fillna(0).astype(int)

    # --- 4. Extract Garage Inclusion (Binary 1 / 0) ---
    df['has_garage'] = df['raw_details'].str.contains('Garaje incluido', case=False, na=False).astype(int)

    # --- 5. Extract Further Details Column ---
    # We strip out the square meters, bedrooms, and garage text to keep this field clean
    def extract_further_details(text):
        parts = [p.strip() for p in text.split('|')]
        cleaned_parts = []
        for part in parts:
            # Drop structural tokens that are already isolated numerical features
            if 'm²' in part or 'hab' in part or 'Garaje' in part:
                continue
            if part:
                cleaned_parts.append(part)
        return ", ".join(cleaned_parts)

    df['further_details'] = df['raw_details'].apply(extract_further_details)

    # --- 6. Drop unneeded raw structural columns ---
    df = df.drop(columns=['raw_title', 'raw_details'])

    # Reorder columns logically for your ML model inputs
    final_order = ['property_type', 'location', 'sq_meters', 'bedrooms', 'has_garage', 'further_details', 'price']
    df = df[final_order]

    # Save to your local clean destination
    df.to_csv("zamora_houses_clean.csv", index=False)
    print("\nProcessing complete! Clean data saved as 'zamora_houses_clean.csv'")
    print("\nPreview of your feature columns layout:")
    print(df.head(3).to_string())

if __name__ == "__main__":
    parse_and_clean_idealista()