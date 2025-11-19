import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_image(url, save_path):
    if not url:
        return
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    print("Fetching Pokemon list...")
    response = requests.get("https://pokeapi.co/api/v2/pokemon?limit=10000")
    data = response.json()
    results = data['results']

    print(f"Found {len(results)} Pokemon. Starting download...")

    tasks = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for i, pokemon in enumerate(results):
            # We need to fetch details to get the image URL
            # Optimization: Construct the URL directly if possible, but official artwork ID might match ID.
            # Let's fetch details to be safe and get high quality images.
            # Actually, fetching details for 1000+ items might be slow sequentially.
            # Let's do it in parallel or use a known URL pattern.
            # URL pattern: https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{id}.png
            
            pokemon_id = pokemon['url'].split('/')[-2]
            image_url = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{pokemon_id}.png"
            save_path = os.path.join(output_dir, f"{pokemon_id}.png")
            
            if not os.path.exists(save_path):
                tasks.append(executor.submit(download_image, image_url, save_path))

        for _ in tqdm(tasks, total=len(tasks)):
            _.result()

    print("Download complete.")

if __name__ == "__main__":
    main()
