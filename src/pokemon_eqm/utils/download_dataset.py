import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def categorize_form(form_name):
    """Categorize a Pokemon form based on its name."""
    name_lower = form_name.lower()
    
    if 'mega' in name_lower:
        return 'mega'
    elif any(region in name_lower for region in ['alola', 'galar', 'hisui', 'paldea']):
        return 'regional'
    elif 'gmax' in name_lower:
        return 'gigantamax'
    elif '-' not in form_name or form_name.endswith('-normal'):
        # Base form (no suffix or just "-normal")
        return 'base'
    else:
        return 'other'

def download_form(form_data, output_base_dir):
    """Downloads a single Pokemon form image at high resolution (official artwork 475x475)."""
    try:
        form_name = form_data['name']
        form_url = form_data['url']

        # Fetch form details to get pokemon ID
        response = requests.get(form_url, timeout=10)
        if response.status_code != 200:
            return

        form_data_json = response.json()

        # Get the pokemon ID from the form data
        pokemon_id = form_data_json.get('id')
        if not pokemon_id:
            return

        # Try high-res sources in order: official-artwork > home > sprite
        image_url = None

        # Official artwork URL pattern (475x475)
        official_url = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{pokemon_id}.png"
        # Home sprite URL pattern (512x512)
        home_url = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/home/{pokemon_id}.png"

        # Try official artwork first
        img_response = requests.get(official_url, timeout=15)
        if img_response.status_code == 200 and len(img_response.content) > 1000:
            img_data = img_response.content
        else:
            # Try home sprites
            img_response = requests.get(home_url, timeout=15)
            if img_response.status_code == 200 and len(img_response.content) > 1000:
                img_data = img_response.content
            else:
                # Fall back to form sprite
                sprites = form_data_json.get('sprites', {})
                sprite_url = sprites.get('front_default')
                if not sprite_url:
                    return
                img_response = requests.get(sprite_url, timeout=10)
                if img_response.status_code != 200:
                    return
                img_data = img_response.content

        # Categorize and save
        category = categorize_form(form_name)
        category_dir = os.path.join(output_base_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        filename = os.path.join(category_dir, f"{form_name}.png")
        with open(filename, 'wb') as handler:
            handler.write(img_data)

    except Exception as e:
        # Silently skip errors to avoid cluttering output
        pass

def download_pokemon_data(output_dir="data/raw"):
    """
    Downloads Pokemon official artwork from PokeAPI, organized by form type.
    Resolution: official-artwork (475x475) > home (512x512) > sprite (96x96)
    """
    print(f"Downloading Pokemon dataset to {output_dir}...")
    print("Resolution: official-artwork (475x475) > home (512x512) > sprite fallback")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all pokemon forms
    print("Fetching Pokemon forms list...")
    response = requests.get("https://pokeapi.co/api/v2/pokemon-form?limit=10000")
    if response.status_code != 200:
        raise Exception("Failed to fetch Pokemon forms list from PokeAPI")
        
    results = response.json()['results']
    print(f"Found {len(results)} Pokemon forms. Starting download...")
    
    # Download in parallel
    with ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(
            executor.map(lambda form: download_form(form, output_dir), results),
            total=len(results),
            desc="Downloading"
        ))
    
    # Print summary
    print("\nDownload complete! Summary:")
    for category in ['base', 'mega', 'regional', 'gigantamax', 'other']:
        category_dir = os.path.join(output_dir, category)
        if os.path.exists(category_dir):
            count = len(os.listdir(category_dir))
            print(f"  {category}: {count} images")

if __name__ == "__main__":
    download_pokemon_data()

