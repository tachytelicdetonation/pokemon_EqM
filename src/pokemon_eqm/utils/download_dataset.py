import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
import io

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

        # Validate image before saving
        try:
            img = Image.open(io.BytesIO(img_data))
            img.verify()  # Verify it's a valid image
            # Re-open after verify (verify() can only be called once)
            img = Image.open(io.BytesIO(img_data))
            # Convert to RGBA to handle palette/transparency issues
            img = img.convert('RGBA')
        except Exception:
            return  # Skip invalid images

        # Categorize and save
        category = categorize_form(form_name)
        category_dir = os.path.join(output_base_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        filename = os.path.join(category_dir, f"{form_name}.png")
        img.save(filename, 'PNG')

    except Exception as e:
        # Silently skip errors to avoid cluttering output
        pass


def cleanup_corrupted_images(data_dir="data/raw"):
    """Remove corrupted images that can't be loaded by PIL."""
    print(f"Scanning {data_dir} for corrupted images...")
    removed = 0
    checked = 0

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.png'):
                filepath = os.path.join(root, f)
                checked += 1
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except Exception:
                    print(f"  Removing corrupted: {filepath}")
                    os.remove(filepath)
                    removed += 1

    print(f"Checked {checked} images, removed {removed} corrupted files.")

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
    
    # Cleanup any corrupted images
    print("\nValidating downloaded images...")
    cleanup_corrupted_images(output_dir)

    # Print summary
    print("\nDownload complete! Summary:")
    for category in ['base', 'mega', 'regional', 'gigantamax', 'other']:
        category_dir = os.path.join(output_dir, category)
        if os.path.exists(category_dir):
            count = len([f for f in os.listdir(category_dir) if f.endswith('.png')])
            print(f"  {category}: {count} images")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        # Just cleanup existing data
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/raw"
        cleanup_corrupted_images(data_dir)
    else:
        download_pokemon_data()

