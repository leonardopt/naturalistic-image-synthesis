"""
Step 1 — Define the stimulus set and save the design DataFrame.

Specifies the 108 object-scenes (6 categories × 18 objects) and the SDXL text
prompt for each. Builds a DataFrame with one row per image to generate
(object × seed, 60 seeds per object), saves it as parquet + CSV under
stimulus_set_designs/, and writes DESIGN_FILE to .env.
"""
import os
import random
from datetime import datetime

import pandas as pd
from naturalistic_image_synthesis.config import DESIGN_DIR, update_env_value
from naturalistic_image_synthesis.utils import generate_ids

categories = {
    "animal": {
        "dog": "picture of a dog sitting in a garden, centered in the scene, high resolution photography, joyful, cinematic",
        "cat": "photo of a cat sitting on a carpet, centered in the scene, high resolution photography, beautiful, cinematic",
        "beaver": "photo of a beaver in a river, lush vegetation, beautiful light, high resolution photography",
        "tropical_bird": "photo of a lonely tropical bird perched on a branch in a lush forest, high resolution photography, spectacular, cinematic",
        "elephant": "award-winning photo of an elephant in the grass of a jungle, centered in the scene, high resolution photography",
        "fish": "award-winning marine photo of a colorful fish in a coral reef, centered in the scene, vibrant underwater scene, high detail",
        "tortoise": "award-winning wildlife photo of a tortoise on arid land, desert in the background, centered in the scene, high resolution photography, cinematic",
        "parrot": "award-winning photo of a parrot perched in a tree, centered in the scene, high resolution, vibrant colors, cinematic",
        "seal": "photo of a seal resting on a rocky beach, centered in the scene, high resolution photography",
        "leopard": "award-winning wildlife photo of a leopard, centered in the scene, mountains in the background, high resolution photography, cinematic",
        "penguin": "photo of a penguin standing on ice, centered in the scene with a snowy backdrop, high resolution photography, endearing, cinematic",
        "butterfly": "photo of a butterfly resting on a flower, centered in the scene, high resolution photography, delicate, cinematic",
        "horse": "award-winning wildlife photo of a wild horse standing in a grassland, centered in the scene, mountains in the background, high resolution photography, cinematic",
        "lizard": "award-winning wildlife photo of a colorful lizard in dry grass, centered in the scene, arid landscape in the background, high resolution photography, cinematic",
        "owl": "photo of a barn owl looking at the camera, centered in the scene on a tree branch, high resolution photography, cinematic",
        "giraffe": "photo of a giraffe standing still in high grass covering its legs, centered in the scene with trees and the sky in the background, high resolution photography, tall, cinematic",
        "koala": "photo of a koala on an eucalyptus tree, beautiful light, high resolution photography",
        "frog": "photo of a colorful frog on a rock, pond setting in the background, high resolution photography, cinematic",
    },

    "plant": {
        "acacia": "film shot of a landscape, an acacia tree in the centre, savanna in the background, bright light, high resolution photography, cinematic",
        "baobab": "film shot of a landscape, lone Baobab tree in the center of the image, frontal view, dry landscape in the background, high resolution photography, bright light",
        "birch": "film shot of a single Birch tree in the centre, snowy ground and a clear sky in the background, high resolution photography, with the crisp light of a winter day",
        "bristlecone": "film shot of a solitary bristlecone pine in the foreground, rocky terrain and blue sky in the background, high resolution photography, timeless, cinematic, under the clear and bright light of midday",
        "cactus": "film shot of a majestic cactus in the center, frontal view, expansive desert and distant mountains at sunset in the background, high resolution photography",
        "cherry": "film shot of a lone blooming cherry tree in the centre, river in the background, high resolution photography, bright",
        "dragon_tree": "film shot of a majestic dragon-tree in the centre, arid landscape in the background, high resolution photography, crisp light",
        "eucalyptus": "film shot of a lone eucalyptus tree in the center, Australian landscape in the background, high resolution photography, clear summer sky",
        "kapok": "film shot of a majestic Kapok tree in the center, massive roots, lush tropical rainforest around it, high resolution photography, bright light",
        "magnolia": "film shot of a beautiful lone magnolia tree in the center, hills in the background, high resolution photography",
        "maple": "film shot of a solitary majestic maple tree in the center, autumnal landscape in the background, high resolution photography, golden hour",
        "oak": "film shot of an ancient oak tree in the center, rolling lush hills in the background, high resolution photography, crisp light",
        "olive": "film shot of an olive tree in the center, countryside in the background with river and rolling hills, high resolution photography, charming, cinematic, under the clear and bright light of a sunny day",
        "palm": "film shot of a lone palm tree on a tropical beach, vegetation in the background, high resolution photography, clear sky",
        "pine": "film shot of a snow-covered pine tree, frontal view, alpine mountains and a clear blue sky in the background, high resolution photography, serene, cinematic, illuminated by the bright but gentle midday sun",
        "pine_med": "film shot of a Mediterranean pine tree on a pristine coast, sea in the background, high resolution photography, cinematic",
        "sequoia": "film shot of a majestic Sequoia tree in the centre, forest and mountains in the background, high resolution photography, crisp clear sky",
        "willow": "film shot of a weeping Willow by a calm lake, high resolution photography",
    },
    "landscape_element": {
        "mountain_ridge": "film shot of a landscape, dramatic alpine ridge in the center with jagged peaks, diagonal view, snow-capped mountains and valleys on the sides, clear sky, high resolution photography, cinematic",
        "desert_arch": "film shot of a landscape, rock formation arch in an arid landscape, high resolution photography, beautiful light",
        "forest_boulder": "film shot of a colossal boulder, moss-covered, weathered, primeval forest, high resolution photography, cinematic, beautiful, bright light",
        "desert_butte": "film shot of a landscape, a solitary rock butte in the center, desert in the background, high resolution photographyy",
        "fjord_pillar": "film shot of a landscape, a solitary pillar-like rock in the center, fjords in the background, high resolution photography",
        "forest_river": "film shot of a frozen river in the center, snow-covered trees on the sides, high resolution photography, crisp winter sky, cinematic",
        "mountain_glacier": "photo of a glacier formation, frontal view, mountains on the sides, cold sunlight, cinematic",
        "grassland_monolith": "film shot of a natural monolith in the center, grassland landscape setting, high resolution photography, cinematic",
        "hill_river": "film shot of a landscape, a towering rocky hill in the center, water flowing on the side, wide hilly meadows in the background, high resolution photography, vibrant natural light",
        "geothermal_spring": "film shot of a single rocky formation in the center, hot spring inside, other steaming ponds in the background, high resolution photography, cinematic",
        "polar_iceberg": "film shot of a single iceberg in the centre, frontal view, polar seas around, high resolution photography, dramatic sky, sunset, cinematic",
        "tropical_karst": "film shot of a landscape, a lone towering limestone karst in the center, lush tropical forest on the horizon, high resolution photography",
        "lake_island": "film shot of a towering rocky lake island in the center, hilly meadows on the horizon, high resolution photography, vibrant natural light",
        "mountain_rock": "film shot of a landscape, a solitary rock formation in the center, wide mountainous landscape on the sides, high resolution photography, cinematic",
        "moorland_tor": "film shot of a landscape, a single majestic granite tor in the center of a vast moorland, high resolution photography, dramatic sky above, cinematic",
        "sea_stack": "film shot of a landscape, solitary sea stack in the center, Mediterranean coast in the background, high resolution photography, vibrant natural light",
        "volcanic_peak": "film shot of a landscape, a single towering volcanic peak in the center, field in the background, high resolution photography, dramatic sky",
        "wetland_tufa": "film shot of a landscape, a solitary tufa formation standing in the center of a wetland, high resolution photography, natural light",
    },
    "building": {
        "adobe_house": "film shot of an Adobe House in the center, frontal view, desert in the background, high resolution photography, natural sunset light",
        "bamboo_house": "film shot of a Bamboo House in the center, tropical forest in the background, high resolution photography, cinematic",
        "barn": "film shot of a traditional barn in the center, surrounded by rolling farmland, with the early morning mist gently enveloping the landscape, high resolution photography, pastoral, cinematic",
        "bell_tower": "film shot of a bell tower in the center, hills in the background, high resolution photography, cinematic",
        "farm": "film shot of a lonely farm in the center, vast open steppe, mountains in the background, high resolution photography, cinematic",
        "fortress": "film shot of a fortress with towering walls, on a hillside, high resolution photography, imposing, cinematic",
        "futuristic_building": "film shot of a futuristic building in an arid landscape, road leading to it, high resolution photography, cinematic",
        "greenhouse": "film shot of a greenhouse building from the outside, garden in the background, high resolution photography, cinematic",
        "cabin": "film shot of a cabin in the mountains, serene setting, high resolution photography, clear sky, cinematic",
        "igloo": "film shot of a solitary Igloo in the center, surrounded by the pristine snow of the Arctic, with the aurora borealis illuminating the night sky in the background, high resolution photography, serene, cinematic",
        "lighthouse": "film shot of a lighthouse on a cliff in the centre, stormy clouds on the horizon, high resolution photography, cinematic",
        "modern_building": "photo of a modern building on a hill by a river, frontal view, high resolution photography, beautiful, cinematic",
        "observatory": "film shot of an observatory on a remote hilly landscape, high resolution photography, exploratory, cinematic",
        "pagoda": "film shot of a traditional pagoda in a pastoral setting, high resolution photography, rustic, cinematic",
        "polar_base": "film  shot of a scientific facility in the polar region, high resolution photography, beautiful, cinematic",
        "skyscraper": "film shot of an art deco skyscraper, coastal city in the background, mountains on the horizon, high resolution photography, cinematic",
        "tea_house": "film shot of a traditional Tea House in the center, surrounded by a lush, meticulously kept garden, high resolution photography, tranquil, cinematic",
        "windmill": "film shot of a historic windmill in a field of wildflowers, high resolution photography, picturesque, cinematic",
    }
,
    "vehicle": {
        'campervan': 'photo of a solitary vintage camper van in a countryside camping spot, mountains in the background, high resolution photography, cinematic',
        'car_vintage': 'photo of a classic car parked on a road, coast at sunset in the background, high resolution photography, nostalgic, cinematic',
        'fishing_boat': 'photo of a fishing boat at dawn in the center, islands on the horizon, high resolution photography, peaceful, cinematic',
        'hot_air_balloon': 'photo of a colorful hot air balloon in the sky, frontal view, mountains on the horizon, high resolution photography, cinematic',
        'motorcycle': 'photo of a motorcycle parked on the side of a desert road, frontal view, high resolution photography, cinematic',
        'lorry': 'photo of a lorry parked on a road, snowy trees in the background, high resolution photography, cinematic',
        'offroad_vehicle': 'photo of an off-road vehicle in a jungle trail, lush vegetation around, high resolution photography, adventurous, cinematic',
        'passenger_train': 'photo of a modern train in the center, mountain landscape in the background, high resolution photography, cinematic',
        'rowing_boat': 'photo of a rowing boat on a calm tropical swamp, high resolution photography, beautiful light, cinematic',
        'sailboat': 'photo of a majestic sailboat on the open sea, frontal view, high resolution photography, natural sunlight, cinematic',
        'scooter': 'photo of a vintage scooter on a narrow historic street, high resolution photography, quaint, cinematic',
        'seaplane': 'photo of a seaplane parked on the water surface in the centre, tropical landscape in the background, frontal view, high resolution photography, industrial, cinematic',
        'jet_plane': 'photo of a small jet plane parked on the runway, high resolution photography, cinematic',
        'space_rocket': 'photo of a space rocket docked on a launchpad, frontal view, high resolution photography, cinematic',
        'sports_car': 'photo of a sports car parked by a scenic viewpoint, mountains in the background, high resolution photography, cinematic',
        'steam_locomotive': 'photo of a classic steam locomotive emitting steam, desert landscape, high resolution photography, nostalgic, cinematic',
        'tourist_helicopter': 'photo of a touristic helicopter, parked on the ground, high resolution photography, cinematic',
        'tractor': 'photo of a tractor parked in the middle of an agricultural field, hills in the background, bright light, high resolution photography, cinematic',
    },
    "item": {
        'basket': 'film shot of a traditional handwoven colorful basket, dry outdoors in the background, high resolution photography',
        'bottle': 'film shot of a vintage glass bottle on a cabinet, high resolution, cinematic',
        'candlestick': 'film shot of an antique candlestick holder centered, with a lit candle, casting flickering shadows, high resolution, atmospheric, cinematic',
        'chair': 'film shot of a wooden chair in a domestic setting, high resolution, nostalgic, cinematic',
        'coffee_mug': 'film shot of a coffee mug centered on a kitchen counter, steam rising from the freshly brewed coffee, high resolution, comforting, cinematic',
        'jar': 'film shot of a Murano glass jar on a shelf, painted wall in the background, high resolution photography',
        'hourglass': 'film shot of a decorated hourglass centered on a shelf, a museum room in the background, high resolution, cinematic',
        'mortar': 'photo of a decorated mortar in the centre, herborist studio in the background, high resolution',
        'lamp_standing': 'film shot of a standing lamp, high resolution, beautiful, cinematic',
        'lamp_table': 'film shot of an art nouveau lamp on a table, high resolution, beautiful, cinematic',
        'lantern': 'film shot of a decorated lantern in the center, high resolution photography, atmospheric, cinematic',
        'pot': 'film shot of a traditional ceramic pot on a fire stove, high resolution, cinematic',
        'shovel': 'film shot of a single shovel in the centre, garden in the background, high resolution, cinematic',
        'sofa': 'film shot of an art deco sofa, minimalistic background, high resolution photography',
        'vase': 'photo of a single traditional kintsugi vase, central view, tatami room in the background, high resolution photography, soft light',
        'teapot': 'film shot of a traditional teapot on a table in the centre, beautiful details, high resolution, cinematic',
        'telescope': 'photo of a modern telescope in the centre, terrace with view on a lake in the background, high resolution photography',
        'wheelbarrow': 'film shot of a wheelbarrow filled with leaves in the centre, a garden in the background, high resolution photography'

    }
}

negative_prompt = 'people, person, human figures, human body parts, humans'
variations_per_prompt = 60
id_counter = 0

# Initialize DataFrame
df = pd.DataFrame(columns=['category', 'object', 'seed', 'ID', 'prompt', 'negative_prompt'])

# Populate DataFrame
for category, objects in categories.items():
    for obj, prompt in objects.items():
        for variation in range(variations_per_prompt):
            seed = random.randint(0, 2**32 - 1)
            new_row = pd.DataFrame([{
                'category': category,
                'object': obj,
                'seed': seed,
                'exclude': 0,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            id_counter += 1
# Generate IDs
df = generate_ids(df, num_initials=3, capitalize=True)


# Save DataFrame to a CSV file
now = datetime.now()
date_time_str = now.strftime('%Y%m%d_%H%M%S')
os.makedirs(DESIGN_DIR, exist_ok=True)
df_dir = os.path.join(DESIGN_DIR, f'stim_design_{date_time_str}')
df.to_parquet(f'{df_dir}.parquet', index=False)
df.to_csv(f'{df_dir}.csv', index=False)
update_env_value('DESIGN_FILE', f'{os.path.basename(df_dir)}.parquet')

print(f'Dataframe saved in {df_dir}.')
