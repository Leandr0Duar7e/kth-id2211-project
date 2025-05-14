import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from src.load_data.load_networks import load_network

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
NETWORKS_DIR = PROJECT_ROOT / 'data' / 'networks'
METADATA_FILE = PROJECT_ROOT / 'data' / 'metadata' / 'subreddits_metadata.json'
PARTY_DIR = PROJECT_ROOT / 'data' / 'processed' / 'network_metadata_visualizations' / 'party'
GUN_DIR = PROJECT_ROOT / 'data' / 'processed' / 'network_metadata_visualizations' / 'gun'
BANNED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'network_metadata_visualizations' / 'banned'
COMBINED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'network_metadata_visualizations' / 'combined'

# Settings
YEARS_TO_PROCESS = ['2016', '2017', '2018', '2019']
SHOW_TITLES = False  # Set to False to remove titles from all plots


for d in [PARTY_DIR, GUN_DIR, BANNED_DIR, COMBINED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load metadata
subreddit_metadata = {}
with open(METADATA_FILE, 'r') as f:
    for line in f:
        data = json.loads(line)
        subreddit_metadata[data['subreddit']] = data

# Color maps
def get_party_color(party):
    if party == 'dem':
        return '#1f77b4'  # blue
    elif party == 'rep':
        return '#d62728'  # red
    else:
        return '#555555'  # darker grey

def get_gun_color(gun):
    return '#222222' if gun else '#555555'

def get_banned_color(banned):
    return '#8B4513' if banned else '#555555'  # brown or darker grey

def draw_and_save(G, node_color_map, title, out_path):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='#000000', width=0.3)
    node_colors = [node_color_map.get(n, '#555555') for n in G.nodes]
    # Separate nodes into grey and colored
    grey_nodes = [n for n, c in zip(G.nodes, node_colors) if c == '#555555']
    colored_nodes = [n for n, c in zip(G.nodes, node_colors) if c != '#555555']
    # Draw grey nodes first (behind)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=grey_nodes,
        node_color=['#555555' for _ in grey_nodes],
        node_size=20,
        alpha=1
    )
    # Draw colored nodes on top
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=colored_nodes,
        node_color=[node_color_map[n] for n in colored_nodes],
        node_size=80,
        alpha=1
    )
    if SHOW_TITLES:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# Process each year
for fname in os.listdir(NETWORKS_DIR):
    if not fname.startswith('networks_') or not fname.endswith('.csv'):
        continue
    year = fname.split('_')[1].split('.')[0]
    if YEARS_TO_PROCESS and year not in YEARS_TO_PROCESS:
        continue
    network_path = NETWORKS_DIR / fname
    G = load_network(str(network_path), view='unweighted')

    # Ensure year subfolders exist
    party_year_dir = PARTY_DIR / year
    gun_year_dir = GUN_DIR / year
    banned_year_dir = BANNED_DIR / year
    party_year_dir.mkdir(parents=True, exist_ok=True)
    gun_year_dir.mkdir(parents=True, exist_ok=True)
    banned_year_dir.mkdir(parents=True, exist_ok=True)

    # Party coloring
    party_map = {n: get_party_color(subreddit_metadata.get(n, {}).get('party', '')) for n in G.nodes}
    draw_and_save(G, party_map, f'Party ({year})', party_year_dir / f'network_party_{year}.png')

    # Gun coloring
    gun_map = {n: get_gun_color(subreddit_metadata.get(n, {}).get('gun', 0)) for n in G.nodes}
    draw_and_save(G, gun_map, f'Gun ({year})', gun_year_dir / f'network_gun_{year}.png')

    # Banned coloring
    banned_map = {n: get_banned_color(subreddit_metadata.get(n, {}).get('banned', 0)) for n in G.nodes}
    draw_and_save(G, banned_map, f'Banned ({year})', banned_year_dir / f'network_banned_{year}.png')

# After generating all individual diagrams, create combined images for each year
for fname in os.listdir(NETWORKS_DIR):
    if not fname.startswith('networks_') or not fname.endswith('.csv'):
        continue
    year = fname.split('_')[1].split('.')[0]
    if YEARS_TO_PROCESS and year not in YEARS_TO_PROCESS:
        continue
    party_img_path = PARTY_DIR / year / f'network_party_{year}.png'
    gun_img_path = GUN_DIR / year / f'network_gun_{year}.png'
    banned_img_path = BANNED_DIR / year / f'network_banned_{year}.png'
    if all(p.exists() for p in [party_img_path, banned_img_path, gun_img_path]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, img_path in zip(
            axes,
            [party_img_path, banned_img_path, gun_img_path],
        ):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(COMBINED_DIR / f'combined_{year}.png', dpi=200)
        plt.close()

print(f"Saved all combined network metadata visualizations to {COMBINED_DIR}")

print(f"Saved all network metadata visualizations to {PROJECT_ROOT / 'data' / 'processed' / 'network_metadata_visualizations'}") 