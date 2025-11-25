import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Set, Dict

# Import from your package (adjust path if needed)
from algorithms.ecga import EnhancedCommunityGreedy
from algorithms.baselines import degree_centrality_baseline, random_selection_baseline
from algorithms.utils import generate_mobile_social_network

def draw_algorithm_graph(G: nx.DiGraph, 
                         communities: List[Set[int]] = None, 
                         seeds: List[int] = None, 
                         title: str = "",
                         ax=None):
    """
    Draw graph with communities (colors) and seeds (red, large).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
    
    # Default gray for no communities
    node_colors = ['lightgray'] * len(G.nodes())
    
    # Color communities if provided
    if communities:
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))  # Distinct colors
        for i, comm in enumerate(communities):
            for node in comm:
                node_colors[node] = colors[i]
    
    # Highlight seeds in red (override color)
    seed_set = set(seeds) if seeds else set()
    final_colors = ['red' if node in seed_set else node_colors[node] for node in G.nodes()]
    sizes = [500 if node in seed_set else 100 for node in G.nodes()]
    
    # Draw
    nx.draw(G, pos, ax=ax, node_color=final_colors, node_size=sizes, 
            with_labels=True, font_size=8, arrows=True, arrowsize=10)
    ax.set_title(title)
    return ax

def main():
    # Config (small for viz)
    n_nodes = 50
    avg_degree = 4
    k = 5  # Seeds
    alpha = 0.2
    theta = 0.3
    num_sim = 100  # Reduced for speed
    
    # Generate graph
    print("Generating graph...")
    G = generate_mobile_social_network(n_nodes, avg_degree, seed=42)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 1. ECGA
    print("\nRunning ECGA...")
    ecga = EnhancedCommunityGreedy(G, alpha=alpha, theta=theta, num_simulations=num_sim, random_state=123)
    ecga.detect_communities()
    ecga_seeds = ecga.select_top_k_nodes(k)
    ecga_communities = ecga.communities
    print(f"ECGA: {len(ecga_communities)} communities, seeds: {ecga_seeds}")
    
    # 2. Degree Baseline
    print("\nRunning Degree...")
    degree_seeds = degree_centrality_baseline(G, k)
    print(f"Degree seeds: {degree_seeds}")
    
    # 3. Random Baseline
    print("\nRunning Random...")
    random_seeds = random_selection_baseline(G, k)
    print(f"Random seeds: {random_seeds}")
    
    # Visualize
    print("\nGenerating plots...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # ECGA
    draw_algorithm_graph(G, ecga_communities, ecga_seeds, "ECGA: Communities (Colors) & Seeds (Red)", axs[0])
    
    # Degree
    draw_algorithm_graph(G, None, degree_seeds, "Degree Centrality: Seeds (Red, No Communities)", axs[1])
    
    # Random
    draw_algorithm_graph(G, None, random_seeds, "Random Selection: Seeds (Red, No Communities)", axs[2])
    
    plt.tight_layout()
    plt.savefig('algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved as 'algorithms_comparison.png'")

if __name__ == "__main__":
    main()