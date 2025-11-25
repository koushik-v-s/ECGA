# algorithms/utils.py

import os
import time
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from algorithms.ecga import EnhancedCommunityGreedy


def generate_mobile_social_network(
    n_nodes: int = 300,
    avg_degree: int = 10,
    seed: int = 42
) -> nx.DiGraph:
    
    if avg_degree < 2:
        raise ValueError("avg_degree must be >= 2")

    np.random.seed(seed)

    G_undirected = nx.barabasi_albert_graph(n_nodes, avg_degree // 2, seed=seed)

    G = nx.DiGraph()
    for u, v in G_undirected.edges():
        if np.random.random() < 0.7:
            weight = np.random.exponential(scale=2.0) + 0.5
            G.add_edge(u, v, weight=weight)
        if np.random.random() < 0.7:
            weight = np.random.exponential(scale=2.0) + 0.5
            G.add_edge(v, u, weight=weight)

    return G


def compare_algorithms(
    graph: nx.DiGraph,
    k: int = 10,
    alpha: float = 0.05,
    theta: float = 0.3,
    num_simulations: int = 200,
) -> Dict:
    """
    Compare ECGA with baseline algorithms: Degree Centrality and Random selection.

    Args:
        graph: The network graph.
        k: Number of seed nodes to select.
        alpha: Diffusion speed parameter.
        theta: Combination entropy threshold.
        num_simulations: Number of Monte Carlo runs.

    Returns:
        Dictionary with result statistics for each algorithm.
    """
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON")
    print("=" * 70)

    results: Dict[str, Dict] = {}

    print("\n1. Running ECGA...")
    start_time = time.time()
    ecga = EnhancedCommunityGreedy(
        graph,
        alpha=alpha,
        theta=theta,
        num_simulations=num_simulations,
        random_state=123,
    )
    ecga_seeds = ecga.select_top_k_nodes(k)
    ecga_spread = ecga.evaluate_influence_spread(ecga_seeds)
    ecga_time = time.time() - start_time
    results['ECGA'] = {
        'seeds': ecga_seeds,
        'spread': ecga_spread,
        'time': ecga_time,
        'communities': len(ecga.communities),
    }
    print(f"   Spread: {ecga_spread:.2f}, Time: {ecga_time:.2f}s")

    print("\n2. Running Degree Centrality...")
    start_time = time.time()
    degree_dict = dict(graph.out_degree())
    degree_seeds = sorted(
        degree_dict.keys(),
        key=lambda x: degree_dict[x],
        reverse=True
    )[:k]
    degree_spread = ecga.evaluate_influence_spread(degree_seeds)
    degree_time = time.time() - start_time
    results['Degree'] = {
        'seeds': degree_seeds,
        'spread': degree_spread,
        'time': degree_time,
    }
    print(f"   Spread: {degree_spread:.2f}, Time: {degree_time:.2f}s")

    print("\n3. Running Random Selection...")
    start_time = time.time()
    random_seeds = list(np.random.choice(list(graph.nodes()), k, replace=False))
    random_spread = ecga.evaluate_influence_spread(random_seeds)
    random_time = time.time() - start_time
    results['Random'] = {
        'seeds': random_seeds,
        'spread': random_spread,
        'time': random_time,
    }
    print(f"   Spread: {random_spread:.2f}, Time: {random_time:.2f}s")

    return results


def visualize_results(results: Dict):
    """
    Visualize comparison results and save them to results/ecga_comparison.png.
    """
    os.makedirs("results", exist_ok=True)

    algorithms = list(results.keys())
    spreads = [results[a]['spread'] for a in algorithms]
    times = [results[a]['time'] for a in algorithms]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Influence Spread Comparison
    axes[0].bar(algorithms, spreads, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0].set_ylabel('Influence Spread')
    axes[0].set_title('Influence Spread Comparison')
    axes[0].set_ylim(0, max(spreads) * 1.2)
    for i, v in enumerate(spreads):
        axes[0].text(i, v + max(spreads)*0.02, f'{v:.1f}',
                     ha='center', va='bottom', fontweight='bold')

    # 2. Runtime Comparison
    axes[1].bar(algorithms, times, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[1].set_ylabel('Runtime (seconds)')
    axes[1].set_title('Runtime Comparison')
    axes[1].set_ylim(0, max(times) * 1.2 if max(times) > 0 else 1)
    for i, v in enumerate(times):
        axes[1].text(i, v + (max(times)*0.02 if max(times) > 0 else 0.02),
                     f'{v:.2f}s',
                     ha='center', va='bottom', fontweight='bold')

    # 3. Efficiency = spread per second
    efficiency = [spreads[i] / max(times[i], 1e-9) for i in range(len(algorithms))]
    axes[2].bar(algorithms, efficiency, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[2].set_ylabel('Spread / Second')
    axes[2].set_title('Efficiency Ratio')
    axes[2].set_ylim(0, max(efficiency) * 1.2)
    for i, v in enumerate(efficiency):
        axes[2].text(i, v + max(efficiency)*0.02, f'{v:.1f}',
                     ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join("results", "ecga_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as '{out_path}'")
    plt.show()



def print_summary(results: Dict, graph: nx.DiGraph):
    """
    Print detailed summary of results to console.
    """
    print("\n" + "=" * 70)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nNetwork Statistics:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
    if graph.number_of_nodes() > 0:
        print(f"  Avg Degree: {2 * graph.number_of_edges() / graph.number_of_nodes():.2f}")

    if 'ECGA' in results and 'communities' in results['ECGA']:
        print(f"  Communities Detected: {results['ECGA']['communities']}")

    print("\nAlgorithm Performance:")
    print(f"{'Algorithm':<15} {'Spread':<15} {'Time (s)':<15} {'Improvement':<15}")
    print("-" * 60)

    baseline_spread = results['Random']['spread']

    for alg in ['ECGA', 'Degree', 'Random']:
        spread = results[alg]['spread']
        time_val = results[alg]['time']
        improvement = ((spread - baseline_spread) / baseline_spread * 100) if baseline_spread > 0 else 0.0
        print(f"{alg:<15} {spread:<15.2f} {time_val:<15.2f} {improvement:>+14.1f}%")

    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)

    if 'ECGA' in results and 'Degree' in results:
        ecga_improvement = ((results['ECGA']['spread'] - results['Degree']['spread'])
                            / results['Degree']['spread'] * 100) if results['Degree']['spread'] > 0 else 0.0
        print(f"✓ ECGA achieves {ecga_improvement:+.1f}% better spread than Degree Centrality")

        ecga_eff = results['ECGA']['spread'] / max(results['ECGA']['time'], 1e-9)
        degree_eff = results['Degree']['spread'] / max(results['Degree']['time'], 1e-9)
        efficiency_gain = ((ecga_eff / degree_eff) - 1) * 100 if degree_eff > 0 else 0.0
        print(f"✓ ECGA efficiency (spread/time): {efficiency_gain:+.1f}% vs Degree Centrality")
