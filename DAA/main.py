import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
from algorithms import (
    generate_mobile_social_network,
    compare_algorithms,
    visualize_results,
    print_summary,
)


def main():
    print("\n🚀 Starting ECGA Project | Design and Analysis of Algorithms")
    print("==============================================================")

    NUM_NODES = 300       
    AVG_DEGREE = 6
    SEED_SIZE = 10
    NUM_SIMULATIONS = 505
    ALPHA = 0.2
    THETA = 0.08
    print("\n📡 Generating Mobile Social Network...")
    graph = generate_mobile_social_network(NUM_NODES, AVG_DEGREE)
    print(f"✅ Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    print("\n🤖 Running ECGA and Baselines...")
    results = compare_algorithms(graph, SEED_SIZE, ALPHA, THETA, NUM_SIMULATIONS)

    visualize_results(results)

    print_summary(results, graph)

if __name__ == "__main__":
    main()
