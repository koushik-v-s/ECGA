# algorithms/ecga.py

import numpy as np
import networkx as nx
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple, Optional


@dataclass
class NetworkStats:
    """Statistics for network and influence spread."""
    nodes: int
    edges: int
    communities: int
    avg_degree: float
    influence_spread: float
    runtime: float


class EnhancedCommunityGreedy:
    """
    Enhanced Community-based Greedy Algorithm (ECGA)
    Implements divide-and-conquer strategy with random walk enhancements,
    community detection, cross-community influence, and DP-based selection.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        alpha: float = 0.05,
        theta: float = 0.3,
        num_simulations: int = 300,   # lowered for classroom demo
        random_state: Optional[int] = None,
    ):
        """
        Initialize ECGA.

        Args:
            graph: Directed weighted graph representing mobile social network.
            alpha: Average diffusion speed.
            theta: Combination entropy threshold for merging communities.
            num_simulations: Number of Monte Carlo simulations.
            random_state: Optional random seed for reproducibility.
        """
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("graph must be a networkx.DiGraph")

        self.graph = graph.copy()
        self.alpha = alpha
        self.theta = theta
        self.num_simulations = num_simulations

        if random_state is not None:
            np.random.seed(random_state)

        self.communities: List[Set[int]] = []
        self.community_labels: Dict[int, int] = {}
        self.border_nodes: Dict[int, Set[int]] = defaultdict(set)
        self.influence_cache: Dict[Tuple[frozenset, Optional[frozenset]], float] = {}

        self._compute_diffusion_probabilities()

    def _compute_diffusion_probabilities(self):
        """
        Compute diffusion probability for each edge based on weight.

        Formula (adapted from paper):
        p_ij = 2 * alpha * w_ij / (w_max + w_min)
        """
        weights = [self.graph[u][v].get('weight', 1.0)
                   for u, v in self.graph.edges()]

        if not weights:
            return

        w_min = min(weights)
        w_max = max(weights)

        if w_max <= 0:
            for u, v in self.graph.edges():
                self.graph[u][v]['prob'] = self.alpha
            return

        for u, v in self.graph.edges():
            w = self.graph[u][v].get('weight', 1.0)
            if w_max != w_min:
                prob = self.alpha * (2.0 * w) / (w_max + w_min)
            else:
                prob = self.alpha
            # Clamp to [0, 1]
            self.graph[u][v]['prob'] = float(min(max(prob, 0.0), 1.0))


    def random_walk_label_propagation(
        self, iterations: int = 10, walk_length: int = 4
    ) -> Dict[int, int]:
        """
        Community detection using random walk-based label propagation.

        Args:
            iterations: Number of label propagation iterations.
            walk_length: Length of random walks for probability estimation.

        Returns:
            Dictionary mapping node to community label.
        """
        labels = {node: node for node in self.graph.nodes()}

        # Compute random walk influence probabilities
        rw_influence = self._compute_random_walk_influence(walk_length)

        for _ in range(iterations):
            nodes = list(self.graph.nodes())
            np.random.shuffle(nodes)

            updated = False
            for node in nodes:
                influenced_neighbors = self._get_influenced_neighbors(
                    node, rw_influence
                )
                if not influenced_neighbors:
                    continue

                label_scores = defaultdict(float)
                for neighbor in influenced_neighbors:
                    neighbor_label = labels[neighbor]
                    prob = rw_influence.get((node, neighbor), 0.0)
                    label_scores[neighbor_label] += prob

                if label_scores:
                    new_label = max(label_scores.items(), key=lambda x: x[1])[0]
                    if new_label != labels[node]:
                        labels[node] = new_label
                        updated = True

            if not updated:
                break

        return labels

    def _compute_random_walk_influence(
        self, walk_length: int
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute influence probabilities using random walks.

        Args:
            walk_length: Length of random walks.

        Returns:
            Dictionary of (source, target) -> influence probability.
        """
        influence: Dict[Tuple[int, int], float] = defaultdict(float)
        n = self.graph.number_of_nodes()
        if n == 0:
            return influence

        num_walks = min(50, n)  # limited for efficiency

        for source in self.graph.nodes():
            walks_hit = defaultdict(int)

            for _ in range(num_walks):
                current = source
                for _ in range(walk_length):
                    neighbors = list(self.graph.successors(current))
                    if not neighbors:
                        break

                    probs = [self.graph[current][n].get('prob', 0.1)
                             for n in neighbors]
                    total = float(sum(probs))
                    if total <= 0:
                        break

                    probs = [p / total for p in probs]
                    current = np.random.choice(neighbors, p=probs)
                    if current != source:
                        walks_hit[current] += 1

            for target, hits in walks_hit.items():
                influence[(source, target)] = hits / float(num_walks)

        return influence

    def _get_influenced_neighbors(
        self,
        node: int,
        rw_influence: Dict[Tuple[int, int], float]
    ) -> List[int]:
        """Get neighbors that can be influenced by node using random walk probs."""
        influenced = []
        for neighbor in self.graph.successors(node):
            prob = rw_influence.get(
                (node, neighbor),
                self.graph[node][neighbor].get('prob', 0.0)
            )
            if np.random.random() < prob:
                influenced.append(neighbor)
        return influenced

    def detect_communities(self, iterations: int = 10) -> List[Set[int]]:
        """
        Detect communities using enhanced random-walk label propagation.
        """
        print(">>> Starting enhanced community detection...")

        labels = self.random_walk_label_propagation(iterations)

        # Group nodes by label
        communities_dict: Dict[int, Set[int]] = defaultdict(set)
        for node, label in labels.items():
            communities_dict[label].add(node)

        initial_communities = list(communities_dict.values())

        # Combine communities based on combination entropy
        self.communities = self._combine_communities(initial_communities)

        # Build community labels mapping
        self.community_labels.clear()
        for comm_id, community in enumerate(self.communities):
            for node in community:
                self.community_labels[node] = comm_id

        # Identify border nodes
        self._identify_border_nodes()

        print(f">>> Detected {len(self.communities)} communities")
        return self.communities

    def _combine_communities(
        self,
        initial_communities: List[Set[int]]
    ) -> List[Set[int]]:
        """
        Combine communities based on combination entropy threshold theta.
        """
        communities = [c.copy() for c in initial_communities]

        max_iterations = 10
        for _ in range(max_iterations):
            combined = False
            i = 0

            while i < len(communities):
                j = i + 1
                while j < len(communities):
                    entropy = self._combination_entropy(communities[i],
                                                        communities[j])

                    if entropy > self.theta:
                        communities[i] |= communities[j]
                        communities.pop(j)
                        combined = True
                    else:
                        j += 1
                i += 1

            if not combined:
                break

        return communities

    def _combination_entropy(self, comm1: Set[int], comm2: Set[int]) -> float:
        """
        Compute combination entropy between two communities.
        """
        max_entropy = 0.0

        if not comm1 or not comm2:
            return 0.0

        for v in comm1:
            for u in comm2:
                if not self.graph.has_edge(v, u):
                    continue

                r_v = self._estimate_influence_increment(v, comm1)
                r_u = self._estimate_influence_increment(u, comm1)

                if r_v > 0:
                    entropy = r_u / r_v
                    max_entropy = max(max_entropy, entropy)

        return max_entropy

    def _estimate_influence_increment(
        self,
        node: int,
        community: Set[int]
    ) -> float:
        """
        Estimate influence increment of a node in a community.
        Uses quick sampling for efficiency.
        """
        if not community:
            return 0.0

        influenced = 0
        samples = min(30, len(community)) 
        community_nodes = set(community)

        for _ in range(samples):
            active = {node}
            queue = deque([node])

            while queue:
                current = queue.popleft()
                for neighbor in self.graph.successors(current):
                    if neighbor not in community_nodes or neighbor in active:
                        continue
                    prob = self.graph[current][neighbor].get('prob', 0.05)
                    if np.random.random() < prob:
                        active.add(neighbor)
                        queue.append(neighbor)

            influenced += (len(active) - 1)

        return influenced / samples if samples > 0 else 0.0

    def _identify_border_nodes(self):
        """Identify border nodes that can influence across communities."""
        self.border_nodes.clear()

        for node in self.graph.nodes():
            node_comm = self.community_labels.get(node)
            if node_comm is None:
                continue

            for neighbor in self.graph.successors(node):
                neighbor_comm = self.community_labels.get(neighbor)
                if neighbor_comm is not None and neighbor_comm != node_comm:
                    self.border_nodes[node_comm].add(node)
                    break

    def independent_cascade(
        self,
        seed_set: Set[int],
        community: Optional[Set[int]] = None
    ) -> Set[int]:
        """
        Simulate Independent Cascade diffusion model.

        Args:
            seed_set: Initial set of active nodes.
            community: Optional subset of nodes allowed to be activated.

        Returns:
            Set of all activated nodes.
        """
        if community is None:
            allowed = set(self.graph.nodes())
        else:
            allowed = set(community)

        active = set(seed_set) & allowed
        new_active = set(active)

        while new_active:
            next_active = set()
            for node in new_active:
                for neighbor in self.graph.successors(node):
                    if neighbor not in allowed or neighbor in active:
                        continue
                    prob = self.graph[node][neighbor].get('prob', 0.05)
                    if np.random.random() < prob:
                        next_active.add(neighbor)

            active |= next_active
            new_active = next_active

        return active

    def compute_influence_spread(
        self,
        seed_set: Set[int],
        community: Optional[Set[int]] = None
    ) -> float:
        """
        Compute average influence spread using Monte Carlo simulations.
        """
        cache_key = (frozenset(seed_set),
                     frozenset(community) if community is not None else None)

        if cache_key in self.influence_cache:
            return self.influence_cache[cache_key]

        total_spread = 0.0
        for _ in range(self.num_simulations):
            active = self.independent_cascade(seed_set, community)
            total_spread += len(active)

        avg_spread = total_spread / self.num_simulations
        self.influence_cache[cache_key] = avg_spread
        return avg_spread

    def compute_marginal_gain(
        self,
        node: int,
        current_seeds: Set[int],
        community: Set[int]
    ) -> float:
        """
        Compute marginal gain of adding a node to seed set.
        (Not used in the optimized DP version, but kept for completeness.)
        """
        current_spread = self.compute_influence_spread(current_seeds, community)
        new_spread = self.compute_influence_spread(
            current_seeds | {node}, community
        )
        return new_spread - current_spread

    def select_top_k_nodes(self, k: int) -> List[int]:
        """
        Select top-K influential nodes using enhanced CGA with dynamic programming.

        Args:
            k: Number of influential nodes to select.

        Returns:
            List of top-K influential nodes.
        """
        print(f"\n>>> Selecting top-{k} influential nodes...")

        if k <= 0:
            return []

        if not self.communities:
            self.detect_communities()

        num_communities = len(self.communities)
        influential_nodes: List[int] = []
        community_seeds: List[Set[int]] = [set() for _ in range(num_communities)]
        community_spread: List[float] = [0.0 for _ in range(num_communities)]

        # DP tables: R[m][step] = best total gain
        R = [[0.0] * (k + 1) for _ in range(num_communities + 1)]
        S = [[0] * (k + 1) for _ in range(num_communities + 1)]

        for step in range(1, k + 1):
            print(f"  - Selecting node {step}/{k}...")

            delta_r: List[Tuple[float, Optional[int], int]] = []

            # For each community, find best node and marginal gain
            for m in range(num_communities):
                community = self.communities[m]
                current_seeds = community_seeds[m]

                base_spread = community_spread[m]
                best_node = None
                best_gain = -1.0

                for node in community:
                    if node in current_seeds:
                        continue

                    new_spread = self.compute_influence_spread(
                        current_seeds | {node},
                        community
                    )
                    gain = new_spread - base_spread

                    # Border nodes: add approximate cross-community influence
                    if node in self.border_nodes.get(m, set()):
                        gain += self._compute_cross_community_influence(node, m)

                    if gain > best_gain:
                        best_gain = gain
                        best_node = node

                delta_r.append((best_gain, best_node, m))

            for m in range(1, num_communities + 1):
                gain, _, comm_idx = delta_r[m - 1]

                option1 = R[m - 1][step]
                option2 = R[num_communities][step - 1] + (gain if gain > 0 else 0.0)

                if option2 > option1:
                    R[m][step] = option2
                    S[m][step] = comm_idx
                else:
                    R[m][step] = option1
                    S[m][step] = S[m - 1][step]

            selected_comm = S[num_communities][step]
            best_gain, selected_node, _ = delta_r[selected_comm]

            if selected_node is not None:
                influential_nodes.append(selected_node)
                community_seeds[selected_comm].add(selected_node)
                community_spread[selected_comm] += max(best_gain, 0.0)

        return influential_nodes

    def _compute_cross_community_influence(
        self,
        node: int,
        comm_id: int
    ) -> float:
        """
        Compute additional influence from cross-community propagation.

        Args:
            node: Border node.
            comm_id: Community ID.

        Returns:
            Additional influence from other communities (down-weighted).
        """
        additional_influence = 0.0

        for neighbor in self.graph.successors(node):
            neighbor_comm = self.community_labels.get(neighbor)
            if neighbor_comm is None or neighbor_comm == comm_id:
                continue

            neighbor_community = self.communities[neighbor_comm]
            samples = max(1, self.num_simulations // 20)

            spread = 0.0
            for _ in range(samples):
                active = self.independent_cascade({node}, neighbor_community)
                spread += len(active)

            additional_influence += spread / samples

        return 0.1 * additional_influence

    def evaluate_influence_spread(self, seed_set: List[int]) -> float:
        """
        Evaluate total influence spread of seed set on entire network.
        """
        return self.compute_influence_spread(set(seed_set))