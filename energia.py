import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import heapq
import timeit
import random


class WeightedGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.edges = []

    def add_edge(self, u, v, weight):
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))
        self.edges.append((weight, u, v))

    def model_energy_infrastructure(self, power_plants, connections):
        for plant in power_plants:
            if plant not in self.graph:
                self.graph[plant] = []

        for u, v, weight in connections:
            self.add_edge(u, v, weight)

    def visualize_graph(self, title="Grafo"):
        G = nx.Graph()
        for u, neighbors in self.graph.items():
            for v, weight in neighbors:
                G.add_edge(u, v, weight=weight)

        pos = nx.spring_layout(G)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=150, font_size=15, font_weight='bold',
                edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title(title)
        plt.show()

    def find_mst_prim(self):
        start_node = list(self.graph.keys())[0]
        visited = {start_node}
        mst_edges = []
        min_heap = [(weight, start_node, neighbor) for neighbor, weight in self.graph[start_node]]
        heapq.heapify(min_heap)

        while min_heap and len(visited) < len(self.graph):
            weight, u, v = heapq.heappop(min_heap)
            if v not in visited:
                visited.add(v)
                mst_edges.append((weight, u, v))
                for neighbor, next_weight in self.graph[v]:
                    if neighbor not in visited:
                        heapq.heappush(min_heap, (next_weight, v, neighbor))
        return mst_edges

    def add_redundancy(self, mst_edges):
        redundancy_edges = []
        visited_nodes = set()

        for weight, u, v in self.edges:

            if (weight, u, v) not in mst_edges and (weight, v, u) not in mst_edges:
                if u not in visited_nodes and v not in visited_nodes:
                    redundancy_edges.append((weight, u, v))
                    visited_nodes.add(u)
                    visited_nodes.add(v)

                    if len(visited_nodes) == len(self.graph):
                        break

        return redundancy_edges

    def find_mst_and_redundancy(self):
        mst_edges = self.find_mst_prim()
        redundancy_edges = self.add_redundancy(mst_edges)
        return mst_edges + redundancy_edges


def generate_complete_graph(num_nodes, weight_range=(1, 100)):
    graph = WeightedGraph()
    nodes = [f"Node{i}" for i in range(num_nodes)]
    edges = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = random.randint(*weight_range)
            edges.append((nodes[i], nodes[j], weight))

    graph.model_energy_infrastructure(nodes, edges)
    return graph



def visualize_graph(graph, title="Grafo"):
    G = nx.Graph()
    for weight, u, v in graph:
        G.add_edge(u, v, weight=weight)

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=150, font_size=15, font_weight='bold',
            edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    randomG = generate_complete_graph(10)
    randomG.visualize_graph(title="Grafo Completo")

    mst_edges = randomG.find_mst_prim()
    print("Árbol de Expansión Mínima (MST):", mst_edges)
    visualize_graph(mst_edges, title="Árbol de Expansión Mínima (MST)")

    mst_redundancy = randomG.add_redundancy(mst_edges)
    mst_result = mst_edges + mst_redundancy
    print("Árbol de Expansión Mínima (MST) con redundancia", mst_result)
    visualize_graph(mst_result, title="Árbol de Expansión Mínima (MST) con redundancia")


    def run_performance_tests():
        sizes = [10, 100, 500, 1000, 1500, 2000]
        results = []

        for size in sizes:
            graph = generate_complete_graph(size)
            time_taken = timeit.timeit(graph.find_mst_and_redundancy, number=1)
            results.append((size, time_taken))
            print(f"Grafo de {size} nodos: {time_taken:.6f} segundos")

        return results


    print("\nResultados de pruebas de performance:")
    results = run_performance_tests()

    sizes, times = zip(*results)
    unique_sizes = sorted(set(sizes))

    size_to_index = {size: index for index, size in enumerate(unique_sizes)}
    x_positions = [size_to_index[size] for size in sizes]

    plt.figure(figsize=(12, 6))
    plt.scatter(x_positions, times, s=100, alpha=0.7)
    plt.xlabel("Número de nodos")
    plt.ylabel("Tiempo de ejecución (segundos)")
    plt.title("Performance del Algoritmo de Prim en Grafos de Diferentes Tamaños")

    plt.xticks(range(len(unique_sizes)), unique_sizes)

    plt.show()
