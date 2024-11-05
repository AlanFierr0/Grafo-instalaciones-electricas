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

    def display_graph(self):
        for node in self.graph:
            print(f"{node}: {self.graph[node]}")

    def visualize_graph(self, additional_edges=None, title="Infraestructura Energética"):
        G = nx.Graph()
        for weight, u, v in self.edges:
            G.add_edge(u, v, weight=weight)

        if additional_edges:
            for weight, u, v in additional_edges:
                G.add_edge(u, v, weight=weight)

        pos = nx.spring_layout(G)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=15, font_weight='bold',
                edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title(title)
        plt.show()


def generate_random_graph(num_nodes, density=0.5, weight_range=(1, 100)):
    graph = WeightedGraph()
    nodes = [f"Node{i}" for i in range(num_nodes)]
    edges = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() <= density:
                weight = random.randint(*weight_range)
                edges.append((nodes[i], nodes[j], weight))

    graph.model_energy_infrastructure(nodes, edges)
    return graph


def run_performance_tests():
    sizes = [10, 100, 1000, 5000]  # Tamaños de grafos
    densities = [0.2, 0.5, 0.8]  # Densidades de grafos
    results = []

    for size in sizes:
        for density in densities:
            graph = generate_random_graph(size, density)
            time_taken = timeit.timeit(graph.find_mst_prim, number=1)
            results.append((size, density, time_taken))
            print(f"Grafo de {size} nodos, densidad {density}: {time_taken:.6f} segundos")

    return results


if __name__ == "__main__":
    graph = WeightedGraph()
    power_plants = ['A', 'B', 'C', 'D', 'E', 'F']
    connections = [
        ('A', 'B', 10),
        ('B', 'C', 15),
        ('C', 'D', 10),
        ('D', 'E', 15),
        ('E', 'F', 20),
        # Conexiones de redundancia
        ('A', 'C', 20),
        ('C', 'E', 25),
        ('B', 'D', 20),
        ('A', 'F', 40),
    ]

    graph.model_energy_infrastructure(power_plants, connections)
    graph.visualize_graph(title="Infraestructura Energética Original")

    mst_edges = graph.find_mst_prim()
    print("Árbol de Expansión Mínima (MST):", mst_edges)

    print("\nResultados de pruebas de performance:")
    results = run_performance_tests()

    sizes, densities, times = zip(*results)
    unique_sizes = sorted(set(sizes))

    size_to_index = {size: index for index, size in enumerate(unique_sizes)}
    x_positions = [size_to_index[size] for size in sizes]

    plt.figure(figsize=(12, 6))
    plt.scatter(x_positions, times, c=densities, cmap="viridis", s=100, alpha=0.7)
    plt.colorbar(label="Densidad")
    plt.xlabel("Número de nodos")
    plt.ylabel("Tiempo de ejecución (segundos)")
    plt.title("Performance del Algoritmo de Prim en Grafos de Diferentes Tamaños y Densidades")

    plt.xticks(range(len(unique_sizes)), unique_sizes)

    plt.show()

