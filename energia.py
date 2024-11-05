import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


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

    def display_graph(self):
        for node in self.graph:
            print(f"{node}: {self.graph[node]}")

    def visualize_graph(self, additional_edges=None, title="Infraestructura Energética"):
        G = nx.Graph()
        for weight, u, v in self.edges:
            G.add_edge(u, v, weight=weight)

        if additional_edges:
            for _, u, v in additional_edges:
                G.add_edge(u, v, weight='Redundante')

        pos = nx.spring_layout(G)  # Layout para posicionar los nodos
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=15, font_weight='bold',
                edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    graph = WeightedGraph()
    power_plants = ['A', 'B', 'C', 'D', 'E', 'F']
    connections = [
        ('A', 'B', 10),
        ('A', 'C', 20),
        ('A', 'D', 25),
        ('A', 'E', 30),
        ('A', 'F', 35),
        ('B', 'C', 15),
        ('B', 'D', 20),
        ('B', 'E', 40),
        ('B', 'F', 45),
        ('C', 'D', 10),
        ('C', 'E', 25),
        ('C', 'F', 50),
        ('D', 'E', 15),
        ('D', 'F', 30),
        ('E', 'F', 20)
    ]

    graph.model_energy_infrastructure(power_plants, connections)

    graph.visualize_graph(title="Original")

