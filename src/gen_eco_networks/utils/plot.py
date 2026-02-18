import networkx as nx
import matplotlib.pyplot as plt


def draw_network(graph: nx.DiGraph) -> None:
    """Draws a directed graph in a hierarchical layout."""
    levels = nx.trophic_levels(graph)
    pos = nx.spring_layout(graph, seed=42)
    for node, level in levels.items():
        pos[node][1] = level  # set y-coordinate to trophic level
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        graph,
        pos,
        with_labels=True,
        node_size=200,
        node_color="lightblue",
        arrowsize=5,
    )
    plt.axis("off")
    plt.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    plt.show()
