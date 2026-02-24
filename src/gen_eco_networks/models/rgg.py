"""
Random Geometric Graphs (RGG) implementation for generating ecological networks.

The RGG model connects pairs of nodes based on their spatial proximity. Each
node is assigned a random position in a unit square, and edges are created
between nodes that are within a specified distance (radius) of each other.
"""

from dataclasses import dataclass
from scipy.spatial import distance

import networkx as nx

from gen_eco_networks.base import (
    EcologicalNetwork,
    NetworkParams,
    AttributeLookup,
)


@dataclass
class RGGParams(NetworkParams):
    """
    Parameters for the RGG model.

    Attributes
    ----------
    radius : float
        The distance threshold for connecting nodes in the RGG model.
    assortative : bool
        Whether to enforce assortative mixing (edges more likely between
        similar nodes) or disassortative mixing (edges more likely between
        dissimilar nodes).
    """

    radius: float = 1.0
    assortative: bool = True


class RGG(EcologicalNetwork):
    def __init__(
        self,
        n_species: int,
        species_attributes: dict[int, AttributeLookup] | None = None,
        n_binary_attributes: int = 0,
        n_numeric_attributes: int = 0,
        radius: float = 1.0,
        assortative: bool = True,
        seed: int | None = None,
    ) -> None:
        if (
            species_attributes is None
            and n_binary_attributes == 0
            and n_numeric_attributes == 0
        ):
            raise ValueError(
                "Must provide at least one of n_binary_attributes or "
                "n_numeric_attributes when species_attributes is not provided"
            )

        super().__init__(
            n_species,
            species_attributes,
            n_binary_attributes,
            n_numeric_attributes,
            seed,
        )
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        self.radius = radius
        self.assortative = assortative
        self.params: RGGParams | None = None

    def generate(self) -> nx.DiGraph:
        """
        Generate a random geometric graph (RGG) as a directed graph.

        Nodes are placed uniformly at random in the unit square, and edges are
        created between nodes that are within self.radius of each other. The
        direction of edges is assigned randomly.

        Returns
        -------
        graph : nx.DiGraph
            Directed graph where an edge (j -> i) indicates species i consumes
            species j.
        params: SBMParams
            Parameters used to generate the graph, including attributes,
            block assignments, and block probabilities
        """
        params = self._initialize_params()
        self.params = params
        graph = self._build_graph(params)
        return graph, params

    def _initialize_params(self) -> RGGParams:
        """Initialize all model parameters."""
        params = RGGParams()

        # attributes
        if self.species_attributes is None:
            raw_attributes = self.generate_random_attributes()
        else:
            raw_attributes = self.species_attributes
        params.attribute_values = self.min_max_scaling(raw_attributes)

        params.radius = self.radius
        params.assortative = self.assortative
        return params

    def _node_distance(self, i: int, j: int) -> float:
        """Compute the Euclidean distance between two nodes based on their attributes."""
        attr_names = self.params.attribute_values[i].keys()
        attrs_i = [
            self.params.attribute_values[i][attr_name]
            for attr_name in attr_names
        ]
        attrs_j = [
            self.params.attribute_values[j][attr_name]
            for attr_name in attr_names
        ]
        return distance.euclidean(attrs_i, attrs_j)

    def _build_graph(self, params: RGGParams) -> nx.DiGraph:
        """
        Build the directed graph based on model parameters.

        For each pair of species (i, j):
        1. Compute the Euclidean distance between their attribute vectors.
        2. If assortative and if the distance is less than params.radius,
        create an edge between them. If disassortative and if the distance is
        greater than params.radius, create an edge between them.
        3. Randomly assign the direction of the edge (i -> j or j -> i).
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n_species))

        # store attribute values as node attributes
        nx.set_node_attributes(graph, params.attribute_values)

        for i in range(self.n_species):
            for j in range(i + 1, self.n_species):
                dist = self._node_distance(i, j)
                if (params.assortative and dist < params.radius) or (
                    not params.assortative and dist > params.radius
                ):
                    if self.rng.random() < 0.5:
                        graph.add_edge(i, j)
                    else:
                        graph.add_edge(j, i)

        return graph

    def __repr__(self) -> str:
        return (
            f"RGG(n_species={self.n_species}, radius={self.radius}, "
            f"assortative={self.assortative})"
        )
