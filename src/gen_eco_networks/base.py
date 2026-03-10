"""
Base class for ecological network generation models.

All network models inherit from EcologicalNetwork and must implement the generate() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeAlias, Union

import os
import networkx as nx
import numpy as np
import pandas as pd

AttributeLookup: TypeAlias = dict[str, Union[int, float]]


@dataclass
class NetworkParams:
    """
    Base class for parameters of ecological network models.
    Specific models can extend this class to include additional parameters.

    Attributes
    ----------
    attribute_values : dict[int, AttributeLookup]
        Attribute values for each species. Keys are species IDs, values are
        dictionaries mapping attribute names to their (scaled) values.
        Attributes can be binary (0 or 1) or numeric (float).

    """

    attribute_values: dict[int, AttributeLookup] = field(default_factory=dict)


class EcologicalNetwork(ABC):
    """
    Abstract base class for ecological network generators.

    Parameters
    ----------
    n_species : int
        Number of species (nodes) in the network. Must be >= 2.
    species_attributes : dict[int, AttributeLookup], optional
        Pre-specified attributes. If provided, n_species, n_binary_attributes,
        and n_numeric_attributes are ignored. Keys are species IDs, values are
        dicts mapping attribute names to numeric values.
    n_binary_attributes : int, optional
        Number of binary (0/1) attributes to generate per species.
        Default is 0.
    n_numeric_attributes : int, optional
        Number of continuous [0,1] attributes to generate per species.
        Default is 0.
    seed : int or None, optional
        Seed for the random number generator. Pass an integer for
        reproducible results, or None for a random seed (default).
    """

    def __init__(
        self,
        n_species: int,
        species_attributes: dict[int, AttributeLookup] | None = None,
        n_binary_attributes: int = 0,
        n_numeric_attributes: int = 0,
        seed: int | None = None,
    ) -> None:
        if n_species < 2:
            raise ValueError(f"n_species must be >= 2, got {n_species}")
        self.n_species = n_species
        self.rng = np.random.default_rng(seed)

        if species_attributes is not None:
            n_species = len(species_attributes)
            self.species_attributes = species_attributes
            self.n_binary_attributes = 0
            self.n_numeric_attributes = 0
        else:
            if n_species is None:
                raise ValueError(
                    "Must provide either species_attributes or n_species"
                )
            self.species_attributes = None
            self.n_binary_attributes = n_binary_attributes
            self.n_numeric_attributes = n_numeric_attributes

    @abstractmethod
    def generate(self) -> tuple[nx.DiGraph, NetworkParams]:
        """
        Generate and return a network.

        Must be implemented by all subclasses. Should return a networkx.DiGraph
        representing the generated ecological network.

        Returns
        -------
        nx.DiGraph
            The generated ecological network as a directed graph. Nodes are
            integer species IDs from 0 to n_species - 1.
        NetworkParams
            An instance of NetworkParams (or a subclass) containing the parameters
            used to generate the network.
        """
        ...

    def initialize_attribute_params(
        self, species_attributes: dict[int, AttributeLookup] | None
    ) -> dict[int, AttributeLookup]:
        """
        Initialize attribute parameters for the model and apply min-max scaling to numeric attributes.
        """
        if species_attributes is None:
            raw_attributes = self._generate_random_attributes()
        else:
            raw_attributes = species_attributes
        attribute_values = self._min_max_scaling(raw_attributes)
        return attribute_values

    def set_node_attributes(
        self, graph: nx.DiGraph, attributes: dict[int, AttributeLookup]
    ) -> None:
        """Set node attributes in the graph in-place."""
        nx.set_node_attributes(graph, attributes)

    def _generate_random_attributes(self) -> dict[int, AttributeLookup]:
        """
        Generate random attributes for species.

        Binary attributes are assigned with equal probability of being 0 or 1.
        Numeric attributes are assigned uniformly from [0, 1].

        Returns
        -------
        dict[int, AttributeLookup]
            A dictionary mapping species IDs to their attributes.
            Attribute keys are of the form "binary_attr_i" or "numeric_attr_i".
        """
        species_attributes: dict[int, AttributeLookup] = {}
        for species in range(self.n_species):
            attributes: AttributeLookup = {}
            for i in range(self.n_binary_attributes):
                attributes[f"binary_attr_{i}"] = int(self.rng.choice([0, 1]))
            for i in range(self.n_numeric_attributes):
                attributes[f"numeric_attr_{i}"] = float(self.rng.uniform(0, 1))
            species_attributes[species] = attributes
        return species_attributes

    def _min_max_scaling(
        self, attributes: dict[int, AttributeLookup]
    ) -> dict[int, AttributeLookup]:
        """
        Apply min-max scaling to numeric attributes across all species.
        For each attribute, scales values to [0, 1]. If all species have the
        same value for an attribute, that attribute is set to 1.0 for all species.

        Parameters
        ----------
        attributes : dict[int, AttributeLookup]
            Raw attribute values.

        Returns
        -------
        dict[int, AttributeLookup]
            Scaled attribute values.
        """
        attribute_names = set(
            name
            for species_attrs in attributes.values()
            for name in species_attrs.keys()
        )

        scaled_attributes: dict[int, AttributeLookup] = {
            species: attrs.copy() for species, attrs in attributes.items()
        }

        for attr_name in attribute_names:
            species_with_attr = [
                species
                for species in attributes
                if attr_name in attributes[species]
            ]
            values = [
                attributes[species][attr_name] for species in species_with_attr
            ]

            if not isinstance(values[0], (int, float)):
                continue  # skip non-numeric attributes

            max_val = max(values)
            min_val = min(values)

            # scale each species' value
            for species in species_with_attr:
                if max_val == min_val:
                    scaled_attributes[species][attr_name] = 1.0
                else:
                    raw_value = attributes[species][attr_name]
                    scaled_attributes[species][attr_name] = round(
                        (raw_value - min_val) / (max_val - min_val), 3
                    )

        return scaled_attributes

    def _write_nodes(self, graph: nx.DiGraph, file: str) -> None:
        """Write nodes to a file."""
        with open(file, "w") as f:
            f.write("species_id\n")
            for node_id in graph.nodes():
                f.write(f"{node_id}\n")

    def _write_edgelist(self, graph: nx.DiGraph, file: str) -> None:
        """Write edges to a file."""
        nx.write_edgelist(graph, file, data=False)

    def _write_attributes(self, graph: nx.DiGraph, file: str) -> None:
        """
        Write node attributes to a file. If no nodes have attributes, then
        no file is written.
        """
        species_attributes = graph.nodes(data=True)
        attribute_keys = set()
        for attr in dict(species_attributes).values():
            attribute_keys.update(attr.keys())

        if len(attribute_keys) == 0:
            return  # no attributes to write

        species_ids = set(graph.nodes())
        with open(file, "w") as f:
            f.write(f"species_id,{','.join(attribute_keys)}\n")
            for species_id in species_ids:
                attribute_values = [
                    str(species_attributes[species_id].get(attribute, ""))
                    for attribute in attribute_keys
                ]
                f.write(f"{species_id},{','.join(attribute_values)}\n")

    def save(self, graph: nx.DiGraph, dir: str, gml: bool = True) -> None:
        """Write the generated web to node, edge, and attribute files."""

        if not os.path.isdir(dir):
            os.makedirs(dir)

        if gml:
            nx.write_gml(
                nx.relabel_nodes(graph, str), os.path.join(dir, "graph.gml")
            )
        else:
            self._write_nodes(graph, os.path.join(dir, "nodes.txt"))
            self._write_edgelist(graph, os.path.join(dir, "edges.txt"))
            self._write_attributes(graph, os.path.join(dir, "attributes.txt"))

    def read(self, dir: str, gml: bool = True) -> nx.DiGraph:
        """
        Read a web from files and return the graph.
        Files must be named "nodes.txt", "edges.txt", and "attributes.txt"
        if gml=False, or "graph.gml" if gml=True.
        """
        if gml:
            return nx.read_gml(os.path.join(dir, "graph.gml"), label="id")

        # read in edges first
        graph = nx.read_edgelist(
            os.path.join(dir, "edges.txt"), create_using=nx.DiGraph
        )

        # add missing nodes
        with open(os.path.join(dir, "nodes.txt"), "r") as f:
            next(f)  # skip header
            for line in f:
                node_id = line.strip()
                if node_id not in graph:
                    graph.add_node(node_id)

        # add node attributes
        if os.path.exists(os.path.join(dir, "attributes.txt")):
            df = pd.read_csv(os.path.join(dir, "attributes.txt"))
            attributes = set(df.columns) - {"species_id"}
            for _, row in df.iterrows():
                species_id = str(int(row["species_id"]))
                for attr in attributes:
                    graph.nodes[species_id][attr] = float(row[attr])

        return graph

    def train_test_split(
        self, graph: nx.DiGraph, test_size: float = 0.2
    ) -> tuple[nx.DiGraph, nx.DiGraph]:
        """
        Split the directed edges into training and testing sets for link prediction.
        TODO: differentiate between negative edges and unobserved edges, and sample negative edges for testing.

        Returns
        -------
        train_subgraph : nx.DiGraph
            A subgraph containing the training edges.
            The subgraph contains (1 - test_size) * total_edges number of edges.
        test_subgraph : nx.DiGraph
            A subgraph containing the testing edges.
            The subgraph contains test_size * total_edges number of edges.
        """
        edges = list(graph.edges())
        self.rng.shuffle(edges)
        split_idx = int(len(edges) * (1 - test_size))

        train_edges = edges[:split_idx]
        train_subgraph = nx.DiGraph()
        train_subgraph.add_nodes_from(graph.nodes(data=True))
        train_subgraph.add_edges_from(train_edges)

        test_edges = edges[split_idx:]
        test_subgraph = nx.DiGraph()
        test_subgraph.add_nodes_from(graph.nodes(data=True))
        test_subgraph.add_edges_from(test_edges)

        return train_subgraph, test_subgraph

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_species={self.n_species})"
