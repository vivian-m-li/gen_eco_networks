"""
Base class for ecological network generation models.

All network models inherit from EcologicalNetwork and must implement the generate() method.
"""

from abc import ABC, abstractmethod

import os
import networkx as nx
import numpy as np
import pandas as pd


class NetworkParams:
    """
    Base class for parameters of ecological network models.

    Specific models can extend this class to include additional parameters.
    """

    pass


class EcologicalNetwork(ABC):
    """
    Abstract base class for ecological network generators.

    Parameters
    ----------
    n_species : int
        Number of species (nodes) in the network. Must be >= 2.
    seed : int or None, optional
        Seed for the random number generator. Pass an integer for
        reproducible results, or None for a random seed (default).

    Attributes
    ----------
    n_species : int
        Number of species in the network.
    rng : numpy.random.Generator
        The seeded random number generator.
    """

    def __init__(self, n_species: int, seed: int | None = None) -> None:
        if n_species < 2:
            raise ValueError(f"n_species must be >= 2, got {n_species}")
        self.n_species = n_species
        self.rng = np.random.default_rng(seed)

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_species={self.n_species})"
