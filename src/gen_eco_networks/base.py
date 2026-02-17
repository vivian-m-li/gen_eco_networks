"""
Base class for ecological network generation models.

All network models inherit from EcologicalNetwork and must implement the generate() method.
"""

from abc import ABC, abstractmethod

import networkx as nx
import numpy as np


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
    def generate(self) -> nx.DiGraph:
        """
        Generate and return a network.

        Must be implemented by all subclasses. Should return a networkx.DiGraph
        representing the generated ecological network.

        Returns
        -------
        nx.DiGraph
            The generated ecological network as a directed graph. Nodes are
            integer species IDs from 0 to n_species - 1.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_species={self.n_species})"
