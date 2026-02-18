"""
Stochastic Block Model (SBM) implementation for generating ecological networks.

The SBM is a generative model for random graphs that can capture community
structure. In the context of ecological networks, it can be used to model groups
of species that interact more frequently with each other than with species in
other groups.

This implementation includes an optional hierarchical bias where edges from
lower-numbered blocks to higher-numbered blocks occur more frqeuently than the
reverse direction (e.g. to capture trophic levels in food webs).
"""

import itertools
from dataclasses import dataclass, field
from typing import TypeAlias, Union

import networkx as nx

from gen_eco_networks.base import EcologicalNetwork, NetworkParams

AttributeLookup: TypeAlias = dict[str, Union[int, float]]


@dataclass
class SBMParams(NetworkParams):
    """
    Parameters for the Stochastic Block Model.

    Attributes
    ----------
    attribute_values : dict[int, AttributeLookup]
        Attribute values for each species. Keys are species IDs, values are
        dictionaries mapping attribute names to their (scaled) values.
        Attributes can be binary (0 or 1) or numeric (float).
    block_sizes : list[int]
        Number of species in each block. Must sum to n_species.
    block_probabilities : list[list[float]]
        n_blocks x n_blocks matrix where entry [i,j] gives the base probability
        of a directed edge from block i to block j (before reciprocal adjustment).
    block_assignments : dict[int, int]
        Mapping from species ID to its assigned block ID.
    """

    attribute_values: dict[int, AttributeLookup] = field(default_factory=dict)
    block_sizes: list[int] = field(default_factory=list)
    block_probabilities: list[list[float]] = field(default_factory=list)
    block_assignments: dict[int, int] = field(default_factory=dict)


class SBM(EcologicalNetwork):
    """
    Stochastic Block Model for generating ecological networks.

    Generates directed networks by partitioning species into blocks and
    connecting them based on block-specific probabilities.
    Can optionally enforce hierarchical structure where edges preferentially
    flow from lower to higher block IDs.

    Parameters
    ----------
    n_species : int
        Number of species in the network.
    n_blocks : int
        Number of blocks (communities) to partition species into.
    n_binary_attributes : int, optional
        Number of binary (0/1) attributes to generate per species.
        Default is 0.
    n_numeric_attributes : int, optional
        Number of continuous [0,1] attributes to generate per species.
        Default is 0.
    species_attributes : dict[int, AttributeLookup], optional
        Pre-specified attributes. If provided, n_species, n_binary_attributes,
        and n_numeric_attributes are ignored. Keys are species IDs, values are
        dicts mapping attribute names to numeric values.
    block_sizes : list[int], optional
        Sizes of each block. Must sum to n_species. If None, blocks are
        assigned uniformly at random.
    block_probabilities : list[list[float]], optional
        n_blocks x n_blocks matrix of base connection probabilities. If None,
        drawn uniformly from [0, 1].
    reciprocal_proportion : float, optional
        For hierarchical structure: edges from lower to higher blocks occur
        at full probability, but reverse edges occur at reciprocal_proportion.
        Must be in [0, 1]. Set to 1.0 to disable hierarchical bias. Default 0.1.
    seed : int or None, optional
        Random seed for reproducibility.

    """

    def __init__(
        self,
        n_species: int | None = None,
        n_blocks: int = 3,
        n_binary_attributes: int = 0,
        n_numeric_attributes: int = 0,
        species_attributes: dict[int, AttributeLookup] | None = None,
        block_sizes: list[int] | None = None,
        block_probabilities: list[list[float]] | None = None,
        reciprocal_proportion: float = 0.1,
        seed: int | None = None,
    ) -> None:
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
            if n_binary_attributes == 0 and n_numeric_attributes == 0:
                raise ValueError(
                    "Must provide at least one of n_binary_attributes or "
                    "n_numeric_attributes when species_attributes is not provided"
                )
            self.species_attributes = None
            self.n_binary_attributes = n_binary_attributes
            self.n_numeric_attributes = n_numeric_attributes

        super().__init__(n_species, seed)

        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {n_blocks}.")
        if not (0 <= reciprocal_proportion <= 1):
            raise ValueError(
                f"reciprocal_proportion must be in [0, 1], got {reciprocal_proportion}."
            )

        # validate block sizes
        if block_probabilities is not None:
            if len(block_probabilities) != n_blocks:
                raise ValueError(
                    f"block_probabilities has {len(block_probabilities)} rows, "
                    f"expected {n_blocks}"
                )
            for row in block_probabilities:
                if len(row) != n_blocks:
                    raise ValueError(
                        f"block_probabilities row has {len(row)} columns, "
                        f"expected {n_blocks}"
                    )
        self.block_probabilities = block_probabilities

        self.n_blocks = n_blocks
        self.block_sizes = block_sizes
        self.reciprocal_proportion = reciprocal_proportion
        self.params: SBMParams | None = None

    def generate(self) -> nx.DiGraph:
        """
        Generate a directed network using the stochastic block model.

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
        graph = self._build_graph(params)
        self.params = params
        return graph, params

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

    def _assign_attributes(self) -> None:
        """
        Assign attributes to species in the generated graph.

        Modifies params.species_attributes in place.
        """
        if self.species_attributes is not None:
            return
        attributes = self._generate_random_attributes()
        self.species_attributes = attributes

    def _generate_block_sizes(self) -> list[int]:
        """
        Generate random block sizes that sum to n_species.
        Uses a multinomial distribution with uniform probabilities.

        Returns
        -------
        list[int]
            Block sizes (length n_blocks).
        """
        return self.rng.multinomial(
            self.n_species, [1 / self.n_blocks] * self.n_blocks
        ).tolist()

    def _generate_block_probabilities(self) -> list[list[float]]:
        """
        Generate random block connection probabilities.
        Each probability is drawn uniformly from [0, 1].

        Returns
        -------
        list[list[float]]
            n_blocks x n_blocks matrix of block connection probabilities.
        """
        return [
            [self.rng.uniform(0, 1) for _ in range(self.n_blocks)]
            for _ in range(self.n_blocks)
        ]

    def _assign_blocks(self, block_sizes: list[int]) -> dict[int, int]:
        """
        Partition species into blocks based on block sizes.
        Species are assigned to blocks sequentally.

        Parameters
        ----------
        block_sizes : list[int]
            Number of species per block.

        Returns
        -------
        dict[int, int]
            Mapping from species ID to block ID.
        """
        block_assignments = {}
        current_species = 0
        for block_id, block_size in enumerate(block_sizes):
            for _ in range(block_size):
                block_assignments[current_species] = block_id
                current_species += 1
        return block_assignments

    def _initialize_params(self) -> SBMParams:
        """Initialize all model parameters."""
        params = SBMParams()

        # attributes
        if self.species_attributes is None:
            raw_attributes = self._generate_random_attributes()
        else:
            raw_attributes = self.species_attributes
        params.attribute_values = self._min_max_scaling(raw_attributes)

        # block structure
        if self.block_sizes is None:
            params.block_sizes = self._generate_block_sizes()
        else:
            params.block_sizes = self.block_sizes

        if self.block_probabilities is None:
            params.block_probabilities = self._generate_block_probabilities()
        else:
            params.block_probabilities = self.block_probabilities

        params.block_assignments = self._assign_blocks(params.block_sizes)
        return params

    def _build_graph(self, params: SBMParams) -> nx.DiGraph:
        """
        Build the directed graph based on model parameters.

        For each pair of species (i, j):
        1. Look up base probability from block_probabilities[block_i][block_j]
        2. Apply hierarchical bias: if block_i < block_j, use full probability;
        if block_i > block_j, use reciprocal_proportion * probability
        3. Add edge (j -> i) with the adjusted probability
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n_species))

        # store attribute values as node attributes
        nx.set_node_attributes(graph, params.attribute_values)

        # store block assignments
        nx.set_node_attributes(graph, params.block_assignments, name="block")

        edges = []
        # iterate over all pairs of species (i, j), avoiding self-loops
        for i, j in itertools.combinations(graph.nodes(), 2):
            block_i = params.block_assignments[i]
            block_j = params.block_assignments[j]

            # assign an edge from i -> j, checking for hierarchical bias
            prob_ij = params.block_probabilities[block_i][block_j]
            if block_i >= block_j:
                prob_ij *= self.reciprocal_proportion
            if self.rng.random() < prob_ij:
                edges.append((i, j))  # i -> j, i.e. j consumes i

            # assign an edge from j -> i, checking for hierarchical bias
            prob_ji = params.block_probabilities[block_j][block_i]
            if block_j >= block_i:
                prob_ji *= self.reciprocal_proportion
            if self.rng.random() < prob_ji:
                edges.append((j, i))  # j -> i, i.e. i consumes j

        graph.add_edges_from(edges)
        return graph

    def __repr__(self) -> str:
        return (
            f"SBMModel(n_species={self.n_species}, n_blocks={self.n_blocks}, "
            f"reciprocal_proportion={self.reciprocal_proportion})"
        )
