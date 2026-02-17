"""
Niche model for food web generation.

Implements the niche model of Williams & Martinez (2000), which generates
synthetic food webs with realistic structural properties by assigning each
species a niche value and a feeding range.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from ecological_networks.base import EcologicalNetwork


@dataclass
class NicheModelParams:
    niche_values: dict[int, float] = field(default_factory=dict)
    range_values: dict[int, float] = field(default_factory=dict)
    center_values: dict[int, float] = field(default_factory=dict)


class NicheModel(EcologicalNetwork):
    def __init__(
        self, n_species: int, connectance: float, seed: int | None = None
    ) -> None:
        super().__init__(n_species, seed)

        if not (0 < connectance < 0.5):
            raise ValueError(
                f"connectance must be in (0, 0.5), got {connectance}. "
                "Values outside this range are not biologically realistic."
            )

        self.connectance = connectance
        self.params: NicheModelParams | None = None

    def generate(self) -> nx.DiGraph:
        """
        Generate a food web using the niche model.

        Iteratively assigns niche parameters and rebuilds the network until all
        species are connected and no two species are trophically identical.

        Returns
        -------
        graph : nx.DiGraph
            Directed graph where an edge (j -> i) indicates species i consumes
            species j.
        """
        params = self._initialize_params()
        graph = self._build_graph(params)

        # Replace problem species one at a time until all species are
        # connected and unique
        problem_species = self._get_problem_species(graph)
        while problem_species:
            species_to_replace = problem_species[0]
            self._reassign_species(params, species_to_replace)
            graph = self._build_graph(params)
            problem_species = self._get_problem_species(graph)

        return graph

    def _beta_param(self) -> float:
        """Compute the beta distribution parameter B from connectance."""
        return (1 - 2 * self.connectance) / (2 * self.connectance)

    def _draw_species_params(self) -> tuple[float, float, float]:
        """
        Draw niche model parameters (n_i, r_i, c_i) for a single species.
        Uses the inverse CDF method to sample from Beta(1, B):
            x = 1 - (1 - U)^(1/B), where U ~ Uniform(0, 1)

        Returns
        -------
        ni, ri, ci : float
            Niche value, feeding range, and feeding center for the species.
        """
        B = self._beta_param()
        ni = self.rng.uniform(0, 1)

        # draw r_i from Beta(1, B)
        u = self.rng.uniform(0, 1)
        x = 1 - (1 - u) ** (1 / B)
        ri = x * ni  # scale by niche value

        # assign the species a center of the range
        # uniformly within the interval [r_i/2, n_i]
        ci = self.rng.uniform(ri / 2, ni)

        return ni, ri, ci

    def _assign_basal(self, params: NicheModelParams) -> None:
        """
        Set r_i = 0 for the species with the smallest niche value.
        This ensures that at least one basal species exists in the network.

        Modifies params.range_values in place.
        """
        basal_id = min(params.niche_values, key=params.niche_values.__getitem__)
        params.range_values[basal_id] = 0.0

    def _initialize_params(self) -> NicheModelParams:
        """Initialize niche parameters for all species."""
        params = NicheModelParams()
        for i in range(self.n_species):
            ni, ri, ci = self._draw_species_params()
            params.niche_values[i] = ni
            params.range_values[i] = ri
            params.center_values[i] = ci
        self._assign_basal(params)
        return params

    def _build_graph(self, params: NicheModelParams) -> nx.DiGraph:
        """
        Build the directed graph based on the niche parameters.

        An edge (j -> i) exists if species i consumes species j, which occurs if
        the niche value of j falls within the feeding interval of i:
            [c_i - r_i/2, c_i + r_i/2]
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n_species))

        for i in range(self.n_species):
            lower = params.center_values[i] - params.range_values[i] / 2
            upper = params.center_values[i] + params.range_values[i] / 2
            for j in range(self.n_species):
                # a species cannot consume itself, so skip if i == j
                # otherwise, add an edge if j's niche value falls within i's
                # feeding interval
                if i != j and lower < params.niche_values[j] < upper:
                    graph.add_edge(j, i)

        return graph

    def _find_trophically_identical_species(
        self, graph: nx.DiGraph
    ) -> list[int]:
        """
        Identify species to replace based on trophic similarity.

        Two species are trophically identical if they have the exact same
        sets of prey and predators. This is ecologically unrealistic and
        violates the competitive exclusion principle. For each group of
        identical species, all but one will be flagged for replacement.

        Returns
        -------
        list[int]
            Species IDs to replace
        """
        # group species by their (prey_set, predator_set) signature
        signature_groups: defaultdict[tuple, list[int]] = defaultdict(list)
        for node in graph.nodes():
            prey = tuple(sorted(j for j, _ in graph.in_edges(node)))
            predators = tuple(sorted(k for _, k in graph.out_edges(node)))
            key = (prey, predators)
            signature_groups[key].append(node)

        # keep one species from each group and flag the rest for replacement
        to_replace = []
        for group in signature_groups.values():
            if len(group) > 1:
                keep = self.rng.choice(group)
                to_replace.extend(s for s in group if s != keep)

        return to_replace

    def _get_problem_species(self, graph: nx.DiGraph) -> list[int]:
        """
        Return a list of species IDs that need to be replaced.

        A species is a "problem species" if it is either:
        1. Unconnected: has no prey and no predators
           (in-degree = out-degree = 0)
        2. Trophically identical to another species:
           shares the same prey and predator sets as another species.
        """
        isolates = list(nx.isolates(graph))
        trophic_identical = self._find_trophically_identical_species(graph)

        # combine isolates and trophic_identical while preserving order
        problem_species = []
        seen = set()
        for species in isolates + trophic_identical:
            if species not in seen:
                seen.add(species)
                problem_species.append(species)
        return problem_species

    def _reassign_species(
        self, params: NicheModelParams, species_id: int
    ) -> None:
        """Replace niche parameters for a single species in-place."""
        ni, ri, ci = self._draw_species_params()
        params.niche_values[species_id] = ni
        params.range_values[species_id] = ri
        params.center_values[species_id] = ci
        self._assign_basal(params)

    def __repr__(self) -> str:
        return (
            f"NicheModel(n_species={self.n_species}, "
            f"connectance={self.connectance})"
        )
