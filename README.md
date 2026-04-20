# gen_eco_networks

Utilities for generating synthetic ecological networks using mechanistic food web models.

# Installation

Install directory from GitHub:

`pip install git+https://github.com/vivian-m-li/gen_eco_networks.git`

# Implemented Models

## Niche Model

The niche model generates a synthetic food web with realistic structural properties by assigning each species a niche value, a feeding center, and a feeding range.

References:
Williams, R. J., & Martinez, N. D. (2000). Simple rules yield complex food webs.
Nature, 404(6774), 180-183. https://doi.org/10.1038/35004572

```
from gen_eco_networks.models import NicheModel

model = NicheModel(n_species=15, connectance=0.15)
web, params = model.generate()
```

## Probabilistic Niche Model (PNM)

The probabilistic niche model extends the niche model, where each species is assigned a niche value, a feeding center, and a feeding range. However, the PNM produces niches that are high probability and highly contiguous in the center of the feeding range and low probability and more fragmented toward the margins, rather than uniform low probability throughout the feeding range.

This implementation also correlates the niche value (n_i) and the feeding center (c_i) of each species based on the correlation coefficient (rho). This allows for more flexible generation of food webs with different structural properties. The magnitude of rho dictates how correlated (vs. random) n_i and c_i are, while the sign of rho dictates the direction of c_i within its allowed range (i.e. if species tend to feed on other species with lower or higher niche values).

```
from gen_eco_networks.models import PNM

model = PNM(
    n_species=20,
    connectance=0.15,
    rho=0.9,
    n_binary_attributes=2,
    n_numeric_attributes=4,
)
web, params = model.generate()
```

## Stochastic Block Model (SBM)

The stochastic block model generates a random graph with community structure. This implementation includes an optional hierarchical bias where edges from lower-numbered blocks to higher-numbered blocks occur more frequently than the reverse direction.
Adapted from the [networkx stochastic_block_model](https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.stochastic_block_model.html) source code.

```
from gen_eco_networks.models import SBM

model = SBM(
    n_species=20,
    n_blocks=3,
    n_binary_attributes=1,
    n_numeric_attributes=2,
    reciprocal_proportion=0.1,
)
web, params = model.generate()
```

## Random Geometric Graph (RGG)

The random geometric graph model generates a random graph with assortative/disassortative structure. The model connects pairs of nodes based on their spatial proximity in trait space. Each node is assigned a random position in a unit square, and edges are created between nodes that are within a specified distance (radius) of each other.

```
from gen_eco_networks.models import RGG

model = RGG(
    n_species=20,
    radius=1.0,
    assortative=True,
    n_binary_attributes=2,
    n_numeric_attributes=4,
)

web, params = model.generate()
```
