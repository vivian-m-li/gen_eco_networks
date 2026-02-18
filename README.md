# gen_eco_networks

Utilities for generating synthetic ecological networks using mechanistic food web models.

# Installation

Install directory from GitHub:

`pip install git+https://github.com/vivian-m-li/gen_eco_networks.git`

# Implemented Models

## Niche Model

The niche model generates a synthetic food web with realistic structural properties by assigning each
species a niche value and a feeding range.

References:
Williams, R. J., & Martinez, N. D. (2000). Simple rules yield complex food webs.
Nature, 404(6774), 180-183. https://doi.org/10.1038/35004572

```
from gen_eco_networks.models import NicheModel

model = NicheModel(n_species=15, connectance=0.15)
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
