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
web = model.generate()
```
