"""
Probabilistic niche model for food web generation.

The probabilistic niche model extends the niche model, where each species is assigned a niche value, a feeding center, and a feeding range. However, the PNM produces niches that are high probability and highly contiguous in the center of the feeding range and low probability and more fragmented toward the margins, rather than uniform low probability throughout the feeding range.

This implementation also correlates the niche value (n_i) and the feeding center (c_i) of each species based on the correlation coefficient (rho). This allows for more flexible generation of food webs with different structural properties. The magnitude of rho dictates how correlated (vs. random) n_i and c_i are, while the sign of rho dictates the direction of c_i within its allowed range (i.e. if species tend to feed on other species with lower or higher niche values).
"""

from gen_eco_networks.models.niche import NicheModel


class ProbabilisticNicheModel(NicheModel):
    """
    Variant of the niche model where n_i and c_i are correlated based on a specified correlation coefficient rho. This allows us to generate food webs with different structural properties by controlling the relationship between species' niche values and their feeding centers.

    Parameters
    ----------
    rho : float
        Correlation between n_i and c_i. Must be in [-1, 1]. A value of 0 means
        no correlation, while values close to 1 or -1 induce strong correlation. Positive values of rho will bias c_i towards the upper bound of its valid range (i.e. species tend to feed on others with higher niche values), while negative values will bias c_i towards the lower bound (i.e. species tend to feed on others with lower niche values).
    **kwargs
        All other arguments are forwarded to NicheModel.
    """

    def __init__(self, *args, rho: float = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rho = rho

    def _draw_species_params(self) -> tuple[float, float, float]:
        """
        Draw niche model parameters (n_i, r_i, c_i) for a single species.
        n_i is drawn from Uniform(0, 1), and c_i is drawn based on the covariance set.
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

        # baseline independent draw of c_i from the valid interval [r_i/2, n_i]
        c_indep = self.rng.uniform(ri / 2, ni)

        if self.rho >= 0:
            # positive rho -> upper bound of ci
            c_corr = ni
        else:
            # negative rho -> lower bound of ci
            c_corr = ri / 2

        # induce correlation between n_i and c_i by combining the aligned and independent draws, weighted by abs(rho)
        alpha = abs(self.rho)  # correlation strength
        ci = alpha * c_corr + (1 - alpha) * c_indep

        return ni, ri, ci

    def __repr__(self) -> str:
        return (
            f"ProbabilisticNicheModel(n_species={self.n_species}, "
            f"connectance={self.connectance}, rho={self.rho})"
        )
