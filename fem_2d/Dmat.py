import numpy as np

from fem_2d import Geometry


def constitutive_matrix(geometry: Geometry.Geometry):
    """
    Build constitutive matrix
    """
    if geometry.config.flag_planeStressorStrain == 1:  # Plane Stress
        l1 = geometry.young_modulus / (1 - geometry.poisson_ratio**2)
        l2 = geometry.poisson_ratio * l1
        l3 = geometry.young_modulus / 2 / (1 + geometry.poisson_ratio)
    else:  # Plane strain
        l1 = (
            geometry.young_modulus
            * (1 - geometry.poisson_ratio)
            / (1 + geometry.poisson_ratio)
            / (1 - 2 * geometry.poisson_ratio)
        )
        l2 = l1 * geometry.poisson_ratio / (1 - geometry.poisson_ratio)
        l3 = geometry.young_modulus / 2 / (1 + geometry.poisson_ratio)
    dmat = np.array([[l1, l2, 0], [l2, l1, 0], [0, 0, l3]])
    return dmat
