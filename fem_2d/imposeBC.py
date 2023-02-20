import numpy as np

from fem_2d import Geometry


def impose_DBCs(geometry: Geometry.Geometry, stiffMat, u_v_displacement):
    """
    Impose Dirichlet BCs
    """
    rhs_fixed_nodes = np.zeros((2 * geometry.Nodes, 1))
    index_fixed = list()
    for i_fix in range(0, geometry.Nfixed):
        loc_nod_x_or_y = 2 * (geometry.fixnodes[i_fix][0] - 1) + (
            geometry.fixnodes[i_fix][1] - 1
        )
        u_v_displacement[loc_nod_x_or_y] = geometry.fixnodes[i_fix][2]
        index_fixed.append(loc_nod_x_or_y)
    rhs_fixed_nodes = np.matmul(stiffMat, u_v_displacement)
    return rhs_fixed_nodes, index_fixed, u_v_displacement
