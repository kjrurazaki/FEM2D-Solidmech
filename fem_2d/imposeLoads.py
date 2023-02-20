import numpy as np

from fem_2d.Geometry import Geometry


def impose_selfweight(geometry: Geometry, Area):
    """Return the self weight for each node with the constribution of all elements"""
    rhs_self_weight = np.zeros((2 * geometry.Nodes, 1))
    if geometry.config.flag_selfWeight == 1:
        for iel in range(0, geometry.Nelem):
            weight = Area[iel] * geometry.density * geometry.thick / 3
            nodes = geometry.triang[iel, 1:]
            ElemFor_selfweight = np.array([0, -weight, 0, -weight, 0, -weight])
            for inod in range(0, 3):
                total_x_force = ElemFor_selfweight[2 * inod]
                loc_nod_x = 2 * (nodes[inod] - 1)
                rhs_self_weight[loc_nod_x] = rhs_self_weight[loc_nod_x] + total_x_force
                total_y_force = ElemFor_selfweight[2 * inod + 1]
                rhs_self_weight[loc_nod_x + 1] = (
                    rhs_self_weight[loc_nod_x + 1] + total_y_force
                )
    return rhs_self_weight


def impose_sideLoad(geometry: Geometry):
    """Return the array of forces needed to apply in each node
    (x, y direction in the same column)"""
    rhs_side_load = np.zeros((2 * geometry.Nodes, 1))

    for i_sl in range(0, geometry.NSload):
        X = (
            geometry.coord[(geometry.sideload[i_sl][0] - 1), 1:]
            - geometry.coord[(geometry.sideload[i_sl][1] - 1), 1:]
        )
        l = (np.matmul(X, np.transpose(X))) ** (1 / 2)  # Length of the side

        # add forces in first node
        loc_nod_x = 2 * (geometry.sideload[i_sl][0] - 1)
        rhs_side_load[loc_nod_x] = (
            rhs_side_load[loc_nod_x] + l * geometry.sideload[i_sl][2] / 2
        )
        rhs_side_load[loc_nod_x + 1] = (
            rhs_side_load[loc_nod_x + 1] + l * geometry.sideload[i_sl][3] / 2
        )

        # add forces in second node
        loc_sec_nod_x = 2 * (geometry.sideload[i_sl][1] - 1)
        rhs_side_load[loc_sec_nod_x] = (
            rhs_side_load[loc_sec_nod_x] + l * geometry.sideload[i_sl][2] / 2
        )
        rhs_side_load[loc_sec_nod_x + 1] = (
            rhs_side_load[loc_sec_nod_x + 1] + l * geometry.sideload[i_sl][3] / 2
        )
    return rhs_side_load


def impose_point_loads(geometry: Geometry):
    """Return the array of forces needed to apply in each node
    (x, y direction in the same column)"""
    rhs_point_load = np.zeros((2 * geometry.Nodes, 1))
    for i_pl in range(0, geometry.Nload):
        loc_nod_x_or_y = 2 * (geometry.pointload[i_pl][0] - 1) + (
            geometry.pointload[i_pl][1] - 1
        )
        rhs_point_load[loc_nod_x_or_y] = (
            rhs_point_load[loc_nod_x_or_y] + geometry.pointload[i_pl][2]
        )
    return rhs_point_load


def impose_LoadCs(geometry: Geometry, Area):
    """Sum all forces in each node"""
    rhs = np.zeros((2 * geometry.Nodes, 1))  # x and y in same column
    # Self-weight
    rhs_self_weight = impose_selfweight(geometry, Area)
    # Side load
    rhs_side_load = impose_sideLoad(geometry)
    # Point load
    rhs_point_load = impose_point_loads(geometry)

    rhs = rhs + rhs_self_weight + rhs_side_load + rhs_point_load
    return rhs
