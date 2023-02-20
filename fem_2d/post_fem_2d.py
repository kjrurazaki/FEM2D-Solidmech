import numpy as np

from fem_2d import Geometry


def run_postComputation(
    geometry: Geometry.Geometry,
    stiffMat,
    bmat,
    dmat,
    rhs,
    u_v_displacement,
    index_fixed,
):
    """
    Calls computation of nodal reactios and stresses (x, y, z and xy)
    """
    post_computations = {
        "Nodal reactions": nodal_reactions(
            geometry, stiffMat, rhs, u_v_displacement, index_fixed
        ),
        "Nodal stresses": nodal_stresses(geometry, bmat, dmat, u_v_displacement),
    }
    return post_computations


def nodal_reactions(geometry, stiffMat, rhs, u_v_displacement, index_fixed):
    """
    Compute Nodal force reactions in the fixednodes
    """
    reactions = np.zeros((2 * geometry.Nodes, 1))
    reactions[index_fixed] = (
        np.matmul(stiffMat[index_fixed, :], u_v_displacement) - rhs[index_fixed]
    )
    return reactions


def nodal_stresses(geometry, bmat, dmat, u_v_displacement):
    """
    Smooths stress by averaging stress in Gauss point
    """
    if geometry.config.flag_planeStressorStrain == 1:
        Nstresses = 3  # Sx, Sy, Txy
    else:
        Nstresses = 4  # Sx, Sy, + Sz, Txy
    nodal_stresses = np.zeros((geometry.Nodes, Nstresses + 1))
    S = element_stresses(geometry, dmat, bmat, u_v_displacement)

    for iel in range(0, geometry.Nelem):
        Elem_stress = S[iel]
        loc_x_and_y_elem = geometry.triang[iel, 1:] - 1
        for i_stress in range(0, Nstresses):
            nodal_stresses[loc_x_and_y_elem, i_stress] = nodal_stresses[
                loc_x_and_y_elem, i_stress
            ] + np.transpose(Elem_stress[i_stress, :])
        nodal_stresses[loc_x_and_y_elem, Nstresses] += 1

    node_stress = np.zeros((geometry.Nodes, Nstresses))
    for i_node in range(0, geometry.Nodes):
        node_stress[i_node, :] = (
            nodal_stresses[i_node, 0:(Nstresses)] / nodal_stresses[i_node, Nstresses]
        )

    arr_vonMises = compute_vonMises(node_stress)
    node_stress = np.append(node_stress, arr_vonMises.reshape(-1, 1), axis=1)
    return node_stress


def element_stresses(geometry: Geometry.Geometry, dmat, bmat, u_v_displacement):
    """
    Computes nodal stresses for the 3-noded triangular element using central Gauss point
    """
    S = list()
    for iel in range(0, geometry.Nelem):
        loc_x_and_y_elem = [
            x
            for y in zip(
                list(2 * (geometry.triang[iel, 1:] - 1)),
                list(2 * (geometry.triang[iel, 1:] - 1) + 1),
            )
            for x in y
        ]
        se = np.matmul(np.matmul(dmat, bmat[iel]), u_v_displacement[loc_x_and_y_elem])
        if geometry.config.flag_planeStressorStrain == 1:
            S.append(se)
        else:
            se = np.array(
                [se[0], se[1], -geometry.poisson_ratio * (se[0] + se[1]), se[2]]
            )
            S.append(se)
    return S


def compute_vonMises(arr_nodalstresses, load_case=False):
    """
    Compute principal stresses and von Mises stress from array of nodal stresses (each line is a node or integration point)
    Column 0 must be x stress
    Column 1 must be y stress
    Column 2 must be xy stress
    Columns 3 must be the identification of the case
    Output is the von Mises computed for each node in each case, case id is the last columns
    """
    stress_principal_I = (
        (arr_nodalstresses[:, 0] + arr_nodalstresses[:, 1]) / 2
        + 1
        / 2
        * np.power(
            [
                np.power(arr_nodalstresses[:, 0] - arr_nodalstresses[:, 1], 2)
                + 4 * np.power(arr_nodalstresses[:, 2], 2)
            ],
            1 / 2,
        )
    ).reshape(-1, 1)
    stress_principal_II = (
        (arr_nodalstresses[:, 0] + arr_nodalstresses[:, 1]) / 2
        - 1
        / 2
        * np.power(
            [
                np.power(arr_nodalstresses[:, 0] - arr_nodalstresses[:, 1], 2)
                + 4 * np.power(arr_nodalstresses[:, 2], 2)
            ],
            1 / 2,
        )
    ).reshape(-1, 1)
    sigma_mises = (
        1
        / np.sqrt(2)
        * np.sqrt(
            np.power(stress_principal_I, 2)
            + np.power(stress_principal_II, 2)
            + np.power(stress_principal_I - stress_principal_II, 2)
        )
    ).reshape(-1, 1)
    if load_case == True:
        arr_vonMises = np.append(
            sigma_mises, arr_nodalstresses[:, -1].reshape(-1, 1), axis=1
        )
    else:
        arr_vonMises = sigma_mises
    return arr_vonMises
