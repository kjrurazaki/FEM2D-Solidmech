import numpy as np

from fem_2d.Geometry import Geometry


def build_stiffMat(geometry: Geometry, dmat):
    """
    Calls local and global stifness matrix building
    Calculate element area and elemental coeffients of basis functions
    """
    bmat, Area = localBasis(geometry)
    stiffMat = stiffBuild(geometry, dmat, bmat, Area)
    assert np.allclose(
        stiffMat, np.transpose(stiffMat), rtol=10e-3, atol=10e-3
    ), "Stiffness matrix should be symmetric"
    return Area, stiffMat, bmat


def stiffBuild(geometry: Geometry, dmat, bmat, Area):
    """
    Buid stiffness matrix (Global assembly)
    Element stiffness matrix: (transpose(bmat)*dmat*bmat)*area*thick
    """
    stiffMat = np.zeros((2 * geometry.Nodes, 2 * geometry.Nodes))
    # print(f'Thickness: {geometry.thick}')
    for iel in range(0, geometry.Nelem):
        ElemMat = (
            np.matmul(np.matmul(np.transpose(bmat[iel]), dmat), bmat[iel])
            * Area[iel]
            * geometry.thick
        )
        for iloc in range(0, 3):
            iglob = geometry.triang[iel, iloc + 1]
            for jloc in range(0, 3):
                jglob = geometry.triang[iel, jloc + 1]
                loc_nod_x_l = 2 * (iglob - 1)
                loc_nod_x_c = 2 * (jglob - 1)
                loc_loc_x_l = 2 * iloc
                loc_loc_x_c = 2 * jloc

                stiffMat[loc_nod_x_l, loc_nod_x_c] = (
                    stiffMat[loc_nod_x_l, loc_nod_x_c]
                    + ElemMat[loc_loc_x_l, loc_loc_x_c]
                )
                stiffMat[loc_nod_x_l, loc_nod_x_c + 1] = (
                    stiffMat[loc_nod_x_l, loc_nod_x_c + 1]
                    + ElemMat[loc_loc_x_l, loc_loc_x_c + 1]
                )
                stiffMat[loc_nod_x_l + 1, loc_nod_x_c] = (
                    stiffMat[loc_nod_x_l + 1, loc_nod_x_c]
                    + ElemMat[loc_loc_x_l + 1, loc_loc_x_c]
                )
                stiffMat[loc_nod_x_l + 1, loc_nod_x_c + 1] = (
                    stiffMat[loc_nod_x_l + 1, loc_nod_x_c + 1]
                    + ElemMat[loc_loc_x_l + 1, loc_loc_x_c + 1]
                )
    return stiffMat


def localBasis(geometry: Geometry):
    """
    Build local P1 basis functions on triangles (3-noded triangular element)
    only the coefficients (b,c) multiplying x and y are built
    phi(x, y) = a + bx + cy
    Built based on page 84 from notes in Putti, Mario and Onate
    b_i = y_j - y_k (i -> j -> k -> i) - convection (e.g. b_2 = y_3 - y_1)
    c_i = x_k - x_j (i -> j -> k -> i)
    """
    Bloc = np.zeros((geometry.Nelem, 3))
    Cloc = np.zeros((geometry.Nelem, 3))
    Area = np.zeros((geometry.Nelem, 1))
    bmat = list()
    for iel in range(0, geometry.Nelem):
        nodes = geometry.triang[iel, 1:]
        p1 = geometry.coord[nodes[0] - 1, 1:].reshape(1, -1)  # coordinate of first node
        p2 = geometry.coord[nodes[1] - 1, 1:].reshape(
            1, -1
        )  # coordinate of second node
        p3 = geometry.coord[nodes[2] - 1, 1:].reshape(1, -1)  # coordinate of third node
        A = np.concatenate(
            (np.ones((3, 1)), (np.concatenate((p1, p2, p3), axis=0))), axis=1
        )
        DetA = np.linalg.det(A)
        Area[iel] = abs(DetA / 2)
        for inod in range(1, 4):
            n1 = mod_n(inod + 1, 3)
            n2 = mod_n(inod + 2, 3)
            Bloc[iel, inod - 1] = (
                geometry.coord[nodes[n1 - 1] - 1, 2]
                - geometry.coord[nodes[n2 - 1] - 1, 2]
            ) / DetA
            Cloc[iel, inod - 1] = (
                geometry.coord[nodes[n2 - 1] - 1, 1]
                - geometry.coord[nodes[n1 - 1] - 1, 1]
            ) / DetA
        bmat.append(
            np.array(
                [
                    [Bloc[iel, 0], 0, Bloc[iel, 1], 0, Bloc[iel, 2], 0],
                    [0, Cloc[iel, 0], 0, Cloc[iel, 1], 0, Cloc[iel, 2]],
                    [
                        Cloc[iel, 0],
                        Bloc[iel, 0],
                        Cloc[iel, 1],
                        Bloc[iel, 1],
                        Cloc[iel, 2],
                        Bloc[iel, 2],
                    ],
                ]
            )
        )
    return bmat, Area


def mod_n(i, j):
    """
    Define the recirculation  (i -> j -> k -> i) - convection (e.g. b_2 = y_3 - y_1)
    """
    aux = np.mod(i, j)
    if aux == 0:
        remainder = j
    else:
        remainder = aux
    return remainder
