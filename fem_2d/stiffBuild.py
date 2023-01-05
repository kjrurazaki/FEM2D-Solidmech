import numpy as np

def stiffBuild(Nelem, Nodes, triang, Bloc, Cloc, Area, Diff):
    '''
    Buid stiff matrix
    D_T - Diffusion coefficient of triangle
    a^T_{ij} = A_T * D_T * (b_i c_i) * (b_j c_j)
            = --- (b_i*b_j + c_i*c_j)
    '''
    stiffMat = np.zeros((Nodes, Nodes))
    for iel in range(0, Nelem):
        for iloc in range(0, 3):
            iglob = triang[iel, iloc]
            for jloc in range(0, 3):
                jglob = triang[iel, jloc]
                stiffMat[iglob - 1, jglob - 1] = stiffMat[iglob - 1, jglob - 1] + \
                    (Bloc[iel, iloc] * Bloc[iel, jloc] + Cloc[iel, iloc] * Cloc[iel, jloc]) * Area[iel] * Diff[iel]
    return stiffMat