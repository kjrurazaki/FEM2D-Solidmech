import numpy as np

def imposeBC(Nodes, NDir, DirNod, DirVal, stiffMat):
    '''
    Impose Dirichlet BCs
    '''

    penalty = 1E15
    rhs = np.zeros((1, Nodes))

    for idir in range(0, NDir):
        iglob = DirNod[idir]
        stiffMat[iglob - 1, iglob - 1] = penalty
        rhs[0, iglob - 1] = penalty * DirVal[idir]

    return stiffMat, rhs