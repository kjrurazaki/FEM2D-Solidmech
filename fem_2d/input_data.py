import numpy as np

def input_data(meshdir):
    triang = np.genfromtxt(meshdir + "/triang.dat", dtype = int)
    coord = np.genfromtxt(meshdir + "/coord.dat")
    DirNod = np.genfromtxt(meshdir + "/DirNod.dat", dtype = int)
    DirVal = np.genfromtxt(meshdir + "/DirVal.dat")
    Diff = np.genfromtxt(meshdir + "/Diff.dat")

    # Remove indices of elements
    triang = triang[:, 1:4]
    coord = coord[:,1:4]
    DirNod = DirNod.reshape(-1,1)
    DirVal = DirVal.reshape(-1,1)
    Diff = Diff.reshape(-1,1)

    Nelem = len(triang)
    Nodes = len(coord)
    NDir = len(DirNod)

    return Nelem, Nodes, NDir, triang, coord, DirNod, DirVal, Diff