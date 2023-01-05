import numpy as np

def input_data(meshdir):
    # Geometry discretized
    triang = np.genfromtxt(meshdir + "/triang.dat", dtype = int)
    coord = np.genfromtxt(meshdir + "/coord.dat")
    thick = 1 # For strain problems unit thickness
    
    # Material properties
    mat_prop = np.genfromtxt(meshdir + "/mat_prop.dat")
    young_modulus = mat_prop[0][1]
    poisson_ratio = mat_prop[1][1]
    density = mat_prop[2][1]

    # Boundary conditions
    fixnodes = np.genfromtxt(meshdir + "/fixnodes.dat")
    pointload = np.genfromtxt(meshdir + "/pointloads.dat")
    sideload = np.genfromtxt(meshdir + "/sideloads.dat")

    # Remove indices of elements
    triang = triang[:, 1:4]
    coord = coord[:,1:4]

    # Dimensions of the problem
    Nelem = len(triang)
    Nodes = len(coord)
    Nfixed = len(fixnodes)
    Nload = len(pointload)
    NSload = len(sideload)


    return [Nelem, Nodes, Nfixed, Nload, NSload,\
           triang, coord, fixnodes, pointload, sideload,\
           thick, young_modulus, poisson_ratio, density]