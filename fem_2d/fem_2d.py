# Finite element code for 2D piecewise Linear Galerkin
# -div(Diff grad u)(x) = f(x) with Dirichlet BC

from fem_2d.input_data import input_data
from fem_2d.localBasis import localBasis
from fem_2d.stiffBuild import stiffBuild
from fem_2d.imposeBC import imposeBC

import numpy as np

def run_2D():
    # Define the 2D mesh and equation BC's and coefficients
    meshdir = "small_mesh"
    data_input = input_data(meshdir)
    Nelem, Nodes, Nfixed, Nload, NSload, *other = data_input
    triang, coord, fixnodes, pointload, sideload, *other = other
    thick, young_modulus, poisson_ratio, density = other

    print(Nelem, Nodes, Nfixed, Nload, NSload,\
           triang, coord, fixnodes, pointload, sideload,\
           thick, young_modulus, poisson_ratio, density)

    # Calculate element area and elemental coeffients of basis functions
    # (b,c)
    Bloc, Cloc, Area = localBasis(Nelem, triang, coord)
    print(Bloc, Cloc, Area)

    # build stiffness matrix (without BCs)
    stiffMat = stiffBuild(Nelem, Nodes, triang, Bloc, Cloc, Area)
    print(stiffMat)

    # Impose BCs
    stiffMat, rhs = imposeBC(Nodes, stiffMat)
    print(stiffMat, rhs)

    uh = np.linalg.solve(stiffMat, rhs.transpose())
    print(uh)

    return triang, coord, uh