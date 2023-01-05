from dbm.dumb import error
from fem_2d import fem_2d

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import tri as mtri

def start_2D():
    # Plane stress problem = 1; Plane strain problem = 0 
    flag_planeStressorStrain = 1

    triang, coord, uh = fem_2d.run_2D()
    triangulation = mtri.Triangulation(coord[:,0], coord[:,1], triang - 1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_trisurf(triangulation, uh.ravel())
    plt.show()

if __name__ == "__main__":
    start_2D()