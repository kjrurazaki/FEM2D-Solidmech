import math

import numpy as np
from scipy.spatial import Delaunay

from fem_2d import Configuration


class Geometry:
    """
    Geometry class, stores all simulation configurations and material and geometry properties
    
    Attributes
    ----------
    Configurations
    rect_lenght: length of rectangular to generate
    rect_height: height of rectangular to generate
    NNodes_edge: Number of nodes in the edge of rectangular
    """

    def __init__(self, configurations: Configuration):
        self.config = configurations
        self.load_structure()
        self.update_thick()
        self.load_material()
        self.generate_loads()
        self.Nelem = len(self.triang)
        self.Nodes = len(self.coord)
        self.Nfixed = len(self.fixnodes)
        self.Nload = len(self.pointload)
        self.NSload = len(self.sideload)
        self.update_weight()

    def load_structure(self):
        if self.config.flag_preLoadedgeometry == 1:
            self.triang = np.genfromtxt(self.config.meshdir + "/triang.dat", dtype=int)
            self.coord = np.genfromtxt(self.config.meshdir + "/coord.dat")
            self.fixnodes = np.genfromtxt(
                self.config.meshdir + "/fixnodes.dat", dtype="int, int, float"
            )
            self.pointload = np.genfromtxt(
                self.config.meshdir + "/pointloads.dat", dtype="int, int, float"
            )
            self.sideload = np.genfromtxt(
                self.config.meshdir + "/sideloads.dat", dtype="int, int, float, float"
            )
        else:
            self.generate_geometry()

    def remove_indices(self):
        self.triang = self.triang[:, 1:4]
        self.coord = self.coord[:, 1:4]

    def load_material(self):
        if self.config.flag_preLoadedgeometry == 1:
            mat_prop = np.genfromtxt(self.config.meshdir + "/mat_prop.dat")
            self.young_modulus = mat_prop[0][1]
            self.poisson_ratio = mat_prop[1][1]
            self.density = mat_prop[2][1]
        else:
            self.young_modulus = self.config.young_modulus
            self.poisson_ratio = self.config.poisson_ratio
            self.density = self.config.density

    def generate_geometry(self):
        """
        Return discretized geometry and loads matrix
        """
        self.coord, self.x_zero_nodes = self.generate_rectangular()
        self.tri_object, self.triang = self.generate_elements()

    def generate_loads(self):
        self.fixnodes = self.set_boundary_nodes()
        self.pointload = self.imposed_load()
        self.sideload = self.imposed_sideload()

    def generate_rectangular(self):
        """
        Generate a rectangular form nodes with NNodes_edge for each edge
        Based on https://github.com/CyprienRusu/Feaforall/blob/master/Simple_Mesh/Simple_mesh.ipynb
        """
        coord = np.array([]).reshape(0, 3)
        x_zero_nodes = np.array([]).reshape(0, 1)
        i_nnode = 1  # Global nodes identification starts from 1
        for x in np.linspace(0, self.config.rect_length, num=self.config.NNodes_edge):
            for y in np.linspace(
                0,
                self.config.rect_heigth,
                num=math.ceil(
                    self.config.NNodes_edge
                    / (self.config.rect_length / self.config.rect_heigth)
                    / 2.0
                )
                * 2
                + 1,
            ):
                coord = np.vstack([coord, np.array([i_nnode, x, y])])
                if x == 0:
                    x_zero_nodes = np.vstack([x_zero_nodes, np.array([i_nnode])])
                i_nnode += 1
        return coord, x_zero_nodes

    def generate_elements(self):
        """
        Generate elements connectivity based on the nodes generated
        """
        tri_object = Delaunay(self.coord[:, 1:])
        triang = tri_object.simplices
        triang = triang + 1  # Global nodes start from 1
        id_array = np.array(range(1, len(triang) + 1)).reshape(-1, 1)
        triang = np.append(id_array, triang, axis=1)
        return tri_object, triang

    def set_boundary_nodes(self):
        """
        Fix x = 0 for x and y displacement
        """
        fixnodes = self.coord[self.coord[:, 1] == 0, 0]
        assert (
            fixnodes == self.x_zero_nodes.reshape(1, -1)
        ).all(), "Error in fixed nodes"
        fixnodes_x = np.append(
            fixnodes.reshape(-1, 1),
            np.array([1, 0] * len(fixnodes)).reshape(-1, 2),
            axis=1,
        )
        fixnodes_y = np.append(
            fixnodes.reshape(-1, 1),
            np.array([2, 0] * len(fixnodes)).reshape(-1, 2),
            axis=1,
        )
        fixnodes = np.append(fixnodes_x, fixnodes_y, axis=0)
        fixnodes = np.core.records.fromarrays(
            fixnodes.transpose(), names="col1, col2, col3", formats="int, int, float"
        )
        return fixnodes

    def imposed_load(self):
        """
        Impose load in the node that correspond to x and y
        direction = 1 for x and 2 for y
        """
        select_x = np.abs(self.coord[:, 1] - self.config.x_load) < 1e-3
        select_y = np.abs(self.coord[:, 2] - self.config.y_load) < 1e-3
        impose_load = self.coord[(select_x) & (select_y), 0]
        pointload = np.append(
            impose_load.reshape(-1, 1),
            np.array([self.config.direction, self.config.magnitude]).reshape(-1, 2),
            axis=1,
        )
        pointload = np.core.records.fromarrays(
            pointload.transpose(), names="col1, col2, col3", formats="int, int, float"
        )
        return pointload

    def imposed_sideload(self):
        """
        Impose sideload if any (Not developed in the project)
        """
        return np.array([])

    def update_thick(self):
        """
        If Strain problem thick == 1
        """
        # Plane stress problem = 1; Plane strain problem = 0
        if self.config.flag_planeStressorStrain == 1:
            self.thick = self.config.thick
            # print('Plane stress problem')
        else:
            self.thick = 1
            # print('Plane strain problem (thick = 1)')

    def update_weight(self):
        """
        Print status
        """
        if self.config.flag_selfWeight == 1:
            print("Self-weight applied in the problem.")
        else:
            print("Self-weight NOT applied in the problem.")

    def update_forces(self, dict_update):
        self.config.update_parameters(dict_update)
        self.generate_loads()
        self.Nfixed = len(self.fixnodes)
        self.Nload = len(self.pointload)
        self.NSload = len(self.sideload)

    def update_material(self, dict_update):
        self.config.update_parameters(dict_update)
        self.load_material()

    def update_geometry(self, dict_update):
        self.config.update_parameters(dict_update)
        self.load_structure()
        self.update_thick()
        self.Nelem = len(self.triang)
        self.Nodes = len(self.coord)
