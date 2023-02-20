"""  Finite element code for 2D piecewise Linear Galerkin
    This module has all functions needed to call to run a 2D FEM, 
    validate the code with case presented in Onate, run grid simulation
    generation and convergence test

    It constains the following functions:
        *generate_environment - starts objects needed in the simulation
        *run_2d - run FEM based on objects from generate_environment and returns
        displacement, nodal reactions and stresses
        *solve_2D - function that calls the two before setting the correct
        environment
        *grid_2D - function that generates many simulations based on a grid
        variation of variables
        *validation_2D - run a simulation for the validation step and saves
        figures of geometr and displacement and stress fields as well as storing
        the value in points A and B needed to the validation
        *convergence_2d - run a simulation and returns the maximum values of the
        interested variables to compare converge with increasing number of elements
"""

from itertools import product

import numpy as np
from scipy.sparse.linalg import factorized

from fem_2d import (
    Configuration,
    Dmat,
    Geometry,
    display_results,
    imposeBC,
    imposeLoads,
    post_fem_2d,
    stiffBuild,
)


def generate_environment(dict_environment):
    """
    Creates two classes to define environment of simulation
    """
    run_configuration = Configuration.Configurations(dict_environment)
    run_configuration.config_rectangular(dict_environment)
    run_configuration.config_pointload(dict_environment)
    run_configuration.config_material(dict_environment)
    rect_geometry = Geometry.Geometry(run_configuration)
    return rect_geometry


def run_2D(geometry: Geometry.Geometry):
    """
    Build the global stiffness matrix and equivalent forces, solve for displacement
    and compute nodal reactions and nodal stresses.
    Returns geometry triangulation and node coordinates, node displcaements, reactions
    and stresses
    """
    # Displacement vector (u, v displacement in same column)
    u_v_displacement = np.zeros((2 * geometry.Nodes, 1))
    # constitutive matrix
    dmat = Dmat.constitutive_matrix(geometry)
    Area, stiffMat, bmat = stiffBuild.build_stiffMat(geometry, dmat)
    # Imposed BCs (loads)
    rhs = imposeLoads.impose_LoadCs(geometry, Area)
    # Impose BCs (fixednodes)
    rhs_fixed_nodes, index_fixed, u_v_displacement = imposeBC.impose_DBCs(
        geometry, stiffMat, u_v_displacement
    )
    rhs = rhs - rhs_fixed_nodes
    # Solve for free nodes (not prescribed)
    freeNodes = [i for i in range(0, 2 * geometry.Nodes) if i not in index_fixed]
    solve = factorized(stiffMat[freeNodes, :][:, freeNodes])  # Makes LU decomposition.
    u_v_displacement[freeNodes] = solve(rhs[freeNodes])
    assert np.allclose(
        rhs[freeNodes],
        np.matmul(stiffMat[freeNodes, :][:, freeNodes], u_v_displacement[freeNodes]),
        rtol=10e-4,
        atol=10e-4,
    ), "Solution not ok"

    # print(pd.DataFrame(stiffMat))
    post_computations = post_fem_2d.run_postComputation(
        geometry, stiffMat, bmat, dmat, rhs, u_v_displacement, index_fixed
    )
    return geometry.triang, geometry.coord, u_v_displacement, post_computations


def solve_2D(NNodes_long_edge=101, dict_environment=None):
    """
    Generate environment and solve for displacement, stresses and reactions
    """
    if dict_environment == None:
        dict_environment = {}

        dict_environment["rect_length"] = 10
        dict_environment["rect_heigth"] = 1
        dict_environment["thick"] = 0.1
        dict_environment["load_magnitude"] = 900
        dict_environment["material_young"] = 2e8
        dict_environment["material_poisson"] = 0.2
        dict_environment["material_density"] = 7750

    dict_environment["flag_preLoadedgeometry"] = 0
    dict_environment["meshdir"] = "small_mesh"
    dict_environment["flag_selfWeight"] = 0
    dict_environment["flag_planeStressorStrain"] = 1
    dict_environment["NNodes_long_edge"] = NNodes_long_edge
    dict_environment["load_direction"] = 2
    dict_environment["load_x"] = dict_environment[
        "rect_length"
    ]  # Load in the tip x_load = rect_lenght
    dict_environment["load_y"] = dict_environment["rect_heigth"] / 2

    rect_geometry = generate_environment(dict_environment)

    # Solve displacement
    triang, coord, u_v_displacement, post_computations = run_2D(rect_geometry)
    return triang, coord, u_v_displacement, post_computations, rect_geometry


def grid_2D():
    """
    Generates many FEM results varying Geometry, load magnitude (direction is not changed)
    and material properties
    """
    # fixed conditions
    dict_environment = {}
    dict_environment["flag_preLoadedgeometry"] = 0
    dict_environment["meshdir"] = "small_mesh"
    dict_environment["flag_selfWeight"] = 0
    dict_environment["flag_planeStressorStrain"] = 1
    dict_environment["NNodes_long_edge"] = 151

    # Geometry dictionary
    list_length = np.linspace(7, 10, num=7)
    list_heigth = np.linspace(0.8, 1.5, num=7)
    list_thick = np.linspace(0.1, 0.25, num=7)
    list_geometries = list(product(list_length, list_heigth, list_thick))
    dict_geometry = {index: x for index, x in enumerate(list_geometries, start=1)}
    v_load_values = np.linspace(85000, 105000, num=30)  # 100 loads UP/Down
    h_load_values = np.linspace(90000, 100000, num=25)  # 25 axial
    dict_load = {"vertical": [2, v_load_values], "horizontal": [1, h_load_values]}
    # 2 materials (Steel, Aluminium) # young in pa, density in kg / m3
    dict_material = {
        "steel": [2e11, 0.3, 7800],
        "aluminium": [7e10, 0.3, 2710],
        "mix_1": [8e10, 0.3, 2710],
        "mix_2": [8.5e10, 0.3, 2710],
        "mix_3": [9e10, 0.3, 2710],
        "mix_4": [9.5e10, 0.3, 2710],
        "mix_5": [1e11, 0.3, 2710],
        "mix_6": [1.2e11, 0.3, 7800],
        "mix_7": [1.4e11, 0.3, 2710],
        "mix_8": [1.6e11, 0.3, 2710],
        "mix_9": [1.8e11, 0.3, 2710],
    }

    # Initialize geometry
    dict_environment["rect_length"] = dict_geometry[list(dict_geometry.keys())[0]][0]
    dict_environment["rect_heigth"] = dict_geometry[list(dict_geometry.keys())[0]][1]
    dict_environment["thick"] = dict_geometry[list(dict_geometry.keys())[0]][2]
    dict_environment["load_x"] = dict_environment[
        "rect_length"
    ]  # Load in the tip x_load = rect_lenght
    dict_environment["load_y"] = dict_environment["rect_heigth"] / 2
    dict_environment["material_young"] = dict_material[list(dict_material.keys())[0]][0]
    dict_environment["material_poisson"] = dict_material[list(dict_material.keys())[0]][
        1
    ]
    dict_environment["material_density"] = dict_material[list(dict_material.keys())[0]][
        2
    ]
    dict_environment["load_direction"] = dict_load["vertical"][0]
    dict_environment["load_magnitude"] = 0

    rect_geometry = generate_environment(dict_environment)
    cnt_case = 72030
    arr_config = np.array([])
    arr_displacement = np.array([]).reshape(-1, 2)
    arr_nodalreactions = np.array([]).reshape(-1, 2)
    arr_nodalstresses = np.array([]).reshape(-1, 4)
    arr_coord = np.array([]).reshape(-1, 3)
    arr_triang = np.array([]).reshape(-1, 4)
    for i_geometry in dict_geometry.keys():
        # grid condition Load
        dict_environment["rect_length"] = dict_geometry[i_geometry][0]
        dict_environment["rect_heigth"] = dict_geometry[i_geometry][1]
        dict_environment["thick"] = dict_geometry[i_geometry][2]
        dict_environment["load_x"] = dict_environment[
            "rect_length"
        ]  # Load in the tip x_load = rect_lenght
        dict_environment["load_y"] = dict_environment["rect_heigth"] / 2
        rect_geometry.update_geometry(dict_environment)
        for i_material in dict_material.keys():
            dict_environment["material_young"] = dict_material[i_material][0]
            dict_environment["material_poisson"] = dict_material[i_material][1]
            dict_environment["material_density"] = dict_material[i_material][2]
            rect_geometry.update_material(dict_environment)
            # constitutive matrix
            dmat = Dmat.constitutive_matrix(rect_geometry)
            Area, stiffMat, bmat = stiffBuild.build_stiffMat(rect_geometry, dmat)
            # Impose BCs (fixednodes)
            u_v_displacement_0 = np.zeros((2 * rect_geometry.Nodes, 1))
            rhs_fixed_nodes, index_fixed, u_v_displacement_0 = imposeBC.impose_DBCs(
                rect_geometry, stiffMat, u_v_displacement_0
            )
            # Solve for free nodes (not prescribed)
            freeNodes = [
                i for i in range(0, 2 * rect_geometry.Nodes) if i not in index_fixed
            ]
            # Factorize stiffMat(for loop force only change rhs)
            solve = factorized(
                stiffMat[freeNodes, :][:, freeNodes]
            )  # Makes LU decomposition.
            arr_config = np.array([]).reshape(-1, 16)
            arr_displacement = np.array([]).reshape(-1, 2)
            arr_nodalreactions = np.array([]).reshape(-1, 2)
            arr_nodalstresses = np.array([]).reshape(-1, 5)
            arr_coord = np.array([]).reshape(-1, 3)
            arr_triang = np.array([]).reshape(-1, 4)
            for i_load_direction in ["vertical"]:
                dict_environment["load_direction"] = dict_load[i_load_direction][0]
                for i_load_magnitude in dict_load[i_load_direction][1]:
                    # Displacement vector (u, v displacement in same column)
                    u_v_displacement = u_v_displacement_0.copy()
                    dict_environment["load_magnitude"] = i_load_magnitude
                    rect_geometry.update_forces(dict_environment)
                    # Imposed BCs (loads)
                    rhs = imposeLoads.impose_LoadCs(rect_geometry, Area)
                    rhs = rhs - rhs_fixed_nodes
                    # Solve for displacement
                    u_v_displacement[freeNodes] = solve(rhs[freeNodes])
                    assert np.allclose(
                        rhs[freeNodes],
                        np.matmul(
                            stiffMat[freeNodes, :][:, freeNodes],
                            u_v_displacement[freeNodes],
                        ),
                        rtol=10e-4,
                        atol=10e-4,
                    ), "Solution not ok"
                    post_computations = post_fem_2d.run_postComputation(
                        rect_geometry,
                        stiffMat,
                        bmat,
                        dmat,
                        rhs,
                        u_v_displacement,
                        index_fixed,
                    )
                    dict_environment.update({"case": cnt_case})
                    arr_config = np.append(
                        arr_config,
                        np.array(list(dict_environment.values())).reshape(-1, 16),
                        axis=0,
                    )
                    arr_displacement = np.append(
                        arr_displacement,
                        np.insert(u_v_displacement, 1, values=cnt_case, axis=1).reshape(
                            -1, 2
                        ),
                        axis=0,
                    )
                    arr_nodalstresses = np.append(
                        arr_nodalstresses,
                        np.insert(
                            post_computations["Nodal stresses"],
                            4,
                            values=cnt_case,
                            axis=1,
                        ).reshape(-1, 5),
                        axis=0,
                    )
                    arr_coord = np.append(
                        arr_coord.reshape(-1, 3),
                        np.insert(
                            rect_geometry.coord[:, 1:], 2, values=cnt_case, axis=1
                        ).reshape(-1, 3),
                        axis=0,
                    )
                    arr_triang = np.append(
                        arr_triang,
                        np.insert(
                            rect_geometry.triang[:, 1:], 3, values=cnt_case, axis=1
                        ).reshape(-1, 4),
                        axis=0,
                    )
                    arr_nodalreactions = np.append(
                        arr_nodalreactions,
                        np.insert(
                            post_computations["Nodal reactions"],
                            1,
                            values=cnt_case,
                            axis=1,
                        ).reshape(-1, 2),
                        axis=0,
                    )
                    cnt_case += 1
                    print(cnt_case)
            np.save("configs_total_" + str(cnt_case) + ".npy", arr_config)
            np.save("coord_total_" + str(cnt_case) + ".npy", arr_coord)
            np.save("displacement_total_" + str(cnt_case) + ".npy", arr_displacement)
            np.save("nodalstresses_total_" + str(cnt_case) + ".npy", arr_nodalstresses)
            np.save(
                "nodalreactions_total_" + str(cnt_case) + ".npy", arr_nodalreactions
            )
            np.save("triang_total_" + str(cnt_case) + ".npy", arr_triang)


def validation_2D(save_id, NNodes_long_edge=101, dict_environment=None):
    """
    Run all functions related to the FEM 2D computation and results generation for the validation problem (Onate)
    """
    triang, coord, u_v_displacement, post_computations, rect_geometry = solve_2D(
        NNodes_long_edge, dict_environment
    )
    display_results.plot_geometry(
        save_id,
        rect_geometry,
        u_v_displacement,
        show_vertices=False,
        show_elements=True,
        deformed=False,
    )
    display_results.plot_geometry(
        save_id,
        rect_geometry,
        u_v_displacement,
        show_vertices=False,
        show_elements=True,
        deformed=True,
    )
    display_results.plot_field(
        save_id,
        rect_geometry,
        u_v_displacement[1::2],
        variable_name="displacement",
        nodal=False,
    )
    display_results.plot_field(
        save_id,
        rect_geometry,
        post_computations["Nodal stresses"],
        stress_direction="x",
    )
    display_results.plot_field(
        save_id,
        rect_geometry,
        post_computations["Nodal stresses"],
        stress_direction="y",
    )
    display_results.plot_field(
        save_id,
        rect_geometry,
        post_computations["Nodal stresses"],
        stress_direction="xy",
    )
    display_results.plot_field(
        save_id,
        rect_geometry,
        post_computations["Nodal stresses"],
        stress_direction="von Mises",
    )
    display_results.plot_field(
        save_id,
        rect_geometry,
        post_computations["Nodal stresses"],
        stress_direction="x",
        nodal=False,
    )
    display_results.plot_field(
        save_id,
        rect_geometry,
        post_computations["Nodal stresses"],
        stress_direction="y",
        nodal=False,
    )
    display_results.plot_field(
        save_id,
        rect_geometry,
        post_computations["Nodal stresses"],
        stress_direction="xy",
        nodal=False,
    )
    display_results.plot_field(
        save_id,
        rect_geometry,
        post_computations["Nodal stresses"],
        stress_direction="von Mises",
        nodal=False,
    )
    point_stress = display_results.f_stress(
        rect_geometry,
        post_computations["Nodal stresses"],
        rect_geometry.config.rect_length / 2,
        0,
    )
    point_displacement = display_results.f_displacement(
        rect_geometry,
        u_v_displacement,
        rect_geometry.config.rect_length,
        rect_geometry.config.rect_heigth / 2,
    )
    print(point_stress)
    print(point_displacement)
    print(f"Number of nodes: {rect_geometry.Nodes}")
    print(f"Number of elements: {rect_geometry.Nelem}")
    return point_stress, point_displacement, rect_geometry.Nodes, rect_geometry.Nelem


def convergence_2D(save_id, NNodes_long_edge=101, dict_environment=None):
    """
    Run all functions related to the FEM 2D computation and results generation to check convergence in some structures
    """
    triang, coord, u_v_displacement, post_computations, rect_geometry = solve_2D(
        NNodes_long_edge, dict_environment
    )
    max_x_stress = np.max(post_computations["Nodal stresses"][:, 0])
    max_y_stress = np.max(post_computations["Nodal stresses"][:, 1])
    max_xy_stress = np.max(post_computations["Nodal stresses"][:, 2])
    max_vonMises = np.max(post_computations["Nodal stresses"][:, 3])
    point_displacement = display_results.f_displacement(
        rect_geometry,
        u_v_displacement,
        rect_geometry.config.rect_length,
        rect_geometry.config.rect_heigth / 2,
    )
    print(f"Number of nodes: {rect_geometry.Nodes}")
    print(f"Number of elements: {rect_geometry.Nelem}")
    return (
        max_x_stress,
        max_y_stress,
        max_xy_stress,
        point_displacement,
        rect_geometry.Nodes,
        rect_geometry.Nelem,
        max_vonMises,
    )
