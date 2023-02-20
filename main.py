import os

import numpy as np

from fem_2d import display_results, fem_2d_code
from NeuralNetwork import NN_data_format


def validation_start():
    """
    Generate table to validate exactiness and convergence of code (FEM example from Onate)
    """
    i_save_id = 0
    val_femresults = np.array([]).reshape(-1, 5)
    for NNodes_long_edge in [21, 41, 61, 81, 101, 121, 151, 181, 211, 241, 271]:
        point_stress, point_displacement, nodes, elements = fem_2d_code.validation_2D(
            i_save_id, NNodes_long_edge
        )
        case_result = np.array(
            [i_save_id, point_stress[0][0], point_displacement[0][3], nodes, elements]
        )
        val_femresults = np.append(val_femresults, case_result.reshape(-1, 5), axis=0)
        i_save_id += 1
    np.savetxt("validation_femresults.csv", val_femresults, delimiter=";")


def validation_results():
    """
    Plots validation results from data generate in validation_star
    """
    validation_femresults = np.genfromtxt("validation_femresults.csv", delimiter=";")
    labels = {"x": "Number of elements", "y": "Displacement in y direction"}
    name = "val_disp_by_nelements"
    display_results.plot_scatter(
        validation_femresults[:, 4],
        validation_femresults[:, 2],
        name,
        labels,
        marker_size=3,
    )
    labels = {"x": "Number of elements", "y": "Stress in x direction"}
    name = "val_xtress_by_nelements"
    display_results.plot_scatter(
        validation_femresults[:, 4],
        validation_femresults[:, 1],
        name,
        labels,
        marker_size=3,
    )


def convergence_start(name, dict_environment):
    """
    Check convergence according to the number of elements
    """
    i_save_id = 0
    val_femresults = np.array([]).reshape(-1, 8)
    for NNodes_long_edge in [21, 41, 61, 81, 101, 121, 151, 181, 211, 241, 271]:
        (
            max_x_stress,
            max_y_stress,
            max_xy_stress,
            point_displacement,
            nodes,
            elements,
            max_vonMises,
        ) = fem_2d_code.convergence_2D(i_save_id, NNodes_long_edge, dict_environment)
        case_result = np.array(
            [
                i_save_id,
                max_x_stress,
                max_y_stress,
                max_xy_stress,
                point_displacement[0][3],
                nodes,
                elements,
                max_vonMises,
            ]
        )
        val_femresults = np.append(val_femresults, case_result.reshape(-1, 8), axis=0)
        i_save_id += 1
    np.savetxt(name + ".csv", val_femresults, delimiter=";")


def convergence_results(name):
    """
    Plots of displacement, y stress and von Mises stress
    by number of elements for the case simulations indicated by name
    """
    validation_femresults = np.genfromtxt(name + ".csv", delimiter=";")
    labels = {"x": "Number of elements", "y": "Displacement in y direction"}
    dict_name = {
        "highestlength_elements": "_len",
        "highestheigth_elements": "_heigth",
        "highestyoung_modulus": "_young",
    }
    display_results.plot_scatter(
        validation_femresults[:, 6],
        validation_femresults[:, 4],
        "convergence_disp_by_nelements" + dict_name[name],
        labels,
        marker_size=3,
    )
    labels = {"x": "Number of elements", "y": "Maximum stress in x direction"}
    display_results.plot_scatter(
        validation_femresults[:, 6],
        validation_femresults[:, 1],
        "convergence_xtress_by_nelements" + dict_name[name],
        labels,
        marker_size=3,
    )
    labels = {"x": "Number of elements", "y": "Maximum stress in y direction"}
    display_results.plot_scatter(
        validation_femresults[:, 6],
        validation_femresults[:, 2],
        "convergence_ytress_by_nelements" + dict_name[name],
        labels,
        marker_size=3,
    )
    labels = {"x": "Number of elements", "y": "Maximum stress in xy direction"}
    display_results.plot_scatter(
        validation_femresults[:, 6],
        validation_femresults[:, 3],
        "convergence_xytress_by_nelements" + dict_name[name],
        labels,
        marker_size=3,
    )
    labels = {"x": "Number of elements", "y": "Maximum von Mises stress"}
    display_results.plot_scatter(
        validation_femresults[:, 6],
        validation_femresults[:, 7],
        "convergence_vonmises_by_nelements" + dict_name[name],
        labels,
        marker_size=3,
    )


if __name__ == "__main__":
    os.chdir(".\\FEM_results_new")
    # Select case
    case = None
    if case == "results generation":
        # Run load to build results from the ones generate in grid_2D()
        fem_2d_code.grid_2D()
        NN_data_format.load_results()
    elif case == "validation":
        # Validation - generate results for validation problem from Onate
        validation_start()
        validation_results()
    elif case == "convergence test":
        # Comparing results with change in elements number
        dict_environment = {}
        dict_environment["rect_length"] = 10
        dict_environment["rect_heigth"] = 0.8
        dict_environment["thick"] = 0.2
        dict_environment["load_magnitude"] = 91896
        dict_environment["material_young"] = 7e10
        dict_environment["material_poisson"] = 0.3
        dict_environment["material_density"] = 2710
        # High length
        name = "highestlength_elements"
        convergence_start(name, dict_environment)
        convergence_results(name)

        # High height
        dict_environment["rect_heigth"] = 1.5
        name = "highestheigth_elements"
        convergence_start(name, dict_environment)
        convergence_results(name)

        # High young modulus
        dict_environment["rect_heigth"] = 0.8
        dict_environment["rect_length"] = 7
        name = "highestyoung_modulus"
        dict_environment["material_young"] = 2e11
        convergence_start(name, dict_environment)
        convergence_results(name)

    # FEM results exploration and Neural networks: Check Jupyter notebook
