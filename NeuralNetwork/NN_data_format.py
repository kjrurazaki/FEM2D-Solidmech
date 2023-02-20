import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from fem_2d import post_fem_2d


def inputs_config(arr_configs_row):
    line_variables = [
        "flag_preLoadedgeometry",
        "meshdir",
        "flag_selfWeight",
        "flag_planeStressorStrain",
        "NNodes_long_edge",
        "rect_length",
        "rect_heigth",
        "thick",
        "load_x",
        "load_y",
        "material_young",
        "material_poisson",
        "material_density",
        "load_direction",
        "load_magnitude",
        "case",
    ]
    line_array = np.array([arr_configs_row[0].get(key) for key in line_variables])
    return line_array


def NN_inputs(only_config=1):
    """
    Loads all inputs of Neural Network
    """
    arr_configs = np.load(os.getcwd() + "\\configs_total.npy", allow_pickle=True)
    # Environment
    arr_input_configs = np.apply_along_axis(
        inputs_config, axis=1, arr=arr_configs.reshape(-1, 1)
    )
    # Conectivity
    arr_coords = np.load(os.getcwd() + "\\coord_total.npy", allow_pickle=True)
    arr_triang = np.load(os.getcwd() + "\\triang_total.npy", allow_pickle=True)

    if only_config == 1:
        arr_input = arr_input_configs
    return arr_input


def NN_output_pivot(arr_output_unpivot, aggregation=["min", "max", "mean"]):
    """
    Return min, max and mean of output
    """
    arr_output = np.array([]).reshape(-1, 4)
    df_output = pd.DataFrame(arr_output_unpivot, columns=["Values", "Case"])
    arr_output = df_output.groupby("Case").agg(aggregation).reset_index()
    arr_output = arr_output.values
    return arr_output


def NN_output_displacement(arr_output_displacement, arr_output_coord):
    """
    Return displacement for rectangle vertices + middle points of long-edge
    """
    df_coord = pd.DataFrame(arr_output_coord, columns=["x", "y", "Case"])
    df_coords = df_coord.groupby("Case").agg(["min", "max", "median"]).reset_index()
    df_coord = df_coord.merge(df_coords, on=["Case"], how="left")
    dict_case_id = {
        "xmin_ymin": 1,
        "xmin_ymax": 2,
        "xmedian_ymax": 3,
        "xmax_ymax": 4,
        "xmax_ymin": 5,
        "xmedian_ymin": 6,
    }
    arr_output_displacement_x_y = np.array([]).reshape(-1, 3)
    for i_x_coord in [("x", "min"), ("x", "max"), ("x", "median")]:
        for i_y_coord in [("y", "min"), ("y", "max")]:
            indices = df_coord[
                (df_coord["x"] == df_coord[i_x_coord])
                & (df_coord["y"] == df_coord[i_y_coord])
            ].index
            indices_x = list(2 * indices)
            indices_y = list(2 * indices + 1)
            arr_displacement_i = np.append(
                arr_output_displacement[indices_x][:, [1, 0]],
                arr_output_displacement[indices_y][:, [0]],
                axis=1,
            )
            if arr_output_displacement_x_y.shape[0] == 0:
                arr_output_displacement_x_y = np.append(
                    arr_output_displacement_x_y, arr_displacement_i, axis=0
                )
            else:
                arr_output_displacement_x_y = np.append(
                    arr_output_displacement_x_y, arr_displacement_i[:, 1:], axis=1
                )
    return arr_output_displacement_x_y


def NN_outputs(
    arr_output_reactions,
    arr_output_stresses,
    arr_output_coord,
    arr_output_displacement,
    nodal_reactions=0,
):
    """
    Formats output for the NNs from FEM simulated data
    """
    if nodal_reactions == 1:
        # Nodal reactions output
        arr_output_reactions = arr_output_reactions[arr_output_reactions[:, 0] != 0]
        arr_output_x_reactions = arr_output_reactions[::2]
        arr_output_y_reactions = arr_output_reactions[1::2]
        arr_output_reactions = np.append(
            NN_output_pivot(arr_output_x_reactions),
            NN_output_pivot(arr_output_y_reactions)[:, 1:],
            axis=1,
        )
    else:
        arr_output_reactions = np.array([])
    # Nodal stresses output
    arr_output_stress_x = NN_output_pivot(
        arr_output_stresses[:, [0, -1]], aggregation=["min", "max"]
    )
    arr_output_stress_y = NN_output_pivot(
        arr_output_stresses[:, [1, -1]], aggregation=["min", "max"]
    )
    arr_output_stress_xy = NN_output_pivot(
        arr_output_stresses[:, [2, -1]], aggregation=["min", "max"]
    )
    arr_output_stresses = np.append(
        np.append(arr_output_stress_x, arr_output_stress_y[:, 1:], axis=1),
        arr_output_stress_xy[:, 1:],
        axis=1,
    )
    # Nodal displacement output
    arr_output_displacement_x = NN_output_pivot(
        arr_output_displacement[::2], aggregation=["max"]
    )
    arr_output_displacement_y = NN_output_pivot(
        arr_output_displacement[1::2], aggregation=["max"]
    )
    arr_output_displacement_x_y = NN_output_displacement(
        arr_output_displacement, arr_output_coord
    )
    arr_output_displacements = np.append(
        np.append(arr_output_displacement_x, arr_output_displacement_y[:, 1:], axis=1),
        arr_output_displacement_x_y[:, 1:],
        axis=1,
    )
    return arr_output_reactions, arr_output_stresses, arr_output_displacements


def load_results():
    """
    Load all results from simulated FEM and call the functions to format in inputs and outputs for NNs
    """
    list_archives = os.listdir()
    list_configs = [i for i in list_archives if i.split("_")[0] == "configs"]

    arr_config = np.array([]).reshape(-1, 16)
    arr_output_reactions = np.array([]).reshape(-1, 7)
    arr_output_stresses = np.array([]).reshape(-1, 7)
    arr_output_vonMises = np.array([]).reshape(-1, 2)
    arr_output_displacement = np.array([]).reshape(-1, 15)
    for i_case in list_configs:
        i_case = "_".join(i_case.split("_")[1:])
        arr_config = np.append(
            arr_config,
            np.load("configs_" + i_case, allow_pickle=True).reshape(-1, 16),
            axis=0,
        )
        arr_coord = np.load("coord_" + i_case, allow_pickle=True)
        arr_nodalreactions = np.load("nodalreactions_" + i_case, allow_pickle=True)
        arr_nodalstresses = np.load("nodalstresses_" + i_case, allow_pickle=True)
        arr_nodaldisplacement = np.load("displacement_" + i_case, allow_pickle=True)
        arr_reactions, arr_stresses, arr_displacement = NN_outputs(
            arr_nodalreactions,
            arr_nodalstresses,
            arr_coord,
            arr_nodaldisplacement,
            nodal_reactions=1,
        )
        arr_output_reactions = np.append(
            arr_output_reactions, arr_reactions.reshape(-1, 7), axis=0
        )
        arr_output_displacement = np.append(
            arr_output_displacement, arr_displacement.reshape(-1, 15), axis=0
        )
        arr_output_stresses = np.append(
            arr_output_stresses, arr_stresses.reshape(-1, 7), axis=0
        )
        arr_vonMises = NN_output_pivot(
            post_fem_2d.compute_vonMises(arr_nodalstresses, load_case=True),
            aggregation=["max"],
        )
        arr_output_vonMises = np.append(
            arr_output_vonMises, arr_vonMises.reshape(-1, 2), axis=0
        )
    np.save("input_configs", arr_config)
    np.save("output_reactions", arr_output_reactions)
    np.save("output_displacement", arr_output_displacement)
    np.save("output_stresses", arr_output_stresses)
    np.save("output_vonMises", arr_output_vonMises)


def construct_df(
    arr_input,
    arr_output_reactions,
    arr_output_stresses,
    arr_output_displacements,
    arr_output_vonMises,
    nodal_reactions=0,
):
    """
    Construct dataframes from input and output array of FEM simulations
    """
    input_dtypes = {
        "flag_preLoadedgeometry": "int",
        "meshdir": "str",
        "flag_selfWeight": "int",
        "flag_planeStressorStrain": "int",
        "NNodes_long_edge": "int",
        "rect_length": "float",
        "rect_heigth": "float",
        "thick": "float",
        "load_x": "float",
        "load_y": "float",
        "material_young": "float",
        "material_poisson": "float",
        "material_density": "float",
        "load_direction": "int",
        "load_magnitude": "float",
        "case": "int",
    }
    reactions_dtypes = {
        "case": "int",
        "min_x_reaction": "float",
        "max_x_reaction": "float",
        "mean_x_reaction": "float",
        "min_y_reaction": "float",
        "max_y_reaction": "float",
        "mean_y_reaction": "float",
    }
    stresses_dtypes = {
        "case": "int",
        "min_x_stress": "float",
        "max_x_stress": "float",
        "min_y_stress": "float",
        "max_y_stress": "float",
        "min_xy_stress": "float",
        "max_xy_stress": "float",
    }
    vonMises_dtypes = {"case": "int", "vonMises": "float"}
    displacement_dtypes = {"case": "int", "max_x": "float", "max_y": "float"}
    point_displacement_dtypes = {
        "case": "int",
        "p1_x": "float",
        "p1_y": "float",
        "p2_x": "float",
        "p2_y": "float",
        "p3_x": "float",
        "p3_y": "float",
        "p4_x": "float",
        "p4_y": "float",
        "p5_x": "float",
        "p5_y": "float",
        "p6_x": "float",
        "p6_y": "float",
    }
    df_input = pd.DataFrame(arr_input, columns=list(input_dtypes.keys())).astype(
        input_dtypes
    )
    df_reactions = pd.DataFrame([])
    if nodal_reactions == 1:
        df_reactions = pd.DataFrame(
            arr_output_reactions, columns=list(reactions_dtypes.keys())
        ).astype(reactions_dtypes)
    df_stresses = pd.DataFrame(
        arr_output_stresses, columns=list(stresses_dtypes.keys())
    ).astype(stresses_dtypes)
    df_vonMises = pd.DataFrame(
        arr_output_vonMises, columns=list(vonMises_dtypes.keys())
    ).astype(vonMises_dtypes)
    df_displacement = pd.DataFrame(
        arr_output_displacements[:, 0:3], columns=list(displacement_dtypes.keys())
    ).astype(displacement_dtypes)
    point_columns = [0] + list(range(3, arr_output_displacements.shape[1]))
    df_point_displacement = pd.DataFrame(
        arr_output_displacements[:, point_columns],
        columns=list(point_displacement_dtypes.keys()),
    ).astype(point_displacement_dtypes)
    return (
        df_input,
        df_reactions,
        df_stresses,
        df_displacement,
        df_point_displacement,
        df_vonMises,
    )


def data_scaling(data):
    """
    Scale data and return transformed data and the scaler
    """
    scaler_data = MinMaxScaler()
    scaler_data.fit(data)
    data = scaler_data.transform(data)
    return data, scaler_data


def scale_dataframe(df, columns_to_scale):
    """
    Scale selected columns and return array of scaled columns, the scaler and original array not scaled
    """
    df_output = df[columns_to_scale]
    arr_output = df_output.values.reshape(-1, df_output.shape[1])
    arr_output_scaled, scaler = data_scaling(arr_output)
    return arr_output_scaled, scaler, arr_output


def data_splitting(df_input, df_output, test_percentage, stratified=None):
    """
    Splits the data in training and test sets considering the pair input, output
    """
    t_size = int(df_input.shape[0] * test_percentage)
    arr_input = df_input.values.reshape(-1, df_input.shape[1])
    arr_output = df_output.values.reshape(-1, df_output.shape[1])
    if stratified == True:
        X_train, X_test, Y_train, Y_test = train_test_split(
            arr_input,
            arr_output,
            test_size=t_size,
            random_state=13,
            stratify=arr_output,
        )
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(
            arr_input, arr_output, test_size=t_size, random_state=13
        )
    return X_train, X_test, Y_train, Y_test


def data_splittingandscaling(
    df,
    features: list,
    target: list,
    test_percentage,
    scale_target=True,
    stratified=None,
):
    """
    Call data splitting (trian and test) and scaling (based on train set)
    """
    X_train, X_test, Y_train, Y_test = data_splitting(
        df[features], df[target], test_percentage, stratified
    )
    X_train, scaler_input = data_scaling(X_train)
    X_test = scaler_input.transform(X_test)
    if scale_target == True:
        Y_train, scaler_output = data_scaling(Y_train)
        Y_test = scaler_output.transform(Y_test)
    else:
        scaler_output = None
    return X_train, X_test, Y_train, Y_test, scaler_input, scaler_output


def split_traindata(X_data, Y_data, stratified=None):
    """
    Cross validation generators
    """
    if stratified == True:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
        print("Stratified CV")
        return skf.split(X_data, Y_data)
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=13)
        print("Not stratified CV")
        return kf.split(X_data)
