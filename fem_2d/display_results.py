import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import tri as mtri
from matplotlib.ticker import FormatStrFormatter

from fem_2d import Geometry

sns.set_theme()
sns.set_style("darkgrid")
sns.set(
    rc={
        "axes.axisbelow": True,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.labelcolor": "black",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "text.color": "black",
        "xtick.bottom": False,
        "xtick.color": "black",
        "xtick.direction": "out",
        "xtick.top": False,
        "ytick.color": "black",
        "ytick.direction": "out",
        "ytick.left": False,
        "ytick.right": False,
        "legend.facecolor": "white",
        "grid.color": "black",
        "grid.linestyle": "--",
    },
)


def f_stress(geometry: Geometry.Geometry, nodal_stresses, x_coord, y_coord):
    """
    Find the stresses of the specified node
    """
    select_x = np.abs(geometry.coord[:, 1] - x_coord) < 1e-3
    select_y = np.abs(geometry.coord[:, 2] - y_coord) < 1e-3
    return nodal_stresses[select_x & select_y]


def f_displacement(geometry: Geometry.Geometry, displacement, x_coord, y_coord):
    """
    Find displacement of x, y coordinate
    """
    displacement_coord = np.append(geometry.coord[:, 1:], displacement[::2], axis=1)
    displacement_coord = np.append(displacement_coord, displacement[1::2], axis=1)
    select_x = np.abs(geometry.coord[:, 1] - x_coord) < 1e-3
    select_y = np.abs(geometry.coord[:, 2] - y_coord) < 1e-3
    return displacement_coord[select_x & select_y]


def plot_geometry(
    save_id,
    geometry: Geometry.Geometry,
    displacement,
    show_vertices=True,
    show_elements=True,
    deformed=False,
):
    """
    Plot geometry with or without displacement and elements
    """
    fig = plt.figure()
    ax = fig.gca()
    if deformed == False:
        if show_vertices == True:
            ax.plot(geometry.coord[:, 1], geometry.coord[:, 2], ".")
        if show_elements == True:
            ax.triplot(
                geometry.coord[:, 1],
                geometry.coord[:, 2],
                geometry.triang[:, 1:] - 1,
                color="k",
                linewidth=0.2,
            )
        suffix = ""

    if deformed == True:
        deformed_coord = geometry.coord[:, 1:].copy()
        deformed_coord[:, 0] = deformed_coord[:, 0] + displacement[::2, 0]
        deformed_coord[:, 1] = deformed_coord[:, 1] + displacement[1::2, 0]
        if show_vertices == True:
            ax.plot(deformed_coord[:, 0], deformed_coord[:, 1], ".")
        if show_elements == True:
            ax.triplot(
                deformed_coord[:, 0],
                deformed_coord[:, 1],
                geometry.triang[:, 1:] - 1,
                color="k",
                linewidth=0.2,
            )
        suffix = "_deformed"
    ax.set_xlim([0, geometry.config.rect_length + 1])
    ax.set_ylim(
        [
            -geometry.config.rect_heigth / 5,
            geometry.config.rect_heigth + geometry.config.rect_heigth / 5,
        ]
    )
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_aspect(1)
    print(os.getcwd())

    plt.savefig(
        f"figures/fig_geo_{save_id}{suffix}.pdf", bbox_inches="tight", format="pdf"
    )


def plot_field(
    save_id,
    geometry: Geometry.Geometry,
    nodal_values,
    stress_direction="x",
    variable_name="stress",
    nodal=True,
):
    """
    Plot color mapping in 2D geometry
    """
    if variable_name == "stress":
        if nodal_values.shape[1] == 4:
            dict_stress_direction = {"x": 0, "y": 1, "xy": 2, "von Mises": 3}
        if nodal_values.shape[1] == 5:
            dict_stress_direction = {"x": 0, "y": 1, "z": 2, "xy": 3}
        colors = nodal_values[:, dict_stress_direction[stress_direction]]
    else:
        colors = nodal_values.copy().flatten()
        stress_direction = variable_name
    triangulation = mtri.Triangulation(
        geometry.coord[:, 1], geometry.coord[:, 2], geometry.triang[:, 1:] - 1
    )

    fig = plt.figure()
    ax = fig.gca()
    if nodal == True:
        c = ax.tricontourf(triangulation, colors)
        suffix = "_nodal"
    else:
        c = ax.tripcolor(triangulation, colors, shading="gouraud", antialiased=True)
        suffix = "_triang"
    cbar = plt.colorbar(c, shrink=0.4, format="%.2e")
    cbar.ax.tick_params(labelsize=6)
    ax.set_xlim([0, geometry.config.rect_length + 1])
    ax.set_ylim(
        [
            -geometry.config.rect_heigth / 5,
            geometry.config.rect_heigth + geometry.config.rect_heigth / 5,
        ]
    )
    ax.set_aspect(1)
    plt.savefig(
        f"figures/fig_stresses_{save_id}_{stress_direction}{suffix}.pdf",
        bbox_inches="tight",
        format="pdf",
    )


def plot_scatter(X, Y, name, axis_labels, marker_size, label=None, ax=None):
    """
    Just a simple scatter plot
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.gca()
    ax.scatter(X, Y, s=marker_size, label=label)

    if axis_labels != None:
        ax.set_xlabel(axis_labels["x"])
        ax.set_ylabel(axis_labels["y"])
    if name != None:
        plt.savefig(f"figures/{name}.pdf", bbox_inches="tight", format="pdf")
        plt.show()


def plot_column_grouped(
    X,
    Y,
    name,
    group_columns,
    group_labels,
    axis_labels,
    y_scale="log",
    x_scale="linear",
):
    """
    Just a line plot with lines grouped by different columns of array
    """
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(X, Y[:, group_columns[0]], label=group_labels[0])
    ax.plot(X, Y[:, group_columns[1]], label=group_labels[1])
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_xlabel(axis_labels["x"])
    ax.set_ylabel(axis_labels["y"])
    plt.savefig(f"figures/{name}.pdf", bbox_inches="tight", format="pdf")
    plt.show()


def plot_line(X, Y, label, axis_labels, y_scale="linear", x_scale="linear", ax=None):
    """
    Just a line plot with lines grouped by different lines
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.gca()
    ax.plot(X, Y, label=label)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_xlabel(axis_labels["x"])
    ax.set_ylabel(axis_labels["y"])


def plot_scatter_group(X, Y, label, axis_labels, marker_size, marker_color, ax=None):
    """
    Just a simple scatter plot with color mapping option
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.gca()
    scatter = ax.scatter(X, Y, s=marker_size, c=marker_color, label=label)
    ax.set_xlabel(axis_labels["x"])
    ax.set_ylabel(axis_labels["y"])
    return scatter


def plot_fitted_line(
    X, Y, degree, line_color, label, subscript, position, full=True, ax=None
):
    """
    Plot a fitted line with order = degree in the X, Y data provided and write the equation in the position
    Plot and returns coefficients and RSS (last one if full = True)
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.gca()
    # Best fit
    (
        coeff,
        RSS,
        *_,
    ) = np.polyfit(X, Y, degree, full=True)
    a, b = coeff[0], coeff[1]
    # Add line
    ax.plot(X, a * X + b, color=line_color, linestyle="--", linewidth=1, label=label)
    if X.shape[0] == 2:  # Exact line (two points)
        RSS = [0]
    # Write equation
    text = (
        "$y_{"
        + subscript
        + "}$ = "
        + "{:.3e}".format(b)
        + " + {:.3e}".format(a)
        + "x"
        + ", RSS = {:.3e}".format(RSS[0])
    )
    ax.text(position[0], position[1], text, size=10)


def plot_mean_geometry(
    df,
    df_fem,
    var_evaluation,
    material,
    geo_measure,
    line_color,
    marker_color,
    var_flag,
):
    """
    Plot mean values for all forces by the different geometries - X should be length, thickness or heigth
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    dict_y_axis = {
        "max_y": "Displacement in y direction",
        "max_x": "Displacement in x direction",
        "max_y_stress": "Maximum stress in y direction",
        "max_x_stress": "Maximum stress in x direction",
        "max_xy_stress": "Maximum stress in xy direction",
        "vonMises": "Maximum von Mises",
    }
    dict_save_name = {
        "max_y": "displacement",
        "max_x": "displacement_x",
        "max_y_stress": "ystress",
        "max_x_stress": "xstress",
        "max_xy_stress": "xystress",
        "vonMises": "vonmises",
    }
    dict_text = {
        "rect_length": ["Length", "L", "lower right"],
        "rect_heigth": ["Heigth", "H", "lower left"],
        "thick": ["thickness", "T", "lower left"],
    }
    axis_labels = {
        "x": f"{dict_text[geo_measure][0]} (m)",
        "y": dict_y_axis[var_evaluation],
    }
    X = df.loc[df["Material"] == material, geo_measure]
    Y = df.loc[df["Material"] == material, [var_evaluation, var_evaluation + "_var"]]
    label = f"Mean"
    scatter = plot_scatter_group(
        X,
        Y[var_evaluation],
        label=label,
        axis_labels=axis_labels,
        marker_size=4,
        marker_color=marker_color,
        ax=ax1,
    )
    if var_flag != 0:
        ax1.fill_between(
            X,
            Y[var_evaluation] - Y[var_evaluation + "_var"],
            Y[var_evaluation] + Y[var_evaluation + "_var"],
            alpha=0.2,
        )
    label, subscript = f"Best fitted line", f"{material}, {dict_text[geo_measure][1]}"
    loc_equation_x = X.min() + (X.max() - X.min()) / 20
    loc_equation_y = Y[var_evaluation].max()
    plot_fitted_line(
        X,
        Y[var_evaluation],
        1,
        line_color,
        label,
        subscript,
        [loc_equation_x, loc_equation_y],
        ax=ax1,
    )

    # Specific geometry
    if geo_measure == "rect_length":
        df_plot = df_fem[
            (df_fem["load_magnitude"] == df_fem["load_magnitude"].unique()[10])
            & (df_fem["rect_heigth"] == 0.8)
            & (df_fem["thick"] == 0.2)
        ]
    elif geo_measure == "rect_heigth":
        df_plot = df_fem[
            (df_fem["rect_length"] == 7)
            & (df_fem["load_magnitude"] == df_fem["load_magnitude"].unique()[10])
            & (df_fem["thick"] == 0.2)
        ]
    elif geo_measure == "thick":
        df_plot = df_fem[
            (df_fem["rect_length"] == 7)
            & (df_fem["rect_heigth"] == 0.8)
            & (df_fem["load_magnitude"] == df_fem["load_magnitude"].unique()[10])
        ]

    plot_scatter(
        df_plot[geo_measure],
        df_plot[var_evaluation],
        name=None,
        axis_labels=axis_labels,
        marker_size=2,
        label="All other measures fixed",
        ax=ax2,
    )

    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    ax1.legend(loc=dict_text[geo_measure][2])
    ax2.legend(loc=dict_text[geo_measure][2])
    plt.savefig(
        f"figures/{material}_meanvar{dict_save_name[var_evaluation]}_{geo_measure}.pdf",
        bbox_inches="tight",
        format="pdf",
    )


def plot_mean_force(
    df, df_fem, var_evaluation, material, line_color, marker_color, var_flag
):
    """
    Plot mean values for all geometries for differente forces
    """
    dict_y_axis = {
        "max_y": "Displacement in y direction",
        "max_x": "Displacement in x direction",
        "max_y_stress": "Maximum stress in y direction",
        "max_x_stress": "Maximum stress in x direction",
        "max_xy_stress": "Maximum stress in xy direction",
        "vonMises": "Maximum von Mises",
    }
    dict_save_name = {
        "max_y": "displacement",
        "max_x": "displacement_x",
        "max_y_stress": "ystress",
        "max_x_stress": "xstress",
        "max_xy_stress": "xystress",
        "vonMises": "vonmises",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axis_labels = {"x": "Force in y direction", "y": dict_y_axis[var_evaluation]}
    X = df.loc[df["Material"] == material, "load_magnitude"]
    Y = df.loc[df["Material"] == material, [var_evaluation, var_evaluation + "_var"]]
    label = f"Mean"
    plot_scatter_group(
        X,
        Y[var_evaluation],
        label=label,
        axis_labels=axis_labels,
        marker_size=4,
        marker_color=marker_color,
        ax=ax1,
    )
    if var_flag != 0:
        ax1.fill_between(
            X,
            Y[var_evaluation] - Y[var_evaluation + "_var"],
            Y[var_evaluation] + Y[var_evaluation + "_var"],
            alpha=0.2,
            label="Variance",
        )
    label, subscript = f"Best fitted line", f"{material}, F"
    loc_equation_x = X.min() + (X.max() - X.min()) / 20
    loc_equation_y = Y[var_evaluation].max()
    plot_fitted_line(
        X,
        Y[var_evaluation],
        1,
        line_color,
        label,
        subscript,
        [loc_equation_x, loc_equation_y],
        ax=ax1,
    )

    # Specific geometry
    df_plot = df_fem[
        (df_fem["rect_length"] == 7)
        & (df_fem["rect_heigth"] == 0.8)
        & (df_fem["thick"] == 0.2)
    ]
    plot_scatter(
        df_plot["load_magnitude"],
        df_plot[var_evaluation],
        name=None,
        axis_labels=axis_labels,
        marker_size=2,
        label="All other measures fixed",
        ax=ax2,
    )
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")
    fig.autofmt_xdate()
    plt.savefig(
        f"figures/{material}_meanvar{dict_save_name[var_evaluation]}_force.pdf",
        bbox_inches="tight",
        format="pdf",
    )


def plot_mean_youngmodulus(
    df, df_fem, var_evaluation, line_color, marker_color, var_flag
):
    """
    Plot mean values for young values
    """
    dict_y_axis = {
        "max_y": "Displacement in y direction",
        "max_y_stress": "Maximum stress in y direction",
        "max_x_stress": "Maximum stress in x direction",
        "max_xy_stress": "Maximum stress in xy direction",
    }
    dict_save_name = {
        "max_y": "displacement",
        "max_y_stress": "ystress",
        "max_x_stress": "xstress",
        "max_xy_stress": "xystress",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axis_labels = {"x": "Young Modulus", "y": dict_y_axis[var_evaluation]}
    X = df["material_young"]
    Y = df[[var_evaluation, var_evaluation + "_var"]]
    label = f"Mean"
    plot_scatter_group(
        X,
        Y[var_evaluation],
        label=label,
        axis_labels=axis_labels,
        marker_size=4,
        marker_color=marker_color,
        ax=ax1,
    )
    if var_flag != 0:
        ax1.fill_between(
            X,
            Y[var_evaluation] - Y[var_evaluation + "_var"],
            Y[var_evaluation] + Y[var_evaluation + "_var"],
            alpha=0.2,
            label="Variance",
        )
    label, subscript = f"Best fitted line", f"E"
    loc_equation_x = X.min() + (X.max() - X.min()) / 4
    loc_equation_y = Y[var_evaluation].max() * 0.997
    plot_fitted_line(
        X,
        Y[var_evaluation],
        1,
        line_color,
        label,
        subscript,
        [loc_equation_x, loc_equation_y],
        ax=ax1,
    )

    # Specific geometry
    df_plot = df_fem[
        (df_fem["rect_length"] == 7)
        & (df_fem["rect_heigth"] == 0.8)
        & (df_fem["thick"] == 0.2)
        & (df_fem["load_magnitude"] == df_fem["load_magnitude"].unique()[10])
    ]
    plot_scatter(
        df_plot["material_young"],
        df_plot[var_evaluation],
        name=None,
        axis_labels=axis_labels,
        marker_size=2,
        label="All other measures fixed",
        ax=ax2,
    )

    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    if (var_evaluation == "max_y_stress") | (var_evaluation == "max_xy_stress"):
        ax1.legend(loc="lower left")
        ax2.legend(loc="lower left")
    else:
        ax1.legend(loc="upper left")
        ax2.legend(loc="lower left")
    plt.savefig(
        f"figures/meanvar{dict_save_name[var_evaluation]}_youngmodulus.pdf",
        bbox_inches="tight",
        format="pdf",
    )


def mean_force_results(df_fem, var_evalutation, load_direction, var_flag):
    """
    Evaluate results based on the mean forces
    """
    df_plot = df_fem[df_fem["load_direction"] == load_direction]
    df_plot = (
        df_plot.groupby(["load_magnitude", "material_young"])[var_evalutation]
        .agg(["mean", "var"])
        .reset_index()
    )
    df_plot.loc[df_plot["material_young"] == 7e10, "Material"] = "Aluminum"
    df_plot.loc[df_plot["material_young"] == 2e11, "Material"] = "Steel"
    df_plot.rename(
        columns={"mean": var_evalutation, "var": var_evalutation + "_var"}, inplace=True
    )
    plot_mean_force(
        df_plot,
        df_fem[df_fem["material_young"] == 7e10],
        var_evalutation,
        material="Aluminum",
        line_color="red",
        marker_color="b",
        var_flag=var_flag,
    )
    plot_mean_force(
        df_plot,
        df_fem[df_fem["material_young"] == 2e11],
        var_evalutation,
        material="Steel",
        line_color="steelblue",
        marker_color="red",
        var_flag=var_flag,
    )


def mean_geometry_results(df_fem, var_evalutation, load_direction, var_flag):
    """
    Evaluate results based on the mean geometry properties
    """
    for geo_measure in ["rect_length", "rect_heigth", "thick"]:
        df_plot = df_fem[df_fem["load_direction"] == load_direction]
        df_plot = (
            df_plot.groupby([geo_measure, "material_young"])[var_evalutation]
            .agg(["mean", "var"])
            .reset_index()
        )
        df_plot["var"] = np.sqrt(df_plot["var"])
        df_plot.loc[df_plot["material_young"] == 7e10, "Material"] = "Aluminum"
        df_plot.loc[df_plot["material_young"] == 2e11, "Material"] = "Steel"
        df_plot.rename(
            columns={"mean": var_evalutation, "var": var_evalutation + "_var"},
            inplace=True,
        )
        plot_mean_geometry(
            df_plot,
            df_fem[df_fem["material_young"] == 7e10],
            var_evalutation,
            material="Aluminum",
            geo_measure=geo_measure,
            line_color="red",
            marker_color="b",
            var_flag=var_flag,
        )
        plot_mean_geometry(
            df_plot,
            df_fem[df_fem["material_young"] == 2e11],
            var_evalutation,
            material="Steel",
            geo_measure=geo_measure,
            line_color="steelblue",
            marker_color="red",
            var_flag=var_flag,
        )


def mean_young_results(df_fem, var_evalutation, load_direction, var_flag):
    """
    Evaluate results based on the mean young moudlus
    """
    df_plot = df_fem[df_fem["load_direction"] == load_direction]
    df_plot = (
        df_plot.groupby(["material_young"])[var_evalutation]
        .agg(["mean", "var"])
        .reset_index()
    )
    df_plot["var"] = np.sqrt(df_plot["var"])
    df_plot.rename(
        columns={"mean": var_evalutation, "var": var_evalutation + "_var"}, inplace=True
    )
    plot_mean_youngmodulus(
        df_plot,
        df_fem,
        var_evalutation,
        line_color="red",
        marker_color="b",
        var_flag=var_flag,
    )


def plot_boxplot(arr_data, y_label, label_boxplots, name_to_save=None):
    fig = plt.figure()
    ax = fig.gca()
    ax.boxplot(arr_data, labels=label_boxplots)
    ax.set_ylabel(y_label)
    if name_to_save != None:
        plt.savefig(f"figures/{name_to_save}.pdf", bbox_inches="tight", format="pdf")


def plot_hist_boxplot(arr_data, x_label, name_to_save=None):
    """
    Plot histogram distribution with boxplot overhead
    Extracted from https://www.python-graph-gallery.com/24-histogram-with-a-boxplot-on-top-seaborn
    """
    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.15, 0.85)}
    )
    ax = f.gca()
    # Assigning a graph to each ax
    sns.boxplot(data=arr_data, ax=ax_box, orient="h")
    sns.histplot(data=arr_data, ax=ax_hist)
    ax_hist.set(xlabel=x_label)
    ax_box.set(xlabel="")
    ax_hist.get_legend().remove()
    if name_to_save != None:
        plt.savefig(f"figures/{name_to_save}.pdf", bbox_inches="tight", format="pdf")
