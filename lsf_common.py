import inspect
from collections import defaultdict
from pathlib import Path

import dyada
import matplot2tikz
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr

from lhb import get_active_functions, get_knot_vectors, get_tensor_bases
from parameters import initial_refinement_level

matplotlib.use("Agg")


def create_meshgrid(size, num_dimensions):
    coords = np.linspace(0.0, 1.0, size)
    grids = np.meshgrid(*[coords] * num_dimensions, indexing="ij")
    pts = np.vstack([g.ravel() for g in grids])
    return pts, coords


def create_discretization(num_dimensions):
    descriptor = dyada.RefinementDescriptor(
        num_dimensions, [initial_refinement_level] * num_dimensions
    )
    return dyada.Discretization(dyada.MortonOrderLinearization(), descriptor)


def apply_refinements(discretization, refinements):
    planner = dyada.PlannedAdaptiveRefinement(discretization)
    for box_index, direction in refinements:
        planner.plan_refinement(box_index, direction)
    return planner.apply_refinements(track_mapping="boxes")[0]


def assemble_matrix(pts, active_bases, tbases, chunk_size=20_000):
    grouped = defaultdict(list)
    for column, (level, local_index) in active_bases.items():
        grouped[level].append((column, local_index))

    rows_parts, columns_parts, data_parts = [], [], []

    for level, functions_at_level in grouped.items():
        tensor_basis = tbases[level]
        local_to_global = np.full(tensor_basis.size(), -1, dtype=np.int64)
        for column, local_index in functions_at_level:
            local_to_global[local_index] = column

        for start in range(0, pts.shape[1], chunk_size):
            stop = min(start + chunk_size, pts.shape[1])
            chunk = pts[:, start:stop]
            active = tensor_basis.active(chunk)
            values = tensor_basis.eval(chunk)
            mapped_columns = local_to_global[active]
            keep = mapped_columns >= 0
            if not np.any(keep):
                continue

            point_rows = np.broadcast_to(
                np.arange(start, stop, dtype=np.int64), mapped_columns.shape
            )
            rows_parts.append(point_rows[keep])
            columns_parts.append(mapped_columns[keep])
            data_parts.append(values[keep])

    if not data_parts:
        raise RuntimeError("La base gerarchica non contiene funzioni attive.")

    matrix = coo_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(rows_parts), np.concatenate(columns_parts)),
        ),
        shape=(pts.shape[1], len(active_bases)),
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix


def approximate(pts, values, discretization, spline_degree, num_dimensions):
    boxes = list(discretization.get_all_boxes_level_indices())
    max_level = max(discretization.descriptor.get_maximum_level())
    knot_vectors = get_knot_vectors(max_level, spline_degree)
    tbases = get_tensor_bases(num_dimensions, knot_vectors)
    active_bases = get_active_functions(tbases, boxes)
    matrix = assemble_matrix(pts, active_bases, tbases)

    coefficients = lsmr(matrix, values)[0]
    return matrix @ coefficients, len(active_bases)


def _plot_colored_surface(axis, x, y, z, values, cmap, norm):
    axis.plot_surface(
        x,
        y,
        z,
        facecolors=cmap(norm(values)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def _draw_error_cutout_3d(axis, error_volume, coords, cmap, norm):
    center = int(np.argmin(np.abs(coords - 0.5)))
    low = slice(0, center + 1)
    high = slice(center, None)
    all_values = slice(None)

    x, y = np.meshgrid(coords, coords[high], indexing="ij")
    _plot_colored_surface(
        axis,
        x,
        y,
        np.ones_like(x),
        error_volume[all_values, high, -1],
        cmap,
        norm,
    )

    x, y = np.meshgrid(coords, coords[low], indexing="ij")
    _plot_colored_surface(
        axis,
        x,
        y,
        np.full_like(x, coords[center]),
        error_volume[all_values, low, center],
        cmap,
        norm,
    )

    x, z = np.meshgrid(coords, coords[low], indexing="ij")
    _plot_colored_surface(
        axis,
        x,
        np.zeros_like(x),
        z,
        error_volume[all_values, 0, low],
        cmap,
        norm,
    )
    x, z = np.meshgrid(coords, coords, indexing="ij")
    _plot_colored_surface(
        axis,
        x,
        np.ones_like(x),
        z,
        error_volume[all_values, -1, all_values],
        cmap,
        norm,
    )

    x, z = np.meshgrid(coords, coords[high], indexing="ij")
    _plot_colored_surface(
        axis,
        x,
        np.full_like(x, coords[center]),
        z,
        error_volume[all_values, center, high],
        cmap,
        norm,
    )

    y, z = np.meshgrid(coords[high], coords, indexing="ij")
    _plot_colored_surface(
        axis,
        np.zeros_like(y),
        y,
        z,
        error_volume[0, high, all_values],
        cmap,
        norm,
    )
    y, z = np.meshgrid(coords[low], coords[low], indexing="ij")
    _plot_colored_surface(
        axis,
        np.zeros_like(y),
        y,
        z,
        error_volume[0, low, low],
        cmap,
        norm,
    )

    y, z = np.meshgrid(coords[high], coords, indexing="ij")
    _plot_colored_surface(
        axis,
        np.ones_like(y),
        y,
        z,
        error_volume[-1, high, all_values],
        cmap,
        norm,
    )
    y, z = np.meshgrid(coords[low], coords[low], indexing="ij")
    _plot_colored_surface(
        axis,
        np.ones_like(y),
        y,
        z,
        error_volume[-1, low, low],
        cmap,
        norm,
    )


def save_results(
    method,
    name,
    pts,
    approximation,
    error,
    discretization,
    dofs,
    mse,
    max_error,
    iteration,
    grid_size,
    num_dimensions,
):
    directory = Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f"{method}_{name}_iter_{iteration:02d}"

    if num_dimensions == 2:
        x = pts[0].reshape((grid_size, grid_size))
        y = pts[1].reshape((grid_size, grid_size))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, approximation.reshape((grid_size, grid_size)))
        ax.axis("off")
        ax.grid(False)
        matplot2tikz.save(f"{prefix}_approximation.tex")
        fig.savefig(f"{prefix}_approximation.png", dpi=500, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots()
        image = ax.pcolormesh(
            x,
            y,
            error.reshape((grid_size, grid_size)),
            cmap="Reds",
            shading="auto",
        )
        fig.colorbar(image, ax=ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(f"{prefix}_error.png", dpi=500, bbox_inches="tight")
        plt.close(fig)

        dyada.plot_all_boxes_2d(
            discretization, labels="boxes", filename=f"{prefix}_boxes.png"
        )

    elif num_dimensions == 3:
        coords = np.linspace(0.0, 1.0, grid_size)
        error_volume = error.reshape((grid_size,) * num_dimensions)
        cmap = plt.get_cmap("Reds")
        maximum = float(np.max(error))
        norm = Normalize(vmin=0.0, vmax=maximum if maximum > 0.0 else 1.0)

        views = (
            ("error_3d", 25, -45),
            ("error_front", 5, -90),
            ("error_side", 5, 0),
        )
        for suffix, elevation, azimuth in views:
            fig = plt.figure(figsize=(7, 6))
            axis = fig.add_axes((0.02, 0.06, 0.76, 0.84), projection="3d")
            _draw_error_cutout_3d(axis, error_volume, coords, cmap, norm)
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_zlim(0.0, 1.0)
            axis.set_box_aspect((1.0, 1.0, 1.0))
            axis.set_axis_off()
            axis.grid(False)

            colorbar_axis = fig.add_axes((0.84, 0.20, 0.035, 0.60))
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis)
            matplot2tikz.save(f"{prefix}_{suffix}.tex")
            plt.close(fig)

    with open(f"{prefix}_results.txt", "w", encoding="utf-8") as output:
        output.write(f"Function: {name}\n")
        output.write(f"Method: {method.upper()}\n")
        output.write(f"Iteration: {iteration}\n")
        output.write(f"DoFs: {dofs}\n")
        output.write(f"MSE: {mse:.16e}\n")
        output.write(f"Max Abs Error: {max_error:.16e}\n")


def get_targets(functions):
    return {
        name: target
        for name, target in inspect.getmembers(functions, inspect.isfunction)
        if target.__module__ == functions.__name__
    }
