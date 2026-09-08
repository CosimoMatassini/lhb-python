import inspect

import numpy as np

import functions
from lsf_common import (
    apply_refinements,
    approximate,
    create_discretization,
    create_meshgrid,
    get_targets,
    save_results,
)
from parameters import *


def run(name, target):
    num_dimensions = len(inspect.signature(target).parameters)

    pts, coords = create_meshgrid(grid_size, num_dimensions)
    values = target(*pts)

    grid_shape = (grid_size,) * num_dimensions
    coords_list = [coords] * num_dimensions
    grads = np.gradient(values.reshape(grid_shape), *coords_list, edge_order=2)

    gradients = np.vstack([np.abs(g).ravel() for g in grads])
    discretization = create_discretization(num_dimensions)

    for iteration in range(1, max_iters + 1):
        print(f"\n=== LHB {name} ITERATION {iteration} ===")
        approximation, dofs = approximate(
            pts, values, discretization, spline_degree, num_dimensions
        )

        error = np.abs(values - approximation)
        max_error = float(np.max(error))
        mse = float(np.mean(error**2))

        print(f"DoFs: {dofs}")
        print(f"MSE: {mse:.3e}")
        print(f"Max Abs Error: {max_error:.3e}")

        save_results(
            "lhb",
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
        )

        if iteration == max_iters:
            print("Max iterations reached.")
            break

        marked_boxes = set()
        points_by_box = {}
        for point_index in range(pts.shape[1]):
            box_index = discretization.get_containing_box(pts[:, point_index])
            if isinstance(box_index, tuple) or box_index is None:
                continue
            points_by_box.setdefault(box_index, []).append(point_index)
            if error[point_index] > tol:
                marked_boxes.add(box_index)

        if not marked_boxes:
            print("No elements to refine.")
            break

        print(f"Marked elements: {len(marked_boxes)}")

        refinements = []
        fallback_direction = "1" * num_dimensions

        for box_index in marked_boxes:
            average = np.mean(gradients[:, points_by_box[box_index]], axis=1)
            maximum = float(np.max(average))
            direction = (
                fallback_direction
                if maximum == 0.0
                else "".join(
                    "1" if value >= eps_dir * maximum else "0" for value in average
                )
            )
            refinements.append((box_index, direction))

        discretization = apply_refinements(discretization, refinements)


if __name__ == "__main__":
    for function_name, function in get_targets(functions).items():
        run(function_name, function)
