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

    pts, _ = create_meshgrid(grid_size, num_dimensions)
    values = target(*pts)
    discretization = create_discretization(num_dimensions)
    refinement_mask = "1" * num_dimensions

    for iteration in range(1, max_iters + 1):
        print(f"\n=== HB {name} ITERATION {iteration} ===")
        approximation, dofs = approximate(
            pts, values, discretization, spline_degree, num_dimensions
        )

        error = np.abs(values - approximation)
        max_error = float(np.max(error))
        mse = float(np.mean(error**2))

        print(f"DoFs: {dofs}")
        print(f"Max Abs Error: {max_error:.3e}")
        print(f"MSE: {mse:.3e}")

        save_results(
            "hb",
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

        marked_boxes = {
            box_index
            for point_index in np.flatnonzero(error > tol)
            if not isinstance(
                box_index := discretization.get_containing_box(pts[:, point_index]),
                tuple,
            )
            and box_index is not None
        }

        if not marked_boxes:
            print("No elements to refine.")
            break

        print(f"Marked elements: {len(marked_boxes)}")

        discretization = apply_refinements(
            discretization, [(box_index, refinement_mask) for box_index in marked_boxes]
        )


if __name__ == "__main__":
    for function_name, function in get_targets(functions).items():
        run(function_name, function)
