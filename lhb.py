from itertools import product

import dyada
import numpy as np
import pygismo as gs
from bitarray import bitarray


def get_knot_vectors(max_refinement_level, polynomial_degree):
    return [
        gs.nurbs.gsKnotVector(
            np.concatenate(
                (
                    np.zeros(polynomial_degree),
                    np.linspace(0, 1, 2**level + 1),
                    np.ones(polynomial_degree),
                )
            ),
            polynomial_degree,
        )
        for level in range(max_refinement_level + 1)
    ]


def get_tensor_bases(num_dimensions, knot_vectors, initial_level=0):
    basis_factory = getattr(gs.nurbs, f"gsTensorBSplineBasis{num_dimensions}")
    return {
        level_tuple: basis_factory(*[knot_vectors[lvl] for lvl in level_tuple])
        for level_tuple in product(
            range(initial_level, len(knot_vectors)), repeat=num_dimensions
        )
    }


def compute_ancestor_grid(target_levels, leaf_boxes):
    """Builds a cell grid storing the element-wise minimum refinement levels of covering leaf boxes.

    Maps the hierarchical tree of leaf boxes into a uniform discrete grid matching
    'target_levels'. Each cell in the grid holds a tuple representing the minimal
    refinement levels of the leaf elements intersecting that cell.

    Parameters
    ----------
    target_levels : tuple of int
        Refinement levels of the target tensor basis per dimension.
    leaf_boxes : list
        Leaf boxes retrieved from the dyada discretization tree.

    Returns
    -------
    ndarray of object
        Multi-dimensional grid containing tuples of minimal leaf refinement levels.
    """
    grid_shape = tuple(1 << lvl for lvl in target_levels)
    ancestor_grid = np.empty(grid_shape, dtype=object)
    ancestor_grid.fill((62,) * len(target_levels))

    for box in leaf_boxes:
        grid_ranges = []
        for target_lvl, box_lvl, box_idx in zip(
            target_levels, box.d_level, box.d_index
        ):
            if target_lvl >= box_lvl:
                diff = target_lvl - box_lvl
                start = box_idx << diff
                end = (box_idx + 1) << diff
            else:
                diff = box_lvl - target_lvl
                start = box_idx >> diff
                end = start + 1
            grid_ranges.append(range(start, end))

        box_level_tuple = tuple(box.d_level)
        for coord in product(*grid_ranges):
            ancestor_grid[coord] = tuple(
                np.minimum(ancestor_grid[coord], box_level_tuple)
            )

    return ancestor_grid


def is_basis_active(
    basis_levels, support_min_corner, support_max_corner, ancestor_grid
):
    """Evaluates whether a B-spline basis function satisfies LHB activity criteria.

    A basis function at 'basis_levels' is defined as active if:
    1. No leaf cell overlapping its support has a lower refinement level than
       `basis_levels` in any dimension.
    2. The function's level is greater than or equal to the closest common ancestor
       level across all overlapping leaf cells in its support.

    Parameters
    ----------
    basis_levels : tuple of int
        Multi-index refinement levels of the basis function.
    support_min_corner : tuple of float
        Lower bound coordinates of the basis function's support domain [0, 1]^d.
    support_max_corner : tuple of float
        Upper bound coordinates of the basis function's support domain [0, 1]^d.
    ancestor_grid : ndarray
        Grid of minimal overlapping leaf box levels.

    Returns
    -------
    bool
        True if the basis function is active, False otherwise.
    """
    tensor_shape = ancestor_grid.shape
    support_slice = tuple(
        slice(
            int(np.floor(min_val * dim_size)),
            int(np.ceil(max_val * dim_size)),
        )
        for min_val, max_val, dim_size in zip(
            support_min_corner, support_max_corner, tensor_shape
        )
    )
    sub_tensor = ancestor_grid[support_slice]

    if sub_tensor.size == 0:
        return False

    closest_common_ancestor = list(sub_tensor.flat[0])

    for cell_box_level in sub_tensor.flat:
        if any(func_l > box_l for func_l, box_l in zip(basis_levels, cell_box_level)):
            return False

        closest_common_ancestor = [
            min(cur_min, box_l)
            for cur_min, box_l in zip(closest_common_ancestor, cell_box_level)
        ]

    return all(
        anc_l <= func_l for anc_l, func_l in zip(closest_common_ancestor, basis_levels)
    )


def get_active_functions(tensor_bases, leaf_boxes):
    """Filters and collects all active Local Hierarchical B-spline (LHB) functions.

    Parameters
    ----------
    tensor_bases : dict
        Mapping of refinement level tuples to tensor product B-spline bases.
    leaf_boxes : list
        Leaf boxes extracted from the dyada discretization.

    Returns
    -------
    dict
        Active functions mapping: global_id -> (basis_levels, local_basis_index).
    """
    active_functions = {}
    global_base_id = 0

    for current_basis_levels, tensor_basis in tensor_bases.items():
        ancestor_grid = compute_ancestor_grid(current_basis_levels, leaf_boxes)

        for local_index in range(tensor_basis.size()):
            support_min_corner, support_max_corner = zip(
                *tensor_basis.support(local_index)
            )

            if is_basis_active(
                current_basis_levels,
                support_min_corner,
                support_max_corner,
                ancestor_grid,
            ):
                active_functions[global_base_id] = (
                    current_basis_levels,
                    local_index,
                )
                global_base_id += 1

    return active_functions


if __name__ == "__main__":
    num_dimensions = 2
    polynomial_degree = 2
    initial_mesh_binary = bitarray("111010000010000001010000010000100000100000")
    initial_level = 1
    discretization = dyada.Discretization(
        dyada.MortonOrderLinearization(),
        dyada.RefinementDescriptor.from_binary(num_dimensions, initial_mesh_binary),
    )

    print(discretization)

    max_refinement_level = max(discretization.descriptor.get_maximum_level())

    leaf_boxes = list(discretization.get_all_boxes_level_indices())

    knot_vectors = get_knot_vectors(max_refinement_level, polynomial_degree)

    tensor_bases = get_tensor_bases(num_dimensions, knot_vectors, initial_level)

    active_bases = get_active_functions(tensor_bases, leaf_boxes)

    print(f"There are {len(active_bases)} active functions")
