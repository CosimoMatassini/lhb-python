from bitarray import bitarray
import dyada
import numpy as np
import pygismo as gs
from itertools import product


def gen_knot_vectors(max_refinement, degree):
    knot_vectors = []
    for i in range(max_refinement + 1):
        knot = np.concatenate(
            (
                np.zeros(degree),
                np.linspace(0, 1, 2**i + 1),
                np.ones(degree),
            )
        )
        knot_vectors.append(gs.nurbs.gsKnotVector(knot, degree))
    return knot_vectors


def gen_tbases(num_dimensions, knot_vectors):
    tbases = {}
    tensorProductGenerator = getattr(gs.nurbs, f"gsTensorBSplineBasis{num_dimensions}")
    for indexes in product(range(len(knot_vectors)), repeat=num_dimensions):
        tbases[indexes] = tensorProductGenerator(*[knot_vectors[i] for i in indexes])
    return tbases


def get_active_bases(tensor_bases, leaf_boxes):
    global_base_id = 0
    active_functions = {}

    def build_ancestor_tensor(target_level, boxes):
        grid_shape = tuple(map(pow, [2] * len(target_level), target_level))

        ancestors_tensor = np.empty(grid_shape, dtype=object)
        ancestors_tensor.fill(tuple([62] * len(target_level)))

        for box in boxes:
            grid_ranges = []
            for target_l, box_l, box_idx in zip(target_level, box.d_level, box.d_index):
                if target_l >= box_l:
                    diff = target_l - box_l
                    start = box_idx << diff
                    end = (box_idx + 1) << diff
                else:
                    diff = box_l - target_l
                    start = box_idx >> diff
                    end = start + 1
                grid_ranges.append(range(start, end))

            box_level_tuple = tuple(box.d_level)
            for coord in product(*grid_ranges):
                ancestors_tensor[coord] = tuple(
                    np.minimum(ancestors_tensor[coord], box_level_tuple)
                )

        return ancestors_tensor

    def check_if_basis_is_active(
        basis_function_levels,
        support_min_bounds,
        support_max_bounds,
        ancestors_matrix,
    ):
        tensor_shape = ancestors_matrix.shape
        grid_indices_start = [
            int(np.floor(continuous_min * axis_size))
            for continuous_min, axis_size in zip(support_min_bounds, tensor_shape)
        ]
        grid_indices_end = [
            int(np.ceil(continuous_max * axis_size))
            for continuous_max, axis_size in zip(support_max_bounds, tensor_shape)
        ]

        support_slice = tuple(
            slice(start, end)
            for start, end in zip(grid_indices_start, grid_indices_end)
        )
        sub_tensor_region = ancestors_matrix[support_slice]

        if sub_tensor_region.size == 0:
            return False

        closest_common_ancestor_level = list(sub_tensor_region.flat[0])

        for cell_box_level in sub_tensor_region.flat:
            # first condition
            if any(
                func_l > box_l
                for func_l, box_l in zip(basis_function_levels, cell_box_level)
            ):
                return False

            closest_common_ancestor_level = [
                min(current_min, current_box_l)
                for current_min, current_box_l in zip(
                    closest_common_ancestor_level, cell_box_level
                )
            ]

        # second condition
        if all(
            ancestor_l <= func_l
            for ancestor_l, func_l in zip(
                closest_common_ancestor_level, basis_function_levels
            )
        ):
            return True

        return False

    for current_basis_levels, tensor_basis in tensor_bases.items():
        ancestors_matrix = build_ancestor_tensor(current_basis_levels, leaf_boxes)

        for basis_function_index in range(tensor_basis.size()):
            support_min_bounds, support_max_bounds = zip(
                *tensor_basis.support(basis_function_index)
            )

            if check_if_basis_is_active(
                current_basis_levels,
                support_min_bounds,
                support_max_bounds,
                ancestors_matrix,
            ):
                active_functions[global_base_id] = (
                    current_basis_levels,
                    basis_function_index,
                )
                global_base_id += 1

    return active_functions


def lhb_constructor(num_dimensions, degree, initial_mesh):
    discretization = dyada.Discretization(
        dyada.MortonOrderLinearization(),
        dyada.RefinementDescriptor.from_binary(num_dimensions, initial_mesh),
    )

    max_refinement_level = max(discretization.descriptor.get_maximum_level())
    leaf_boxes = list(discretization.get_all_boxes_level_indices())

    knot_vectors = gen_knot_vectors(max_refinement_level, degree)
    tbases = gen_tbases(num_dimensions, knot_vectors)

    active_bases = get_active_bases(tbases, leaf_boxes)

    return active_bases


if __name__ == "__main__":
    num_dimensions = 2
    initial_mesh = bitarray("100110000100100001100001000000001000010001010000010000")
    degree = 2

    active_bases = lhb_constructor(num_dimensions, degree, initial_mesh)

    print("Active functions:", len(active_bases))
