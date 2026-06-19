from bitarray import bitarray
import dyada
import numpy as np
import pygismo as gs
from itertools import product


def apply_multiple_refinements(
    discretization, *refinements: tuple[int, str], check_validity=False
):
    p = dyada.PlannedAdaptiveRefinement(discretization)
    for level, direction in refinements:
        p.plan_refinement(level, direction)
    discretization, index_mapping = p.apply_refinements(track_mapping="boxes")
    if check_validity:
        dyada.plot_all_boxes_2d(discretization, labels="boxes")
    return discretization, index_mapping


def get_boxes_in_area_naive(discretization, bb_min, bb_max):
    box_trovati = []

    for box_idx, level_index in enumerate(discretization.get_all_boxes_level_indices()):
        box_coords = dyada.get_coordinates_from_level_index(level_index)
        b_min = box_coords.lower_bound
        b_max = box_coords.upper_bound

        if np.all(b_min < bb_max) and np.all(b_max > bb_min):
            box_trovati.append(box_idx)

    return box_trovati


def active(func_levels, leaf_boxes_in_support):
    closest_common_ancestor = leaf_boxes_in_support[0].d_level
    for box in leaf_boxes_in_support:
        box_levels = box.d_level
        if np.any(func_levels > box_levels):  # first condition
            return False
        closest_common_ancestor = np.minimum(closest_common_ancestor, box_levels)
    if np.all(closest_common_ancestor <= func_levels):  # second condition
        return True
    return False


def gen_tbases(num_dimensions, knot_vectors):
    tbases = {}
    # può essere ottimizzato utilizzando i generatori, ma va bene così, tanto questa funzione andrà a morire
    func = getattr(gs.nurbs, f"gsTensorBSplineBasis{num_dimensions}")
    # può essere ottimizzato utilizzando i generatori, ma va bene così, tanto questa funzione andrà a morire
    for indexes in product(range(len(knot_vectors)), repeat=num_dimensions):
        tbases[indexes] = func(*[knot_vectors[i] for i in indexes])
    return tbases


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


if __name__ == "__main__":
    num_dimensions = 2
    initial_mesh = bitarray("100110000100100001100001000000001000010001010000010000")
    degree = 2

    discretization = dyada.Discretization(
        dyada.MortonOrderLinearization(),
        dyada.RefinementDescriptor.from_binary(num_dimensions, initial_mesh),
    )

    print(discretization)

    max_refinement = max(discretization.descriptor.get_maximum_level())
    boxes_and_bounds = list(discretization.get_all_boxes_level_indices())

    knot_vectors = gen_knot_vectors(max_refinement, degree)
    tbases = gen_tbases(num_dimensions, knot_vectors)

    global_base_id = 0
    active_functions = {}

    for indexes, tbasis in tbases.items():
        for i in range(tbasis.size()):
            bb_min, bb_max = zip(*tbasis.support(i))

            boxes_in_support_idx = get_boxes_in_area_naive(
                discretization, bb_min, bb_max
            )
            leaf_boxes_in_support = [
                boxes_and_bounds[idx] for idx in boxes_in_support_idx
            ]

            if active(indexes, leaf_boxes_in_support):
                active_functions[global_base_id] = (indexes, i)
                global_base_id += 1

    print("Active functions:", len(active_functions))
