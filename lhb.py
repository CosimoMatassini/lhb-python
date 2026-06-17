import dyada
import numpy as np
import pygismo as gs
from itertools import product

def apply_multiple_refinements(discretization, *refinements: tuple[int, str], check_validity=False):
    p = dyada.PlannedAdaptiveRefinement(discretization)
    for level, direction in refinements:
        p.plan_refinement(level, direction)
    discretization, index_mapping = p.apply_refinements(track_mapping="boxes")
    if check_validity:
        dyada.plot_all_boxes_2d(discretization, labels='boxes')
    return discretization, index_mapping

#implementare una sorta di pruning delle foglie non attive
def get_boxes_in_area(discretization, bb_min, bb_max, strict=False):
    box_trovati = []

    for box_idx, level_index in enumerate(discretization.get_all_boxes_level_indices()):
        
        box_coords = dyada.get_coordinates_from_level_index(level_index)
        b_min = box_coords.lower_bound
        b_max = box_coords.upper_bound

        if strict:
            if np.all(b_min >= bb_min) and np.all(b_max <= bb_max):
                box_trovati.append(box_idx)
        else:
            if np.all(b_min < bb_max) and np.all(b_max > bb_min):
                box_trovati.append(box_idx)

    return box_trovati

def active(func_levels, boxes_in_support):
    closest_common_ancestor = boxes_in_support[0].d_level
    for box in boxes_in_support:
        box_levels = box.d_level
        if np.any(func_levels > box_levels): #first condition
            return False
        closest_common_ancestor = np.minimum(closest_common_ancestor, box_levels)
    if np.all(closest_common_ancestor <= func_levels): #second condition
        return True
    return False

def gen_all_tbases(num_dimensions, knot_vectors):
    tbases = {}
    #può essere ottimizzato utilizzando i generatori, ma va bene così, tanto questa funzione andrà a morire
    for indexes in product(range(len(knot_vectors)), repeat=num_dimensions):
        tbases[indexes] = gs.nurbs.gsTensorBSplineBasis2(*[knot_vectors[i] for i in indexes])
    return tbases

def gen_tbases(boxes_and_refinement_levels, knot_vectors):
    tbases = {}

    for box in boxes_and_refinement_levels:
        box_level = tuple(box.d_level)
        if box_level in tbases:
            continue
        stack = [box_level]
        while stack:
            current_level = stack.pop()
            if current_level in tbases:
                continue
            tbases[current_level] = gs.nurbs.gsTensorBSplineBasis2(*[knot_vectors[i] for i in current_level])
            
            for i in range(len(current_level)):
                if current_level[i] > 0:
                    ancestor = current_level[:i] + (current_level[i] - 1,) + current_level[i+1:]
                    
                    if ancestor not in tbases:
                        stack.append(ancestor)
                        
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
    degree = 2

    descriptor = dyada.RefinementDescriptor(num_dimensions, [1, 0])
    discretization = dyada.Discretization(dyada.MortonOrderLinearization(), descriptor)

    discretization, _ = apply_multiple_refinements(discretization, (0, "01"), (1, "10"))
    discretization, _ = apply_multiple_refinements(discretization, (0, "10"), (3, "01"))
    discretization, _ = apply_multiple_refinements(discretization, (1, "01"), (5, "01"))
    discretization, _ = apply_multiple_refinements(discretization, (2, "10"), (6, "01"), (7, "01"))

    discretization, _ = dyada.apply_single_refinement(discretization, 3, "01")
    discretization, _ = dyada.apply_single_refinement(discretization, 3, "10")
    discretization, _ = dyada.apply_single_refinement(discretization, 4, "01")

    print(discretization)

    #dyada.plot_all_boxes_2d(discretization, labels='boxes')

    max_refinement = max(discretization.descriptor.get_maximum_level())
    knot_vectors = gen_knot_vectors(max_refinement, degree)

    boxes_and_bounds = list(discretization.get_all_boxes_level_indices())

    tbases_all = gen_all_tbases(num_dimensions, knot_vectors)
    tbases_opt = gen_tbases(boxes_and_bounds, knot_vectors)

    tbases = tbases_opt

    global_base_id = 0
    active_functions = {}
    total_basis_functions = 0

    for indexes, tbasis in tbases.items():
        total_basis_functions += tbasis.size()
        
        for i in range(tbasis.size()):
            bb_min, bb_max = zip(*tbasis.support(i))

            boxes_in_support_idx = get_boxes_in_area(discretization, bb_min, bb_max, strict=False)
            boxes_in_support_obj = [boxes_and_bounds[idx] for idx in boxes_in_support_idx]
            
            if active(indexes, boxes_in_support_obj):
                active_functions[global_base_id] = (indexes, i)
                global_base_id += 1

    print("Total basis functions:", total_basis_functions)
    print("Active functions:", len(active_functions))