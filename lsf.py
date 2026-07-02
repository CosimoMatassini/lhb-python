import numpy as np
import pygismo as gs
import matplotlib.pyplot as plt
from lhb import get_active_bases, gen_knot_vectors, gen_tbases
from bitarray import bitarray
import dyada

n_intervals_tp = 10
spline_degree = 3
tol = 1e-2
max_iters = 6
grid_size = 1000


def f1(x, y):
    b_0_7 = (1 - x) ** 7
    b_1_7 = 7 * x * (1 - x) ** 6
    b_7_7 = x**7
    sin_120x = np.sin(120 * x)
    sin_2pi_x = np.sin(2 * np.pi * x)
    term1 = b_0_7 * (sin_120x * sin_2pi_x)
    term2 = b_1_7 * (2 * sin_120x * sin_2pi_x)
    term3 = b_7_7 * (2 - 2 * (1 + 0.4 * np.sin(60 * y)) * np.abs(np.cos(2 * np.pi * y)))
    return 0.1 * (term1 + term2 + term3)


def f2(x, y):
    sigma = 10**2
    xc = 0.5
    w = 0.1
    term1 = 1 + np.tanh(sigma * (x - xc + w / 2))
    term2 = 1 + np.tanh(sigma * (xc - x + w / 2))
    return 0.25 * term1 * term2


def create_meshgrid(pts_per_dim):
    coords = np.linspace(0, 1, pts_per_dim)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    return np.vstack([xx.ravel(), yy.ravel()])


def apply_refinements(discretization, refinements):
    p = dyada.PlannedAdaptiveRefinement(discretization)
    for level, direction in refinements:
        p.plan_refinement(level, direction)
    return p.apply_refinements(track_mapping="boxes")[0]


def assemble_matrix(pts, active_bases, tbases):
    matrix = np.zeros((pts.shape[1], len(active_bases)))
    for global_idx, (level_tuple, local_idx) in active_bases.items():
        tbasis = tbases[level_tuple]
        vals = tbasis.eval(pts)
        act = tbasis.active(pts)
        mask = act == local_idx
        rows, cols = np.where(mask)
        matrix[cols, global_idx] = vals[rows, cols]
    return matrix


pts = create_meshgrid(grid_size)
f_vals = f1(pts[0], pts[1])


knots = np.concatenate(
    (
        np.zeros(spline_degree),
        np.linspace(0, 1, n_intervals_tp + 1),
        np.ones(spline_degree),
    )
)
knot_vec_tp = gs.nurbs.gsKnotVector(knots, spline_degree)
tbasis_tp = gs.nurbs.gsTensorBSplineBasis2(knot_vec_tp, knot_vec_tp)
n_basis_tp = tbasis_tp.size()
matrix_tp = np.zeros((pts.shape[1], n_basis_tp))
vals_tp = tbasis_tp.eval(pts)
act_tp = tbasis_tp.active(pts)

for i in range(vals_tp.shape[0]):
    matrix_tp[np.arange(pts.shape[1]), act_tp[i]] = vals_tp[i]

coeffs_tp = np.linalg.lstsq(matrix_tp, f_vals, rcond=None)[0]
f_hat_tp = matrix_tp @ coeffs_tp
err_tp = f_vals - f_hat_tp

print(f"TP DoFs: {n_basis_tp}")
print(f"TP Max Abs Error: {np.max(np.abs(err_tp))}")
print(f"TP MSE: {np.mean(err_tp**2)}")
print(f"TP Rel Error: {np.max(np.abs(err_tp) / (np.abs(f_vals) + 1e-15))}")

discretization = dyada.Discretization(
    dyada.MortonOrderLinearization(),
    dyada.RefinementDescriptor.from_binary(2, bitarray("00")),
)

iteration = 0
while True:
    iteration += 1
    print(f"\n=== ADAPTIVE REFINEMENT ITERATION {iteration} ===")

    max_refinement_level = max(discretization.descriptor.get_maximum_level())
    boxes = list(discretization.get_all_boxes_level_indices())

    knot_vectors_lhb = gen_knot_vectors(max_refinement_level, spline_degree)
    tbases_lhb = gen_tbases(2, knot_vectors_lhb)
    active_bases_lhb = get_active_bases(tbases_lhb, boxes)

    matrix_lhb = assemble_matrix(pts, active_bases_lhb, tbases_lhb)
    coeffs_lhb = np.linalg.lstsq(matrix_lhb, f_vals, rcond=None)[0]

    f_hat_lhb = matrix_lhb @ coeffs_lhb
    err_lhb = np.abs(f_vals - f_hat_lhb)

    max_err_lhb = np.max(err_lhb)
    mse_lhb = np.mean(err_lhb**2)
    rel_err_lhb = np.max(err_lhb / (np.abs(f_vals) + 1e-15))

    print(f"LHB DoFs: {len(active_bases_lhb)}")
    print(f"LHB Max Abs Error: {max_err_lhb}")
    print(f"LHB MSE: {mse_lhb}")
    print(f"LHB Rel Error: {rel_err_lhb}")

    marked_indices = np.where(err_lhb > tol)[0]
    marked_points = pts[:, marked_indices]

    print(f"Points above tolerance: {len(marked_indices)}")

    marked_boxes = set()
    for coord in marked_points.T:
        box_idx = discretization.get_containing_box(coord)
        if not isinstance(box_idx, tuple) and box_idx is not None:
            marked_boxes.add(box_idx)

    if len(marked_indices) == 0:
        break

    if iteration >= max_iters:
        print("Max iterations reached.")
        break

    if not marked_boxes:
        print("Marked points only on boundaries. Stopping.")
        break

    print(f"{len(marked_boxes)} box(es) to be refined")
    # qui ci va la logica per trovare le dimensioni da raffinare, per ora raffiniamo tutte le dimensioni
    refinements = [(box_idx, "11") for box_idx in marked_boxes]
    discretization = apply_refinements(discretization, refinements)

x_plot = pts[0].reshape((grid_size, grid_size))
y_plot = pts[1].reshape((grid_size, grid_size))
z_true = f_vals.reshape((grid_size, grid_size))
z_hat_tp = f_hat_tp.reshape((grid_size, grid_size))
z_hat_lhb = f_hat_lhb.reshape((grid_size, grid_size))

fig1 = plt.figure(figsize=(18, 6))
ax1 = fig1.add_subplot(131, projection="3d")
ax1.plot_surface(x_plot, y_plot, z_true, cmap="viridis", edgecolor="none")
ax1.set_title("Funzione Originale")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax2 = fig1.add_subplot(132, projection="3d")
ax2.plot_surface(x_plot, y_plot, z_hat_tp, cmap="plasma", edgecolor="none")
ax2.set_title(f"Stima TP B-Splines (DoFs: {n_basis_tp})")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

ax3 = fig1.add_subplot(133, projection="3d")
ax3.plot_surface(x_plot, y_plot, z_hat_lhb, cmap="inferno", edgecolor="none")
ax3.set_title(f"Stima LHB (DoFs: {len(active_bases_lhb)})\nIterazioni: {iteration}")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")

plt.tight_layout()
plt.show()

abs_err_tp = np.abs(z_true - z_hat_tp)
abs_err_lhb = np.abs(z_true - z_hat_lhb)

fig1, (ax1_err, ax2_err) = plt.subplots(1, 2, figsize=(14, 6))
im1 = ax1_err.pcolormesh(x_plot, y_plot, abs_err_tp, cmap="Reds", shading="auto")
ax1_err.set_title(f"Errore Assoluto TP\n(Max: {np.max(np.abs(err_tp)):.4f})")
ax1_err.set_xlabel("X")
ax1_err.set_ylabel("Y")
fig1.colorbar(im1, ax=ax1_err, label="Magnitudo Errore Assoluto")

im2 = ax2_err.pcolormesh(x_plot, y_plot, abs_err_lhb, cmap="Reds", shading="auto")
ax2_err.set_title(f"Errore Assoluto LHB Finale\n(Max: {max_err_lhb:.4f})")
ax2_err.set_xlabel("X")
ax2_err.set_ylabel("Y")
fig1.colorbar(im2, ax=ax2_err, label="Magnitudo Errore Assoluto")

plt.tight_layout()
plt.show()

dyada.plot_all_boxes_2d(discretization, labels="boxes")
