import numpy as np


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


def f3(x, y, z):
    SIGMA = 100.0
    RIDGE_WIDTH = 0.1
    RIDGE_CENTER = 0.5
    RIDGE_LENGTH = 0.2

    def tau(gamma, sigma=SIGMA):
        return np.tanh(sigma * gamma)

    def phi_x(x, y, yc=RIDGE_CENTER, width=RIDGE_WIDTH):
        """Ridge parallela all'asse x, equazione (11)."""
        return (
            0.25 * (1.0 + tau(y - yc + width / 2.0)) * (1.0 + tau(yc - y + width / 2.0))
        )

    def phi_y(x, y, xc=RIDGE_CENTER, width=RIDGE_WIDTH):
        """Ridge parallela all'asse y, equazione (12)."""
        return (
            0.25 * (1.0 + tau(x - xc + width / 2.0)) * (1.0 + tau(xc - x + width / 2.0))
        )

    def psi_x(x, y, length=RIDGE_LENGTH):
        """Ridge in direzione x troncata alla lunghezza ``length``."""
        return 0.5 * phi_x(x, y) * (1.0 - tau(x - length))

    def psi_y(x, y, length=RIDGE_LENGTH):
        """Ridge in direzione y troncata alla lunghezza ``length``."""
        return 0.5 * phi_y(x, y) * (1.0 - tau(y - length))

    value = psi_x(x, y) + psi_x(1.0 - x, y) + psi_y(x, y) + psi_y(x, 1.0 - y)
    return value + np.zeros_like(z, dtype=float)
