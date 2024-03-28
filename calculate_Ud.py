import numpy as np
from scipy.integrate import cumulative_trapezoid 

def calculate_Ud(Hmean, Amean, rhoi, g, alpha, n):

    # Discretize depth
    z = np.arange(0, round(Hmean)+1, 1)

    # Perform the cumulative integration using trapezoidal rule
    Ud = np.abs(2 * Amean * (rhoi * g * alpha)**n * cumulative_trapezoid((Hmean - z)**n, z))

    # Output is a cumulative array, total sum is final value
    Ud = Ud[-1] * (365 * 24 * 60 * 60)

    return Ud

