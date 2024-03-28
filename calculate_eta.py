import numpy as np
import scipy

def calculate_eta(Hmean, Amean, rhoi, g, alpha, n, Ud):

    # I want to test two different ways to cacluate eta to see if they match

    # The first way is how Jeremy handles it in his MATLAB code

    eta_jeremy = 1/(Amean * (rhoi * g * Hmean * alpha)**(n-1))

    # The second way is how Gudmundsson defines deformational velocity in his 2008 paper.
    # I rearrange Ud and eta to solve for eta

    eta_gudmundsson = (1/(2 * Ud)) * (rhoi * g * np.sin(alpha) * Hmean**2)

    return [eta_jeremy, eta_gudmundsson]

