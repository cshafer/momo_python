# Calculate A using GULL borehole temperature and depth vectors

# GULL borehole temperatures and depths

from scipy.interpolate import pchip_interpolate, Akima1DInterpolator
import numpy as np
import matplotlib.pyplot as plt
from stretch import *

def calculate_A(Hmean, Tborehole, Zborehole, interp):

    # If the average thickness of our region of interest is deeper than the depth of the available
    # borehole, stretch the dataset so that the max value is equal to the mean thickness we want
    if (Hmean > Zborehole[-1]):

        # Strecth the dataset so that it extends to the full thickness
        Zborehole = stretch(Zborehole, Zborehole[0], Hmean)


    # Use the mean thickness of our region and discretize it equally:
    z = np.arange(0, round(Hmean)+1, 0.5)

    # Then, using different interpolation methods (to test), we interpolate and extrapolate
    # for depths greater than the max depth provided in the sample data. For depths greater
    # than the provided depth, this method does not work as well.
    
    if interp == 'pchip':
        Tvector = pchip_interpolate(Zborehole, Tborehole, z)
    if interp == 'linear':
        Tvector = np.interp(z, Zborehole, Tborehole)
    if interp == 'akima':
        akima = Akima1DInterpolator(Zborehole, Tborehole)
        akima.extrapolate = True
        Tvector = akima(z)

    # The following table is the A flow factor lookup table from Cuffey & Paterson (Table 3.4). Using our
    # interpolated temperatures, we then interpolate (and extrapolate) for an A vector
        
    Alookup = [2.6*(10**-27), 5.2*(10**-27), 1.0*(10**-26), 2.0*(10**-26), 3.7*(10**-26), 6.8*(10**-26), 1.2*(10**-25), 2.1*(10**-25), 3.5*(10**-25), 9.3*(10**-25), 1.7*(10**-24), 2.4*(10**-24)]
    Tlookup = [-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, -2, 0]

    # Then interpolate from T to A to get an A vector
    A = pchip_interpolate(Tlookup, Alookup, Tvector)

    # Then we calculate the average A value of the A vector:
    Amean = np.mean(A)

    return Tvector, Amean



