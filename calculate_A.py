# Calculate A using GULL borehole temperature and depth vectors

# GULL borehole temperatures and depths

from scipy.interpolate import pchip_interpolate, Akima1DInterpolator
import numpy as np
import matplotlib.pyplot as plt

def calculate_A(Hmean, Tborehole, Zborehole, interp):
    
    # Our goal is to come up with an A value that fits our region of study that we
    # can use to calculate deformational velocity, Ud. To do that, we need a temperature
    # profile from the region - this comes from temperature measurements made at depth 
    # within boreholes. Depending on the thickness of our region (as long as it is relatively
    # close or within the range of the provided borehole depth), we can interpolate the
    # temperature values and apply it to our thickness range. Then, we use the A-T lookup
    # table (Table 3.4 Cuffey & Paterson) and match our interpolated temperatures to 
    # a range of A values. Then we take the mean of all of the A values to get a final
    # mean A flow factor for our region. 

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

    return [Tvector, Amean]



