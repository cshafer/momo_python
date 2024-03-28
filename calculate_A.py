# Calculate A using GULL borehole temperature and depth vectors

# GULL borehole temperatures and depths

from scipy.interpolate import pchip_interpolate, Akima1DInterpolator
import numpy as np
import matplotlib.pyplot as plt

def calculate_A(Hmean, Tborehole, Zborehole, Tlookup, Alookup, interp):
    
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

    z = np.arange(0, round(Hmean), 1)

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
        
    #Alookup = [2.6*(10**-27), 5.2*(10**-27), 1.0*(10**-26), 2.0*(10**-26), 3.7*(10**-26), 6.8*(10**-26), 1.2*(10**-25), 2.1*(10**-25), 3.5*(10**-25), 9.3*(10**-25), 1.7*(10**-24), 2.4*(10**-24)]
    #Tlookup = [-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, -2, 0]

    # Then interpolate from T to A to get an A vector
    A = pchip_interpolate(Tlookup, Alookup, Tvector)

    # Then we calculate the average A value of the A vector:
    Amean = np.mean(A)

    return [Tvector, Amean]

# The following arrays are the temperature and depth datapoints taken from the GULL
# borehole (Ryser et al., 2014, Luthi et al., 2015)
tempGULL = [-0.6500, -7.7500, -11.2700, -11.9500, -14.1300, -13.5700, -12.7400, -11.6900, 
            -10.1100, -8.4900, -6.5500, -4.7400, -2.7300, -1.5200, -0.8300, -0.6000,
            -0.5600 , -0.4900, -0.5400, -0.4200, -0.4700, -0.3900, -0.5000]

depthGULL = [4, 255, 307, 355, 407, 455, 497, 515, 537, 555, 577, 595, 622, 645, 667, 
             676, 687, 690, 697, 699, 702, 705, 707]

# Cuffey & Paterson (Table 3.4)
# The following table is the A flow factor lookup table from Cuffey & Paterson (Table 3.4). Using our
# interpolated temperatures, we then interpolate from T to A to get an A vector
Alookup = [2.6*(10**-27), 5.2*(10**-27), 1.0*(10**-26), 2.0*(10**-26), 3.7*(10**-26), 6.8*(10**-26), 1.2*(10**-25), 2.1*(10**-25), 3.5*(10**-25), 9.3*(10**-25), 1.7*(10**-24), 2.4*(10**-24)]
Tlookup = [-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, -2, 0]

Hmean = 700
z = np.arange(0, round(Hmean), 1)

[Tvector_pchip, Amean_pchip] = calculate_A(Hmean, tempGULL, depthGULL, Tlookup, Alookup, interp = 'pchip')
[Tvector_linear, Amean_linear] = calculate_A(Hmean, tempGULL, depthGULL, Tlookup, Alookup, interp = 'linear')
[Tvector_akima, Amean_akima] = calculate_A(Hmean, tempGULL, depthGULL, Tlookup, Alookup, interp = 'akima' )

# Plot the figures

fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (4,7))

ax1.plot(tempGULL, depthGULL, 'bo', label = 'GULL sample data')
ax1.plot(Tvector_linear, z, label = 'linear, A = ' + str(Amean_linear))
ax1.plot(Tvector_pchip, z, label = 'pchip, A = ' + str(Amean_pchip))
ax1.plot(Tvector_akima, z, label = 'akima, A = ' + str(Amean_akima))

ax1.set_xlabel('Temperature (C)')
ax1.set_ylabel('Depth (m)')
ax1.invert_yaxis()
ax1.set_title('Temperature interpolation, Hmean = ' + str(Hmean))
ax1.axvline(x = 0, color = 'black', linewidth = 0.5)
ax1.legend(loc = 'lower center', bbox_to_anchor=(.5, -0.27))

#ax2.plot(Tlookup, Alookup, 'ko', label = 'A-T lookup table')
#ax2.plot(Tvector_linear, A_linear, label = 'linear/pchip')
#ax2.plot(Tvector_pchip, A_pchip,  label = 'pchip/pchip')
#ax2.plot(Tvector_pchip, A_pchip2, label = 'pchip/linear')
#ax2.plot(Tvector_akima, A_akima, label = 'akima/pchip')
#ax2.plot(Tvector_akima,A_akima2,  label = 'akima/linear')
#ax2.plot(Tvector_spline,A_spline,  label = 'spline/pchip')
#ax2.plot(Tvector_spline, A_spline2, label = 'spline/linear')
#ax2.axvline(x = 0, color = 'black', linewidth = 0.5)
#ax2.legend(loc = 'lower left')
#ax2.set_xlabel('Temperature (C)')
#ax2.set_ylabel('A ($s^{-1}$ $Pa^{-3}$)')
#ax2.set_title('Flow factor A interpolation, Hmean = ' + str(Hmean))

plt.show()


