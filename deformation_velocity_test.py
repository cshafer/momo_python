# Deformation velocity test

from calculate_A import *
from calculate_Ud import *
from calculate_eta import *
from get_GPS_data import *

# Our goal is to come up with an A value that fits our region of study that we
# can use to calculate deformational velocity, Ud.

# To do that, we need a temperature profile from the region - this comes from 
# temperature measurements made at depth within boreholes. Depending on the thickness
# of our region (less than the max depth recorded in the borehole), we can interpolate the
# temperature values and apply it to our thickness range. Then, we use the A-T lookup
# table (Table 3.4 Cuffey & Paterson) and match our interpolated temperatures to 
# a range of A values. Then we take the mean of all of the A values to get a final
# mean A flow factor for our region.

# With our calculated mean A value, we can calculate deformation velocity, Ud. Then,
# using either A or Ud, we can calculate an eta value. 

Hmean = 707
rhoi = 917
g = 9.80665
alpha = 0.02268    # At GULL borehole, surface slope measured to be 1.3 +- 0.2 degrees (Ryser et al., 2014)
n = 3
secperyear = 365*24*60*60

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

[Tvector_pchip, Amean_pchip] = calculate_A(Hmean, tempGULL, depthGULL, 'pchip')
[Tvector_linear, Amean_linear] = calculate_A(Hmean, tempGULL, depthGULL, 'linear')
[Tvector_akima, Amean_akima] = calculate_A(Hmean, tempGULL, depthGULL, 'akima' )

print('Possible flow factors (A):')
print('Linear mean A: ' + str(Amean_linear) + ' s^-1 Pa^-3')
print('PCHIP mean A:  ' + str(Amean_pchip) + ' s^-1 Pa^-3')
print('Akima mean A:  ' + str(Amean_akima) + ' s^-1 Pa^-3')

Ud_linear = calculate_Ud(Hmean, Amean_linear, rhoi, g, alpha, n)
Ud_pchip = calculate_Ud(Hmean, Amean_pchip, rhoi, g, alpha, n)
Ud_akima = calculate_Ud(Hmean, Amean_akima, rhoi, g, alpha, n)

print('')
print('Possible deformational velocities (Ud):')
print('Linear Ud: ' + str(Ud_linear*secperyear) + ' m/yr')
print('PCHIP Ud:  ' + str(Ud_pchip*secperyear) + ' m/yr')
print('Akima Ud:  ' + str(Ud_akima*secperyear) + ' m/yr')

[eta_jer_lin, eta_gud_lin] = calculate_eta(Hmean, Amean_linear, rhoi, g, alpha, n, Ud_linear)
[eta_jer_pchip, eta_gud_pchip] = calculate_eta(Hmean, Amean_pchip, rhoi, g, alpha, n, Ud_pchip)
[eta_jer_akima, eta_gud_akima] = calculate_eta(Hmean, Amean_akima, rhoi, g, alpha, n, Ud_akima)

print('')
print('Possible eta values:')
print('Jeremy Linear Eta: ' + str("%.7g" % eta_jer_lin) + ' kg m^-1 s^-1)' + '    Gudmundsson Linear Eta: ' + str("%.7g" %(eta_gud_lin)) + ' kg m^-1 s^-1)' )
print('Jeremy PCHIP Eta:  ' + str("%.7g" % eta_jer_pchip) + ' kg m^-1 s^-1)' + '   Gudmundsson PCHIP Eta:  ' + str("%.7g" % (eta_gud_pchip)) + ' kg m^-1 s^-1)' )
print('Jeremy Akima Eta:  ' + str("%.7g" % eta_jer_akima) + ' kg m^-1 s^-1)' + '   Gudmundsson Akima Eta:  ' + str("%.7g" %(eta_gud_akima)) + ' kg m^-1 s^-1)' )

#--------------------------------------------------------------------
# Calculating C from surface and deformation velocity

# Using Ud that we calculated earlier, we can then determine the slip ratio C of our region
# by getting the surface velocities

GPS2011_1h_GULL = get_GPS_data(2011, 1, 'GULL')
GPS2012_1h_GULL = get_GPS_data(2012, 1, 'GULL')
t2011 = GPS2011_1h_GULL[0]
t2012 = GPS2012_1h_GULL[0]
v2011 = GPS2011_1h_GULL[4]
v2012 = GPS2012_1h_GULL[4]
Us2011 = np.nanmean(v2011)
Us2012 = np.nanmean(v2012)

print('')
print('2011 mean Surface velocity: ' + str(Us2011) + ' m/yr')
print('2012 mean Surface velocity: ' + str(Us2012) + ' m/yr')

Ub2011 = Us2011 - (Ud_pchip*secperyear)
Ub2012 = Us2012 - (Ud_pchip*secperyear)

print('')
print('2011 Basal velocity: ' + str(Ub2011) + ' m/yr')
print('2012 Basal velocity: ' + str(Ub2012) + ' m/yr')

C2011 = Ub2011/(Ud_pchip*secperyear)
C2012 = Ub2012/(Ud_pchip*secperyear)

print('')
print('2011 Slip ratio C: ' + str(C2011))
print('2012 Slip ratio C: ' + str(C2012))

#--------------------------------------------------------------------
# Here, I'm testing the different interpolations

# Set mean thickness (less than or equal to maximum depth measured in the borehole)
# Discretize the depth
z = np.arange(0, round(Hmean)+1, 0.5)

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

#----------------------------------------------------------------------------------
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20,6))

ax.plot(t2011, v2011, 'b+', label = '2011')
ax.plot(t2012, v2012, 'g+', label = '2012')
ax.axhline(y = Us2011, color = 'red', linewidth = 1, label = 'mean 2011 vel')
ax.axhline(y = Us2012, color = 'orange', linewidth = 1, label = 'mean 2012 vel')

ax.set_xlabel('Day')
ax.set_ylabel('Velocity (m/yr)')
ax.set_title('Velocity profile at GULL borehole')
ax.legend()

plt.show()