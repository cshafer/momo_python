import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import ifft2, fft2
from scipy.ndimage import rotate
from transfer_function import TSB, TSC_transient, TUB, TUC_transient, TVB, TVC_transient # Python functions from Ockenden
from create_gaussian_perturbation import *
from read_bedmachine import *
from get_surface_slope import *
from mirror_pad_data import *
from calculate_A import *
from calculate_Ud import *
from get_GPS_data import *
from correct_and_transform_bedrock import *
from calculate_eta import *

#####################################
# Define physical parameters to use #
#####################################

n = 3                       # Glen's law exponent
m = 1                       # Flow law exponent (m = 1 is linear/Newtonian)
rhoi = 917                  # [kg/m^3] density of ice
g = 9.80665                 # [m/s^2] gravity
secperyear = 365*24*60*60   # [s/yr] conversion between seconds and years

###################################################################################
# Define information about region of study that will be extracted from BedMachine #
###################################################################################

# Define center of region and size of region with radius
x_center = -181636      # [m], polarstereographic x
y_center = -2237217     # [m], polarstereographic y # Lake 13 in the dataset
radius = 10500          # [m], x and y extent from the center which defines the bounding box
width = radius*2        # [m], twice the radius 
cell_spacing = 150      # [m], cell resolution of the dataset you're interested in (150 for BedMachine)

# Define BedMachine datafile location 
bedmachine_filepath = "C:\\Users\\casha\\Documents\\Research-Courtney_PC\\Moulin Model\\LAKE_DRAINAGE_CREVASSE_MODEL_JSTOCK\\BedMachineGreenland-2017-09-20.nc"

################################
# Extract data from Bedmachine #
################################

# Read data from BedMachine file
(bedrock, bedrock_error, surface, thickness, x, y) = read_bedmachine(bedmachine_filepath, x_center, y_center, radius, cell_spacing)

# Determine cell length in x and y
length_x = len(x)
length_y = len(y)

##############################################################################
# Calculate mean values (thickness and bedrock) and mean surface slope plane #
##############################################################################

# Calculate mean values
h_bar = np.mean(thickness)      # Get mean thickness
b_bar = np.mean(bedrock)        # Get mean bedrock

# Create mean surface plane 
xp = np.arange(-radius, radius + cell_spacing, cell_spacing)    # [m], x vector with radius extent 
yp = np.arange(-radius, radius + cell_spacing, cell_spacing)    # [m], y vector with radius extent
[x_plane, y_plane] = np.meshgrid(xp, yp)                        # [m, m], 2d mesh of xp and yp
alpha, alpha_x, alpha_y, flow_dir = get_surface_slope(surface)  # Get surface information
mean_sloped_plane = x_plane * alpha_x + y_plane * alpha_y       # [m], mean sloped plane

fig02 = plt.figure()
im02 = plt.imshow(np.flipud(mean_sloped_plane), extent = [-192125, -171125, -2247725, -2226725])
plt.title('Mean sloped plane')
plt.colorbar()
plt.xlabel('polarstereographic x (m)')
plt.ylabel('polarstereographic y (m)')
plt.show()

#####################################################################################################
# Calculate background slip ratio C for the region of interest to be used within transfer functions #
#####################################################################################################

# Slip ratio calculation routine:
# (1) Use Temperature/depth data to calculate -> (2) A factor to calculate -> (3) Deformation velocity Ud.
# (4) Load GPS velocity data to determine -> (5) Average surface velocity, Us to calculate -> (6) Slip ratio, C.


# (1) Borehole temperature data [C degrees] at GULL
tempGULL = [-0.6500, -7.7500, -11.2700, -11.9500, -14.1300, -13.5700, -12.7400, -11.6900, 
            -10.1100, -8.4900, -6.5500, -4.7400, -2.7300, -1.5200, -0.8300, -0.6000,
            -0.5600 , -0.4900, -0.5400, -0.4200, -0.4700, -0.3900, -0.5000]

# (1) Borehole depth [m] that corresponds with each temperature value
depthGULL = [4, 255, 307, 355, 407, 455, 497, 515, 537, 555, 577, 595, 622, 645, 667, 
             676, 687, 690, 697, 699, 702, 705, 707]

# (2) Calculate A factor
Tvector, Amean = calculate_A(h_bar, tempGULL, depthGULL, 'pchip') # 'pchip' is the interpolation method

# (3) Calculate deformation velocity, Ud and eta
Ud = calculate_Ud(h_bar, Amean, rhoi, g, alpha, n) * secperyear # [m/yr], deformation velocity Ud
eta = calculate_eta(h_bar, Amean, rhoi, g, alpha, n)            # [kg m^-1 s^-1], eta 

# (4) Load GPS data and get velocity data

GPS_data_filepath = 'C:\\Users\\casha\\Documents\\Research-Courtney_PC\\Moulin Model\\LAKE_DRAINAGE_CREVASSE_MODEL_JSTOCK\\GPS_Data\\'

GPS2011_1h_GULL = get_GPS_data(GPS_data_filepath, 2011, 1, 'GULL')     # GPS data  
v2011 = GPS2011_1h_GULL[4]
winter_v2011 = v2011[13921:]

# (5) Calculate average surface velocity and (6) slip ratio, C     
Us = np.nanmean(v2011)      # [m/yr] Average surface velocity for entire time period
Ub = Us - Ud                # [m/yr] Basal velocity
C = Ub/Ud                   # [unitless], Slip ratio, C

#winter_Us = np.nanmean(winter_v2011)    # [m/yr], Average surface velocity for winter time period (starting Oct. 1st)
#winter_Ub = winter_Us - Ud              # [m/yr], Winter basal velocity
#winter_C = winter_Ub/Ud                 # [unitless], Winter slip ratio, C

###########################################################################
# Begin bedrock data pre-processing for input into the transfer functions #
###########################################################################

# Non-dimensionalize cell spacing 
cell_spacing_nd = cell_spacing/h_bar   # [unitless], non-dimensional cell-spacing
 
# Bedrock perturbation Δb
# Within the correct function, bedrock is
    # 1. Normalized around the mean B = B - mean(B)
    # 2. Non-dimensionalized using h_bar
    # 3. Mirror padded to reduce edge effects
    # 4. Cosine tapered to set up for rotation
    # 5. Padded with zeros to further set up for rotation
    # 6. Rotated to match flow direction
    # 7. Fourier transformed (fft2)

bedrock_perturbation = correct_and_transform_bedrock(bedrock, b_bar, h_bar, flow_dir)

##################################################
# Create Gaussian slippery patch perturbation Δc #
##################################################

A_slip = 100        # Amplitude of Gaussian perturbation
sigma_x = 2300      # [m], Approximate average blister radius from Lai et al., (2021) for the 9 lakes (Fig. 3)
sigma_y = 2300      # [m], Approximate average blister radius from Lai et al., (2021) for the 9 lakes (Fig. 3)
theta = 0
                    # And then we use x_plane and y_plane from earlier to set the x and y extents of the grid
                    # that we place the gaussian perturbation on

slip_perturbation = create_gaussian_perturbation(A_slip, sigma_x, sigma_y, theta, x_plane, y_plane)

# Now correct and pre-process slip perturbation to get it ready for transfer function multiplication
slip_perturbation = slip_perturbation - np.mean(slip_perturbation)  # Normalize with the mean
slip_pert_mirror_padded = mirror_pad_data(slip_perturbation)        # Mirror padding

# Create cosine window to taper padded slip perturbation
cosine_window1d = np.abs(tukey(len(slip_pert_mirror_padded)))  # tukey is a kind of window function within the scipy library
cosine_window2d = np.sqrt(np.outer(cosine_window1d, cosine_window1d))
slip_pert_tapered = slip_pert_mirror_padded * cosine_window2d

# Add zero region to prep for rotation
slip_pert_zeros = np.pad(slip_pert_tapered, pad_width = len(slip_perturbation)) 

# Rotate
slip_pert_rotate = rotate(slip_pert_zeros, 180 - flow_dir)

# Fourier transform
slip_perturbation = fft2(slip_pert_rotate)                         # Fourier transform

# How long is the slippery patch "on" for?
t = 0.5     # [days], time here affects the surface response due to the patch

######################
# Get k and l values #
######################

ar1 = np.fft.fftfreq(bedrock_perturbation.shape[1], cell_spacing_nd)
ar2 = np.fft.fftfreq(bedrock_perturbation.shape[0], cell_spacing_nd)
k, l = np.meshgrid(ar1, ar2)

##########################################
# Apply transfer functions to get ratios #
##########################################

Tsb = TSB(k, l, m, C, alpha)
Tub = TUB(k, l, m, C, alpha)
Tvb = TVB(k, l, m, C, alpha)
Tsc_t = TSC_transient(k, l, m, C, t, alpha)
Tuc_t = TUC_transient(k, l, m, C, t, alpha)
Tvc_t = TVC_transient(k, l, m, C, t, alpha)

# Correct the first value in the array
Tsb[0,0] = 0
Tub[0,0] = 0
Tvb[0,0] = 0
Tsc_t[0,0] = 0
Tuc_t[0,0] = 0
Tvc_t[0,0] = 0

##########################################################################
# Multiply surface to bed ratio Tsb with bedrock to get surface response #
##########################################################################

Sb = Tsb * bedrock_perturbation
Ub = Tub * bedrock_perturbation
Vb = Tvb * bedrock_perturbation

Sc = Tsc_t * slip_perturbation
Uc = Tuc_t * slip_perturbation
Vc = Tvc_t * slip_perturbation

#############################################
# Get modeled surface from surface response #
#############################################

# Corrections
S = ifft2(Sb + Sc)                      # Inverse fourier transform
S = S * h_bar                           # Redimensionalize
S = rotate(S, -(180 - flow_dir))        # Rotate back to normal

U = ifft2(Ub + Uc)
U = U * (1/(2*eta)) * (rhoi * g * h_bar * alpha) * h_bar    # Redimensionalize 
U = rotate(U, -(180 - flow_dir))

new_center = (len(S) - 1)/2                         # Find the new center to find where to trim
new_start = int(new_center - (length_x - 1)/2)      # left/bottom extent of region of interest to crop
new_end = int(new_center + (length_x - 1)/2 + 1)    # right/top extent of region of interest to crop

S = S[new_start:new_end, new_start:new_end]               # Trim
S = S + h_bar + b_bar + np.flipud(mean_sloped_plane)      # Adjust to match real surface elevation

############################################################
# Take difference between modeled surface and real surface #
############################################################

# Corrections
difference = surface - S.real



#

#print("Mean 2011 Winter velocity: " + str(winter_Us))
#print('')
print("Mean 2011 yearly velocity: " + str(Us))
print('')
#print("Winter 2011 slip Ratio C: " + str(winter_C))
#print('')
print("Yearly 2011 slip ratio C: " + str(C))
print('')
print('Mean thickness H: ' + str(h_bar) + ' m')
print('')
print('Surface slope alpha: ' + str(alpha) + ' radians')
print('')
print('mean A: ' + str(Amean) + ' s^-1 Pa^-3')
print('')
print('Deformation velocity Ud: ' + str(Ud) + ' m/yr')
print('')
print('Mean surface velocity Us: ' + str(Us) + ' m/yr')
print('')
print('Basal velocity Ub: ' + str(Ub) + ' m/yr')
print('')
print('Slip ratio C: ' + str(C))
print('')
print('Eta: ' + str(eta) + ' kg m^-1 s^-1')