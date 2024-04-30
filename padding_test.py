
# This is a testing demo to determine how adding corrections such as mirrored padding,
# window tapering, rotation of bedrock, and surface slope affects the difference between
# The modeled surface and the true surface. 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import ifft2
from scipy.ndimage import rotate
from transfer_function import TSB # Python functions from Ockenden
from create_gaussian_perturbation import *
from read_bedmachine import *
from get_surface_slope import *
from mirror_pad_data import *
from calculate_A import *
from calculate_Ud import *
from get_GPS_data import *
from correct_and_transform_bedrock import *

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
[x_plane, y_plane] = np.meshgrid(xp, yp)                        # [m, m], 2d mesh of xp and yp to create sloped plane
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
[Tvector, Amean] = calculate_A(h_bar, tempGULL, depthGULL, 'pchip') # 'pchip' is the interpolation method

# (3) Calculate deformation velocity, Ud
Ud = calculate_Ud(h_bar, Amean, rhoi, g, alpha, n) * secperyear     # [m/yr], deformation velocity Ud

# (4) Load GPS data and get velocity data

GPS_data_filepath = 'C:\\Users\\casha\\Documents\\Research-Courtney_PC\\Moulin Model\\LAKE_DRAINAGE_CREVASSE_MODEL_JSTOCK\\GPS_Data\\'

GPS2011_1h_GULL = get_GPS_data(GPS_data_filepath, 2011, 1, 'GULL')     # GPS data  
v2011 = GPS2011_1h_GULL[4]
winter_v2011 = v2011[13921:]

# (5) Calculate average surface velocity and (6) slip ratio, C     
Us = np.nanmean(v2011)      # [m/yr] Average surface velocity for entire time period
Ub = Us - Ud                # [m/yr] Basal velocity
C = Ub/Ud                   # [unitless], Slip ratio, C

winter_Us = np.nanmean(winter_v2011)    # [m/yr], Average surface velocity for winter time period (starting Oct. 1st)
winter_Ub = winter_Us - Ud              # [m/yr], Winter basal velocity
winter_C = winter_Ub/Ud                 # [unitless], Winter slip ratio, C

###########################################################################
# Begin bedrock data pre-processing for input into the transfer functions #
###########################################################################

# Create two different bedrock arrays to test. The first one will be without corrections, so
# no padding, no window tapering, no rotation, and later no correction to the surface slope. The second 
# one, using the correct_and_transform_bedrock function will apply padding, window tapering, and rotation. 
# Finally the bedrock is run through the fft2 function to get it ready to read the k and l values

# Non-dimensionalize cell spacing 
cell_spacing_nd = cell_spacing/h_bar   # [unitless], non-dimensional cell-spacing

# No corrections
bedrock_no_corr = bedrock - b_bar           # Normalize around mean
bedrock_no_corr = bedrock_no_corr/h_bar     # Non-dimensionalize
bedrock_no_corr = fft2(bedrock_no_corr)     # Fourier transform 

# Corrections
bedrock_corr = correct_and_transform_bedrock(bedrock, b_bar, h_bar, flow_dir)

######################
# Get k and l values #
######################

# No corrections
ar1_no_corr = np.fft.fftfreq(bedrock_no_corr.shape[1], cell_spacing_nd)
ar2_no_corr = np.fft.fftfreq(bedrock_no_corr.shape[0], cell_spacing_nd)
k_no_corr, l_no_corr = np.meshgrid(ar1_no_corr, ar2_no_corr)

# Corrections
ar1 = np.fft.fftfreq(bedrock_corr.shape[1], cell_spacing_nd)
ar2 = np.fft.fftfreq(bedrock_corr.shape[0], cell_spacing_nd)
k, l = np.meshgrid(ar1, ar2)

###############################################################
# Apply Tsb transfer function to get surface to bed ratio Tsb #
###############################################################

# No corrections
Tsb_no_corr = TSB(k_no_corr, l_no_corr, m, C, alpha)
Tsb_no_corr[0,0] = 0

# Corrections
Tsb = TSB(k, l, m, C, alpha)
Tsb[0,0] = 0

##########################################################################
# Multiply surface to bed ratio Tsb with bedrock to get surface response #
##########################################################################

# No corrections
Sb_no_corr = Tsb_no_corr * bedrock_no_corr

# Corrections
Sb = Tsb * bedrock_corr

#############################################
# Get modeled surface from surface response #
#############################################

# No corrections
S_no_corr = ifft2(Sb_no_corr)           # Inverse fourier transform
S_no_corr = S_no_corr * h_bar           # Redimensionalize
S_no_corr = S_no_corr + h_bar + b_bar   # Adjust to match real surface elevation

# Corrections
S = ifft2(Sb)                      # Inverse fourier transform
S = S * h_bar                      # Redimensionalize
S = rotate(S, -(180 - flow_dir))   # Rotate back to normal

new_center = (len(S) - 1)/2                         # Find the new center to find where to trim
new_start = int(new_center - (length_x - 1)/2)      # left/bottom extent of region of interest to crop
new_end = int(new_center + (length_x - 1)/2 + 1)    # right/top extent of region of interest to crop

S = S[new_start:new_end, new_start:new_end]               # Trim
S = S + h_bar + b_bar + np.flipud(mean_sloped_plane)      # Adjust to match real surface elevation

############################################################
# Take difference between modeled surface and real surface #
############################################################

# No corrections
difference_no_corr = surface - S_no_corr.real

# Corrections
difference = surface - S.real

######################################################
# Generate figures to display the difference between #
# using non-corrected and corrected bedrock data     #
######################################################

# Create true surface plot figure
fig0 = plt.figure()
im0 = plt.imshow(surface, extent = [x[0], x[-1], y[-1], y[0]])
plt.title('True surface')
plt.colorbar()
plt.xlabel('polarstereographic x (m)')
plt.ylabel('polarstereographic y (m)')

# Create subplots figure and define axes within figure
fig1 = plt.figure(figsize= (18,23))
ax1 = plt.subplot2grid((3, 2), (0, 0)) 
ax2 = plt.subplot2grid((3, 2), (0, 1)) 
ax3 = plt.subplot2grid((3, 2), (1,0))
ax4 = plt.subplot2grid((3, 2), (1,1))
ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2) 

# Top left, plot 1
im1 = ax1.imshow(S_no_corr.real, extent = [x[0], x[-1], y[-1], y[0]])
cbar1 = fig1.colorbar(im1, ax = ax1)
cbar1.ax.get_yaxis().labelpad = 3
cbar1.ax.set_ylabel('Height (m)')
ax1.set_title('Modeled Surface (uncorrected)', fontsize = 20)
ax1.set_xlabel('polarstereographic x (m)')
ax1.set_ylabel('polarstereographic y (m)')

# Top right, plot 2
im2 = ax2.imshow(S.real, extent = [x[0], x[-1], y[-1], y[0]])
cbar2 = fig1.colorbar(im2, ax = ax2)
cbar2.ax.get_yaxis().labelpad = 3
cbar2.ax.set_ylabel('Height (m)')
ax2.set_title('Modeled Surface (corrected)', fontsize = 20)
ax2.set_xlabel('polarstereographic x (m)')
ax2.set_ylabel('polarstereographic y (m)')

# Middle left, plot 3
norm = mpl.colors.TwoSlopeNorm(vcenter = 0)
im3 = ax3.imshow(difference_no_corr, cmap = 'seismic', extent = [x[0], x[-1], y[-1], y[0]], norm = norm)
cbar3 = fig1.colorbar(im3, ax = ax3)
cbar3.ax.get_yaxis().labelpad = 3
cbar3.ax.set_ylabel('Difference (m)')
ax3.set_title('Error/Difference (uncorrected)', fontsize = 20)
ax3.axhline(y=y[int(radius/cell_spacing)], color = 'cyan')
ax3.set_xlabel('polarstereographic x (m)')
ax3.set_ylabel('polarstereographic y (m)')

# Middle right, plot 4
norm = mpl.colors.TwoSlopeNorm(vcenter = 0)
im4 = ax4.imshow(difference, cmap = 'seismic', extent = [x[0], x[-1], y[-1], y[0]], norm = norm)
cbar4 = fig1.colorbar(im4, ax = ax4)
cbar4.ax.get_yaxis().labelpad = 3
cbar4.ax.set_ylabel('Difference (m)')
ax4.set_title('Error/Difference (corrected)', fontsize = 20)
ax4.axhline(y=y[int(radius/cell_spacing)], color = 'cyan')
ax4.set_xlabel('polarstereographic x (m)')
ax4.set_ylabel('polarstereographic y (m)')

# Bottom, plot 5
ax5.plot(difference_no_corr[int(radius/cell_spacing)], color = 'r', label = 'uncorrected')
ax5.plot(difference[int(radius/cell_spacing)], color = 'b', label = 'corrected')
ax5.set_title('Error (difference) along x transect', fontsize = 20)
ax5.set_ylabel('Difference (m)')
ax5.set_xlabel('x extent (cells)')
ax5.axhline(y=0, color = 'black')
ax5.legend(fontsize = 15)




print("Mean 2011 Winter velocity: " + str(winter_Us))
print("Mean 2011 yearly velocity: " + str(Us))

print("Winter 2011 slip Ratio C: " + str(C))
print("Yearly 2011 slip ratio C: " + str(winter_C))

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