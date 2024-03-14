
# This is a testing demo to determine how padding affects the error. I 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft2, ifft2
from transfer_function import TSB # Python functions from Ockenden
from create_gaussian_perturbation import *
from read_bedmachine import *
from get_surface_slope import *
from pad_data import *

m = 1
C = 10

# Load data
filepath = "C:\\Users\\casha\\Documents\\Research-Courtney_PC\\Moulin Model\\Greenland_BedMacine\\244470242\\BedMachineGreenland-v5.nc"

# Define cell center and radius size
x_center = 264000    # [m], polarstereographic x
y_center = -2583000     # [m], polarstereographic y # Lake 13 in the dataset
radius = 9000      # [m], x and y extent from the center which defines the bounding box

# Get bedmachine data and calculate necessary values
(bedrock, bedrock_error, surface, thickness, x, y) = read_bedmachine(filepath, x_center, y_center, radius)
xm, ym = np.meshgrid(x,y)           # Get mesh of x and y vector ranges
h_bar = np.mean(thickness)          # Get mean thickness
ALPHA = get_surface_slope(surface)  # Get surface slope
cell_spacing = np.abs(x[0] - x[1])  # Get cell resolution (BedMachine == 150)

# Non-dimensionalize cell spacing 
cell_spacing_nd = cell_spacing/h_bar   # [unitless], non-dimensional cell-spacing

# Correct bedrock with mean value, non-dimensionalize, pad, and take fourier transform
bedrock_corr = bedrock - np.mean(bedrock) #- xm * cell_spacing_nd * ALPHA
bedrock_corr_padded = pad_data(bedrock_corr)

plt.imshow(bedrock_corr)
plt.colorbar()
#plt.imshow(bedrock_corr_padded)

bedrock_corr_nd = bedrock_corr/h_bar
bedrock_corr_padded_nd = bedrock_corr_padded/h_bar

bedrock_corr_padded_nd_ft = fft2(bedrock_corr_padded_nd)
bedrock_corr_nd_ft = fft2(bedrock_corr_nd)

# Get k and l values 
ar1_padded = np.fft.fftfreq(bedrock_corr_padded_nd_ft.shape[1], cell_spacing_nd)
ar2_padded = np.fft.fftfreq(bedrock_corr_padded_nd_ft.shape[0], cell_spacing_nd)
k_padded,l_padded = np.meshgrid(ar1_padded,ar2_padded)

ar1 = np.fft.fftfreq(bedrock_corr_nd_ft.shape[1], cell_spacing_nd)
ar2 = np.fft.fftfreq(bedrock_corr_nd_ft.shape[0], cell_spacing_nd)
k, l = np.meshgrid(ar1, ar2)

# Apply Tsb transfer function and get Tsb ratio
Tsb_padded = TSB(k_padded, l_padded, m, C, ALPHA)
Tsb_padded[0,0] = 0

Tsb = TSB(k, l, m, C, ALPHA)
Tsb[0,0] = 0

# Multiply transfer function to transformed bedrock 
Sb_padded_nd_ft = Tsb_padded * bedrock_corr_padded_nd_ft
Sb_nd_ft = Tsb * bedrock_corr_nd_ft

# Take inverse fourier transform and redimensionalize and shift back
Sb_padded = ifft2(Sb_padded_nd_ft) * h_bar + h_bar + np.mean(bedrock)
Sb = ifft2(Sb_nd_ft) * h_bar + h_bar + np.mean(bedrock)

# Trim padding
Sb_padded = Sb_padded[120:240, 120:240] #+ xm * cell_spacing_nd * ALPHA 

# Take difference between actual surface and real surface
difference_padded = surface - Sb_padded.real
difference = surface - Sb.real

#plt.imshow(Sb_padded.real)
#plt.colorbar
#plt.imshow(Sb.real)
#plt.colorbar
#plt.imshow(surface)
#plt.colorbar()

# Create true surface plot figure
fig1 = plt.figure()
im6 = plt.imshow(surface, extent = [x[0], x[-1], y[-1], y[0]])
plt.title('True surface')
plt.colorbar()
plt.xlabel('polarstereographic x (m)')
plt.ylabel('polarstereographic y (m)')

# Create subplots figure and define axes within figure
fig = plt.figure(figsize= (15,20))
ax1 = plt.subplot2grid((3, 2), (0, 0)) 
ax2 = plt.subplot2grid((3, 2), (0, 1)) 
ax3 = plt.subplot2grid((3, 2), (1,0))
ax4 = plt.subplot2grid((3, 2), (1,1))
ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2) 

# Top left, plot 1
im1 = ax1.imshow(Sb_padded.real, extent = [x[0], x[-1], y[-1], y[0]])
cbar1 = fig.colorbar(im1, ax = ax1)
cbar1.ax.get_yaxis().labelpad = 3
cbar1.ax.set_ylabel('Height (m)')
ax1.set_title('Modeled Surface (padded)')
ax1.set_xlabel('polarstereographic x (m)')
ax1.set_ylabel('polarstereographic y (m)')


# Top right, plot 2
im2 = ax2.imshow(Sb.real, extent = [x[0], x[-1], y[-1], y[0]])
cbar2 = fig.colorbar(im2, ax = ax2)
cbar2.ax.get_yaxis().labelpad = 3
cbar2.ax.set_ylabel('Height (m)')
ax2.set_title('Modeled Surface (not padded)')
ax2.set_xlabel('polarstereographic x (m)')
ax2.set_ylabel('polarstereographic y (m)')

# Middle left, plot 3
norm = mpl.colors.TwoSlopeNorm(vcenter = 0)
im3 = ax3.imshow(difference_padded, cmap = 'seismic', extent = [x[0], x[-1], y[-1], y[0]], norm = norm)
cbar3 = fig.colorbar(im3, ax = ax3)
cbar3.ax.get_yaxis().labelpad = 3
cbar3.ax.set_ylabel('Difference (m)')
ax3.set_title('True Surface - Sb (padded)')
ax3.axhline(y=y[60], color = 'cyan')
ax3.set_xlabel('polarstereographic x (m)')
ax3.set_ylabel('polarstereographic y (m)')

# Middle right, plot 4
norm = mpl.colors.TwoSlopeNorm(vcenter = 0)
im4 = ax4.imshow(difference, cmap = 'seismic', extent = [x[0], x[-1], y[-1], y[0]], norm = norm)
cbar4 = fig.colorbar(im4, ax = ax4)
cbar4.ax.get_yaxis().labelpad = 3
cbar4.ax.set_ylabel('Difference (m)')
ax4.set_title('True surface - Sb (not padded)')
ax4.axhline(y=y[60], color = 'cyan')
ax4.set_xlabel('polarstereographic x (m)')
ax4.set_ylabel('polarstereographic y (m)')

# Bottom, plot 5
ax5.plot(difference_padded[60], color = 'r', label = 'padded')
ax5.plot(difference[60], color = 'b', label = 'not padded')
ax5.set_title('Error (difference) along x transect')
ax5.set_ylabel('Difference (m)')
ax5.set_xlabel('x extent (cells)')
ax5.axhline(y=0, color = 'black')
ax5.legend()