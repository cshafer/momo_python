import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft2, ifft2
from transfer_function import TSB, TUB, TVB, TSC, TUC, TVC # These are from Ockenden
from create_gaussian_perturbation import *
from read_bedmachine import *
from get_surface_slope import *
from pad_data import *

# IDEALIZED EXPERIMENT

## |---------- 1.1 Ice properties setup --------| ##

# Define properties of the ice
ALPHA = 0.002                               # [unitless]    , surface slope of the ice sheet
HMEAN = 1000                                # [m]           , mean thickness of the ice sheet, used in non-dimensionalization
#THETA = 90                                  # [unitless]    , angle between wave vector k_hat = (k,l) and x axis
C = 1                                     # [unitless]    , mean non-dimensional slipperiness ub_bar/ud_bar
RHOI = 917                                  # [kg m^-3]     , density of ice
G = 9.80665                                 # [m s^-2], gravitational force
TAU_D = RHOI * G * HMEAN * np.sin(ALPHA)    # [Pa or N m^-2], driving stress
ETA = 0                                     # [kg m^-1 s^-1]    , effective viscosity
m = 1

## |----------- 1.2 Domain setup ----------------| ##

# Define properties of the domain 
cell_spacing = 150                          # [m], resolution of the domain, 150 is the resolution of BedMachine
x_window_length = 60000                     # [m], length of the domain in x
y_window_length = 60000                     # [m], length of the domain in y

# Non-dimensionalize the domain using mean thickness
cell_spacing_nd = cell_spacing/HMEAN        # [unitless], non-dimensional cell-spacing
x_window_length_nd = x_window_length/HMEAN  # [unitless], non-dimensional x domain
y_window_length_nd = y_window_length/HMEAN  # [unitless], non-dimensional y domain

# Create x-y grid using non-dimensional x and y window lengths
left_x = -x_window_length_nd/2
right_x = (x_window_length_nd/2) + cell_spacing_nd
bottom_y = -y_window_length_nd/2
top_y = (y_window_length_nd/2) + cell_spacing_nd

x_array = np.arange(left_x, right_x, cell_spacing_nd)
y_array = np.arange(bottom_y, top_y, cell_spacing_nd)
[x,y] = np.meshgrid(x_array, y_array) 

## |---------- 1.3 Create Bedrock Perturbation ---------------| ##

# Define the amplitude and the width of a single Gaussian bed bump
bedrock_gaussian_amplitude = 1     # [unitless]    , amplitude of the Gaussian bedrock perturbation
bedrock_gaussian_sigmax = 7000      # [m]           , width of the Gaussian in the x direction
bedrock_gaussian_sigmay = 15000      # [m]           , width of the Guassian in the y direction
theta1 = 0
theta2 = np.pi/4

# Non-dimensionalize width of Gaussian bed bump by dividing by Hmean
bedrock_gaussian_sigmax_nd = bedrock_gaussian_sigmax/HMEAN   # [unitless]    , non-dimensional Gaussian width in x-direction
bedrock_gaussian_sigmay_nd = bedrock_gaussian_sigmay/HMEAN   # [unitless]    , non-dimensional Gaussian width in y-direction

# Create non-dimensional Gaussian bedrock perturbation delta b
bedrock_topg_delta_b_nd = create_gaussian_perturbation(bedrock_gaussian_amplitude, bedrock_gaussian_sigmax_nd, bedrock_gaussian_sigmay_nd, theta1, x, y)
bedrock_topg_delta_b_nd_theta = create_gaussian_perturbation(bedrock_gaussian_amplitude, bedrock_gaussian_sigmax_nd, bedrock_gaussian_sigmay_nd, theta2, x, y)

## |--------- 1.5 Transform non-d perturbations to Fourier Space ----------| ##

delta_b_nd_ft = fft2(bedrock_topg_delta_b_nd)               # Fourier transformed bedrock perturbation
delta_b_nd_ft_theta2 = fft2(bedrock_topg_delta_b_nd_theta) # Fourier transofrmed bedrock perturbation with angle change

## \---------- 1.6 Transfer Functions ----------\ ##
# We need to get arrays of k and l to input into the transfer functions
ar1 = np.fft.fftfreq(bedrock_topg_delta_b_nd.shape[1], cell_spacing_nd)
ar2 = np.fft.fftfreq(bedrock_topg_delta_b_nd.shape[0], cell_spacing_nd)
k,l = np.meshgrid(ar1,ar2)

# Bed topography to surface transfer functions
Tsb = TSB(k, l, m, C, ALPHA)   # Bed B to surface expression S

# Because both k and l have zeros in them, the transfer functions return a NaN in the first [0,0] entry
# due to division by 0. We set this value to 0 to avoid this issue.
Tsb[0,0] = 0

# Multiply transfer function output (fourier space) to bedrock and slip perturbations (also fourier space)
Sb_nd_ft = Tsb * delta_b_nd_ft
Sb_nd_ft_theta = Tsb * delta_b_nd_ft_theta2

# Take the inverse fourier transform to go from spectral to x,y space and then redimensionalize
Sb = ifft2(Sb_nd_ft) * HMEAN
Sb_theta = ifft2(Sb_nd_ft_theta) * HMEAN


fig = plt.figure(figsize= (14,11))
ax1 = plt.subplot2grid((2, 2), (0, 0)) 
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0)) 
ax4 = plt.subplot2grid((2, 2), (1, 1))

extent = [left_x, right_x, bottom_y, top_y]

im1 = ax1.imshow(bedrock_topg_delta_b_nd, extent = extent, interpolation = 'none')
cbar1 = fig.colorbar(im1, ax = ax1)
ax1.set_title('Gaussian Bedrock Perturbation')
ax1.set_xlabel('x (km)')
ax1.set_ylabel('y (km)')
ax1.axhline(y=0, color = 'black', linewidth = '0.5')
ax1.axvline(x=0, color = 'black', linewidth = '0.5')
CS = ax1.contour(bedrock_topg_delta_b_nd, [0.2, 0.4, 0.6, 0.8], origin = 'upper', colors='k', extent = extent, linewidths = 0.5)
ax1.clabel(CS, inline=1, fontsize=10)

norm = mpl.colors.TwoSlopeNorm(vcenter = 0)
im2 = ax2.imshow(Sb.real, cmap = 'jet', norm = norm, extent = extent, interpolation = 'none')
cbar2 = fig.colorbar(im2, ax = ax2)
ax2.set_title('Surface response, Sb')
ax2.set_xlabel('x (km)')
ax2.set_ylabel('y (km)')
ax2.axhline(y=0, color = 'red', linewidth = '0.5')
ax2.axvline(x=0, color = 'red', linewidth = '0.5')
CS = ax2.contour(Sb.real, [-80, -70,-60, -50,-40, -30, -20, -10, 0,10, 20, 30, 40, 50, 60, 70, 80], origin = 'upper', colors='k', extent = extent, linewidths = 0.5)
ax2.clabel(CS, inline=1, fontsize=10)

im3 = ax3.imshow(bedrock_topg_delta_b_nd_theta, extent = extent, interpolation = 'none')
cbar3 = fig.colorbar(im3, ax = ax3)
ax3.set_title('Gaussian Bedrock Perturbation, 45 degree rotation')
ax3.set_xlabel('x (km)')
ax3.set_ylabel('y (km)')
ax3.axhline(y=0, color = 'black', linewidth = '0.5')
ax3.axvline(x=0, color = 'black', linewidth = '0.5')
CS = ax3.contour(bedrock_topg_delta_b_nd_theta, [0.2, 0.4, 0.6, 0.8], origin = 'upper', colors='k', extent = extent, linewidths = 0.5)
ax3.clabel(CS, inline=1, fontsize=10)

norm = mpl.colors.TwoSlopeNorm(vcenter = 0)
im4 = ax4.imshow(Sb_theta.real, cmap = 'jet', norm = norm, extent = extent, interpolation = 'none')
cbar4 = fig.colorbar(im2, ax = ax4)
ax4.set_title('Surface response, Sb')
ax4.set_xlabel('x (km)')
ax4.set_ylabel('y (km)')
ax4.axhline(y=0, color = 'red', linewidth = '0.5')
ax4.axvline(x=0, color = 'red', linewidth = '0.5')
CS = ax4.contour(Sb_theta.real,[-80, -70,-60, -50,-40, -30, -20, -10, 0,10, 20, 30, 40, 50, 60, 70, 80], origin = 'upper', colors='k', extent = extent, linewidths = 0.5)
ax4.clabel(CS, inline=1, fontsize=10)

fig.suptitle("Effects of rotation of Gaussian Perturbation on surface response, C = 1", fontsize = 25)

