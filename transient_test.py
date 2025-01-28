import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft2, ifft2
from transfer_function import *
from create_gaussian_perturbation import *
from read_bedmachine import *
from get_surface_slope import *
import imageio

# This is a test to see how the transient slip perturbation behaves as a function of time 

# Create a set of time values and see how the transfer functions behave when the slip perturbation is
# a function of time

time = np.linspace(0, 1000, 1000)

#####################################
# Define physical parameters to use #
#####################################

n = 3                           # Glen's law exponent
m = 1                           # Flow law exponent (m = 1 is linear/Newtonian)
rhoi = 917                      # [kg/m^3] density of ice
g = 9.80665                     # [m/s^2] gravity
Hmean = 1000                    # [m] thickness
C = 1                           # [unitless] slip ratio
alpha = 0.002                   # [unitless] surface slope
tau_d = rhoi*g*Hmean*np.sin(alpha)  # [Pa or N*m^-2] driving stress 
secperyear = 365*24*60*60       # [s/yr] conversion between seconds and years

time = np.linspace(0, 7, 10000)  # [days] 


################
# Domain setup #
################

# Define properties of the domain 
cell_spacing = 150                          # [m], resolution of the domain, 150 is the resolution of BedMachine
x_window_length = 60000                     # [m], length of the domain in x
y_window_length = 60000                     # [m], length of the domain in y

# Non-dimensionalize the domain using mean thickness
cell_spacing_nd = cell_spacing/Hmean        # [unitless], non-dimensional cell-spacing
x_window_length_nd = x_window_length/Hmean  # [unitless], non-dimensional x domain
y_window_length_nd = y_window_length/Hmean  # [unitless], non-dimensional y domain

# Create x-y grid using non-dimensional x and y window lengths
left_x = -x_window_length_nd/2
right_x = (x_window_length_nd/2) + cell_spacing_nd
bottom_y = -y_window_length_nd/2
top_y = (y_window_length_nd/2) + cell_spacing_nd

x_array = np.arange(left_x, right_x, cell_spacing_nd)
y_array = np.arange(bottom_y, top_y, cell_spacing_nd)
[x,y] = np.meshgrid(x_array, y_array)

#####################################################
# Create Bedrock Perturbation and slip perturbation #
#####################################################

# Define the amplitude and the width of a single Gaussian bed bump and Gaussian slip perturbation
bedrock_gaussian_amplitude = 1     # [unitless]    , amplitude of the Gaussian bedrock perturbation
bedrock_gaussian_sigmax = 7000      # [m]           , width of the Gaussian in the x direction
bedrock_gaussian_sigmay = 7000      # [m]           , width of the Gaussian in the y direction
theta_b = 0                         # [radians]     , counter-clockwise rotation of perturbation

slip_gaussian_amplitude = 1         # [unitless], amplitude of slip patch perturbation
slip_gaussian_sigmax = 5000         # [m]       , width of Gaussian in the x direction
slip_gaussian_sigmay = 5000         # [m]       , width of Gaussian in the y direction
theta_c = 0

# Non-dimensionalize width of Gaussian bed bump and slip patch by dividing by Hmean
bedrock_gaussian_sigmax_nd = bedrock_gaussian_sigmax/Hmean   # [unitless], non-dimensional Gaussian width in x-direction
bedrock_gaussian_sigmay_nd = bedrock_gaussian_sigmay/Hmean   # [unitless], non-dimensional Gaussian width in y-direction

slip_gaussian_sigmax_nd = slip_gaussian_sigmax/Hmean        # [unitless], non-dimensional slip Gaussian x width
slip_gaussian_sigmay_nd = slip_gaussian_sigmay/Hmean        # [unitless], non-dimensional slip Gaussian y width

# Create 2D non-dimensional Gaussian perturbation delta b and delta c
bedrock_topg_delta_b_nd = create_gaussian_perturbation(bedrock_gaussian_amplitude, bedrock_gaussian_sigmax_nd, bedrock_gaussian_sigmay_nd, theta_b, x, y)
slip_topg_delta_c_nd = create_gaussian_perturbation(slip_gaussian_amplitude, slip_gaussian_sigmax, slip_gaussian_sigmay, theta_c, x, y)

# Normalize by the mean
bedrock_topg_delta_b_nd_norm = bedrock_topg_delta_b_nd - np.mean(bedrock_topg_delta_b_nd)
slip_patch_delta_c_nd_norm = slip_topg_delta_c_nd - np.mean(slip_topg_delta_c_nd)

##################################################
# Transform non-d perturbations to Fourier Space #
##################################################

# Fourier transform bedrock perturbation to set up for getting k and l values
delta_b_nd_norm_ft = fft2(bedrock_topg_delta_b_nd_norm)
delta_c_nd_norm_ft = fft2(slip_patch_delta_c_nd_norm)

# We need to get arrays of k and l to input into the transfer functions
# The shape of the dataset can be from anything, we just need the form of the data with the correct
# cell resolution/# of cells and the cell spacing
ar1 = np.fft.fftfreq(bedrock_topg_delta_b_nd_norm.shape[1], cell_spacing_nd)
ar2 = np.fft.fftfreq(bedrock_topg_delta_b_nd_norm.shape[0], cell_spacing_nd)
k,l = np.meshgrid(ar1,ar2)

# Create empty arrays to store the Sc arrays into (to make a gif)
Sc_array = []
Uc_array = []
Vc_array = []

for t in time:

    # Run transfer functions
    Tsc_transient = TSC_transient(k, l, m, C, t, alpha)
    Tuc_transient = TUC_transient(k, l, m, C, t, alpha)
    Tvc_transient = TVC_transient(k, l, m, C, t, alpha)

    # Correct value
    Tsc_transient[0,0] = 0
    Tuc_transient[0,0] = 0
    Tvc_transient[0,0] = 0

    # Multiply Txc ratio with perturbation to get surface response (untransformed)
    Sc_transient = Tsc_transient * delta_c_nd_norm_ft
    Uc_transient = Tuc_transient * delta_c_nd_norm_ft
    Vc_transient = Tvc_transient * delta_c_nd_norm_ft

    # Transform Surface response (transformed)
    Sc_transient = ifft2(Sc_transient)
    Uc_transient = ifft2(Uc_transient)
    Vc_transient = ifft2(Vc_transient)

    # Surface response (Redimensionalized)



    # Append array to list of arrays
    Sc_array.append(Sc_transient)
    Uc_array.append(Uc_transient)
    Vc_array.append(Vc_transient)

imageio.mimsave('SC_gif.gif', Sc_array, fps=10)
imageio.mimsave('UC_gif.gif', Uc_array, fps=10)
imageio.mimsave('VC_gif.gif', Vc_array, fps=10)