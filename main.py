#main.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from transfer_function import TSB, TUB, TVB, TSC, TUC, TVC # These are from Ockenden
from create_gaussian_bed import *
import sys

np.set_printoptions(threshold=sys.maxsize)

# Choose between idealized experiment or a real world experiment.
# idealized == 0 ; real world == 1;
idealized_or_realworld = 0

if idealized_or_realworld == 0:
    
    # IDEALIZED EXPERIMENT

    ## |---------- 1.1 Ice properties setup --------| ##

    # Define properties of the ice
    surface_slope_alpha = 0.002         # [unitless]    , surface slope of the ice sheet
    mean_thickness_Hmean = 1000         # [m]           , mean thickness of the ice sheet, used in non-dimensionalization
    flow_direction_theta = 90           # [unitless]    , angle between wave vector k_hat = (k,l) and x axis
    slip_ratio_C = 100                  # [unitless]    , mean non-dimensional slipperiness ub_bar/ud_bar
    time_days_t = 0                     # []            , dimensional time
    flow_law_coef_A = 0                 # []            , flow law coefficient in Glen's flow law
    m = 1                               # [unitless]    , slip law exponent

    ## |----------- 1.2 Domain setup ----------------| ##

    # Define properties of the domain 
    cell_spacing_dx = 150       # [m], resolution of the domain, 150 is the resolution of BedMachine
    x_domain_length = 30000     # [m], length of the domain in x
    y_domain_length = 30000     # [m], length of the domain in y

    # Non-dimensionalize the domain using mean thickness
    cell_spacing_dx_nd = cell_spacing_dx/mean_thickness_Hmean   # [unitless], non-dimensional cell-spacing
    x_domain_length_nd = x_domain_length/mean_thickness_Hmean   # [unitless], non-dimensional x domain
    y_domain_length_nd = y_domain_length/mean_thickness_Hmean   # [unitless], non-dimensional y domain
    
    ## |---------- 1.3 Create Bedrock Perturbation ---------------| ##

    # Define the amplitude and the width of a single Gaussian bed bump
    bedrock_gaussian_amplitude = 40     # [unitless]    , amplitude of the Gaussian bedrock perturbation
    bedrock_gaussian_sigmax = 2000      # [m]           , width of the Gaussian in the x direction
    bedrock_gaussian_sigmay = 2000      # [m]           , width of the Guassian in the y direction

    # Non-dimensionalize width of Gaussian bed bump by dividing by Hmean
    bedrock_gaussian_sigmax_nd = bedrock_gaussian_sigmax/mean_thickness_Hmean   # [unitless]    , non-dimensinoal Gaussian width in x-direction
    bedrock_gaussian_sigmay_nd = bedrock_gaussian_sigmay/mean_thickness_Hmean   # [unitless]    , non-dimensional Gaussian width in y-direction

    # Create non-dimensional Gaussian bedrock perturbation delta b
    bedrock_topg_delta_b_nd = create_gaussian_bed(bedrock_gaussian_amplitude, bedrock_gaussian_sigmax_nd, bedrock_gaussian_sigmay_nd, x_domain_length_nd, y_domain_length_nd, cell_spacing_dx_nd)

    ## |---------- 1.4 Create Slippery Patch Perturbation -----------| ##

    # Define the amplitude and width of a single Gaussian slippery patch
    slip_patch_amplitude_Ac = 0         # []
    slip_patch_x_width_sigmax = 1000    # []
    slip_patch_y_width_sigmay = 1000    # []

    # Non-dimensionalize width of Gaussian slippery patch by dividing by Hmean

    # Create non-dimensional Gaussian slip perturbation delta c
    # slippery_patch_delta_c_nd = create_gaussian_slippery_patch()

    ## |--------- 1.5 Transform non-d perturbations to Fourier Space ----------| ##

    delta_b_nd_ft = fft2(bedrock_topg_delta_b_nd)       # Fourier transformed bedrock perturbation
    #delta_c_nd_ft = fft2(slippery_path_delta_c_nd)     # Fourier transformed slip patch perturbation

    ## \---------- 1.6 Transfer Functions ----------\ ##
    # We need to get arrays of k and l to input into the transfer functions
    ar1 = np.fft.fftfreq(bedrock_topg_delta_b_nd.shape[1], cell_spacing_dx_nd)
    ar2 = np.fft.fftfreq(bedrock_topg_delta_b_nd.shape[0], cell_spacing_dx_nd)
    k,l = np.meshgrid(ar1,ar2)

    # Bed topography to surface transfer functions
    Tsb = TSB(k, l, m, slip_ratio_C, surface_slope_alpha)   # Bed B to surface expression S
    Tub = TUB(k, l, m, slip_ratio_C, surface_slope_alpha)   # Bed B to surface velocity in x direction U
    Tvb = TVB(k, l, m, slip_ratio_C, surface_slope_alpha)   # Bed B to surface velocity in y direction V
    #Tsc = TSC(k, l, m, slip_ratio_C, surface_slope_alpha)   # Slip C to surface expression S
    #Tuc = TUC(k, l, m, slip_ratio_C, surface_slope_alpha)   # Slip C to surface velocity in x direction U
    #Tvc = TVC(k, l, m, slip_ratio_C, surface_slope_alpha)   # Slip C to surface velocity in y direction V

    # Because both k and l have zeros in them, the transfer functions return a NaN in the first [0,0] entry
    # due to division by 0. We set this value to 0 to avoid this issue.
    Tsb[0,0] = 0
    Tub[0,0] = 0
    Tvb[0,0] = 0
    #Tsc[0,0] = 0
    #Tuc[0,0] = 0
    #Tvc[0,0] = 0

    # Multiply transfer function output (fourier space) to bedrock and slip perturbations (also fourier space)
    Sb_ft = Tsb * delta_b_nd_ft
    Sb = ifft2(Sb_ft) * mean_thickness_Hmean
    plt.imshow(Sb.real)

else:
    # Here run the real world experiment
    test = 1

    # 1. Here load bedmachine and produce predicted surface profile from BedMachine bedrock
    # 2. Compare against bedmachine surface within  

# Then here the rest of the code will run as expected
