#main.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from transfer_function import TSB, TUB, TVB, TSC, TUC, TVC # These are from Ockenden
from create_gaussian_perturbation import *
from read_bedmachine import *
from get_surface_slope import *


m = 1       # [unitless], slip law exponent

# Choose between idealized experiment or a real world experiment.
# idealized == 0 ; real world == 1;
idealized_or_realworld = 1

if idealized_or_realworld == 0:
    
    # IDEALIZED EXPERIMENT

    ## |---------- 1.1 Ice properties setup --------| ##

    # Define properties of the ice
    surface_slope_alpha = 0.002         # [unitless]    , surface slope of the ice sheet
    mean_thickness_Hmean = 1000         # [m]           , mean thickness of the ice sheet, used in non-dimensionalization
    #flow_direction_theta = 90           # [unitless]    , angle between wave vector k_hat = (k,l) and x axis
    slip_ratio_C = 100                  # [unitless]    , mean non-dimensional slipperiness ub_bar/ud_bar
    #time_days_t = 0                     # []            , dimensional time
    #flow_law_coef_A = 0                 # []            , flow law coefficient in Glen's flow law

    ## |----------- 1.2 Domain setup ----------------| ##

    # Define properties of the domain 
    cell_spacing = 150       # [m], resolution of the domain, 150 is the resolution of BedMachine
    x_window_length = 30000     # [m], length of the domain in x
    y_window_length = 30000     # [m], length of the domain in y

    # Non-dimensionalize the domain using mean thickness
    cell_spacing_nd = cell_spacing/mean_thickness_Hmean   # [unitless], non-dimensional cell-spacing
    x_window_length_nd = x_window_length/mean_thickness_Hmean   # [unitless], non-dimensional x domain
    y_window_length_nd = y_window_length/mean_thickness_Hmean   # [unitless], non-dimensional y domain
    
    # Create x-y grid using non-dimensional x and y window lengths
    left_x = -x_window_length_nd/2
    right_x = (x_window_length_nd/2) + 1
    bottom_y = -y_window_length_nd/2
    top_y = (y_window_length_nd/2) + 1
    [x,y] = np.mgrid[left_x:right_x:cell_spacing_nd, bottom_y:top_y:cell_spacing_nd]

    ## |---------- 1.3 Create Bedrock Perturbation ---------------| ##

    # Define the amplitude and the width of a single Gaussian bed bump
    bedrock_gaussian_amplitude = 40     # [unitless]    , amplitude of the Gaussian bedrock perturbation
    bedrock_gaussian_sigmax = 2000      # [m]           , width of the Gaussian in the x direction
    bedrock_gaussian_sigmay = 2000      # [m]           , width of the Guassian in the y direction

    # Non-dimensionalize width of Gaussian bed bump by dividing by Hmean
    bedrock_gaussian_sigmax_nd = bedrock_gaussian_sigmax/mean_thickness_Hmean   # [unitless]    , non-dimensinoal Gaussian width in x-direction
    bedrock_gaussian_sigmay_nd = bedrock_gaussian_sigmay/mean_thickness_Hmean   # [unitless]    , non-dimensional Gaussian width in y-direction

    # Create non-dimensional Gaussian bedrock perturbation delta b
    bedrock_topg_delta_b_nd = create_gaussian_perturbation(bedrock_gaussian_amplitude, bedrock_gaussian_sigmax_nd, bedrock_gaussian_sigmay_nd, x, y)

    ## |---------- 1.4 Create Slippery Patch Perturbation -----------| ##

    # Define the amplitude and width of a single Gaussian slippery patch
    slip_gaussian_amplitude_Ac = 10 # []
    slip_gaussian_sigmax = 2000     # []
    slip_gaussian_sigmay = 2000     # []

    # Non-dimensionalize width of Gaussian slippery patch by dividing by Hmean
    slip_gaussian_sigmax_nd = slip_gaussian_sigmax/mean_thickness_Hmean     # [unitless], non-dimensional Gaussian width in the x-direction
    slip_gaussian_sigmay_nd = slip_gaussian_sigmay/mean_thickness_Hmean     # [unitless], non-dimensional Gaussian width in the y-direction

    # Create non-dimensional Gaussian slip perturbation delta c
    slippery_patch_delta_c_nd = create_gaussian_perturbation(slip_gaussian_amplitude_Ac, slip_gaussian_sigmax_nd, slip_gaussian_sigmay_nd, x, y)

    ## |--------- 1.5 Transform non-d perturbations to Fourier Space ----------| ##

    delta_b_nd_ft = fft2(bedrock_topg_delta_b_nd)       # Fourier transformed bedrock perturbation
    delta_c_nd_ft = fft2(slippery_patch_delta_c_nd)     # Fourier transformed slip patch perturbation

    ## \---------- 1.6 Transfer Functions ----------\ ##
    # We need to get arrays of k and l to input into the transfer functions
    ar1 = np.fft.fftfreq(bedrock_topg_delta_b_nd.shape[1], cell_spacing_nd)
    ar2 = np.fft.fftfreq(bedrock_topg_delta_b_nd.shape[0], cell_spacing_nd)
    k,l = np.meshgrid(ar1,ar2)

    # Bed topography to surface transfer functions
    Tsb = TSB(k, l, m, slip_ratio_C, surface_slope_alpha)   # Bed B to surface expression S
    Tub = TUB(k, l, m, slip_ratio_C, surface_slope_alpha)   # Bed B to surface velocity in x direction U
    Tvb = TVB(k, l, m, slip_ratio_C, surface_slope_alpha)   # Bed B to surface velocity in y direction V
    Tsc = TSC(k, l, m, slip_ratio_C, surface_slope_alpha)   # Slip C to surface expression S
    Tuc = TUC(k, l, m, slip_ratio_C, surface_slope_alpha)   # Slip C to surface velocity in x direction U
    Tvc = TVC(k, l, m, slip_ratio_C, surface_slope_alpha)   # Slip C to surface velocity in y direction V

    # Because both k and l have zeros in them, the transfer functions return a NaN in the first [0,0] entry
    # due to division by 0. We set this value to 0 to avoid this issue.
    Tsb[0,0] = 0
    Tub[0,0] = 0
    Tvb[0,0] = 0
    Tsc[0,0] = 0
    Tuc[0,0] = 0
    Tvc[0,0] = 0

    # Multiply transfer function output (fourier space) to bedrock and slip perturbations (also fourier space)
    Sb_nd_ft = Tsb * delta_b_nd_ft
    Ub_nd_ft = Tub * delta_b_nd_ft
    Vb_nd_ft = Tvb * delta_b_nd_ft
    Sc_nd_ft = Tsc * delta_c_nd_ft
    Uc_nd_ft = Tuc * delta_c_nd_ft
    Vc_nd_ft = Tvc * delta_c_nd_ft

    # Take the inverse fourier transform to go from spectral to x,y space and then redimensionalize
    Sb = ifft2(Sb_nd_ft) * mean_thickness_Hmean
    Ub = ifft2(Ub_nd_ft) * mean_thickness_Hmean #wrong I need to redimensionalize using mean velocity
    Vb = ifft2(Vb_nd_ft) * mean_thickness_Hmean #wrong
    Sc = ifft2(Sc_nd_ft) * mean_thickness_Hmean
    Uc = ifft2(Uc_nd_ft) * mean_thickness_Hmean #wrong
    Vc = ifft2(Vc_nd_ft) * mean_thickness_Hmean #wrong 

    plt.imshow(Sb.real, extent=[left_x, right_x, bottom_y, top_y])
    #plt.imshow(Ub.real, extent=[left_x, right_x, bottom_y, top_y])
    #plt.imshow(Vb.real, extent=[left_x, right_x, bottom_y, top_y])
    #plt.imshow(Sc.real, extent=[left_x, right_x, bottom_y, top_y])
    #plt.imshow(Uc.real, extent=[left_x, right_x, bottom_y, top_y])
    #plt.imshow(Vc.real, extent=[left_x, right_x, bottom_y, top_y])

####################################################################################################################################

# REAL-WORLD EXPERIMENT

else: # Run the real world experiment here

    # Load bedmachine and produce predicted surface profile from BedMachine bedrock
    filepath = "C:\\Users\\casha\\Documents\\Research-Courtney_PC\\Moulin Model\\Greenland_BedMacine\\244470242\\BedMachineGreenland-v5.nc"
    
    # Define the center in polarstereographic coords and the radius limits of the region you want to look at within BedMachine
    x_center = -14960    # [m], polarstereographic x
    y_center = -2116315     # [m], polarstereographic y # Lake 13 in the dataset
    radius = 9000          # [m], x and y extent from the center which defines the bounding box

    # Run the read_bedmachine function to output corresponding data arrays
    (bedrock, bedrock_error, surface, thickness, x, y) = read_bedmachine(filepath, x_center, y_center, radius)

    # Get mean thickness and mean surface slope from bedmachine data
    h_bar = np.mean(thickness)
    print(h_bar)
    bedrock_corrected = bedrock - np.mean(bedrock)
    surface_slope_alpha = get_surface_slope(surface)
    slip_ratio_C = 10 ### STILL NEED TO GET A REAL VALUES FOR THIS

    # Non-dimensionalized cell-spacing
    cell_spacing = 150       # [m], resolution of the domain, 150 is the resolution of BedMachine
    cell_spacing_nd = cell_spacing/h_bar   # [unitless], non-dimensional cell-spacing

    # Non-dimensionalize bedrock topography
    bedrock_nd = bedrock_corrected/h_bar

    # Fourier transform non-dimensionalized bedrock topography
    bedrock_nd_ft = fft2(bedrock_nd) 

    # Getting k and l
    ar1 = np.fft.fftfreq(bedrock_nd_ft.shape[1], cell_spacing_nd)
    ar2 = np.fft.fftfreq(bedrock_nd_ft.shape[0], cell_spacing_nd)
    k,l = np.meshgrid(ar1,ar2)

    # Transfer function to the surface 
    Tsb = TSB(k, l, m, slip_ratio_C, surface_slope_alpha)   # Bed B to surface expression S
    
    # Because both k and l have zeros in them, the transfer functions return a NaN in the first [0,0] entry
    # due to division by 0. We set this value to 0 to avoid this issue.
    Tsb[0,0] = 0

    # Multiply transfer function output (fourier space) to bedrock and slip perturbations (also fourier space)
    Sb_nd_ft = Tsb * bedrock_nd_ft

    # Take the inverse fourier transform to go from spectral to x,y space and then redimensionalize
    Sb = ifft2(Sb_nd_ft) * h_bar + h_bar + np.mean(bedrock)

    #plt.imshow(Sb.real)
    #plt.imshow(surface - Sb.real, cmap="seismic")
    #plt.colorbar()
    plt.imshow(surface)
    #plt.imshow(bedrock)
# Then here the rest of the code will run as expected
