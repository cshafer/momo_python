#main.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from transfer_function import TSB, TUB, TVB, TSC, TUC, TVC # These are from Ockenden
from create_gaussian_perturbation import *
from read_bedmachine import *
from get_surface_slope import *
from mirror_pad_data import *
from calculate_A import *
from calculate_Ud import *
from calculate_eta import *
from get_GPS_data import *

m = 1       # [unitless], slip law exponent

# Choose between idealized experiment or a real world experiment.
# idealized == 0 ; real world == 1;
idealized_or_realworld = 1

if idealized_or_realworld == 0:
    
    # IDEALIZED EXPERIMENT

    ## |---------- 1.1 Ice properties setup --------| ##

    # Define properties of the ice
    ALPHA = 0.002                               # [unitless]    , surface slope of the ice sheet
    HMEAN = 1000                                # [m]           , mean thickness of the ice sheet, used in non-dimensionalization
    THETA = 90                                  # [unitless]    , angle between wave vector k_hat = (k,l) and x axis
    C = 100                                     # [unitless]    , mean non-dimensional slipperiness ub_bar/ud_bar
    #time_days_t = 0                            # []            , dimensional time
    #flow_law_coef_A = 0                        # []            , flow law coefficient in Glen's flow law
    RHOI = 917                                  # [kg m^-3]     , density of ice
    G = 9.80665                                 # [m s^-2], gravitational force
    TAU_D = RHOI * G * HMEAN * np.sin(ALPHA)    # [Pa or N m^-2], driving stress
    ETA = 0                                     # [kg m^-1 s^-1]    , effective viscosity

    ## |----------- 1.2 Domain setup ----------------| ##

    # Define properties of the domain 
    cell_spacing = 150                          # [m], resolution of the domain, 150 is the resolution of BedMachine
    x_window_length = 30000                     # [m], length of the domain in x
    y_window_length = 30000                     # [m], length of the domain in y

    # Non-dimensionalize the domain using mean thickness
    cell_spacing_nd = cell_spacing/HMEAN        # [unitless], non-dimensional cell-spacing
    x_window_length_nd = x_window_length/HMEAN  # [unitless], non-dimensional x domain
    y_window_length_nd = y_window_length/HMEAN  # [unitless], non-dimensional y domain
    
    # Create x-y grid using non-dimensional x and y window lengths
    left_x = -x_window_length_nd/2
    right_x = (x_window_length_nd/2) + cell_spacing_nd
    bottom_y = -y_window_length_nd/2
    top_y = (y_window_length_nd/2) + cell_spacing_nd
    [y,x] = np.mgrid[left_x:right_x:cell_spacing_nd, bottom_y:top_y:cell_spacing_nd]

    ## |---------- 1.3 Create Bedrock Perturbation ---------------| ##

    # Define the amplitude and the width of a single Gaussian bed bump
    bedrock_gaussian_amplitude = 40     # [unitless]    , amplitude of the Gaussian bedrock perturbation
    bedrock_gaussian_sigmax = 2000      # [m]           , width of the Gaussian in the x direction
    bedrock_gaussian_sigmay = 2000      # [m]           , width of the Guassian in the y direction

    # Non-dimensionalize width of Gaussian bed bump by dividing by Hmean
    bedrock_gaussian_sigmax_nd = bedrock_gaussian_sigmax/HMEAN   # [unitless]    , non-dimensional Gaussian width in x-direction
    bedrock_gaussian_sigmay_nd = bedrock_gaussian_sigmay/HMEAN   # [unitless]    , non-dimensional Gaussian width in y-direction

    # Create non-dimensional Gaussian bedrock perturbation delta b
    bedrock_topg_delta_b_nd = create_gaussian_perturbation(bedrock_gaussian_amplitude, bedrock_gaussian_sigmax_nd, bedrock_gaussian_sigmay_nd, x, y)

    ## |---------- 1.4 Create Slippery Patch Perturbation -----------| ##

    # Define the amplitude and width of a single Gaussian slippery patch
    slip_gaussian_amplitude_Ac = 10 # []
    slip_gaussian_sigmax = 2000     # []
    slip_gaussian_sigmay = 2000     # []

    # Non-dimensionalize width of Gaussian slippery patch by dividing by Hmean
    slip_gaussian_sigmax_nd = slip_gaussian_sigmax/HMEAN     # [unitless], non-dimensional Gaussian width in the x-direction
    slip_gaussian_sigmay_nd = slip_gaussian_sigmay/HMEAN     # [unitless], non-dimensional Gaussian width in the y-direction

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
    Tsb = TSB(k, l, m, C, ALPHA)   # Bed B to surface expression S
    Tub = TUB(k, l, m, C, ALPHA)   # Bed B to surface velocity in x direction U
    Tvb = TVB(k, l, m, C, ALPHA)   # Bed B to surface velocity in y direction V
    Tsc = TSC(k, l, m, C, ALPHA)   # Slip C to surface expression S
    Tuc = TUC(k, l, m, C, ALPHA)   # Slip C to surface velocity in x direction U
    Tvc = TVC(k, l, m, C, ALPHA)   # Slip C to surface velocity in y direction V

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
    Sb = ifft2(Sb_nd_ft) * HMEAN
    Ub = ifft2(Ub_nd_ft) * HMEAN #wrong I need to redimensionalize using mean velocity
    Vb = ifft2(Vb_nd_ft) * HMEAN #wrong
    Sc = ifft2(Sc_nd_ft) * HMEAN
    Uc = ifft2(Uc_nd_ft) * HMEAN #wrong
    Vc = ifft2(Vc_nd_ft) * HMEAN #wrong 

    plt.imshow(Sb.real, extent=[left_x, right_x, bottom_y, top_y])
    #plt.imshow(Ub.real, extent=[left_x, right_x, bottom_y, top_y])
    #plt.imshow(Vb.real, extent=[left_x, right_x, bottom_y, top_y])
    #plt.imshow(Sc.real, extent=[left_x, right_x, bottom_y, top_y])
    #plt.imshow(Uc.real, extent=[left_x, right_x, bottom_y, top_y])
    #plt.imshow(Vc.real, extent=[left_x, right_x, bottom_y, top_y])

####################################################################################################################################

# REAL-WORLD EXPERIMENT

else: # Run the real world experiment here

    # Constants
    m = 1
    n = 3
    rhoi = 917
    g = 9.80665
    secperyear = 365*24*60*60

    # Borehole temperature data at GULL
    tempGULL = [-0.6500, -7.7500, -11.2700, -11.9500, -14.1300, -13.5700, -12.7400, -11.6900, 
            -10.1100, -8.4900, -6.5500, -4.7400, -2.7300, -1.5200, -0.8300, -0.6000,
            -0.5600 , -0.4900, -0.5400, -0.4200, -0.4700, -0.3900, -0.5000]

    depthGULL = [4, 255, 307, 355, 407, 455, 497, 515, 537, 555, 577, 595, 622, 645, 667, 
                676, 687, 690, 697, 699, 702, 705, 707]

    # Load BedMachine dataset
    filepath = "C:\\Users\\casha\\Documents\\Research-Courtney_PC\\Moulin Model\\LAKE_DRAINAGE_CREVASSE_MODEL_JSTOCK\\BedMachineGreenland-2017-09-20.nc"
    
    # Choose the center in polarstereographic coords and the radius limits of the region you want to look at within BedMachine
    x_center = -184984    # [m], polarstereographic x
    y_center = -2241487     # [m], polarstereographic y
    radius = 6000       # [m], x and y extent from the center which defines the bounding box
    width = radius*2
    cell_spacing = 150

    # Run the read_bedmachine function to output corresponding data arrays
    (bedrock, bedrock_error, surface, thickness, x, y) = read_bedmachine(filepath, x_center, y_center, radius, cell_spacing)
    xm, ym = np.meshgrid(x,y)
    h_bar = np.mean(thickness)
    alpha = get_surface_slope(surface)

    # Calculate slip ratio C
    Tvector_pchip, Amean_pchip = calculate_A(h_bar, tempGULL, depthGULL, 'pchip')
    Ud_pchip = calculate_Ud(h_bar, Amean_pchip, rhoi, g, alpha, n)
    GPS2011_1h_GULL = get_GPS_data(2011, 1, 'GULL')
    v2011 = GPS2011_1h_GULL[4]
    Us = np.nanmean(v2011)
    Ub = Us - (Ud_pchip*secperyear)
    C = Ub/(Ud_pchip*secperyear)

    # Non-dimensionalize cell-spacing
    cell_spacing_nd = cell_spacing/h_bar   # [unitless], non-dimensional cell-spacing

    # Correct bed topography using mean and slope then non-dimensionalize
    bedrock_corr = bedrock - np.mean(bedrock)
    bedrock_corr_nd = bedrock_corr/h_bar
    bedrock_corr_nd_padded = mirror_pad_data(bedrock_corr_nd)
    bedrock_corr_nd_padded_ft = fft2(bedrock_corr_nd_padded)

    # Get k and l
    ar1_padded = np.fft.fftfreq(bedrock_corr_nd_padded_ft.shape[1], cell_spacing_nd)
    ar2_padded = np.fft.fftfreq(bedrock_corr_nd_padded_ft.shape[0], cell_spacing_nd)
    k_padded,l_padded = np.meshgrid(ar1_padded,ar2_padded)

    # Perform bedrock transfer functions to get surface outputs
    Tsb_padded = TSB(k_padded, l_padded, m, C, alpha)   # Bed B to surface expression S

    # Because both k and l have zeros in them, the transfer functions return a NaN in the first [0,0] entry
    # due to division by 0. We set this value to 0 to avoid this issue.
    Tsb_padded[0,0] = 0

    # Multiply transfer function output (fourier space) to bedrock and slip perturbations (which are also fourier space)
    Sb_padded_nd_ft = Tsb_padded * bedrock_corr_nd_padded_ft

    # Take the inverse fourier transform to go from spectral to x,y space and then redimensionalize
    Sb_padded = ifft2(Sb_padded_nd_ft) * h_bar + h_bar + np.mean(bedrock)
    Sb_padded = Sb_padded[int(width/cell_spacing):int(width/cell_spacing)*2 , int(width/cell_spacing):int(width/cell_spacing)*2] + xm * cell_spacing_nd * alpha

    difference_padded_slope = surface - Sb_padded.real

    #plt.imshow(Sb_slope.real)
    #plt.imshow(difference_slope, cmap="seismic")
    #plt.colorbar()
    #plt.imshow(surface)
    #plt.imshow(difference_padded_mean.real, cmap='seismic')
    #plt.colorbar()
    #plt.imshow(bedrock)
    #plt.plot(difference_padded_slope[60], color = 'r')
    #plt.plot(difference_padded_mean[60], color = 'b')
    #plt.axhline(y=0, color = 'r')
