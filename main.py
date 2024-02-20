#main.py

import numpy as np
import matplotlib.pyplot as plt
from transfer_function import TSB, TUB, TVB, TSC, TUC, TVC # These are from Ockenden
from create_gaussian_bed import *

# Choose between idealized experiment or a real world experiment.
# idealized == 0 ; real world == 1;
idealized_or_realworld = 0

if idealized_or_realworld == 0:
    
    # IDEALIZED EXPERIMENT

    ## \------ 1.1 Domain setup ----------------\ ##

    # Define properties of domain
    cell_spacing_dx = 150               # [m], resolution of the domain, 150 is the resolution of BedMachine
    x_domain_length = 30000             # [m], length of the domain in x
    y_domain_length = 30000             # [m], length of the domain in y

    ## \------ 1.2 Ice properties setup --------\ ##

    # Define properties of the ice
    surface_slope_alpha = 0.002         # [unitless]    , surface slope of the ice sheet
    mean_thickness_Hmean = 1000         # [m]           , mean thickness of the ice sheet, used in non-dimensionalization
    flow_direction_theta = 90           # [unitless]    , angle between wave vector k_hat = (k,l) and x axis
    slip_ratio_C = 100                  # [unitless]    , mean non-dimensional slipperiness ub_bar/ud_bar
    time_days_t = 0                     # []            , dimensional time
    flow_law_coef_A = 0                 # [], flow law coefficient in Glen's flow law
    m = 1                               # [unitless]    , slip law exponent

    ## \---------- 1.3 Bedrock ---------------\ ##
    # Create idealized bed topography that contains single gaussian bedrock bump
    bedrock_gaussian_amplitude = 40     # [unitless]    , amplitude of the Gaussian bedrock perturbation
    bedrock_gaussian_sigmax = 2000      # [m]           , width of the Gaussian in the x direction
    bedrock_gaussian_sigmay = 2000      # [m]           , width of the Guassian in the y direction

    # Non-dimensionalizing
    x_domain_length_nd = x_domain_length/mean_thickness_Hmean                   # [unitless]    , non-dimensionalized x domain
    y_domain_length_nd = y_domain_length/mean_thickness_Hmean                   # [unitless]    , non-dimensionalized y domain
    bedrock_gaussian_sigmax_nd = bedrock_gaussian_sigmax/mean_thickness_Hmean   # [unitless]    , non-dimensinoalized Gaussian width in x-direction
    bedrock_gaussian_sigmay_nd = bedrock_gaussian_sigmay/mean_thickness_Hmean   # [unitless]    , non-dimensionalized Gaussian width in y-direction
    cell_spacing_dx_nd = cell_spacing_dx/mean_thickness_Hmean

    # Create non-dimensional Gaussian bedrock perturbation delta b
    bedrock_topg_delta_b = create_gaussian_bed(bedrock_gaussian_amplitude, bedrock_gaussian_sigmax_nd, bedrock_gaussian_sigmay_nd, x_domain_length_nd, y_domain_length_nd, cell_spacing_dx_nd)

    ## \------ 1.4 Slippery Patch ## -----------\ ##
    # Create idealized slippery patch
    slip_patch_amplitude_Ac = 0         # []
    slip_patch_x_width_sigmax = 1000    # []
    slip_patch_y_width_sigmay = 1000    # []   
    slippery_patch_delta_c = create_gaussian_slippery_patch()  # Need to write this function still

    ## \------ 1.5 Transform data to Fourier Space ------\ ##


    ## \------ 1.6 Transfer Functions ---------------\ ##

    # Bed topography to surface transfer functions
    Tsb = TSB(cell_spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha) 
    Tub = TUB(cell_spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
    Tvb = TVB(cell_spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
    
    # Slipperiness patch to surface transfer functions
    Tsc = TSC(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
    Tuc = TUC(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
    Tvc = TVC(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)

    # Multiply transfer function output (fourier space) to bedrock and slip perturbations (also fourier space)


else:
    # Here run the real world experiment
    test = 1


# Then here the rest of the code will run as expected
