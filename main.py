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

    # Parameters for idealized experiment
    surface_slope_alpha = 0.002
    mean_thickness_Hmean = 1000
    slip_ratio_C = 100
    slip_patch_amplitude_Ac = 40  
    slip_patch_x_width_sigmax = 1000
    slip_patch_y_width_sigmay = 1000
    flow_direction_theta = 90
    time_days_t = 0
    flow_law_coef_A = 0
    spacing_dx = 150 # meters, this is the resolution of BedMachine
    m = 1 # unitless, slip law exponenet

    # Create idealized bed topography single gaussian bedrock bump
    bedrock_gaussian_amplitude = 40
    bedrock_gaussian_sigmax = 2000
    bedrock_gaussian_sigmay = 2000
    bedrock_topg_B = create_gaussian_bed(bedrock_gaussian_amplitude, bedrock_gaussian_sigmax, bedrock_gaussian_sigmay)

    # Create idealized surface

    # Create idealized slippery patch 

    # Bed topography to surface transfer functions
    Tsb = TSB(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha) 
    Tub = TUB(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
    Tvb = TVB(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)

    sB = Tsb * bedrock_topg_B
    
    # Slipperiness patch to surface transfer functions
   # Tsc = TSC(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
   # Tuc = TUC(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
   # Tvc = TVC(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)

else:
    # Here run the real world experiment
    test = 1


# Then here the rest of the code will run as expected
