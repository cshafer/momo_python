#main.py

import numpy as np
import matplotlib.pyplot as plt
from transfer_function import TSB, TUB, TVB, TSC, TUC, TVC # These are from Ockenden 

# Choose between idealized experiment or a real world experiment.
# idealized == 0 ; real world == 1;
idealized_or_realworld = 0

if idealized_or_realworld == 0:
    
    # Parameters for idealized experiment
    surface_slope_alpha = 0.002
    bedrock_topg_B = 0 ############### I need to create a sample bedrock base
    thickness_H = 0    ############### I need to create a thickness field
    slip_ratio_C = 100
    slip_patch_amplitude_Ac = 0    ### I need to think about how I'm going to incorporate the gaussian bumps
    slip_patch_x_width_sigmax = 0  ### 
    slip_patch_y_width_sigmay = 0
    flow_direction_theta = 90
    time_days_t = 0
    flow_law_coef_A = 0
    spacing_dx = 150 # meters, this is the resolution of BedMachine
    m = 1 # unitless, slip law exponenet

    # Bed to surface transfer function
    Tsb = TSB(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha) 
    Tub = TUB(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
    Tvb = TVB(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
    Tsc = TSC(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
    Tuc = TUC(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)
    Tvc = TVC(spacing_dx, flow_direction_theta, m, slip_ratio_C, surface_slope_alpha)

    # Choose specific values for the parameters that we need or update parameters within text file
else:
    # Here run the real world experiment
    test = 1


# Then here the rest of the code will run as expected
