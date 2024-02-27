import numpy as np

def get_surface_slope(surface):
    
    [slope_x, slope_y] = np.gradient(surface, 150) # Spacing here is 150 because BedMachine has resolution of 150m
    alpha_x = np.mean(slope_x)
    alpha_y = np.mean(slope_y)
    surface_slope_alpha = np.sqrt((alpha_x)**2 + (alpha_y)**2)

    return surface_slope_alpha