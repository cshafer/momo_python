import numpy as np

def get_surface_slope(surface):
    
    # Flip the surface upside down so that the x and y axes are oriented in the positive directions

    #     Before        ->      After            ==    What's really happening
    #    ----->                ----->               
    # | [ n1  n2  n3 ]       | [ n7  n8  n9 ]             [ n1  n2  n3 ]
    # | [ n4  n5  n6 ]       | [ n4  n5  n6 ]         y ^ [ n4  n5  n6 ]
    # v [ n7  n8  n9 ]       v [ n1  n2  n3 ]           | [ n7  n8  n9 ]
    #                                                      ---> x

    ud_surface = np.flipud(surface)

    # Get the y and x slopes from the flipped surface. Numpy gradient calculates gradient 
    # in the column direction first (y) and then in the row direction second (x). The sign of
    # slope_y and slope_x tells you whether it is positive or negative in the positive x and y directions.
    # For example, a positive x and y slope means that it is sloping upward in the 1st quadrant. A negative x
    # and y slope is sloping upward in the 3rd quadrant. 

    [slope_y, slope_x] = np.gradient(ud_surface, 150) # Spacing here is 150 because BedMachine has resolution of 150m

    # Get average slope
    alpha_x = np.mean(slope_x)
    alpha_y = np.mean(slope_y)

    # Get magnitude of the average slope
    surface_slope_alpha = np.sqrt((alpha_x)**2 + (alpha_y)**2)

    # Calculate ascent direction from surface slopes. The angle points in the direction of upwards slope.
    ascent_dir_radians = np.arctan2(alpha_y, alpha_x)

    # Convert ascent direction into degrees
    ascent_dir_degrees = ascent_dir_radians * 180/np.pi

    # Add 180 degrees to flip the direction to get descent, or flow direction
    flow_dir_degrees = ascent_dir_degrees + 180


    return surface_slope_alpha, alpha_x, alpha_y, flow_dir_degrees