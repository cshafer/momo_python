import numpy as np
import matplotlib.pyplot as plt



def create_gaussian_bed(amplitude_A, sigma_x, sigma_y, x_length_domain, y_length_domain, spacing):
    
    # Create domain
    [x,y] = np.mgrid[-x_length_domain/2:(x_length_domain/2) + 1:spacing, -y_length_domain/2:(y_length_domain/2) + 1:spacing]
    # Create the 2d gaussian grid for the bedrock
    gaussian_bed = amplitude_A * np.exp(-((np.power(x,2)/(2*sigma_x**2)) + (np.power(y,2)/(2*sigma_y**2))))
    return gaussian_bed


# testing
#B = create_gaussian_bed(40, 4, 8, 30, 30, 150/1000)
#print(B)
#plt.imshow(B, extent=[-30/2, 30/2, -30/2, 30/2], interpolation='none')
