import numpy as np
import matplotlib.pyplot as plt

def create_gaussian_perturbation(A, sigma_x, sigma_y, x, y):
    
    # Create the 2d gaussian grid for the bedrock
    gaussian_perturbation = A * np.exp(-((np.power(x,2)/(2*sigma_x**2)) + (np.power(y,2)/(2*sigma_y**2))))
    return gaussian_perturbation


# testing
#B = create_gaussian_bed(40, 4, 8, 30, 30, 150/1000)
#print(B)
#plt.imshow(B, extent=[-30/2, 30/2, -30/2, 30/2], interpolation='none')
