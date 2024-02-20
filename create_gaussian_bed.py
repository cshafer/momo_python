import numpy as np
import matplotlib.pyplot as plt

def create_gaussian_bed(amplitude_A, sigma_x, sigma_y):
    
    # Create domain
    [x,y] = np.mgrid[-100:101:10, -100:101:10]
    # Create the 2d gaussian grid for the bedrock
    gaussian_bed = amplitude_A * np.exp(-((np.power(x,2)/(2*(sigma_x^2))) + (np.power(y,2)/(2*(sigma_y^2)))))
    return gaussian_bed

B = create_gaussian_bed(40, 2000, 2000)
print(B)
plt.imshow(B)
