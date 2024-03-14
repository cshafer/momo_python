import numpy as np

def create_gaussian_perturbation(A, sigma_x, sigma_y, theta, x, y):
    
    '''
    A:  Amplitude
    sigma_x: width of gaussian distribution in x direction
    sigma_y: width of gaussian distribution in y direction
    theta: rotation of distribution in counter-clockwise direction, 0 is no rotation
    x: x array mesh
    y: y array mesh

    '''

    # Define the coefficients for a general 2d gaussian distribution
    a = (np.cos(theta)**2/2*sigma_x**2) + (np.sin(theta)**2/2*sigma_y**2)
    b = (-np.sin(theta)*np.cos(theta)/2*sigma_x**2) + (np.sin(theta)*np.cos(theta)/2*sigma_y**2)
    c = (np.sin(theta)**2/2*sigma_x**2) + (np.cos(theta)**2/2*sigma_y**2)

    # Calculate the gaussian distribution
    gaussian_perturbation = A * np.exp(-(a*(np.power(x,2)) + 2*b*(x*y) + c*(np.power(y,2))))
    return gaussian_perturbation


# testing
#B = create_gaussian_bed(40, 4, 8, 30, 30, 150/1000)
#print(B)
#plt.imshow(B, extent=[-30/2, 30/2, -30/2, 30/2], interpolation='none')
