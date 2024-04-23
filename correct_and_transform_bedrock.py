import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey 
from scipy.ndimage import rotate
from scipy.fft import fft2
from mirror_pad_data import *

def correct_and_transform_bedrock(bedrock, b_bar, h_bar, flow_dir):

    # Normalize bedrock around mean
    bedrock_norm = bedrock - b_bar
    plt.imshow(bedrock_norm)
    plt.show()

    # Non-dimensionalize bedrock using h_bar
    bedrock_non_d = bedrock_norm / h_bar
    plt.imshow(bedrock_non_d)
    plt.show()

    # Add mirrored padding around bedrock
    bedrock_mirror_padded = mirror_pad_data(bedrock_non_d)
    plt.imshow(bedrock_mirror_padded)
    plt.show()

    # Create cosine window to taper padded bedrock
    cosine_window1d = np.abs(tukey(len(bedrock_mirror_padded)))
    cosine_window2d = np.sqrt(np.outer(cosine_window1d, cosine_window1d))

    bedrock_tapered = bedrock_mirror_padded * cosine_window2d
    plt.imshow(bedrock_tapered)
    plt.show()

    # Add zero region to prep for rotation
    bedrock_zeros = np.pad(bedrock_tapered, pad_width = len(bedrock))
    plt.imshow(bedrock_zeros)
    plt.show()    

    # Rotate bedrock so that the model's default flow direction matches actual ice flow direction
    bedrock_rotate = rotate(bedrock_zeros, 180 - flow_dir)
    plt.imshow(bedrock_rotate)
    plt.show()

    # Fourier transform bedrock
    bedrock_fourier = fft2(bedrock_rotate)

    return bedrock_fourier

