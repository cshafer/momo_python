import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey 
from scipy.ndimage import rotate
from scipy.fft import fft2
from mirror_pad_data import *

def correct_and_transform_bedrock(bedrock, b_bar, h_bar, flow_dir):

    # Normalize bedrock around mean
    bedrock_norm = bedrock - b_bar

    # Non-dimensionalize bedrock using h_bar
    bedrock_non_d = bedrock_norm / h_bar

    # Add mirrored padding around bedrock
    bedrock_mirror_padded = mirror_pad_data(bedrock_non_d)

    # Create cosine window to taper padded bedrock
    cosine_window1d = np.abs(tukey(len(bedrock_mirror_padded)))  # tukey is a kind of window function within the scipy library
    cosine_window2d = np.sqrt(np.outer(cosine_window1d, cosine_window1d))

    bedrock_tapered = bedrock_mirror_padded * cosine_window2d

    # Add zero region to prep for rotation
    bedrock_zeros = np.pad(bedrock_tapered, pad_width = len(bedrock))  

    # Rotate bedrock so that the model's default flow direction matches actual ice flow direction
    bedrock_rotate = rotate(bedrock_zeros, 180 - flow_dir)

    # Fourier transform bedrock
    bedrock_fourier = fft2(bedrock_rotate)

    return bedrock_fourier

