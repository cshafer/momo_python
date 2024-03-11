import numpy as np

def pad_data(bedrock):
    bedrock_ud = np.flipud(bedrock)
    bedrock_lr = np.fliplr(bedrock)
    bedrock_corner = np.flipud(bedrock_lr)

    top = np.hstack((bedrock_corner, bedrock_ud, bedrock_corner))
    middle = np.hstack((bedrock_lr, bedrock, bedrock_lr))
    bottom = np.hstack((bedrock_corner, bedrock_ud, bedrock_corner))

    padded_bedrock = np.vstack((top, middle, bottom))

    return padded_bedrock