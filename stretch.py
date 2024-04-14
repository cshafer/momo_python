import numpy as np

def stretch(dataset, new_min, new_max):
  """Stretches a dataset defined on some range to a new larger range.

  Args:
    dataset: A NumPy array containing the dataset to be stretched.
    new_min: The new minimum value for the dataset.
    new_max: The new maximum value for the dataset.

  Returns:
    A NumPy array containing the stretched dataset.
  """

  old_min = np.min(dataset)
  old_max = np.max(dataset)

  # Calculate the scaling factor
  scaling_factor = (new_max - new_min) / (old_max - old_min)

  # Stretch the dataset
  stretched_dataset = scaling_factor * (dataset - old_min) + new_min

  return stretched_dataset

# Create a dataset.
depthGULL = [4, 255, 307, 355, 407, 455, 497, 515, 537, 555, 577, 595, 622, 645, 667, 
             676, 687, 690, 697, 699, 702, 705, 707]

Hmean = 1000

# Stretch the dataset to a new range.
stretched_dataset = stretch(depthGULL, depthGULL[0], Hmean)

# Print the stretched dataset.
print(stretched_dataset)