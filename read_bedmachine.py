import numpy as np
import netCDF4 as nc
import xarray as xr 


def read_bedmachine(filepath, x_center, y_center, radius):
    
    # Because bedmachine v5 has a resolution of 150m, we first round the given radius to the nearest 150m
    rounded_radius = radius - (radius % 150)

    # Get the x and y limits that we're interested in
    xmin = x_center - rounded_radius
    xmax = x_center + rounded_radius
    ymin = y_center - rounded_radius
    ymax = y_center + rounded_radius

    # Open the NetCDF file
    bedmachine = xr.open_dataset(filepath)

    # Select the region within bedmachine that we're interested in. This is our "box".
    box = bedmachine.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

    # Read the data
    bedrock = box.variables['bed'].values
    bedrock_error = box.variables['errbed'].values
    surface = box.variables['surface'].values
    thickness = box.variables['thickness'].values
    x = box.variables['x'].values
    y = box.variables['y'].values

    # Close the file
    bedmachine.close()

    return bedrock, bedrock_error, surface, thickness, x, y