#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import pickle
import os
import sys
import glob
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import urllib.request
from datetime import datetime
import copy
from skimage import io as skio
from skimage import util as skutil
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)
from clustring_lr_gapfilling import SpatioTemporalGapFilling
from dtw_seg import DTWSegmentation
from pixel_weighting_Gap_filling import PixelWeightingGapFilling 
from ssrf_gap_filling import SSRF_GapFilling
from rasterio.transform import from_origin
import os
import numpy as np
import rasterio
from rasterio.mask import mask


# In[10]:


def extract_bitmask(value, bit, num_bits=1):
    """Extract specific bits from an integer value."""
    return (value.astype(np.uint16) >> bit) & ((1 << num_bits) - 1)

def generate_cloud_mask_modis(input_tif, output_tif):
    """Generate a cloud mask from the MODIS state_1km dataset."""
    with rasterio.open(input_tif) as src:
        state_data = src.read(1).astype(np.uint16)  # Ensure data is integer
        profile = src.profile

        # Extract cloud state bits (Bits 0-1)
        cloud_state = extract_bitmask(state_data, 0, 2)  # Extract bits 0-1
        internal_cloud = extract_bitmask(state_data, 10, 1)  # Extract bit 10
        adjacent_to_cloud = extract_bitmask(state_data, 13, 1)  # Extract bit 13

        # Cloudy pixels are where cloud state is 1 or 2, or internal cloud is set
        cloud_mask = (cloud_state == 1) | (cloud_state == 2) | (internal_cloud == 1) | (adjacent_to_cloud == 1)

        # Convert boolean mask to integer (1 for cloudy, 0 for clear)
        cloud_mask = cloud_mask.astype(np.uint8)

        # Update profile for the output file
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

        # Save mask to a new GeoTIFF file
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(cloud_mask, 1)

def generate_cloud_mask_landsat(input_tif, output_tif):
    """Generate a cloud mask from the Landsat QA_PIXEL dataset."""
    with rasterio.open(input_tif) as src:
        qa_data = src.read(1).astype(np.uint16)  # Ensure data is integer
        profile = src.profile

        # Extract relevant bits
        dilated_cloud = extract_bitmask(qa_data, 1, 1)  # Bit 1
        cirrus = extract_bitmask(qa_data, 2, 1)  # Bit 2
        cloud = extract_bitmask(qa_data, 3, 1)  # Bit 3
        cloud_shadow = extract_bitmask(qa_data, 4, 1)  # Bit 4
        snow = extract_bitmask(qa_data, 5, 1)  # Bit 5
        clear = extract_bitmask(qa_data, 6, 1)  # Bit 6

        # Cloudy pixels are where cloud, dilated cloud, cirrus, cloud shadow, or snow is set
        cloud_mask = (dilated_cloud == 1) | (cirrus == 1) | (cloud == 1) | (cloud_shadow == 1) | (snow == 1)

        # Convert boolean mask to integer (1 for cloudy, 0 for clear)
        cloud_mask = cloud_mask.astype(np.uint8)

        # Update profile for the output file
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

        # Save mask to a new GeoTIFF file
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(cloud_mask, 1)

def process_folder(input_folder, output_folder, sensor_type="landsat"):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith(".tif"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, f"cloud_mask_{file}")

            if sensor_type.lower() == "modis":
                generate_cloud_mask_modis(input_path, output_path)
            elif sensor_type.lower() == "landsat":
                generate_cloud_mask_landsat(input_path, output_path)

            print(f"Processed: {file} -> {output_path}")


# In[11]:


input_folder = "../../lustre/rspa-kyyj3mesntk/users/said.grich/Rs_data/Robot_data/landsat_state/"
output_folder = "../../lustre/rspa-kyyj3mesntk/users/said.grich/Rs_data/Robot_data/landsat_masks/"
process_folder(input_folder, output_folder)


# In[12]:


def find_tif_files(folder_path):
    # Use glob to recursively find all .tif files in the specified folder and subfolders
    tif_files = glob.glob(os.path.join(folder_path, '**', '*.tif'), recursive=True)
    return tif_files

def brighten(band):
    alpha=0.13
    beta=0
    return np.clip(alpha*band+beta, 0,255)


def read_tif_band(tif_path, band_index):

    with rasterio.open(tif_path) as dataset:
        # Read the specified band (band_index is 1-based)
        band = dataset.read(band_index)
    return band 

def plot_rgb_with_cloud_mask(data):
    for item in data:
        # Extract RGB bands and cloud mask
        B1 = item['B3']
        B4 = item['B2']
        B3 = item['B1']
        cloud_mask = item['cloud_mask']
        date = item['date']

        # Stack the RGB bands to create an RGB image
        rgb_image = np.stack([B1, B4, B3], axis=-1)

        # Plot the RGB image and the cloud mask overlay
        plt.figure(figsize=(15, 15))

        # Plot the RGB image
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title('RGB Image')

        # Plot the cloud mask overlay
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_image)
        plt.imshow(cloud_mask, cmap='jet', alpha=0.5)  # Overlay the cloud mask with transparency
        plt.title(f'RGB Image with Cloud Mask Overlay\nDate: {date}')


        plt.show()


# In[6]:


def save_multi_band(bands, filename, metadata, transform, crs):
    count = bands.shape[0]
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=bands.shape[1],
        width=bands.shape[2],
        count=count,
        dtype=bands.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(count):
            dst.write(bands[i], i + 1)

# Define a function to read a TIFF file and extract metadata
def read_tif_file(tif_file):
    with rasterio.open(tif_file) as src:
        metadata = src.profile
        bands = src.read()
    return bands, metadata

def read_landsat_files(masks_dir, bands_dir):
    """Read MODIS bands and cloud masks for each date and return as a sorted list of dictionaries."""
    data_list = []

    # Get all mask and band files
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.tif')]
    band_files = [f for f in os.listdir(bands_dir) if f.endswith('.tif')]

    # Create a dictionary mapping dates to mask files
    mask_dict = {}
    for file in mask_files:
        date_part = file.split('_')[6:9][0]+"-"+file.split('_')[6:9][1]+"-"+file.split('_')[6:9][2].split(".")[0]  # Extract date part (e.g., '2023-01-01')
        mask_dict[date_part] = os.path.join(masks_dir, file)

    # Process each band file and match it with the mask
    for file in band_files:
        date_part = file.split('_')[4:7][0]+"-"+file.split('_')[4:7][1]+"-"+file.split('_')[4:7][2].split(".")[0]  # Extract date part (e.g., '2023-01-01')
        date = datetime.strptime(date_part, '%Y-%m-%d')

        data_entry = {
            'date': date,
            'is_cloudy': False,
            'cloud_mask': None
        }

        # Read the cloud mask if available for the date
        if date_part in mask_dict:
            mask_file = mask_dict[date_part]
            with rasterio.open(mask_file) as src:
                cloud_mask = src.read(1)
                data_entry['cloud_mask'] = cloud_mask
                data_entry['is_cloudy'] = np.max(cloud_mask) == 1  # Check if there's any cloud (1)

        # Read the bands
        with rasterio.open(os.path.join(bands_dir, file)) as src:
            for band_index in range(1, src.count + 1):
                band_name = f'B{band_index}'
                data_entry[band_name] = src.read(band_index).astype(np.float32)

        data_list.append(data_entry)

    # Sort the list by the 'date' field
    data_list_sorted = sorted(data_list, key=lambda x: x['date'])

    return data_list_sorted


# In[44]:


masks_dir = '../../lustre/rspa-kyyj3mesntk/users/said.grich/Rs_data/Robot_data/landsat_masks/'
bands_dir = '../../lustre/rspa-kyyj3mesntk/users/said.grich/Rs_data/Robot_data/landsat_agdal/'
data_dict = read_landsat_files(masks_dir, bands_dir)


# In[8]:


data_dict = sorted(data_dict, key=lambda x: x['date'])


# In[45]:


plot_rgb_with_cloud_mask(data_dict)


# In[43]:


# def delete_landsat_images_by_date(tifs_folder, masks_folder, dates_to_delete):
#     """Delete Landsat images and their corresponding masks based on a list of dates."""
#     formatted_dates = [date.replace("-", "_") for date in dates_to_delete]    
#     for file in os.listdir(tifs_folder):
#         for date in formatted_dates:
#             if date in file and file.startswith("Landsat_agdal_data"):

#                 file_path = os.path.join(tifs_folder, file)
#                 mask_file = file.replace("Landsat_agdal_data", "cloud_mask_Landsat_agdal_quality")
#                 mask_path = os.path.join(masks_folder, mask_file)

#                 if os.path.exists(file_path):
#                     os.remove(file_path)
#                     print(f"Deleted: {file_path}")

#                 if os.path.exists(mask_path):
#                     os.remove(mask_path)
#                     print(f"Deleted: {mask_path}")

# # Example usage
# dates_to_delete = [
#     "2024-12-29", "2024-12-21", "2024-10-26", "2024-09-16", "2024-08-31", "2024-06-12",
#     "2024-05-19", "2024-05-11", "2024-04-01", "2024-02-05", "2024-01-20", "2023-11-25",
#     "2023-05-17", "2023-03-22", "2023-02-18", "2023-02-26", "2022-10-05", "2022-09-19", "2022-06-15"
# ]
# delete_landsat_images_by_date(bands_dir,masks_dir, dates_to_delete)


# In[48]:


rgb1=np.stack([data_dict[12]["B3"], data_dict[12]["B2"], data_dict[12]["B1"]], axis=-1)
plt.figure(figsize=(10, 10))  # Adjust the figsize to make the image larger
plt.imshow(rgb1)
plt.axis('off')  # Hide the axes
plt.grid(False)  # Hide the grid linesQ
plt.show()


# In[49]:


rgb1=np.stack([data_dict[12]["B3"], data_dict[12]["B2"], data_dict[12]["B1"]], axis=-1)
plt.figure(figsize=(10, 10))  # Adjust the figsize to make the image larger
plt.imshow(rgb1)
plt.imshow(data_dict[12]["cloud_mask"], cmap='jet', alpha=0.3)
plt.axis('off')  # Hide the axes
plt.grid(False)  # Hide the grid lines
plt.show()


# In[50]:


import numpy as np
import copy
from datetime import datetime

# Initialize dictionaries for each band and cloud masks
bands_dict = {band: {} for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']}
cloud_mask_dict = {}

# Deep copy the original data to avoid modifying it
data_dict_copy = copy.deepcopy(data_dict)

# Sort the data by date
data_dict_copy = sorted(data_dict_copy, key=lambda x: x['date'])

# Iterate over the sorted data
for item in data_dict_copy:
    # Extract the date from the 'date' field
    date = item['date']

    # Apply cloud mask to each band: set cloudy areas to NaN
    for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6',]:
        if band in item:
            masked_array = np.where(item['cloud_mask'], np.nan, item[band])
            bands_dict[band][date] = masked_array

    # Store the cloud mask for the date
    cloud_mask_dict[date] = item['cloud_mask']

# Combine the bands and cloud masks into a single dictionary
bands_dict_gap = {**bands_dict, "cloud_mask": cloud_mask_dict}

# Example: Print a summary to verify the contents
print("Sorted Data:")
for date, b2_data in bands_dict_gap["B2"].items():
    print(f"Date: {date}")


# In[52]:


gapFilling=SpatioTemporalGapFilling()
clr_results={}
for band in bands_dict_gap.keys():
    if band !="cloud_mask":
        filled_result = gapFilling.reconstruct_filled_stack_images(bands_dict_gap[band], 20)
        clr_results[band]=filled_result


# In[54]:


import os
import rasterio
import numpy as np
from datetime import datetime

def save_dict_as_multiband_tifs(data_dict, reference_tif, output_dir="./output/s2"):

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract metadata from the first reference TIFF
    with rasterio.open(reference_tif) as src:
        metadata = src.meta.copy()
        transform = src.transform
        crs = src.crs

    # Collect all unique dates from the data dictionary
    unique_dates = set(date for band_dict in data_dict.values() for date in band_dict.keys())

    # Iterate over each unique date
    for date in sorted(unique_dates):
        date_str = date.strftime('%Y%m%d')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"landsat_GP_{date_str}.tif")

        # Gather the arrays for all bands for the current date
        band_arrays = []
        for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']:
            if date in data_dict[band]:
                band_arrays.append(data_dict[band][date])
            else:
                print("nan")
                # If a band is missing for a given date, fill with NaNs
                reference_shape = list(data_dict.values())[0][date].shape
                band_arrays.append(np.full(reference_shape, np.nan, dtype=np.float32))

        # Convert the list of arrays to a 3D array (Bands, Height, Width)
        stacked_array = np.stack(band_arrays, axis=0)

        # Update metadata for multi-band TIFF
        metadata.update({
            "count": len(band_arrays),  # Number of bands
            "height": stacked_array.shape[1],
            "width": stacked_array.shape[2],
            "transform": transform,
            "crs": crs,
            "dtype": str(stacked_array.dtype)
        })

        # Save the multi-band TIFF
        with rasterio.open(output_path, 'w', **metadata) as dst:
            for i in range(stacked_array.shape[0]):
                dst.write(stacked_array[i], i + 1)  # Write each band

        print(f"Saved {output_path}")


# In[55]:


save_dict_as_multiband_tifs(clr_results,"../../lustre/rspa-kyyj3mesntk/users/said.grich/Rs_data/Robot_data/landsat_agdal/Landsat_agdal_data_LC08_2022_01_22.tif", output_dir="../../lustre/rspa-kyyj3mesntk/users/said.grich/Rs_data/Robot_data/landsat_gf/")

