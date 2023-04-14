#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script reads GeoTIFF files each of which is for one spectral 
# band of a Sentinel-21 image patch in the BigEarthNet Archive.
# 
# The script is capable of reading either  all spectral bands of one patch 
# folder (-p option) or all bands for all patches (-r option).
# 
# After reading files, Sentinel-1 image patch values can be used as numpy array 
# for further purposes.
# 
# read_patch --help can be used to learn how to use this script.
#
# Date: 22 Dec 2020
# Version: 1.0.1
# Usage: read_patch.py [-h] [-p PATCH_FOLDER] [-r ROOT_FOLDER]

from __future__ import print_function
import argparse
import os
import json
import numpy as np
from scipy.ndimage import map_coordinates
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description='This script reads the BigEarthNet image patches')
parser.add_argument('-r1', '--root_folder_s1', dest='root_folder_s1',
                    help='root folder path contains multiple patch folders of BigEarthNet-S1')
parser.add_argument('-r2', '--root_folder_s2', dest='root_folder_s2',
                    help='root folder path contains multiple patch folders of BigEarthNet-S2')
parser.add_argument('-o', '--output', dest='output',
                    help='folder where the dataset will be stored')
parser.add_argument('-s', '--split', dest='split',
                    help='split', choices=["train", "test", "val"])

args = parser.parse_args()

# Checks the existence of patch folders and populate the list of patch folder paths
s1_patch_names = []
if args.root_folder_s1 and args.root_folder_s2:
    if not os.path.exists(args.root_folder_s1):
        print('ERROR: folder', args.root_folder_s1, 'does not exist')
        exit()
    elif not os.path.exists(args.root_folder_s2):
        print('ERROR: folder', args.root_folder_s2, 'does not exist')
        exit()
    else:
        if args.split == "train":
            patches_df = pd.read_csv("splits/train.csv", names=["S2", "S1"])
        elif args.split == "test":
            patches_df = pd.read_csv("splits/test.csv", names=["S2", "S1"])
        elif args.split == "val":
            patches_df = pd.read_csv("splits/val.csv", names=["S2", "S1"])
        s1_patch_names = patches_df.iloc[:,1].to_list()
else:
    print("ERROR: invalid arguments")

# Checks the existence of required python packages
gdal_existed = rasterio_existed = georasters_existed = False
try:
    import gdal
    gdal_existed = True
    print('INFO: GDAL package will be used to read GeoTIFF files')
except ImportError:
    try:
        import rasterio
        rasterio_existed = True
        print('INFO: rasterio package will be used to read GeoTIFF files')
    except ImportError:
        print('ERROR: please install either GDAL or rasterio package to read GeoTIFF files')
        exit()

# radar and spectral band names to read related GeoTIFF files
band_names_s1 = ['VV', 'VH']
band_names_s2 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']


def resize_interp(arr, dim):
    new_dims = []
    for original_length, new_length in zip(arr.shape, (dim,dim)):
        new_dims.append(np.linspace(0, original_length-1, new_length))

    coords = np.meshgrid(*new_dims, indexing='ij')
    arr_resized = map_coordinates(arr, coords)
    return arr_resized


def read_bands(patch_folder_path, patch_name, band_names):
    bands_data = []
    for band_name in band_names:
        # First finds related GeoTIFF path and reads values as an array
        band_path = os.path.join(
            patch_folder_path, patch_name + '_' + band_name + '.tif')
        if gdal_existed:
            band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
            raster_band = band_ds.GetRasterBand(1)
            band_data = raster_band.ReadAsArray()
        elif rasterio_existed:
            band_ds = rasterio.open(band_path)
            band_data = band_ds.read(1)
        # band_data keeps the values of band band_name for the patch patch_name
        # print('INFO: band', band_name, 'of patch', patch_name, 'is ready with size', band_data.shape)
        band_data_resized = resize_interp(band_data, 120)
        bands_data.append(band_data_resized)
    return bands_data


def convert_labels_to_19(original_labels, label_indices):
    label_conversion = label_indices['label_conversion']
    BigEarthNet_19_label_idx = {v: k for k, v in label_indices['BigEarthNet-19_labels'].items()}

    original_labels_multi_hot = np.zeros(len(label_indices['original_labels'].keys()), dtype=int)
    BigEarthNet_19_labels_multi_hot = np.zeros(len(label_conversion),dtype=int)

    for label in original_labels:
        original_labels_multi_hot[label_indices['original_labels'][label]] = 1

    for i in range(len(label_conversion)):
        BigEarthNet_19_labels_multi_hot[i] = (
                np.sum(original_labels_multi_hot[label_conversion[i]]) > 0
                ).astype(int)
    
    BigEarthNet_19_labels = []
    for i in np.where(BigEarthNet_19_labels_multi_hot == 1)[0]:
        BigEarthNet_19_labels.append(BigEarthNet_19_label_idx[i])

    BigEarthNet_19_label = BigEarthNet_19_labels[0] # Choice : only keep first label

    return BigEarthNet_19_label




# Reads spectral bands of all patches whose folder names are populated before
from datetime import datetime
begin_time = datetime.now()
print(f"{begin_time} - INFO: creating {args.split} datasest...")
nb_images = len(patches_df)
patches_df["image_path"] = None
patches_df["category"] = None
means = []
stds = []

for i, s1_patch_name in enumerate(s1_patch_names[20000:]):

    if i%100==0:
        print(f"Completion percentage:", round(i*100/nb_images, 2), "%")

    folder_s1 = os.path.join(args.root_folder_s1, s1_patch_name)

    # read S1 bands
    bands_s1 = read_bands(folder_s1, s1_patch_name, band_names_s1)

    labels_metadata_path = os.path.join(folder_s1, 
                            s1_patch_name + '_labels_metadata.json')
    try:
        # Reads labels_metadata json file 
        with open(labels_metadata_path, 'r') as f:
            labels_metadata = json.load(f)
    except:
        continue

    # get corresponding s2 patch
    s2_patch_name = labels_metadata["corresponding_s2_patch"]
    original_labels = labels_metadata["labels"]
    
    # read corresponding S2 bands
    folder_s2 = os.path.join(args.root_folder_s2, s2_patch_name)
    patch_folder_path = os.path.join(folder_s2, s2_patch_name)
    bands_s2 = read_bands(folder_s2, s2_patch_name, band_names_s2)

    # write .tif
    bands = np.array(bands_s2 + bands_s1)
    mm_patch_name = s2_patch_name.split("_")[0]+"_"+s1_patch_name
    target_folder = os.path.join(args.output, args.split)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    filename = os.path.join(target_folder, mm_patch_name+".tif")
    with rasterio.open(filename, "w", height=bands.shape[1], width=bands.shape[2], count=bands.shape[0], dtype=bands.dtype) as dst:
        dst.write(bands, range(1,15))

    # update csv
    patches_df.at[i,"image_path"] = filename
    with open("label_indices.json") as jsonfile:
        label_indices = json.load(jsonfile)
    BigEarthNet_19_label = convert_labels_to_19(original_labels, label_indices)
    patches_df.at[i,"category"] = BigEarthNet_19_label

    # updates stats
    mean, std = bands.mean(axis=(1,2)), bands.std(axis=(1,2))
    means.append(mean)
    stds.append(std)

patches_df.dropna(inplace=True)
patches_df.to_csv(f"{args.output}/{args.split}.csv")
end_time = datetime.now()
print(f"{end_time} - INFO: Dataset successfully created. Size: {len(patches_df)}. Dropped images: {nb_images-len(patches_df)}. Time elapsed: {end_time - begin_time}")

mean = sum(means)/len(patches_df)
std = sum(stds)/len(patches_df)
print(f"Means on {args.split} dataset:\n {mean}\n")
print(f"Stds on {args.split} dataset:\n {std}")