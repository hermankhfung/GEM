#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Find closest positive pixel in segmentation given refined particle coordinates

Search for segmentation (MRC format) in specified directory for each tomogram in STAR file.
Find closest positive pixel in segmentation for each refined particle coordinate
Output CSV tabulating particle and closest pixel coordinates and distance in Angstroms

Contact Rasmus K Jensen for the Python module star_tools for STAR file parsing

Copyright (C) 2023  EMBL/Herman Fung, EMBL/Julia Mahamid

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os, glob, re, csv
import numpy as np
import pandas as pd

from math import floor
from scipy.spatial.distance import cdist

import mrcfile
import star_tools.star_tools as st

import warnings

def fix_origins(star, pixel_size, origin_labels=['_rlnOriginX', '_rlnOriginY', '_rlnOriginZ']):
    for origin_label in origin_labels:
        if f'{origin_label}Angst' in star.columns:
            star = st._modCol(
                star, f'{origin_label}Angst', float(pixel_size), 'division')
            star = star.rename(columns={f'{origin_label}Angst': origin_label})
    return star

if __name__ == '__main__':
    
    starpath = '/(directory)/run_data.star'
    starpxsize = 6.85

    outcsv = '/(directory)/mito_gem_origin_dist.csv'

    segmemdir = '/(directory)/membrane_segmentation'

    segmempxsize_dict = {
        '211206' : 13.7,
        '220330' : 13.7,
        '220404' : 13.7,
        '220504' : 13.48,
        '220506' : 13.7,
        '220720' : 13.7
    }

    segmemflipZ_dict = {
        '211206' : False,
        '220330' : False,
        '220404' : False,
        '220504' : False,
        '220506' : False,
        '220720' : False
    }

    stardf = st._open_star(starpath, mode='new')  # Open star file
    stardf = fix_origins(stardf, starpxsize)

    tomoname_pattern = re.compile('(.*_TS_*[0-9_]+)((?:.mrc)+.tomostar)')

    neighbourlist = []

    for micrograph in stardf['_rlnMicrographName'].unique():
        idx = stardf.index[stardf['_rlnMicrographName'] == micrograph]
        tomodate = micrograph.split('_')[0]
        try:
            segmempxsize = segmempxsize_dict[tomodate]
            tomoname = tomoname_pattern.match(micrograph).group(1)
            matches = glob.glob(''.join([segmemdir,'/',tomoname,'*.mrc']))
            if matches == []:
                print(f'Segementation not found for {tomodate} {tomoname}')
            elif len(matches) > 1:
                print(f'Multiple possible segmentations found for {tomodate} {tomoname}')
            else:
                segmempath = matches[0]
                print(f'Segmentation found: {segmempath}')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with mrcfile.open(segmempath, mode='r', permissive=True) as mrc:
                        segmem = mrc.data
                nz,ny,nx = segmem.shape
                pd.set_option('display.max_columns', None)
                if segmem.min() == -128:
                    segmem = segmem + 128
                if segmem.max() != 1:
                    print(f'Segmentation not binary. Terminating...')
                    continue
                memb_coords = np.transpose(np.nonzero(segmem)) + 0.5  # all labelled pixels, take center of pixels
                print(f'Number of nonzero pixels: {memb_coords.size}')
                for k in idx:
                    x = (stardf['_rlnCoordinateX'][k] - stardf['_rlnOriginX'][k]) * starpxsize / segmempxsize
                    y = (stardf['_rlnCoordinateY'][k] - stardf['_rlnOriginY'][k]) * starpxsize / segmempxsize
                    z = (stardf['_rlnCoordinateZ'][k] - stardf['_rlnOriginZ'][k]) * starpxsize / segmempxsize
                    if segmemflipZ_dict[tomodate]:
                        z = nz - z
                    if memb_coords.size != 0:
                        distances = cdist(memb_coords,[[z,y,x]])
                        min_dist = np.amin(distances)
                        closest_px = memb_coords[np.argmin(distances)]
                    else:
                        min_dist = np.nan
                        closest_px = [np.nan, np.nan, np.nan]

                    neighbourlist.append([ micrograph, segmempxsize, x, y, z, closest_px[2], closest_px[1], closest_px[0], min_dist*segmempxsize ])
        except KeyError:
            pass

    with open(outcsv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['#_rlnMicrographName', 'PixelSize', 'gemCoordX', 'gemCoordY', 'gemCoordZ', 'membCoordX', 'membCoordY', 'membCoordZ','distAngst'])
        writer.writerows(neighbourlist)

