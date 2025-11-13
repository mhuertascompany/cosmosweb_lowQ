'''
test matching the GZ CEERS classifications with the catalog
make stamps out of the rest of the classifications
'''

import pandas as pd
import os
import numpy as np
import re
# import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord

from PIL import Image
from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D
import os
from skimage.transform import resize

filter='f150w'

# path for classifications
class_dir = "/n03data/huertas/COSMOS-Web/zoobot"
class_name = "jwst-ceers-v0-5-aggregated-class-singlechoicequestionsonly.csv"

# path for catalog
cat_dir = "/n03data/huertas/CEERS/cats"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"

# directory for cutout images
image_dir = f'/n03data/huertas/CEERS/zoobot/stamps/{filter}'
file_loc = [os.path.join(image_dir,path) for path in os.listdir(image_dir)]
ids = np.array([int(re.findall(r'\d+',path)[1]) for path in os.listdir(image_dir)])

cla = pd.read_csv(os.path.join(class_dir,class_name))
cat = pd.read_csv(os.path.join(cat_dir,cat_name)).iloc[ids]


# cols = cla.columns

# for col in cols:
#     print(repr(col))
# col_names = ['RA_1','DEC_1']

N = len(cla)

# retrieve coordinates for comparison
cat_ra = np.round(cat['RA_1'].values,6)
cat_dec = np.round(cat['DEC_1'].values,6)
cla_ra = np.round(cla['RA'].values,6)
cla_dec = np.round(cla['Dec'].values,6)

c = SkyCoord(ra=cla_ra*u.degree, dec=cla_dec*u.degree)

catalog = SkyCoord(ra=cat_ra*u.degree, dec=cat_dec*u.degree)

idx, d2d, d3d = c.match_to_catalog_sky(catalog)

cat2class = -1*np.ones(len(ids)).astype(int)

mask = d2d<0.2*u.arcsec # to be smaller

# check if any catalog object is matched multiple times (successful match labelled by mask)
for i in range(N):
    if mask[i]:
        if cat2class[idx[i]] >= 0:
            prev_match = cat2class[idx[i]]
            if d2d[i] < d2d[prev_match]:
                cat2class[idx[i]] = i
                mask[prev_match] = False
            else: 
                mask[i] = False
        else:
            cat2class[idx[i]] = i
        
print("Total matches: %i"%np.sum(mask))



# convert columns in classifications into matched catalog

gz_answers = [
    't0_smooth_or_featured__features_or_disk',
    't0_smooth_or_featured__smooth',
    't0_smooth_or_featured__star_artifact_or_bad_zoom',
    't1_how_rounded_is_it__cigarshaped',
    't1_how_rounded_is_it__in_between',
    't1_how_rounded_is_it__completely_round',
    't2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk',
    't2_could_this_be_a_disk_viewed_edgeon__no_something_else',
    't3_edge_on_bulge_what_shape__boxy',
    't3_edge_on_bulge_what_shape__no_bulge',
    't3_edge_on_bulge_what_shape__rounded',
    't4_is_there_a_bar__no_bar',
    't4_is_there_a_bar__strong_bar',
    't4_is_there_a_bar__weak_bar',
    't5_is_there_any_spiral_arm_pattern__yes',
    't5_is_there_any_spiral_arm_pattern__no',
    't6_spiral_how_tightly_wound__loose',
    't6_spiral_how_tightly_wound__medium',
    't6_spiral_how_tightly_wound__tight',
    't7_how_many_spiral_arms_are_there__1',
    't7_how_many_spiral_arms_are_there__2',
    't7_how_many_spiral_arms_are_there__3',
    't7_how_many_spiral_arms_are_there__4',
    't7_how_many_spiral_arms_are_there__more_than_4',
    't7_how_many_spiral_arms_are_there__cant_tell',
    't8_not_edge_on_bulge__dominant',
    't8_not_edge_on_bulge__moderate',
    't8_not_edge_on_bulge__no_bulge',
    't8_not_edge_on_bulge__large',
    't8_not_edge_on_bulge__small',
    't11_is_the_galaxy_merging_or_disturbed__major_disturbance',
    't11_is_the_galaxy_merging_or_disturbed__merging',
    't11_is_the_galaxy_merging_or_disturbed__minor_disturbance',
    't11_is_the_galaxy_merging_or_disturbed__none',
    't12_are_there_any_obvious_bright_clumps__yes',
    't12_are_there_any_obvious_bright_clumps__no',
    't19_what_problem_do___e_with_the_image__nonstar_artifact',
    't19_what_problem_do___e_with_the_image__bad_image_zoom',
    't19_what_problem_do___e_with_the_image__star'
]

gz_counts = [a+'__count' for a in gz_answers]
match_catalog = cla.loc[mask, gz_counts]
match_catalog.columns = [col[:-7] for col in match_catalog.columns]

# The columns "id_str" and "file_loc" are required for finetuning. 
match_catalog['id_str'] = ids[idx[mask]]
match_catalog['file_loc'] = [file_loc[k] for k in idx[mask]]

# match_catalog.to_csv("bot/match_catalog_F200W.csv")

# for col in cols:
#     print(col)


radius = cla['flux_rad_0p50'].values
pointing = cla['which_nircam'].values.astype(int)

# convert a greyscale numpy array to [0,255] jpg image
def array2img(arr,clipped_percent=0):
    arr = np.arcsinh(arr)
    # max_val = np.percentile(arr,100-clipped_percent/2)
    # min_val = np.percentile(arr,clipped_percent/2)
    # arr = np.clip(arr,min_val,max_val)
    # arr = (arr-min_val)/(max_val-min_val)*255
    max = np.max(arr)
    min = np.min(arr)
    arr = (arr-min)/(max-min)*300.5-0.5
    arr = np.clip(arr,0.,255.)
    return Image.fromarray(arr.astype(np.uint8))

def zero_pix_fraction(img):
    zeros = np.sum(np.max(img,axis=0)==0.)+np.sum(np.max(img,axis=1)==0.)
    size = img.shape[0]
    return zeros/size

raw_img_dir = "/n03data/huertas/CEERS/images"

N_POINTINGS = 10
POINTING1 = [1,2,3,6]
POINTING2 = [4,5,7,8,9,10]

pointing_name = f"fullceers_ddta_{filter}_v0.51_30mas_sci.fits.gz"
hdul=fits.open(os.path.join(raw_img_dir,pointing_name))
added = 0
for i in range(N):
    if mask[i] == False:
        
        # "hlsp_ceers_jwst_nircam_nircam%i_f200w_dr0.5_i2d.fits"%n
       
        # hdul.info()
        hdr = hdul[0].header
        w = wcs.WCS(hdr)
        data = hdul[0].data

        ymax, xmax = data.shape
        pixels = w.wcs_world2pix([[cla_ra[i],cla_dec[i]]],0)
        pix_size = 0.031
        pix = pixels[0]
            

        size = 212*np.maximum(0.04*radius[i],0.1)
        up = int(pix[0]+size)
        down = up-size*2
        right = int(pix[1]+size)
        left = right-size*2
        if all([up<xmax,down>-1,right<ymax,left>-1]):   
                # cut = data[left:right,down:up]
            cut = Cutout2D(data,pix,wcs=w,size=size*2).data

            if zero_pix_fraction(cut)<0.1:
                resized_cut = resize(cut,output_shape=(424,424))

                image = array2img(resized_cut,clipped_percent=1.)

                image.save(f'/n03data/huertas/CEERS/zoobot/stamps/{filter}_training/{filter}_%i_a.jpg'%i)

                new_record = cla.loc[[i], gz_counts]
                new_record.columns = [col[:-7] for col in new_record.columns]
                new_record['id_str'] = 20000+i
                new_record['file_loc'] = f'/n03data/huertas/CEERS/zoobot/stamps/{filter}_training/{filter}_%i_a.jpg'%i

                match_catalog = pd.concat([match_catalog,new_record],ignore_index=True)
                added += 1

# save the matched catalog
match_catalog.to_csv(f"/n03data/huertas/CEERS/zoobot/match_catalog_{filter}.csv")
print(f"Successfully matched {np.sum(mask)}, additionally cut out {added}")