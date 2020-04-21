from scipy.ndimage.morphology import generate_binary_structure, iterate_structure
from scipy.ndimage import grey_dilation
from scipy.stats import binned_statistic_2d
import numpy as np
import torch

def generate_image(combined_freqs, combined_deg, delta_nu, epsilon, star_numax):

    imgs = []
    
    dimx = 128
    dimy = 64
    degs = [0,1,2]# np.unique(combined_deg).astype(int) #[0,1]
    imgs_array = np.empty((dimx, dimy, len(degs)))
    for k, ell in enumerate(degs):
        nu = combined_freqs[combined_deg == ell]
        if len(nu) == 0:
            print('Empty Frequency Array for Degree %d !' %ell)
        reduced_freqs = (nu % delta_nu) - epsilon*delta_nu


        binx = np.linspace(-delta_nu, delta_nu, dimx+1).squeeze()
        biny = np.linspace(star_numax-7*delta_nu, star_numax+7*delta_nu, dimy+1).squeeze()
        reduced_freqs_extended = np.concatenate([reduced_freqs-delta_nu, reduced_freqs, delta_nu+reduced_freqs])
        nu_extended = np.concatenate([nu,nu,nu])
        nu_extended = nu_extended[(reduced_freqs_extended <=delta_nu)&(reduced_freqs_extended >= -delta_nu)]
        reduced_freqs_extended = reduced_freqs_extended[(reduced_freqs_extended <= delta_nu)&(reduced_freqs_extended >= -delta_nu)]
                
        img, xedge, yedge, binnumber = binned_statistic_2d(reduced_freqs_extended, nu_extended, None, 'count', bins=[binx,biny])
        
        st = generate_binary_structure(2, 2) # square expansion
        dilated = grey_dilation(img, footprint = iterate_structure(st,1), mode='constant') # arg 2 = repeat twice, 1=3x3; 2=5x5
        imgs_array[:,:,k] = dilated

        imgs +=[imgs_array]
    imgs = np.array(imgs)
    #     imgs = np.squeeze(imgs, 0) # if dipole only
    imgs = np.maximum(np.maximum(imgs[0], imgs[1]), imgs[2]) # if all spherical degs

    return imgs
