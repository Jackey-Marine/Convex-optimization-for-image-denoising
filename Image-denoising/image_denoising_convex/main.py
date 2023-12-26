'''Main file for running experiments.'''

import time
import pickle
import cv2

import numpy as np
import matplotlib.pyplot as plt

from filters.quadratic_filter import quadratic_filter
from filters.TV_filter import TV_filter
from filters.TV_filter_pd import TV_filter_pd
from filters.non_local_means_filter import non_local_means_filter
from filters.non_local_wnnm_filter import non_local_wnnm_filter
from filters.exact_matrix_completion import complete_psd_symmetric, complete_matrix, mask_out_matrix
from utilities.utils import read_image, add_gaussian_noise, add_poisson_noise, add_salt_pepper_noise, PSNR, create_results_directory, normalize_image


def main(noise_type, plot=False, savefigs=True):
    '''Runs the various filters on the provided images with varying noise levels
       and saves the results.

    Args:
        noise_type (str): The type of noise, 'gaussian', 'poisson' or 'salt-pepper'.
        plot (bool): Whether to plot the noisy and cleaned images.
        savefigs (bool): Whether to save the generated images.
    '''

    np.random.seed(0)

    PSNR_results = {'quad': {}, 'TV': {}, 'nlm': {}, 'wnnm': {}, 'VMF': {}, 'MTM': {}, 'GF': {},}
    time_results = {'quad': {}, 'TV': {}, 'nlm': {}, 'wnnm': {}, 'VMF': {}, 'MTM': {}, 'GF': {},}

    # PSNR_results = {'quad': {}, 'TV': {}, 'nlm': {}, 'wnnm': {}, 'EMC': {}, 'VMF': {}, 'MTM': {}, 'GF': {},}
    # time_results = {'quad': {}, 'TV': {}, 'nlm': {}, 'wnnm': {}, 'EMC': {}, 'VMF': {}, 'MTM': {}, 'GF': {},}


    images = ['Insulators_1', 'Insulators_2', 'Insulators_3', 'Redlight_1', 'Redlight_2']

    if noise_type == 'gaussian':
        hyperparameters = [0.01, 0.025, 0.05]
    elif noise_type == 'poisson':
        hyperparameters = [50, 20, 10]
    elif noise_type == 'salt-pepper':
        hyperparameters = [0.0005, 0.001, 0.01]

    create_results_directory(noise_type, images, hyperparameters)

    for im_name in images:
        print('Now denoising image is ',im_name)
        for key in PSNR_results.keys():
            PSNR_results[key][im_name] = []
            time_results[key][im_name] = []

        for param in hyperparameters:
            str_var = str(param).replace('.', '_')

            im = read_image(im_name)
            if noise_type == 'gaussian':
                noisy_im = add_gaussian_noise(im, mean=0, var=param)
                variance = param
            elif noise_type == 'poisson':
                noisy_im = add_poisson_noise(im, photons=param)
                variance = 0.5/param
            elif noise_type == 'salt-pepper':
                noisy_im = add_salt_pepper_noise(im)
                variance = param
            print('Noise type: ', noise_type,)
            print('Noise parameters: ', param)

            start_time = time.time()
            # quad
            print('quad denoising...')
            quad_im = quadratic_filter(noisy_im, 5)
            quad_time = time.time()
            quad_im = normalize_image(quad_im)

            # TV
            print('TV denoising...')
            # TV_im = TV_filter(noisy_im, 0.3)
            TV_im = TV_filter_pd(noisy_im, 6)
            TV_time = time.time()
            TV_im = normalize_image(TV_im)

            # nlm 
            print('nlm denoising...')
            nlm_im = non_local_means_filter(noisy_im, 7, 10, 0.1)
            nlm_time = time.time()
            nlm_im = normalize_image(nlm_im)

            # wnnm
            print('wnnm denoising...')
            x = noisy_im
            y = noisy_im
            delta = 0.3
            for _ in range(1):
                y = x + delta*(noisy_im - y)
                x = non_local_wnnm_filter(y, 7, 10, variance)
                x = normalize_image(x)
            wnnm_im = x
            wnnm_time = time.time()

            # # EMC
            # print('EMC denoising...')
            # EMC_difference = cv2.absdiff(im, noisy_im)
            # EMC_threshold = 0.15
            # # Create a binary mask indicating noisy pixels
            # EMC_noisy_pixel_mask = EMC_difference > EMC_threshold
            # EMC_inverted_mask = ~EMC_noisy_pixel_mask
            # EMC_noisy_pixel_coordinates = np.where(EMC_inverted_mask)
            # EMC_coords = []
            # for row, col in zip(*EMC_noisy_pixel_coordinates):
            #     print(f"Row: {row}, Column: {col}")
            #     EMC_coords.append((row,col))
            # EMC_updated_coords = EMC_coords
            # EMC_im = complete_matrix(noisy_im, EMC_updated_coords)
            # EMC_time = time.time()

            # VMF 
            print('VMF denoising...')
            VMF_im = cv2.medianBlur(np.uint8(noisy_im * 255), ksize=3)
            VMF_im = VMF_im.astype(np.float32) / 255.0
            VMF_time = time.time()

            # MTM
            print('MTM denoising...')
            MTM_im = cv2.fastNlMeansDenoising(np.uint8(noisy_im * 255))
            MTM_im = MTM_im.astype(np.float32) / 255.0
            MTM_time = time.time()

            # GF
            print('GF denoising...')
            GF_im = cv2.GaussianBlur(np.uint8(noisy_im * 255), (3, 3), 0)
            GF_im = GF_im.astype(np.float32) / 255.0
            GF_time = time.time()


            # PSNR result calculation
            PSNR_results['quad'][im_name].append(PSNR(original_im=im, cleaned_im=quad_im))
            PSNR_results['TV'][im_name].append(PSNR(original_im=im, cleaned_im=TV_im))
            PSNR_results['nlm'][im_name].append(PSNR(original_im=im, cleaned_im=nlm_im))
            PSNR_results['wnnm'][im_name].append(PSNR(original_im=im, cleaned_im=wnnm_im))
            # PSNR_results['EMC'][im_name].append(PSNR(original_im=im, cleaned_im=EMC_im))
            PSNR_results['VMF'][im_name].append(PSNR(original_im=im, cleaned_im=VMF_im))
            PSNR_results['MTM'][im_name].append(PSNR(original_im=im, cleaned_im=MTM_im))
            PSNR_results['GF'][im_name].append(PSNR(original_im=im, cleaned_im=GF_im))

            # time result calculation
            time_results['quad'][im_name].append(quad_time - start_time)
            time_results['TV'][im_name].append(TV_time - quad_time)
            time_results['nlm'][im_name].append(nlm_time - TV_time)
            time_results['wnnm'][im_name].append(wnnm_time - nlm_time)
            # time_results['EMC'][im_name].append(EMC_time - wnnm_time)
            # time_results['VMF'][im_name].append(VMF_time - EMC_time)

            time_results['VMF'][im_name].append(VMF_time - wnnm_time)
            
            time_results['MTM'][im_name].append(MTM_time - VMF_time)
            time_results['GF'][im_name].append(GF_time - MTM_time)

            if plot is True or savefigs is True:
                _, ax_original = plt.subplots()
                ax_original.imshow(im, cmap='gray')
                ax_original.set_title('Original Image')

                fig_noisy, ax_noisy = plt.subplots()
                ax_noisy.imshow(noisy_im, cmap='gray')
                ax_noisy.set_title(f'Noisy Image, PSNR={round(PSNR(original_im=im, cleaned_im=noisy_im), 2)}')

                fig_quad, ax_quad = plt.subplots()
                ax_quad.imshow(quad_im, cmap='gray')
                ax_quad.set_title(f'Quadratic Image, PSNR={round(PSNR(original_im=im, cleaned_im=quad_im), 2)}')

                fig_tv, ax_tv = plt.subplots()
                ax_tv.imshow(TV_im, cmap='gray')
                ax_tv.set_title(f'TV Image, PSNR={round(PSNR(original_im=im, cleaned_im=TV_im), 2)}')

                fig_nlm, ax_nlm = plt.subplots()
                ax_nlm.imshow(nlm_im, cmap='gray')
                ax_nlm.set_title(f'Non-local means Image, PSNR={round(PSNR(original_im=im, cleaned_im=nlm_im), 2)}')

                fig_wnnm, ax_wnnm = plt.subplots()
                ax_wnnm.imshow(wnnm_im, cmap='gray')
                ax_wnnm.set_title(f'Weighted Nuclear Norm Minimization Image, PSNR={round(PSNR(original_im=im, cleaned_im=wnnm_im), 2)}')

                # fig_EMC, ax_EMC = plt.subplots()
                # ax_EMC.imshow(EMC_im, cmap='gray')
                # ax_EMC.set_title(f'Exact-matrix Completion Image, PSNR={round(PSNR(original_im=im, cleaned_im=EMC_im), 2)}')

                fig_VMF, ax_VMF = plt.subplots()
                ax_VMF.imshow(VMF_im, cmap='gray')
                ax_VMF.set_title(f'Medium Filtering Image, PSNR={round(PSNR(original_im=im, cleaned_im=VMF_im), 2)}')

                fig_MTM, ax_MTM = plt.subplots()
                ax_MTM.imshow(MTM_im, cmap='gray')
                ax_MTM.set_title(f'Mean Filtering Image, PSNR={round(PSNR(original_im=im, cleaned_im=MTM_im), 2)}')

                fig_GF, ax_GF = plt.subplots()
                ax_GF.imshow(GF_im, cmap='gray')
                ax_GF.set_title(f'Gaussian Filtering Image, PSNR={round(PSNR(original_im=im, cleaned_im=GF_im), 2)}')

                if savefigs is True:
                    fig_noisy.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/noisy.png')
                    fig_quad.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/quad.png')
                    fig_tv.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/tv.png')
                    fig_nlm.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/nlm.png')
                    fig_wnnm.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/wnnm.png')
                    # fig_EMC.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/EMC.png')
                    fig_VMF.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/VMF.png')
                    fig_MTM.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/MTM.png')
                    fig_GF.savefig(f'./results/{noise_type}/{im_name}/var_{str_var}/GF.png')

                if plot is True:
                    plt.show()

    print('PSNR results: ',PSNR_results)
    print('Time results: ',time_results)
    with open(f'./results/{noise_type}/PSNR_results.pkl', 'wb') as f:
        pickle.dump(PSNR_results, f)
    with open(f'./results/{noise_type}/time_results.pkl', 'wb') as f:
        pickle.dump(time_results, f)


if __name__ == '__main__':
    # 'gaussian' 'poisson' 'salt-pepper'
    main('gaussian')
