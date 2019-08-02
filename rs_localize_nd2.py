'''
rs_localize_nd2.py
'''
import click 
import numpy as np
from nd2reader import ND2Reader
from scipy.ndimage import uniform_filter
from scipy.optimize import minimize
from scipy.stats import chi2
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import time

@click.command()
@click.argument('nd2_file_or_dir', type = str)
@click.option('-w', '--window_size', type = int, default = 9, help = 'default 9 pixels')
@click.option('-p', '--psf_scale', type = float, default = 1.35, help = 'default 1.35')
@click.option('-y', '--wavelength', type = float, default = 0.664, help = 'default 0.664 um')
@click.option('-n', '--na', type = float, default = 1.49, help = 'default 1.49')
@click.option('-s', '--pixel_size_um', type = float, default = 0.16, help = 'default 0.16 um/pixel')
@click.option('-e', '--error_rate', type = float, default = 7.0, help = 'default 7.0')
@click.option('-r', '--rs_window_size', type = int, default = 9, help = 'default 9')
@click.option('-m', '--max_particles', type = int, default = 1000000, help = 'default 1000000 / file')
def localize(
	nd2_file_or_dir,
	window_size = 9,
	psf_scale = 1.35,
	wavelength = 0.664,
	na = 1.49,
	pixel_size_um = 0.16,
	error_rate = 7.0,
	rs_window_size = 9,
	max_particles = 1000000
):
	if os.path.isdir(nd2_file_or_dir):
		nd2_paths = ['%s/%s' % (nd2_file_or_dir, i) for i \
			in os.listdir(nd2_file_or_dir) if '.nd2' in i]
	elif os.path.isfile(nd2_file_or_dir):
		nd2_paths = [nd2_file_or_dir]
	else:
		print('Could not identify file/directory %s' % nd2_file_or_dir)
		exit(1)

	for nd2_path in nd2_paths:
		rs_localize_nd2(
			nd2_path,
			window_size = window_size,
			psf_scale = psf_scale,
			wavelength = wavelength,
			na = na,
			pixel_size_um = pixel_size_um,
			error_rate = error_rate,
			rs_window_size = rs_window_size,
			max_particles = max_particles
		)

def rs_localize_nd2(
	nd2_file,
	out_txt = None,
	window_size = 9,
	psf_scale = 1.35,
	wavelength = 0.664,
	na = 1.49,
	pixel_size_um = 0.16,
	error_rate = 7.0,
	rs_window_size = 9,
	max_particles = 1000000
):
	f = ND2Reader(nd2_file)

	particle_idx = 0
	frame_idx = 0
	result = np.zeros((max_particles, 10), dtype = 'double')
	while particle_idx < max_particles:
		try:
			image_2d = f.get_frame_2D(t = frame_idx).astype('double')
			detections = detect(
				image_2d,
				window_size = window_size,
				psf_scale = psf_scale,
				wavelength = wavelength,
				na = na,
				pixel_size_um = pixel_size_um,
				error_rate = error_rate,
			)
			fit_positions = rs_localize_frame(
				image_2d,
				rs_window_size,
				detections[:, :2].astype('uint16'),
			)
			n_fit = fit_positions.shape[0]
			result[particle_idx : particle_idx + n_fit, 1] = frame_idx 
			result[particle_idx : particle_idx + n_fit, 2:4] = fit_positions[:, :2]
			result[particle_idx : particle_idx + n_fit, 4] = detections[:, 2]
			result[particle_idx : particle_idx + n_fit, 5] = detections[:, 3]
			result[particle_idx : particle_idx + n_fit, 6] = fit_positions[:, 2]
			result[particle_idx : particle_idx + n_fit, 7:9] = detections[:, :2]
			result[particle_idx : particle_idx + n_fit, 9] = fit_positions[:, 3]

			frame_idx += 1
			particle_idx += n_fit 

			sys.stdout.write('Finished with %d frames...\r' % (frame_idx))
			sys.stdout.flush()
		except KeyError: # end of file
			break
	result = result[:particle_idx, :]
	result = result[~result[:,6].astype('bool'), :]
	result[:, 0] = np.arange(result.shape[0])

	if out_txt == None:
		out_txt = nd2_file.replace('.nd2', '_rs_locs.txt')
	print('Writing to %s...' % out_txt)
	result = pd.DataFrame(result, columns = [
		'particle_idx',             #0
		'frame_idx',                #1
		'y_coord_pixels',           #2
		'x_coord_pixels',           #3
		'alpha',                    #4
		'sig2',                     #5
		'result_ok',                #6
		'y_coord_pixels_detected',  #7
		'x_coord_pixels_detected',  #8
		'im_part_var',              #9
	])
	result['particle_idx'] = result['particle_idx'].astype('uint16')
	result['frame_idx'] = result['frame_idx'].astype('uint16')
	result['result_ok'] = result['result_ok'].astype('bool')
	result.to_csv(out_txt, sep = '\t', index = False)

	return result 

# Helper functions
def chi2inv(log_likelihood_threshold):
	def min_function(X):
		return (log_likelihood_threshold + np.log10(1-chi2.cdf(X,1)))**2
	return minimize(min_function, 20.0).x[0]

def local_max_2d(image):
	N, M = image.shape
	ref = image[1:N-1, 1:M-1]
	pos_max_h = (image[0:N-2, 1:M-1] < ref) & (image[2:N, 1:M-1] < ref)
	pos_max_v = (image[1:N-1, 0:M-2] < ref) & (image[1:N-1, 2:M] < ref)
	pos_max_135 = (image[0:N-2, 0:M-2] < ref) & (image[2:N, 2:M] < ref)
	pos_max_45 = (image[2:N, 0:M-2] < ref) & (image[0:N-2, 2:M] < ref)
	peaks = np.zeros((N, M))
	peaks[1:N-1, 1:M-1] = pos_max_h & pos_max_v & pos_max_135 & pos_max_45
	peaks = peaks * image
	return peaks

def gaussian_model(psf_std, window_size):
	refi = (0.5 + np.arange(0, window_size) - window_size / 2).astype('int8')
	ii = np.outer(refi, np.ones(window_size, dtype = 'int8'))
	jj = np.outer(np.ones(window_size, dtype = 'int8'), refi)
	g = np.exp(-((ii**2) + (jj**2)) / (2 * (psf_std**2))) / (np.sqrt(np.pi) * psf_std)
	return g

def expand_window(image_2d, N, M):
	N_in, M_in = image_2d.shape
	out = np.zeros((N, M))
	nc = np.floor(N/2 - N_in/2).astype(int)
	mc = np.floor(M/2 - M_in/2).astype(int)
	out[nc:nc+N_in, mc:mc+M_in] = image_2d
	return out

# Main detection / localization functions
def detect(
	image,
	window_size = 9,
	psf_scale = 1.35,
	wavelength = 0.664,
	na = 1.49,
	pixel_size_um = 0.16,
	error_rate = 7.0
):
	image = image.astype('double')
	N, M = image.shape
	psf_std = psf_scale * 0.55 * wavelength / (na * 1.17 * pixel_size_um * 2)
	pfa = chi2inv(error_rate)
	T = window_size ** 2
	hm = expand_window(
		np.ones((window_size, window_size)),
		N, M
	)
	tfhm = np.fft.fft2(hm)
	g = gaussian_model(psf_std, window_size)
	gc = g - g.sum() / T
	Sgc2 = (gc ** 2).sum()
	hgc = expand_window(gc, N, M)
	tfhgc = np.fft.fft2(hgc)
	tfim = np.fft.fft2(image)
	m0 = np.real(np.fft.fftshift(np.fft.ifft2(tfhm * tfim))) / T
	tfim2 = np.fft.fft2(image ** 2)
	Sim2 = np.real(np.fft.fftshift(np.fft.ifft2(tfhm * tfim2)))
	T_sig0_2 = Sim2 - T * (m0 ** 2)
	alpha = np.real(np.fft.fftshift(np.fft.ifft2(tfhgc * tfim))) / Sgc2
	test = 1 - (Sgc2 * (alpha ** 2)) / T_sig0_2
	test = (test > 0) * test + (test <= 0)
	peaks = -T * np.log(test)
	peaks[np.isnan(peaks)] = 0
	detections = peaks > pfa
	detect_pfa = local_max_2d(peaks).astype('bool') & detections
	detected_positions = np.asarray(np.nonzero(detect_pfa)).T
	n_detect = detected_positions.shape[0]
	
	# Adjust for the off-by-1 error
	detected_positions += 1

	sig1_2 = (T_sig0_2 - (alpha ** 2) * Sgc2) / T 
	result = np.zeros((detected_positions.shape[0], 4), dtype = 'double')
	result[:, :2] = detected_positions
	result[:, 2] = alpha[detected_positions[:, 0], detected_positions[:, 1]]
	result[:, 3] = sig1_2[detected_positions[:, 0], detected_positions[:, 1]]

	return result  #0: y_pos, 1: x_pos, 2: alpha, 3: sig2

def rs_localize(
	image_2d,
):
	'''
	INPUT
		image_2d		:	np.array, square
	
	'''
	#image_2d = image_2d.astype('double')
	time_0 = time.time()
	Ny, Nx = image_2d.shape
	Ny_half = int(Ny // 2)
	Nx_half = int(Nx // 2)
	ym_one_col = np.arange(Ny - 1) - Ny_half + 0.5
	xm_one_row = np.arange(Nx - 1) - Nx_half + 0.5
	ym = np.outer(ym_one_col, np.ones(Ny - 1))
	xm = np.outer(np.ones(Nx - 1), xm_one_row)

	dI_du = image_2d[:Ny-1, 1:] - image_2d[1:, :Nx-1]
	dI_dv = image_2d[:Ny-1, :Nx-1] - image_2d[1:, 1:]

	fdu = uniform_filter(dI_du, 3)
	fdv = uniform_filter(dI_dv, 3)

	dI2 = (fdu ** 2) + (fdv ** 2)
	m = -(fdv + fdu) / (fdu - fdv)
	m[np.isinf(m)] = 9e9

	b = ym - m * xm

	sdI2 = dI2.sum()
	ycentroid = (dI2 * ym).sum() / sdI2
	xcentroid = (dI2 * xm).sum() / sdI2
	w = dI2 / np.sqrt((xm - xcentroid)**2 + (ym - ycentroid)**2)

	# Correct nan / inf values
	w[np.isnan(m)] = 0
	b[np.isnan(m)] = 0
	m[np.isnan(m)] = 0

	# Least-squares analytical solution (equiv. to lsradialcenterfit)
	wm2p1 = w / ((m**2) + 1)
	sw = wm2p1.sum()
	smmw = ((m**2) * wm2p1).sum()
	smw = (m * wm2p1).sum()
	smbw = (m * b * wm2p1).sum()
	sbw = (b * wm2p1).sum()
	det = (smw ** 2) - (smmw * sw)
	xc = (smbw*sw - smw*sbw)/det
	yc = (smbw*smw - smmw*sbw)/det

	# Adjustment of coordinates
	yc = (yc + (Ny + 1) / 2.0) - 1
	xc = (xc + (Nx + 1) / 2.0) - 1

	fit_vector = np.array([yc, xc])
	time_1 = time.time()

	return fit_vector

def rs_localize_frame(
	full_image,
	window_size,
	positions,  #np.array of shape (N_points, 2), the starting positions for the fit
):
	full_image = full_image.astype('double')
	half_window = int(window_size // 2)
	fit_positions = np.zeros((positions.shape[0], 4), dtype = 'double')
	for pos_idx in range(positions.shape[0]):
		sub_image = full_image[ \
			positions[pos_idx, 0] - half_window : \
			positions[pos_idx, 0] + half_window + 1, \
			positions[pos_idx, 1] - half_window : \
			positions[pos_idx, 1] + half_window + 1, \
		]
		fit_positions[pos_idx, 3] = sub_image.var()
		try:
			fit_positions[pos_idx, :2] = rs_localize(sub_image) + \
				positions[pos_idx, :] - half_window
		except ValueError:  #particle on edge of frame
			fit_positions[pos_idx, 2] = 1  # flag for edge-of-frame
	return fit_positions #0: fit_y, 1: fit_x, 2: border_flag, 3: subimage variance

if __name__ == '__main__':
	localize()

