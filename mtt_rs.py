'''
mtt_v2.py -- implementation of the multiple target tracing algorithm 
to localize and track fluorophores in single molecule microscopy
data.

THEORY: See supplemental materials of
Arnauld, Nicolas, Hervé & Didier. "Dynamic multiple-target tracing to
probe spatiotemporal cartography of cell membranes." Nature Methods
5, 687 - 694 (2008).

Updated 190723.

'''
__author__ = 'Alec Heckert'
import numpy as np 
import pyfftw
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift, ifftshift, rfft2, irfft2
from nd2reader import ND2Reader 
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, uniform_filter 
from scipy.stats import chi2 
import pandas as pd 
import sys 
import random
from scipy import io as sio 
from copy import copy 
import click
import os 
import time 
import matplotlib.pyplot as plt 
from munkres import Munkres
munkres_solver = Munkres()

@click.group()
def cli():
	'''
	Localization and tracking utilities for single molecule tracking
	data in Nikon ND2 files. For usage information on a particular
	command, run

		python mtt.py <command_name> --help

	'''
	pass

@cli.command()
@click.argument('nd2_file_or_directory', type = str)
@click.option('-m', '--mat_save_suffix', type = str, default = '_Tracked.mat', help = 'default _Tracked.mat')
@click.option('-w', '--window_size', type = int, default = 9, help = 'default 9 pixels')
@click.option('-e', '--error_rate', type = float, default = 7.0, help = 'default 7.0')
@click.option('-p', '--psf_scale', type = float, default = 1.35, help = 'default 1.35')
@click.option('-s', '--pixel_size_um', type = float, default = 0.16, help = 'default 0.16 um/pixel')
@click.option('-y', '--wavelength', type = float, default = 0.664, help = 'default 0.664 um')
@click.option('-r', '--frame_interval', type = float, default = 0.00748, help = 'default 0.00748 s')
@click.option('-n', '--na', type = float, default = 1.49, help = 'default 1.49')
@click.option('-d', '--d_max', type = float, default = 15.0, help = 'default 15.0 um^2 s^-1')
@click.option('-a', '--naive_d_bound', type = float, default = 0.1, help = 'default 0.1 um^2 s^-1')
@click.option('-f', '--search_exp_fac', type = int, default = 1, help = 'default 1')
@click.option('-b', '--max_blinks', type = int, default = 1, help = 'default 1')
@click.option('-i', '--min_int', type = float, default = 0.0, help = 'default 0.0')
def localize_and_track(
	nd2_file_or_directory,
	mat_save_suffix = '_Tracked.mat',
	window_size = 9,
	error_rate = 7.0,
	psf_scale = 1.35,
	pixel_size_um = 0.16,
	wavelength = 0.664,
	frame_interval = 0.00748,
	na = 1.49,
	d_max = 15.0,
	naive_d_bound = 0.1,
	search_exp_fac = 1,
	max_blinks = 2,
	min_int = 0.0
):
	'''
	Localize and track single molecules in an ND2 file or a
	directory of ND2 files. If given as a directory, automatically
	generates output files for each ND2 file in the directory.

	'''
	if os.path.isdir(nd2_file_or_directory):
		path_list = ['%s/%s' % (nd2_file_or_directory, i) for i in os.listdir(nd2_file_or_directory) if '.nd2' in i]
	elif os.path.isfile(nd2_file_or_directory):
		path_list = [nd2_file_or_directory]
	for nd2_path in path_list:
		loc_file = nd2_file.replace('.nd2', '_locs.txt')
		mat_save_file = nd2_file.replace('.nd2', mat_save_suffix)
		localizeND2file(
			nd2_path,
			outfile = loc_file,
			window_size = window_size,
			error_rate = error_rate,
			psf_scale = psf_scale,
			pixel_size_um = pixel_size_um,
			wavelength = wavelength,
			NA = na 
		)
		trajectories = track_locs(
			loc_file,
			chi2inv(error_rate),
			frame_interval,
			mat_save_file = mat_save_file,
			Dmax = d_max,
			naive_Dbound = naive_d_bound,
			searchExpFac = search_exp_fac,
			max_blinks = max_blinks,
			minInt = min_int,
			pixel_size_um = pixel_size_um
		)
		return trajectories 

@cli.command()
@click.argument('localization_txt', type = str)
@click.option('-e', '--error_rate', type = float, default = 7.0, help = 'default 7.0')
@click.option('-r', '--frame_interval', type = float, default = 0.00748, help = 'default 0.00748 s')
@click.option('-m', '--mat_save_suffix', type = str, default = '_Tracked.mat', help = 'default _Tracked.mat')
@click.option('-d', '--d_max', type = float, default = 15.0, help = 'default 15.0 um^2 s^-1')
@click.option('-a', '--naive_d_bound', type = float, default = 0.1, help = 'default 0.1 um^2 s^-1')
@click.option('-f', '--search_exp_fac', type = int, default = 1, help = 'default 1')
@click.option('-b', '--max_blinks', type = int, default = 1, help = 'default 1')
@click.option('-i', '--min_int', type = float, default = 0.0, help = 'default 0.0')
@click.option('-s', '--pixel_size_um', type = float, default = 0.16, help = 'default 0.16 um/pixel')
def track(
	localization_txt,
	error_rate,
	frame_interval,
	mat_save_suffix,
	d_max,
	naive_d_bound,
	search_exp_fac,
	max_blinks,
	min_int,
	pixel_size_um
):
	'''
	Reconnect localizations into trajectories. Here, localization_txt
	is a tab-delimited file with columns y_coord_pixels and x_coord_pixels.

	'''
	mat_save_file = localization_txt.replace('.txt', mat_save_suffix)
	trajectories = track_locs(
		localization_txt,
		chi2inv(error_rate),
		frame_interval,
		mat_save_file = mat_save_file,
		Dmax = d_max,
		naive_Dbound = naive_d_bound,
		searchExpFac = search_exp_fac,
		max_blinks = max_blinks,
		minInt = min_int,
		pixel_size_um = pixel_size_um
	)

@cli.command()
@click.argument('nd2_file', type = str)
@click.argument('frame_idx', type = int)
@click.option('-w', '--window_size', type = int, default = 9, help = 'default 9 pixels')
@click.option('-r', '--rs_window_size', type = int, default = 9, help = 'default 9 pixels')
@click.option('-k', '--rs_filter_kernel', type = float, default = 1.4, help = 'default 1.4')
@click.option('-e', '--error_rate', type = float, default = 7.0, help = 'default 7.0')
@click.option('-p', '--psf_scale', type = float, default = 1.35, help = 'default 1.35')
@click.option('-s', '--pixel_size_um', type = float, default = 0.16, help = 'default 0.16 um/pixel')
@click.option('-y', '--wavelength', type = float, default = 0.664, help = 'default 0.664 um')
@click.option('-n', '--na', type = float, default = 1.49, help = 'default 1.49')
@click.option('-l', '--radial_symmetry_weights', type = str, default = 'centroid_distance_and_gradient_magnitude')
def visualize_locs(
	nd2_file,
	frame_idx,
	window_size,
	rs_window_size,
	rs_filter_kernel,
	error_rate,
	psf_scale,
	pixel_size_um,
	wavelength,
	na,
	radial_symmetry_weights,
):
	'''
	Visualize localizations to check that the localization algorithm
	is working correctly.

	'''
	f = ND2Reader(nd2_file)
	image_2d = f.get_frame_2D(t = frame_idx).astype('double')
	locs = localize_single_frame(
		image_2d,
		window_size = window_size,
		rs_window_size = rs_window_size,
		rs_filter_kernel = rs_filter_kernel,
		error_rate = error_rate,
		psf_scale = psf_scale,
		pixel_size_um = pixel_size_um,
		wavelength = wavelength,
		na = na,
		radial_symmetry_weights = radial_symmetry_weights,
		plot_detection = True 
	)
	n_detect = locs.shape[0]
	if n_detect == 0:
		print('No localizations detected')
	else:
		# First, plot individual localizations
		rs_half = int(rs_window_size // 2)
		ax_len = int(np.ceil(np.sqrt(n_detect)))
		fig, ax = plt.subplots(ax_len, 2 * ax_len, figsize = (12, 6), squeeze = False)
		markersize = 60 / ax_len
		for detect_idx in range(n_detect):
			y_det, x_det = locs[detect_idx, 7:9].astype('uint16')
			im_part = image_2d[y_det - rs_half + 1: y_det + rs_half + 2, \
				x_det - rs_half + 1: x_det + rs_half + 2]
			y_ax = detect_idx % ax_len
			x_ax = int(detect_idx // ax_len)
			ax[y_ax, x_ax * 2].imshow(im_part.copy(), cmap = 'gray')
			ax[y_ax, x_ax * 2 + 1].imshow(im_part.copy(), cmap = 'gray')
			fit_y = locs[detect_idx, 2] - (y_det - rs_half)
			fit_x = locs[detect_idx, 3] - (x_det - rs_half)
			ax[y_ax, x_ax * 2 + 1].plot([fit_y], [fit_x], marker = '.', markersize = markersize, color = 'r')
		for j in range(ax_len):
			ax[0, 2 * j].set_title('Original', fontsize = 8)
			ax[0, 2 * j + 1].set_title('Fitted', fontsize = 8)
		plt.tight_layout()
		out_png = 'visualize_locs_%s_frame_%d.png' % (nd2_file.replace('.nd2', ''), frame_idx)
		plt.savefig(out_png, dpi = 800)
		plt.close()
		os.system('open %s' % out_png)

@cli.command()
@click.argument('nd2_file', type = str)
@click.option('-w', '--window_size', type = int, default = 9, help = 'default 9 pixels')
@click.option('-r', '--rs_window_size', type = int, default = 9, help = 'default 9 pixels')
@click.option('-k', '--rs_filter_kernel', type = float, default = 3, help = 'default 3 pixels')
@click.option('-e', '--error_rate', type = float, default = 7.0, help = 'default 7.0')
@click.option('-p', '--psf_scale', type = float, default = 1.35, help = 'default 1.35')
@click.option('-s', '--pixel_size_um', type = float, default = 0.16, help = 'default 0.16 um/pixel')
@click.option('-y', '--wavelength', type = float, default = 0.664, help = 'default 0.664 um')
@click.option('-n', '--na', type = float, default = 1.49, help = 'default 1.49')
@click.option('-l', '--radial_symmetry_weights', type = str, default = 'centroid_distance_and_gradient_magnitude')
def localize(
	nd2_file,
	out_txt = None,
	window_size = 9,
	rs_window_size = 9,
	rs_filter_kernel = 3,
	error_rate = 7.0,
	psf_scale = 1.35,
	pixel_size_um = 0.16,
	wavelength = 0.664,
	na = 1.49,
	radial_symmetry_weights = 'centroid_distance_and_gradient_magnitude'
):
	time_0 = time.time()
	result = localize_nd2(
		nd2_file,
		window_size = window_size,
		rs_window_size = rs_window_size,
		rs_filter_kernel = rs_filter_kernel,
		error_rate = error_rate,
		psf_scale = psf_scale,
		pixel_size_um = pixel_size_um,
		wavelength = wavelength,
		na = na,
		radial_symmetry_weights = radial_symmetry_weights
	)
	result = pd.DataFrame(result, columns = [
		'particle_idx',
		'frame_idx',
		'y_coord_pixels',
		'x_coord_pixels',
		'alpha',
		'sig2',
		'result_ok',
		'y_coord_pixels_detected',
		'x_coord_pixels_detected',
		'im_part_var',
	])
	if out_txt == None:
		out_txt = nd2_file.replace('.nd2', '_locs.txt')
	result.to_csv(out_txt, sep = '\t', index = False)

def localize_nd2(
	nd2_file,
	window_size = 9,
	rs_window_size = 9,
	rs_filter_kernel = 1.4,
	error_rate = 7.0,
	psf_scale = 1.35,
	pixel_size_um = 0.16,
	wavelength = 0.664,
	na = 1.49,
	max_particles = 100000,
	radial_symmetry_weights = 'centroid_distance_and_gradient_magnitude',
):
	f = ND2Reader(nd2_file)
	n_frames = 0
	while 1:
		try:
			image_2d = f.get_frame_2D(t = n_frames)
			n_frames += 1
		except KeyError:
			break 

	# Precomputation to increase speed
	image_2d = f.get_frame_2D(t = 0).astype('double')
	N_sample, M_sample = image_2d.shape
	sizes = np.array([128, 192, 256])   #good frame sizes for FFT
	test_N = (sizes / N_sample) > 1
	if test_N.any():
		N = sizes[np.nonzero(test_N)][0]
	else:
		N = N_sample
	test_M = (sizes / M_sample) > 1
	if test_M.any():
		M = sizes[np.nonzero(test_M)][0]
	else:
		M = M_sample

	print('sample shape: %d by %d' % (N_sample, M_sample))
	print('FFT shape: %d by %d' % (N, M))

	T = window_size ** 2
	pfa = chi2inv(error_rate)
	psf_std = psf_scale * 0.55 * wavelength / (na * 1.17 * pixel_size_um * 2)

	m_one = np.ones((window_size, window_size))
	hm = expand_window(m_one, N, M)
	tfhm = pyfftw.empty_aligned((N, M), dtype = 'complex128')
	tfhm[:] = fft2(hm)

	g = gaussian_model(psf_std, window_size)
	gc = g - g.sum() / T
	Sgc2 = (gc ** 2).sum()
	hgc = expand_window(gc, N, M)
	tfhgc = pyfftw.empty_aligned((N, M), dtype = 'complex128')
	tfhgc[:] = fft2(hgc)

	bord = np.ceil(rs_window_size // 2) + 2
	rs_half = int(rs_window_size // 2)
	sub_n = rs_window_size - 1
	y_field, x_field = np.mgrid[:sub_n, :sub_n]

	# Static typing for FFT
	image_2d_complex = pyfftw.empty_aligned((N, M), dtype = 'complex128')
	tfim = pyfftw.empty_aligned((N, M), dtype = 'complex128')
	tfim2 = pyfftw.empty_aligned((N, M), dtype = 'complex128')
	m0 = np.zeros((N, M), dtype = 'double')
	Sim2 = np.zeros((N, M), dtype = 'double')
	T_sig0_2 = np.zeros((N, M), dtype = 'double')
	alpha = np.zeros((N, M), dtype = 'double')

	frame_idx = 0
	particle_idx = 0
	result = np.zeros((max_particles, 10), dtype = 'float')

	# Format of result:
		# result[0]	:	particle_idx
		# result[1]	:	frame_idx
		# result[2]	:	y_coord_pixels (fitted)
		# result[3]	:	x_coord_pixels (fitted)
		# result[4]	:	alpha (intensity)
		# result[5]	:	Sig2 (variance of gaussian model)
		# result[6]	:	result_ok (inside borders)
		# result[7]	:	y_coord_pixels (detection)
		# result[8]	:	x_coord_pixels (detection)
		# result[0]	:	variance of local image window (used in tracking)

	while particle_idx < max_particles - 1:
		try:
			image_2d = f.get_frame_2D(t = frame_idx).astype('double')
			image_2d_complex[:N_sample, :M_sample] = image_2d

			#hypothesis 1: no particles in the window
			tfim[:] = fft2(image_2d_complex)
			m0[:] = np.real(fftshift(ifft2(tfhm * tfim))) / T
			tfim2[:] = fft2(image_2d_complex ** 2)
			Sim2[:] = np.real(fftshift(ifft2(tfhm * tfim2)))
			T_sig0_2[:] = Sim2 - T * (m0**2)

			#hypothesis 2: a particle in the center of the window
			alpha[:] = np.real(fftshift(ifft2(tfhgc * tfim))) / Sgc2

			#conduct test
			test = 1 - (Sgc2 * (alpha**2)) / T_sig0_2
			test = (test > 0) * test + (test <= 0)
			peaks = -T * np.log(test)
			peaks[np.isnan(peaks)] = 0
			detections = peaks > pfa

			# Take the peaks of the GLLR test as the initial guesses for
			#   the spot centers
			detect_pfa = local_max_2d(peaks).astype('bool') & detections
			detected_positions = np.asarray(np.nonzero(detect_pfa)).T
			n_detect = detected_positions.shape[0]
			if n_detect + particle_idx >= max_particles:
				n_detect = max_particles - particle_idx
				detected_positions = detected_positions[:n_detect, :]

			# Record into the result dataframe
			result[particle_idx : particle_idx + n_detect, 1] = frame_idx 
			result[particle_idx : particle_idx + n_detect, 7:9] = detected_positions
			alpha_detect = alpha[result[particle_idx : particle_idx + n_detect, 7].astype('uint16'), \
				result[particle_idx : particle_idx + n_detect, 8].astype('uint16')]
			sig1_2 = (T_sig0_2 - (alpha ** 2) * Sgc2) / T 
			sig2_detect = sig1_2[result[particle_idx : particle_idx + n_detect, 7].astype('uint16'), \
				result[particle_idx : particle_idx + n_detect, 8].astype('uint16')]
			result[particle_idx : particle_idx + n_detect, 4] = alpha_detect
			result[particle_idx : particle_idx + n_detect, 5] = sig2_detect

			# Discard particles too close to the borders
			result[particle_idx : particle_idx + n_detect, 6] = np.logical_and(
				(detected_positions > bord).all(axis = 1),
				np.logical_and(
					(detected_positions[:,0] < N_sample - bord),
					(detected_positions[:,1] < M_sample - bord)
				)
			)

			for detect_idx in range(n_detect):
				global_idx = particle_idx + detect_idx
				if (result[global_idx, 4] > 0.0) and (result[global_idx, 6] != 0):
					detect_y = int(result[global_idx, 7])
					detect_x = int(result[global_idx, 8])
					im_part = image_2d[detect_y - rs_half + 1 : detect_y + rs_half + 2, \
						detect_x - rs_half + 1: detect_x + rs_half + 2]
					result[global_idx, 9] = im_part.var()

					im_part -= (im_part.min() - 1) / 2
					#im_part = gaussian_filter(im_part, rs_filter_kernel)
					im_part = uniform_filter(im_part, rs_filter_kernel)
					
					cross_a = im_part[1:, 1:] - im_part[:sub_n, :sub_n]
					cross_b = im_part[1:, :sub_n] - im_part[:sub_n, 1:]
					m = (cross_a + cross_b) / (cross_a - cross_b)

					# Remove divide-by-zero errors
					m[np.isinf(m)] = 0

					I = (im_part[:sub_n, :sub_n] + im_part[1:, :sub_n] + \
						im_part[:sub_n, 1:] + im_part[1:, 1:]) / 4
					I_sum = I.sum()
					centroid_y = (I.sum(axis = 1) * np.arange(sub_n)).sum() / I_sum
					centroid_x = (I.sum(axis = 0) * np.arange(sub_n)).sum() / I_sum 
					d = np.sqrt(((y_field - centroid_y)**2 + (x_field - centroid_x)**2))
					w = (cross_a ** 2 + cross_b ** 2) / (d + 0.5)
					term_0 = w / ((m ** 2) + 1)

					term_1 = term_0 * m
					term_2 = term_1 * m
					term_3 = y_field - m * x_field
					term_4 = term_0 * term_3
					term_5 = term_1 * term_3

					term_0_sum = term_0.sum()
					term_1_sum = term_1.sum()
					term_2_sum = term_2.sum()
					term_4_sum = term_4.sum()
					term_5_sum = term_5.sum()

					D = (term_1_sum ** 2) - (term_2_sum * term_0_sum)

					xc = (term_5_sum * term_0_sum - term_1_sum * term_4_sum) / D
					yc = (term_5_sum * term_1_sum - term_2_sum * term_4_sum) / D

					xc = xc + (detect_y - rs_half + 0.5)
					yc = yc + (detect_x - rs_half + 0.5)

					result[global_idx, 2] = xc
					result[global_idx, 3] = yc 
					result[global_idx, 9] = im_part.var()

			frame_idx += 1
			particle_idx += n_detect
			if particle_idx == max_particles - 1:
				break
			sys.stdout.write('finished with %d/%d frames; %d particles localized...\r' % (frame_idx, n_frames, particle_idx))
			sys.stdout.flush()
		except KeyError as e2:  #end of file
			break
	result[:particle_idx, 0] = np.arange(particle_idx)
	result[particle_idx:, 0] = np.nan
	return result 

def localize_single_frame(
	image_2d,
	window_size = 9,
	rs_window_size = 9,
	rs_filter_kernel = 1.4,
	error_rate = 7.0,
	psf_scale = 1.35,
	pixel_size_um = 0.16,
	wavelength = 0.664,
	na = 1.49,
	max_particles = 100000,
	radial_symmetry_weights = 'centroid_distance_and_gradient_magnitude',
	plot_detection = False
):
	N_sample, M_sample = image_2d.shape
	sizes = np.array([128, 192, 256])   #good frame sizes for FFT
	test_N = (sizes / N_sample) > 1
	if test_N.any():
		N = sizes[np.nonzero(test_N)][0]
	else:
		N = N_sample
	test_M = (sizes / M_sample) > 1
	if test_M.any():
		M = sizes[np.nonzero(test_M)][0]
	else:
		M = M_sample

	T = window_size ** 2
	pfa = chi2inv(error_rate)
	psf_std = psf_scale * 0.55 * wavelength / (na * 1.17 * pixel_size_um * 2)

	m = np.ones((window_size, window_size))
	hm = expand_window(m, N, M)
	tfhm = pyfftw.empty_aligned((N, M), 'complex128')
	tfhm[:] = pyfftw.interfaces.numpy_fft.fft2(hm)

	g = gaussian_model(psf_std, window_size)
	gc = g - g.sum() / T
	Sgc2 = (gc ** 2).sum()
	hgc = expand_window(gc, N, M)

	tfhgc = pyfftw.empty_aligned((N, M), 'complex128')
	tfhgc[:] = pyfftw.interfaces.numpy_fft.fft2(hgc)

	bord = int(rs_window_size // 2)
	rs_half = int(rs_window_size // 2)
	sub_n = rs_window_size - 1
	y_field, x_field = np.mgrid[:sub_n, :sub_n]

	frame_idx = 0
	particle_idx = 0
	result = np.zeros((max_particles, 10), dtype = 'float')

	# Some static typing to increase speed
	tfim = pyfftw.empty_aligned((N, M), 'complex128')
	tfim2 = pyfftw.empty_aligned((N, M), 'complex128')
	image_2d_complex = pyfftw.empty_aligned((N, M), 'complex128')
	image_2d_complex[:N_sample, :M_sample] = image_2d

	# Format of result:
		# result[0]	:	particle_idx
		# result[1]	:	frame_idx
		# result[2]	:	y_coord_pixels (fitted)
		# result[3]	:	x_coord_pixels (fitted)
		# result[4]	:	alpha (intensity)
		# result[5]	:	Sig2 (variance of gaussian model)
		# result[6]	:	result_ok
		# result[7]	:	y_coord_pixels (detection)
		# result[8]	:	x_coord_pixels (detection)
		# result[9]	:	variance of local image window (used in tracking)

	#hypothesis 1: no particles in the window
	tfim[:] = fft2(image_2d_complex)
	m0 = np.real(fftshift(ifft2(tfhm * tfim))) / T	
	tfim2[:] = fft2(image_2d_complex ** 2)
	Sim2 = np.real(fftshift(ifft2(tfhm * tfim2)))
	T_sig0_2 = Sim2 - T * (m0**2)

	#hypothesis 2: a particle in the center of the window
	alpha = np.real(fftshift(ifft2(tfhgc * tfim))) / Sgc2

	#conduct test
	test = (1 - (Sgc2 * (alpha**2)) / T_sig0_2)[:N_sample, :M_sample]
	test = (test > 0) * test + (test <= 0)
	peaks = -T * np.log(test)
	peaks[np.isnan(peaks)] = 0
	detections = peaks > pfa

	# Take the peaks of the GLLR test as the initial guesses for
	#   the spot centers
	detect_pfa = local_max_2d(peaks).astype('bool') & detections
	detected_positions = np.asarray(np.nonzero(detect_pfa)).T
	n_detect = detected_positions.shape[0]

	if plot_detection:
		fig, ax = plt.subplots(1, 4, figsize = (12, 3))
		ax[0].imshow(image_2d, cmap = 'gray')
		ax[1].imshow(peaks, cmap = 'gray')
		ax[2].imshow(detections, cmap = 'gray')
		ax[3].imshow(detect_pfa, cmap = 'gray')
		plt.show(); plt.close()

	result = np.zeros((n_detect, 10), dtype = 'float')

	# Record into the result dataframe
	result[:, 0] = frame_idx 
	result[:, 7:9] = detected_positions
	alpha_detect = alpha[result[:, 7].astype('uint16'), \
		result[:, 8].astype('uint16')]
	sig1_2 = (T_sig0_2 - (alpha ** 2) * Sgc2) / T 
	sig2_detect = sig1_2[result[:, 7].astype('uint16'), \
		result[:, 8].astype('uint16')]
	result[:, 4] = alpha_detect
	result[:, 5] = sig2_detect

	# Discard particles too close to the borders
	result[:, 6] = np.logical_and(
		(detected_positions > bord).all(axis = 1),
		np.logical_and(
			(detected_positions[:,0] < N_sample - bord),
			(detected_positions[:,1] < M_sample - bord)
		)
	)

	for detect_idx in range(n_detect):
		if (result[detect_idx, 4] > 0.0) and (result[detect_idx, 6] != 0):
			detect_y = int(result[detect_idx, 7])
			detect_x = int(result[detect_idx, 8])
			im_part = image_2d[detect_y - rs_half + 1 : detect_y + rs_half + 2, \
				detect_x - rs_half + 1 : detect_x + rs_half + 2].copy()
			result[detect_idx, 9] = im_part.var()

			im_part -= (im_part.min() - 1) / 2
			#im_part = gaussian_filter(im_part, rs_filter_kernel)
			im_part = uniform_filter(im_part, rs_filter_kernel)
			
			cross_a = im_part[1:, 1:] - im_part[:sub_n, :sub_n]
			cross_b = im_part[1:, :sub_n] - im_part[:sub_n, 1:]

			m = (cross_a + cross_b) / (cross_a - cross_b)

			# Remove divide by zero errors
			m[np.isinf(m)] = 0

			I = (im_part[:sub_n, :sub_n] + im_part[1:, :sub_n] + \
				im_part[:sub_n, 1:] + im_part[1:, 1:]) / 4
			I_sum = I.sum()
			centroid_y = (I.sum(axis = 1) * np.arange(sub_n)).sum() / I_sum
			centroid_x = (I.sum(axis = 0) * np.arange(sub_n)).sum() / I_sum 
			d = np.sqrt(((y_field - centroid_y)**2 + (x_field - centroid_x)**2))
			w = (cross_a ** 2 + cross_b ** 2) / (d + 0.5)
			term_0 = w / ((m ** 2) + 1)

			term_1 = term_0 * m
			term_2 = term_1 * m
			term_3 = y_field - m * x_field
			term_4 = term_0 * term_3
			term_5 = term_1 * term_3

			term_0_sum = term_0.sum()
			term_1_sum = term_1.sum()
			term_2_sum = term_2.sum()
			term_4_sum = term_4.sum()
			term_5_sum = term_5.sum()

			D = (term_1_sum ** 2) - (term_2_sum * term_0_sum)

			xc = (term_5_sum * term_0_sum - term_1_sum * term_4_sum) / D
			yc = (term_5_sum * term_1_sum - term_2_sum * term_4_sum) / D

			xc = xc + (detect_y - rs_half + 0.5)
			yc = yc + (detect_x - rs_half + 0.5)

			result[detect_idx, 2] = xc
			result[detect_idx, 3] = yc 
			result[detect_idx, 9] = im_part.var()


	result[:, 0] = np.arange(n_detect)
	return result 

def expand_window(
	image_2d,
	N,
	M
):
	'''
	INPUT
		image_2d	:	np.array, image to be expanded
		N, M		:	int, the size to expand to

	RETURNS
		np.array
	'''
	N_in, M_in = image_2d.shape
	out = np.zeros((N, M))
	nc = np.floor(N/2 - N_in/2).astype(int)
	mc = np.floor(M/2 - M_in/2).astype(int)
	out[nc:nc+N_in, mc:mc+M_in] = image_2d
	return out

def gaussian_model(
	psf_std,
	window_size,
):
	'''
	INPUT
		psf_std		:	float, expected point spread function
						radius in um
		window_size	:	int

	RETURNS
		numpy.array (2D), a centered Gaussian
	'''
	refi = (0.5 + np.arange(0, window_size) - window_size / 2).astype('int8')
	ii = np.outer(refi, np.ones(window_size, dtype = 'int8'))
	jj = np.outer(np.ones(window_size, dtype = 'int8'), refi)
	g = np.exp(-((ii**2) + (jj**2)) / (2 * (psf_std**2))) / (np.sqrt(np.pi) * psf_std)
	return g

def local_max_2d(image):
	'''
	INPUT
		image	:	numpy.array (2D), a non-binary image

	RETURNS
		numpy.array (2D) of the same shape as *image*, a
			binary image with the local maxima
	'''
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

def chi2inv(log_likelihood_threshold):
	'''
	Numerically compute the inverse chi-squared cumulative distribution
	function.

	INPUT
		log_likelihood
	Input is assumed positive: e.g. 7.0 for a log-error threshold of 
	10^-7.0.
	'''
	def min_function(X):
		return (log_likelihood_threshold + np.log10(1-chi2.cdf(X,1)))**2
	return minimize(min_function, 20.0).x[0]

def connectedComponents(semigraph):
	'''
	Given a semigraph, divide it into smaller graphs with
	connected components.

	INPUT
		semigraph		:	numpy.array (2D)

	RETURNS
		list, list, list :
			list of numpy.array (2D), the subgraphs;
			list of list of int, the x-indices of the subgraphs
				in the coordinates of the original semigraph;
			list of list of int, the y-indices of the subgraphs
				in the coordinates of the original semigraph
	'''
	where_x, where_y = np.where(semigraph)
	vertices = [(where_x[i], where_y[i]) for i in range(len(where_x))]
	component_x_labels = []
	component_y_labels = []
	subgraphs = []
	while len(vertices) > 0:
		component = [random.choice(vertices)]
		current_x_labels, current_y_labels = [component[0][0]], [component[0][1]]
		vertices.remove(component[0])
		prev_len = 0
		current_len = 1
		while current_len != prev_len:
			component += [v for v in vertices if any(
				[(component[i][0] == v[0] or component[i][1] == v[1]) \
					for i in range(len(component))]
			)]
			prev_len = current_len
			current_len = len(component)
			for j in range(prev_len, current_len):
				if component[j][0] not in current_x_labels:
					current_x_labels.append(component[j][0])
				if component[j][1] not in current_y_labels:
					current_y_labels.append(component[j][1])
				vertices.remove(component[j])

		current_x_labels.sort()
		current_y_labels.sort()

		subgraphs.append(semigraph[current_x_labels, :][:, current_y_labels])
		component_x_labels.append(current_x_labels)
		component_y_labels.append(current_y_labels)

	return subgraphs, component_x_labels, component_y_labels

def permuteString(characters, current = ''):
	'''
	INPUT
		characters	:	list of str

	RETURNS
		list of str, all permutation of *characters*

	'''
	if len(characters) == 1:
		return ['%s%s' % (current, characters[0])]
	result = []
	for character in characters:
		result += permuteString(
			[j for j in characters if j != character],
			current = '%s%s' % (current, character)
		)
	return result

def generatePermutationMatrices(n):
	'''
	Generate all permutation matrices of order *n*.
	
	INPUT
		n	:	int

	RETURNS
		list of numpy.array (2D; m*m), the permutation matrices

	'''
	integers = [str(i) for i in range(n)]
	strings = permuteString(integers)
	result = []
	for string in strings:
		matrix = np.zeros((n, n), dtype = 'uint8')
		for i in range(len(string)):
			matrix[i, int(string[i])] = 1
		result.append(matrix)
	return result

def minimizeTrace(matrix):
	'''
	Find the left permutation matrix P such that P * matrix
	has a minimized trace (diagonal sum).

	INPUT
		matrix		:	numpy.array (2D), an M by N matrix

	RETURNS
		numpy.array (2D), an M by M permutation matrix

	'''
	permutations = generatePermutationMatrices(matrix.shape[0])
	current_value, current_idx = 0.0, 0
	for p_idx in range(len(permutations)):
		test_value = permutations[p_idx].dot(matrix).trace()
		if test_value < current_value:
			current_idx = p_idx
			current_value = test_value
	return permutations[current_idx]

class Trajectory(object):
	def __init__(
		self,
		positions = [],  #list of np.array([y_coord_pixels, x_coord_pixels])
		frames = [],   #list of frame indices
		active = True,
		N_blinks = 0,
		D = None
	):
		self.positions = positions
		self.frames = frames
		self.active = active
		self.N_blinks = N_blinks
		self.D = D 

def track_locs(
	localization_file,
	pfa,
	frame_interval,
	mat_save_file = None,
	Dmax = 20.0,
	naive_Dbound = 0.1,
	searchExpFac = 9,
	max_blinks = 2,
	minInt = 0.0,
	pixel_size_um = 0.16
):
	var_free = Dmax * 4 * frame_interval
	naive_var_bound = naive_Dbound * 4 * frame_interval

	locs = np.asarray(pd.read_csv(localization_file, sep = '\t'))

	#convert pixels to um
	locs[:, 2] = locs[:, 2] * pixel_size_um
	locs[:, 3] = locs[:, 3] * pixel_size_um

	N_frames = int(locs[:,1].max()) + 1
	active_trajectories = [Trajectory(
		positions = [loc[2:4]],
		frames = [0]) for loc in locs[locs[:,1] == 0]
	]
	all_finished_trajectories = []
	for frame_idx in range(1, N_frames):
		active_trajectories, finished_trajectories, adjacency_graph = \
			reconnectFrame(
				active_trajectories,
				locs,
				frame_idx,
				var_free,
				naive_var_bound,
				pfa,
				searchExpFac = searchExpFac,
				max_blinks = max_blinks,
				minInt = minInt
			)
		all_finished_trajectories += finished_trajectories
		sys.stdout.write('Tracked through %d/%d frames...\r' % (frame_idx+1, N_frames))
		sys.stdout.flush()

	print('Finished tracking %s' % localization_file)
	all_finished_trajectories += active_trajectories

	if mat_save_file:
		saveTrajectoriesAsMat(all_finished_trajectories, mat_save_file, frame_interval)

	return all_finished_trajectories

def saveTrajectoriesAsMat(
	trajectories,
	mat_file_name,
	frame_interval
):
	result_list = []
	for traj_idx in range(len(trajectories)):
		frames = trajectories[traj_idx].frames
		times = [i * frame_interval for i in frames]
		positions = [[i[0], i[1]] for i in trajectories[traj_idx].positions]
		result_list.append([positions, frames, times])
	result = {'trackedPar' : result_list}
	sio.savemat(mat_file_name, result)
	return result

def reconnectFrame(
	active_trajectories,
	locs,
	frame_idx,
	var_free,
	naive_var_bound,
	pfa,
	searchExpFac = 9,
	max_blinks = 2,
	minInt = 0.0
):
	'''
	Extend trajectories by looking for matching localizations in the next
	frame.

	INPUT
		active_trajectories	:	list of Trajectory objects
		locs				:	numpy.array (2D), all localizations
								from the movie
		frame_idx			:	int, the frame to look for locs in
		var_free			:	float, variance of displacements
								expected for particle diffusing at
								Dmax (in um^2)
		naive_var_bound		:	float, variance of displacements
								expected for a bound particle. Used
								if no prior info is available for 
								that trajectory. Good guess: 0.003 um^2,
								which corresponds to D(bound) = 0.1 um^2 s^-1 
								and frame rate = 7.48 ms.
		pfa					:	float, the threshold for detection in GLRT
		searchExpFac		:	float, modifier for squared search radius of each
								trajectory. If 9, then looks in a radius equal
								to 3 * sqrt(var_free).
		max_blinks			:	int, the maximum number of off frames tolerated
								before a trajectory is discarded
		minInt				:	float, grayvalue threshold for a localization
								to start a new trajectory

	RETURNS
		list of Trajectory, the new active trajectories with updated info;
		list of Trajectory, finished trajectories to be saved;
		numpy.array (2D), the adjacency matrix used for detection

	'''
	locs_in_frame = locs[locs[:,1] == frame_idx]
	N_traj = len(active_trajectories)
	N_loc = locs_in_frame.shape[0]
	original_traj_indices = np.arange(N_traj)
	original_loc_indices = np.arange(N_loc)

	#Build the adjacency matrix. adjacency_graph[traj_idx, loc_idx] = 1
	#if loc is within the search radius of traj, and 0 otherwise.
	adjacency_graph = np.zeros((N_traj, N_loc), dtype = 'uint8')
	for traj_idx in range(N_traj):
		traj = active_trajectories[traj_idx]
		search_radius_squared = searchExpFac * var_free * (1 + traj.N_blinks)
		adjacency_graph[traj_idx, np.where(((locs_in_frame[:,2:4] - \
			traj.positions[-1])**2).sum(axis = 1) <= search_radius_squared)] = 1

	#Set trajectories without a corresponding localization to blink
	criterion = adjacency_graph.sum(axis = 1) == 0
	for traj_idx in np.where(criterion)[0]:
		if active_trajectories[traj_idx].N_blinks > max_blinks:
			active_trajectories[traj_idx].active = False
		else:
			active_trajectories[traj_idx].N_blinks += 1

	#Consider localizations without a corresponding trajectory for 
	#new trajectories
	criterion = adjacency_graph.sum(axis = 0) == 0
	new_trajectories = []
	for loc_idx in np.where(criterion)[0]:
		log_likelihood = logLikelihoodReconnection(
			None,
			locs_in_frame,
			int(locs[loc_idx, 0]),
			var_free = var_free,
			naive_var_bound = naive_var_bound
		)
		if log_likelihood >= pfa and locs_in_frame[loc_idx, 4] > minInt:
			new_trajectories.append(
				Trajectory(
					positions = [locs_in_frame[loc_idx, 2:4]],
					frames = [frame_idx]
				)
			)

	#Break the remaining adjacency graph into smaller connected
	#components that are easier to solve
	subgraphs, traj_subgraph_labels, loc_subgraph_labels = \
		connectedComponents(adjacency_graph)

	for subgraph_idx in range(len(subgraphs)):
		#If the tracking is unambiguous, assign the localization to the trajectory
		if subgraphs[subgraph_idx].shape == (1, 1):
			traj_idx = traj_subgraph_labels[subgraph_idx][0]
			loc_idx = loc_subgraph_labels[subgraph_idx][0]
			active_trajectories[traj_idx].positions.append(
				locs_in_frame[loc_idx, 2:4]
			)
			active_trajectories[traj_idx].frames.append(frame_idx)
			active_trajectories[traj_idx].N_blinks = 0
		#If there are ambiguities, find the maximum likelihood scenario
		else:
			traj_indices = traj_subgraph_labels[subgraph_idx]
			loc_indices = loc_subgraph_labels[subgraph_idx]
			N = len(traj_indices)
			M = len(loc_indices)
			max_dim = max([N, M])
			LL = np.zeros((max_dim, max_dim))
			where_x, where_y = np.where(subgraphs[subgraph_idx])
			#likelihood of reconnection
			for i in where_x:
				for j in where_y:
					traj_idx = traj_indices[i]
					loc_idx = loc_indices[j]
					LL[i,j] = logLikelihoodReconnection(
						active_trajectories[traj_idx],
						locs,
						int(locs_in_frame[loc_idx, 0]),
						var_free,
						naive_var_bound
					)
			#likelihood of each localization starting its own trajectory
			for i in range(N, max_dim):
				for j in range(M):
					LL[i,j] = logLikelihoodReconnection(
						None,
						locs,
						int(locs_in_frame[loc_indices[j], 0]),
						var_free,
						naive_var_bound
					)
			#P = minimizeTrace(LL)
			assignments = np.asarray(munkres_solver.compute(LL), dtype = 'uint8')
			P = np.zeros((max_dim, max_dim), dtype = 'uint8')
			P[assignments[:,0], assignments[:,1]] = 1

			expanded_traj_indices = np.zeros(max_dim, dtype = 'int8')
			expanded_traj_indices[:N] = traj_indices
			expanded_traj_indices[N:] = -1
			reordered_traj_indices = P.dot(expanded_traj_indices)

			#update the trajectories by walking along the diagonal
			for i in range(max_dim):
				traj_idx = reordered_traj_indices[i]
				if i < M:  #corresponds to a localization
					loc_idx = loc_indices[i]
					if traj_idx == -1:  #no trajectory connected; start a new one
						new_trajectories.append(Trajectory(
							positions = [locs_in_frame[loc_idx, 2:4]],
							frames = [frame_idx]
						))
					else:  #link up trajectory with new localization
						active_trajectories[traj_idx].positions.append(
							locs_in_frame[loc_idx, 2:4]
						)
						active_trajectories[traj_idx].frames.append(frame_idx)
				else:	#does not correspond to a localization
					active_trajectories[traj_idx].N_blinks += 1

	#remove trajectories that have been in blink too long, and 
	#add new trajectories to active trajectory stack
	finished_trajs = [traj for traj in active_trajectories if traj.N_blinks > max_blinks]
	active_trajectories = [traj for traj in active_trajectories if traj.N_blinks <= max_blinks] \
		+ new_trajectories

	return active_trajectories, finished_trajs, adjacency_graph

def initializeTrajectories(locs):
	result = []
	first_frame_locs = locs[locs[:,1] == 0]
	for loc in first_frame_locs:
		result.append(Trajectory(
			positions = [loc[2:4]],
			frames = [0]
		))
	return result 

def logLikelihoodReconnection(
	trajectory,
	locs,
	particle_idx,
	var_free,
	naive_var_bound,
	intensity_law_weight = 0.9,
	diff_law_weight = 0.5,
	N_pixels = 81,
	T_off = -1,
):
	'''
	INPUT
		trajectory		:	class Trajectory object or *None*. If
							*None*, the log-likelihood of starting
							a new trajectory is returned 
		locs			:	np.array (2D) indexed to [particle_idx,
							attribute_idx], the array of all localizations
		particle_idx	:	int, index of the particle in *locs* to
							consider for reconnection to *trajectory*
		var_free		:	float, expected variance of radial
							displacements according to a particle
							diffusing with diffusion constant Dmax
							(pixels^2)
		naive_var_bound	:	float, the expected variance in um^2 for
							bound population diffusion. Used if no
							prior information for a trajectory
							is available. A good default value
							(corresponding to 7.48 ms frame interval
							and D_bound = 0.1 um^2 s^-1) is 0.003 um^2.
		intensity_law_weight	:	float in [0, 1],  weighting
							for the intensity likelihood calculation
		diff_law_weight	:	float in [0, 1], the expected bound
							fraction of the molecule
		N_pixels		:	int, the number of pixels in the detection
							window
		T_off			:	int, the *negative* expected inverse rate
							constant for the return-from-blink process
							in frames

	RETURNS
		float, the log-likelihood of reconnection
	'''

	#new trajectories
	if trajectory == None:
		LxH1 = -(N_pixels / 2) * np.log(locs[particle_idx, 5])
		LxH0 = -(N_pixels / 2) * np.log(locs[particle_idx, 9])
		return -2 * (LxH0 - LxH1)

	#return-from-blink likelihood
	if trajectory.N_blinks != 0:
		sigma_blink = -T_off / 3
		P_blink = (2 / (sigma_blink * np.sqrt(2 * np.pi))) * \
			np.exp(- (trajectory.N_blinks**2) / (2 * sigma_blink**2))
		L_blink = np.log(P_blink)
	else:
		L_blink = 0.0

	#intensity likelihood (bootstrapped from population mean)
	particle_intensity = locs[particle_idx, 4]
	intensity_mean = locs[:, 4].mean()
	intensity_var = locs[:, 4].var()
	if particle_intensity < intensity_mean:
		P_int_univ = 1.0 / intensity_mean
	else:
		P_int_univ = 0.0
	P_int_gaussian = (1 / (intensity_var * np.sqrt(2 * np.pi))) * \
		np.exp(-(particle_intensity - intensity_mean)**2 / (2 * intensity_var))

	L_intensity = np.log(intensity_law_weight * P_int_gaussian + \
		(1 - intensity_law_weight) * P_int_univ)

	#displacement likelihood (need to check how this is handled for length-1 trajectories)
	var_bound = np.asarray(trajectory.positions)[:,0].var()
	if var_bound == 0.0:
		var_bound = naive_var_bound
	if trajectory.N_blinks != 0:
		var_bound = var_bound * np.sqrt(1 + trajectory.N_blinks)
		var_free = var_free * np.sqrt(1 + trajectory.N_blinks)

	squared_displacement = (locs[particle_idx, 2] - trajectory.positions[-1][0])**2 + \
		(locs[particle_idx, 3] - trajectory.positions[-1][1])**2
	P_displacement_bound = (1 / (var_bound * np.sqrt(2 * np.pi))) * \
		np.exp(-squared_displacement / (2 * var_bound))
	P_displacement_free = (1 / (var_free * np.sqrt(2 * np.pi))) * \
		np.exp(-squared_displacement / (2 * var_free))
	L_displacement = np.log(diff_law_weight * P_displacement_bound + \
		(1 - diff_law_weight) * P_displacement_free)

	return L_blink + L_intensity + L_displacement

if __name__ == '__main__':
	cli()
