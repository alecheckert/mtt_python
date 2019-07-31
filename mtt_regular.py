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
from nd2reader import ND2Reader 
from scipy.optimize import minimize
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
@click.argument('nd2_file_or_directory', type = str)
@click.option('-w', '--window_size', type = int, default = 9, help = 'default 9 pixels')
@click.option('-e', '--error_rate', type = float, default = 7.0, help = 'default 7.0')
@click.option('-p', '--psf_scale', type = float, default = 1.35, help = 'default 1.35')
@click.option('-s', '--pixel_size_um', type = float, default = 0.16, help = 'default 0.16 um/pixel')
@click.option('-y', '--wavelength', type = float, default = 0.664, help = 'default 0.664 um')
@click.option('-n', '--na', type = float, default = 1.49, help = 'default 1.49')
def localize(
	nd2_file_or_directory,
	window_size,
	error_rate,
	psf_scale,
	pixel_size_um,
	wavelength,
	na
):
	'''
	Localize single molecules in an ND2 movie, saving the result
	as a tab-delimited file with the suffix '_locs.txt'. If run
	on a directory of ND2 files, runs localization on each file
	individually.

	'''
	if os.path.isdir(nd2_file_or_directory):
		path_list = ['%s/%s' % (nd2_file_or_directory, i) for i in os.listdir(nd2_file_or_directory) if '.nd2' in i]
	elif os.path.isfile(nd2_file_or_directory):
		path_list = [nd2_file_or_directory]
	for nd2_path in path_list:
		loc_file = nd2_path.replace('.nd2', '_locs.txt')
		try:
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
			print('Finished localizing %s' % nd2_path)
		except ValueError:
			print('Error localizing %s; aborting' % nd2_path)

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

@cli.command()
@click.argument('nd2_file', type = str)
@click.argument('frame_idx', type = int)
@click.option('-w', '--window_size', type = int, default = 9, help = 'default 9 pixels')
@click.option('-e', '--error_rate', type = float, default = 7.0, help = 'default 7.0')
@click.option('-p', '--psf_scale', type = float, default = 1.35, help = 'default 1.35')
@click.option('-s', '--pixel_size_um', type = float, default = 0.16, help = 'default 0.16 um/pixel')
@click.option('-y', '--wavelength', type = float, default = 0.664, help = 'default 0.664 um')
@click.option('-n', '--na', type = float, default = 1.49, help = 'default 1.49')
def visualize_locs(
	nd2_file,
	frame_idx,
	window_size,
	error_rate,
	psf_scale,
	pixel_size_um,
	wavelength,
	na
):
	'''
	Visualize localizations to check that the localization algorithm
	is working correctly.

	'''
	f = ND2Reader(nd2_file)
	image_2d = f.get_frame_2D(t = frame_idx)
	locs, detection_parameters = localize(
		image_2d,
		window_size = window_size,
		error_rate = error_rate,
		psf_scale = psf_scale,
		pixel_size_um = pixel_size_um,
		wavelength = wavelength,
		NA = na,
		plot = True
	)

def localizeND2file(
	nd2_file,
	outfile = None,
	window_size = 9,
	error_rate = 7.0,
	psf_scale = 1.35,
	pixel_size_um = 0.16,
	wavelength = 0.664,
	NA = 1.49
):
	time_0 = time.time()
	f = ND2Reader(nd2_file)
	N_frames = 0
	while 1:
		try:
			frame = f.get_frame_2D(t = N_frames)
			N_frames += 1
		except KeyError:
			break
	if not outfile:
		outfile = nd2_file.replace('.nd2', '_locs.txt')
	format_string = '%f\t%f\t%f\t%f\t%f\t%f\t%r\t%f\n'
	with open(outfile, 'w') as o:
		o.write('particle_idx\tframe_idx\ty_coord_pixels\tx_coord_pixels\talpha\tSig2\toffset\tr\tresult_ok\tim_part_var\n')
		particle_idx = 0
		for frame_idx in range(N_frames):
			image_2d = f.get_frame_2D(t = frame_idx)
			locs, detection_parameters = localize(
				image_2d,
				window_size = window_size,
				error_rate = error_rate,
				psf_scale = psf_scale,
				pixel_size_um = pixel_size_um,
				wavelength = wavelength,
				NA = NA
			)
			for loc in locs:
				o.write('%d\t' % particle_idx)
				o.write('%d\t' % frame_idx)
				o.write(format_string % loc)
				particle_idx += 1

			sys.stdout.write('Finished localizing %d/%d frames...\r' % (frame_idx + 1, N_frames))
			sys.stdout.flush()


def localize(
	image_2d,
	window_size = 9,
	error_rate = 7.0,   #log-likelihood threshold: 10^-error_rate
	psf_scale = 1.35,
	pixel_size_um = 0.16,
	wavelength = 0.664,  #um
	NA = 1.49,
	plot = False, 
):
	image_2d = image_2d.astype('double')
	psf_std = psf_scale * 0.55 * wavelength / (NA * 1.17 * pixel_size_um * 2)
	N, M = image_2d.shape
	T = window_size ** 2
	pfa = chi2inv(error_rate)

	#hypothesis 1: no particles in the window
	m = np.ones((window_size, window_size))
	hm = expand_window(m, N, M)
	tfhm = np.fft.fft2(hm)
	tfim = np.fft.fft2(image_2d)
	m0 = np.real(np.fft.fftshift(np.fft.ifft2(tfhm * tfim))) / T
	im2 = image_2d ** 2
	tfim2 = np.fft.fft2(im2)
	Sim2 = np.real(np.fft.fftshift(np.fft.ifft2(tfhm * tfim2)))
	T_sig0_2 = Sim2 - T * (m0**2)

	#hypothesis 2: a particle in the center of the window
	g = gaussianModel(psf_std, window_size)
	gc = g - g.sum() / T
	Sgc2 = (gc**2).sum()
	hgc = expand_window(gc, N, M)
	tfhgc = np.fft.fft2(hgc)
	alpha = np.real(np.fft.fftshift(np.fft.ifft2(tfhgc * tfim))) / Sgc2

	#conduct test
	test = 1 - (Sgc2 * (alpha**2)) / T_sig0_2
	test = (test > 0) * test + (test <= 0)
	peaks = -T * np.log(test)
	peaks[np.isnan(peaks)] = 0
	detections = peaks > pfa

	#record results
	result = []
	if detections.sum() == 0:
		detection_parameters = np.zeros(6)
		detect_pfa = np.zeros(detections.shape)
	else:
		detect_pfa = localMax2D(peaks).astype('bool') & detections
		di, dj = np.nonzero(detect_pfa)
		n_detect = len(di)
		alpha_detect = alpha[di, dj]

		sig1_2 = (T_sig0_2 - (alpha ** 2) * Sgc2) / T
		sig2_detect = sig1_2[di, dj]

		detection_parameters = np.zeros((n_detect, 7))
		detection_parameters[:,0] = np.arange(n_detect)
		detection_parameters[:,1] = di
		detection_parameters[:,2] = dj
		detection_parameters[:,3] = alpha_detect
		detection_parameters[:,4] = sig2_detect
		detection_parameters[:,5] = psf_std
		detection_parameters[:,6] = np.ones(n_detect)

		Nestime = 0
		lestime = np.zeros((n_detect, 7))
		#bord = np.ceil(window_size / 2)
		bord = np.ceil(window_size / 2) + 1

		for i in range(n_detect):
			test_bord = \
				(detection_parameters[i,1] < bord) | \
				(detection_parameters[i,1] > N - bord) | \
				(detection_parameters[i,2] < bord) | \
				(detection_parameters[i,2] > M - bord)
			if detection_parameters[i,3] > 0.0 and not test_bord:
				y_coord, x_coord, alpha, Sig2, offset, r, result_ok, naive_var = fitGaussian(
					image_2d,
					int(detection_parameters[i,1]),
					int(detection_parameters[i,2]),
					psf_std
				)
				result.append((y_coord, x_coord, alpha, Sig2, offset, r, result_ok, naive_var))

		if plot:
			ax_len = int(np.ceil(np.sqrt(n_detect)))
			half_window = int(window_size // 2)
			fig, ax = plt.subplots(ax_len, 2 * ax_len, figsize = (12, 6))
			for detect_idx in range(n_detect):
				y_det, x_det = detection_parameters[detect_idx, 1:3].astype('uint16')
				im_part = image_2d[y_det - half_window + 1 : y_det + half_window + 2, \
					x_det - half_window + 1 : x_det + half_window + 2]
				y_ax = detect_idx % ax_len
				x_ax = int(detect_idx // ax_len)
				ax[y_ax, x_ax * 2].imshow(im_part.copy(), cmap = 'gray')
				ax[y_ax, x_ax * 2 + 1].imshow(im_part.copy(), cmap = 'gray')
				test_bord = \
					(detection_parameters[detect_idx,1] < bord) | \
					(detection_parameters[detect_idx,1] > N - bord) | \
					(detection_parameters[detect_idx,2] < bord) | \
					(detection_parameters[detect_idx,2] > M - bord)
				if detection_parameters[i, 3] > 0.0 and not test_bord:
					y_coord, x_coord, alpha, Sig2, offset, r, result_ok, naive_var = fitGaussian(
						image_2d,
						int(detection_parameters[detect_idx,1]),
						int(detection_parameters[detect_idx,2]),
						psf_std,
					)
					y_coord_subim = y_coord - (y_det - half_window)
					x_coord_subim = x_coord - (x_det - half_window)
					ax[y_ax, x_ax * 2 + 1].plot([copy(y_coord_subim)], [copy(x_coord_subim)], marker = '.', markersize = 15, color = 'r')
			for j in range(ax_len):
				ax[0,2 * j].set_title('Original', fontsize = 8)
				ax[0,2 * j + 1].set_title('Fitted', fontsize = 8)
			plt.tight_layout()
			plt.savefig('subpixel_localization.png', dpi = 800)
			plt.close()
			os.system('open subpixel_localization.png')


	if plot:
		fig, ax = plt.subplots(2, 2, figsize = (6, 6))
		ax[0,0].imshow(image_2d, cmap = 'gray', vmax = image_2d.mean() + 4.5 * image_2d.std())
		ax[0,1].imshow(peaks, cmap = 'gray', vmax = peaks.mean() + 3.5 * peaks.std())
		ax[1,0].imshow(detections, cmap = 'gray')
		ax[1,1].imshow(detect_pfa, cmap = 'gray', vmax = detect_pfa.mean() + 3.5 * detect_pfa.std())
		ax[0,0].set_title('Original frame')
		ax[0,1].set_title('GLLR test')
		ax[1,0].set_title('GLLR test > PFA')
		ax[1,1].set_title('Detections')
		plt.tight_layout()
		plt.savefig('visualize_locs_detections.png', dpi = 800)
		plt.close(); os.system('open visualize_locs_detections.png')

	return result, detection_parameters

def fitGaussian(
	image,
	Pi,
	Pj,
	psf_std,
	window_size = 9,
	ITER_MAX = 50,
):
	half_window = int(np.floor(window_size / 2))
	im_part = image[Pi - half_window + 1: Pi + half_window + 2, \
		Pj - half_window + 1: Pj + half_window + 2]
	prec_rel = 0.01
	wn_i, wn_j = im_part.shape
	N = wn_i * wn_j
	refi = 0.5 + np.arange(wn_i) - wn_i / 2
	refj = 0.5 + np.arange(wn_j) - wn_j / 2

	def fitGaussianIter(
		r0,
		i0,
		j0,
		x, #the image
		sig2init,
		p_dr,
		p_di,
		p_dj,
		current_iter = 0,
		max_iter = 1000,
	):
		pp_r = r0 - p_dr
		pp_i = i0 - p_di
		pp_j = j0 - p_dj

		again = 1
		while again:
			i = refi - i0
			j = refj - j0
			#ii = np.outer(i, np.ones(window_size))
			#jj = np.outer(np.ones(window_size), j)
			ii = np.outer(i, np.ones(wn_j))
			jj = np.outer(np.ones(wn_i), j)
			iiii = ii ** 2
			jjjj = jj ** 2
			g = np.exp(-(1/(2 * r0**2)) * (iiii + jjjj)) / np.sqrt(np.pi) * r0
			gc = (g - g.sum() / N).astype('float64')
			Sgc2 = (gc**2).sum()
			g_div_sq_r0 = g / (r0**2)
			if Sgc2 != 0.0:
				alpha = (x * gc).astype('float64').sum() / Sgc2
				#alpha = (x * gc).sum() / Sgc2
			else:
				alpha = 0
			x_alphag = x - alpha * g
			m = x_alphag.sum() / N
			err = x_alphag - m
			sig2 = (err**2).sum() / N
			current_iter += 1
			if (sig2 > sig2init):
				p_di = p_di / 10.0
				p_dj = p_dj / 10.0
				i0 = pp_i + p_di
				j0 = pp_j + p_dj
				current_iter += 1
				if (max([abs(p_dr), abs(p_di), abs(p_dj)]) > prec_rel):
					n_r = r0
					n_i = i0  
					n_j = j0 
					dr = 0
					di = 0
					dj = 0
					return n_r, n_i, n_j, dr, di, dj, alpha, sig2, m 
			else:
				again = 0
			current_iter += 1
			if current_iter > max_iter:
				again = 0

		d_g_i0 = ii * g_div_sq_r0
		d_g_j0 = jj * g_div_sq_r0
		dd_g_i0 = (-1 + iiii/(r0**2)) * g_div_sq_r0
		dd_g_j0 = (-1 + jjjj/(r0**2)) * g_div_sq_r0

		d_J_i0 = alpha * (d_g_i0 * err).sum()
		d_J_j0 = alpha * (d_g_j0 * err).sum()

		dd_J_i0 = alpha * (dd_g_i0 * err).sum() - (alpha**2) * (dd_g_i0**2).sum()
		dd_J_j0 = alpha * (dd_g_j0 * err).sum() - (alpha**2) * (dd_g_j0**2).sum()

		dr = 0
		n_r = r0 
		di = -d_J_i0 / dd_J_i0
		dj = -d_J_j0 / dd_J_j0
		n_i = i0 + di
		n_j = j0 + dj

		return n_r, n_i, n_j, dr, di, dj, alpha, sig2, m 


	localization_flags = [-1.5, 1.5, -1.5, 1.5, psf_std - 50 * psf_std / 100, \
		psf_std + 50 * psf_std / 100]
	test = True 
	r = psf_std
	i = 0.0
	j = 0.0
	dr = 1
	di = 1
	dj = 1
	fin = 0.01
	sig2 = np.inf 
	cpt = 0

	while test:
		r, i, j, dr, di, dj, alpha, sig2, offset = \
			fitGaussianIter(r, i, j, im_part, sig2, dr, di, dj)
		cpt += 1
		test = max([abs(di), abs(dj)]) > fin 
		if cpt > ITER_MAX:
			test = False
		result_ok = not ((i < localization_flags[0]) or (i > localization_flags[1]) \
			or (j < localization_flags[2]) or (j > localization_flags[3]) or \
			(r < localization_flags[4]) or (r > localization_flags[5]))
		test = test and result_ok

	return (
		Pi + i, #y-coordinate
		Pj + j, #x-coordinate
		alpha,  #mean amplitude
		sig2,   #noise power
		offset, #background level
		r,      #r0
		result_ok,
		im_part.var()
	)

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

def gaussianModel(
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

def localMax2D(image):
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

	N_frames = locs[:,1].max() + 1
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
			locs[loc_idx, 0],
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
						locs_in_frame[loc_idx, 0],
						var_free,
						naive_var_bound
					)
			#likelihood of each localization starting its own trajectory
			for i in range(N, max_dim):
				for j in range(M):
					LL[i,j] = logLikelihoodReconnection(
						None,
						locs,
						locs_in_frame[loc_indices[j], 0],
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
	particle_intensity = locs[particle_idx, 3]
	intensity_mean = locs[:,4].mean()
	intensity_var = locs[:,4].var()
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
