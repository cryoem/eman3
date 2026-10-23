#!/usr/bin/env python
#
# Author: Steven Ludtke  06/07/2023
# Copyright (c) 2023- Baylor College of Medicine
#
# This software is issued under a joint BSD/GNU license. You may use the
# source code in this file under either license. However, note that the
# complete EMAN2 and SPARX software packages have some GPL dependencies,
# so you are responsible for compliance with the licenses of these packages
# if you opt to use BSD licensing. The warranty disclaimer below holds
# in either instance.
#
# This complete copyright notice must be included in any revised version of the
# source code. Additional authorship citations may be added, but existing
# author citations must be preserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA
#
# Ported to pure Python (EMAN3 modules + JAX, no C++ dependencies)

import sys
import os
import time
import traceback

import numpy as np
import jax.numpy as jnp

from EMAN3.EMAN3 import (EMArgumentParser, E3init, E3end, E3progress,
                         error_exit, warning, good_size, base_name, info_name)
from EMAN3.EMAN3jax import (jax_fft2d, jax_ift2d, jax_pointfilt_2d,
                            jax_downsample_2d, jax_phaseorigin2d, jax_set_device)
from EMAN3.io.imageio import ImageIO
from EMAN3.EMAN3jsondb import js_open_dict

EMANVERSION = "e3movie (EMAN3)"


# ---------------------------------------------------------------------------
# Low level helpers (numpy real-space image: shape (ny,nx))
# ---------------------------------------------------------------------------

def _read_range(f, a, b):
	"""Read frames a..b-1 from file f, return (numpy {N,Y,X}, headers list)."""
	io = ImageIO(f, "r")
	data, headers = io.read_images(list(range(a, b)))
	io.close()
	return np.asarray(data), headers


def _read_frame(f, i):
	"""Read a single frame i from file f, return (numpy (Y,X), header dict)."""
	io = ImageIO(f, "r")
	data, hdr = io.read_image(i)
	io.close()
	return np.asarray(data), hdr


def _fixgain_counting(img, gain, gainmin=2.0, gainmax=2.0):
	"""Port of EMAN2's "math.fixgain.counting" for a single (ny,nx) image.

	Divides the image by the gain image. Pixels whose gain falls outside the
	[gainmin, gainmax] window (expressed in units of the gain sigma about the
	gain mean, gainmax/gainmin = n*sigma, gainmin floored at 0.1) are considered
	bad sensor pixels: a good pixel is found by walking the diagonal toward the
	image center, and the (image/gain) ratio from that good pixel is substituted.
	"""
	img = np.asarray(img, dtype=np.float64)
	gain = np.asarray(gain, dtype=np.float64)
	ny, nx = gain.shape
	gmean = float(gain.mean())
	gsig = float(gain.std())
	hi = gmean + gainmax * gsig
	lo = max(0.1, gmean - gainmin * gsig)

	good = (gain >= lo) & (gain <= hi)
	out = np.zeros_like(img)
	with np.errstate(divide="ignore", invalid="ignore"):
		out[good] = img[good] / gain[good]

	# bad pixels: walk the diagonal toward the center to find a good gain value
	for (y, x) in np.argwhere(~good):
		dyy = -1 if y < ny / 2 else 1
		dxx = -1 if x < nx / 2 else 1
		found = False
		for i in range(1, max(nx, ny) // 2 + 1):
			yy, xx = y + dyy * i, x + dxx * i
			if 0 <= yy < ny and 0 <= xx < nx and lo <= gain[yy, xx] <= hi:
				out[y, x] = img[yy, xx] / gain[yy, xx]
				found = True
				break
		if not found:
			out[y, x] = gmean  # fallback to the gain mean
	return out


def _translate2(im, dx, dy):
	"""Zero-fill integer translate: content moves by +dx in x and +dy in y
	(out[p] = in[p - shift]), matching EMAN2's integer translate."""
	im = np.asarray(im)
	out = np.zeros_like(im)
	ny, nx = im.shape
	sx0, sx1 = max(0, -dx), min(nx, nx - dx)
	sy0, sy1 = max(0, -dy), min(ny, ny - dy)
	out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = im[sy0:sy1, sx0:sx1]
	return out


def _edgemean(img):
	"""Port of EMAN2's "normalize.edgemean": set the mean over a 3-pixel border
	around each edge to zero and normalize to unit standard deviation."""
	img = np.asarray(img, dtype=np.float64)
	ny, nx = img.shape
	edgemean = (img[0:ny - 3, 0:3].mean() +
	            img[ny - 3:ny, 0:nx - 3].mean() +
	            img[3:ny, nx - 3:nx].mean() +
	            img[0:3, 3:nx].mean()) / 4.0
	return (img - edgemean) / img.std()


def _center_crop(img, cx, cy):
	"""Centrally crop a single image or stack to (cy, cx), trimming both edges.
	If the requested size exceeds the image in an axis, that axis is left as-is."""
	img = np.asarray(img)
	ny, nx = img.shape[-2:]
	cy = min(cy, ny)
	cx = min(cx, nx)
	sy = (ny - cy) // 2
	sx = (nx - cx) // 2
	if img.ndim == 2:
		return img[sy:sy + cy, sx:sx + cx].copy()
	return img[:, sy:sy + cy, sx:sx + cx].copy()


def _resample(img, factor):
	"""Fourier-downsample a real-space image by a (possibly fractional) factor,
	producing an even-sized image. Analogous to EMAN2's "math.fft.resample n=factor"."""
	nx = img.shape[1]
	newsize = int(round(nx / factor))
	if newsize % 2:
		newsize += 1
	if newsize < 2:
		newsize = 2
	return np.asarray(jax_ift2d(jax_downsample_2d(jax_fft2d(img[np.newaxis, ...]), newsize))[0])


def _write_image(fsp, img, apix=None):
	"""Write a single (ny,nx) image to a new file at index 0."""
	hdr = {"nx": img.shape[1], "ny": img.shape[0], "nz": 1}
	if apix is not None:
		hdr["apix_x"] = hdr["apix_y"] = hdr["apix_z"] = apix
	io = ImageIO(fsp, "rw")
	io.write_image(0, np.asarray(img, dtype=np.float32), hdr)
	io.close()


def _write_stack(fsp, stack, apix=None):
	"""Write an {N,Y,X} stack to a new file."""
	io = ImageIO(fsp, "rw")
	for i in range(stack.shape[0]):
		hdr = {"nx": stack.shape[2], "ny": stack.shape[1], "nz": 1}
		if apix is not None:
			hdr["apix_x"] = hdr["apix_y"] = hdr["apix_z"] = apix
		io.write_image(i, np.asarray(stack[i], dtype=np.float32), hdr)
	io.close()


def _peak_shifts(ccfr, half):
	"""Given a real-space CCF/ACF stack {N, 2*half, 2*half}, extract the peak
	shift of each (relative to center) as a (2, N) int array (rows: y then x)."""
	flat = np.argmax(ccfr.reshape(ccfr.shape[0], -1), axis=1)
	py, px = np.unravel_index(flat, (half * 2, half * 2))
	return np.array([py, px]) - half  # shape (2, N): row 0 = y, row 1 = x


def main():
	progname = os.path.basename(sys.argv[0])
	usage = """e3movie.py <movie stack> ...

This will estimate gain for counting mode cameras, and has a "first draft" of frame alignment.
	"""
	parser = EMArgumentParser(usage=usage, version=EMANVERSION)
	parser.add_argument("--est_gain", type=str, help="specify output file for gain image. Estimates a gain image when given a set of many movies via hierarchical median estimation", default=None)
	parser.add_argument("--seqavg", type=int, default=-1, help="Average N frames in sequence before alignment")
	parser.add_argument("--shrinkout", type=float, default=-1, help="Fourier downsample the output movie stack by a factor of N (may be fractional)")
	parser.add_argument("--apix", type=float, help="Override A/pixel from header on all inputs", default=None)
	parser.add_argument("--alignbyccf", action="store_true", default=False, help="Performs movie alignment via sequential ccfs ")
	parser.add_argument("--alignbyacfccf", action="store_true", default=False, help="Performs movie alignment via progressive ")
	parser.add_argument("--align_gain", type=str, help="Gain image for correcting movie images before alignment. Applied correction is to divide by the gain image.", default=None)
	parser.add_argument("--keepunali", action="store_true", default=False, help="If set, stores unaligned as well as aligned micrograph average")
	parser.add_argument("--noinvert", action="store_true", default=False, help="Default behavior is to invert averaged image contrast. This disables that behavior.")
	parser.add_argument("--clip", type=str, default=None, help="nx,ny output image size. Trims both edges equally as necessary")
	parser.add_argument("--frames", type=str, default=None, help="<first>,<last+1> movie frames to use, first frame is 0, '0,3' will use frames 0,1,2")
	parser.add_argument("--acftest", action="store_true", default=False, help="compute ACF images for input stack")
	parser.add_argument("--ccftest", action="store_true", default=False, help="compute CCF between each image and the middle image in the movie")
	parser.add_argument("--ccfdtest", type=str, default=None, help="compute the CCF between each image and the next image in the movie, length n-1, provide the filename of the gain correction image")
	parser.add_argument("--ccftiletest", action="store_true", default=False, help="test on tiled average of CCF")
	parser.add_argument("--gpudev", type=int, help="GPU Device, default 0", default=0)
	parser.add_argument("--gpuram", type=int, help="Maximum GPU ram to allocate in MB, default=4096", default=4096)
	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n", type=int, default=0, help="verbose level [0-9], higher number means higher level of verboseness")

	(options, args) = parser.parse_args()
	jax_set_device(dev=options.gpudev, maxmem=options.gpuram)

	pid = E3init(sys.argv)
	nmov = len(args)
	if options.clip is not None:
		clip = (int(options.clip.split(",")[0]), int(options.clip.split(",")[1]))
	else:
		clip = None
	if options.frames is not None:
		options.frames = (int(options.frames.split(",")[0]), int(options.frames.split(",")[1]))
	else:
		options.frames = (0, 100000)

	# Load the gain image once if it is used for correction
	gain = None
	if options.align_gain is not None:
		io = ImageIO(options.align_gain, "r")
		gain, _ = io.read_image(0)
		io.close()
		gain = np.asarray(gain)

	# ------------------------------------------------------------------
	# --est_gain : estimate a gain image across all provided movies
	# ------------------------------------------------------------------
	if options.est_gain is not None:
		sumim = None
		sumsq = None
		n = 0
		for i in range(nmov):
			if (i % 10 == 0):
				E3progress(pid, i / nmov)
			f = args[i]
			nimg = ImageIO(f, "r").nimg
			print(f"{f}: {nimg}")
			for j in range(0, nimg, 50):
				data, _ = _read_range(f, j, min(j + 50, nimg))
				for img in data:
					img = np.asarray(img, dtype=np.float64)
					if sumim is None:
						sumim = np.zeros_like(img)
						sumsq = np.zeros_like(img)
					sumim += img
					sumsq += img * img
					n += 1
		avg = sumim / n
		sig = np.sqrt(sumsq / n - avg ** 2)
		# ensure that on average e- counts are still counts (shared scale factor)
		A = float(avg.mean())
		avg = avg / A
		sig = sig / A  # scale the sigma the same way as the average
		# pixels with very small or very large standard deviations are likely bad
		slow = max(0.1, sig.mean() - 4.0 * sig.std())
		shigh = sig.mean() + 4.0 * sig.std()
		sigm = ((sig >= slow) & (sig <= shigh)).astype(np.float64)
		avg = avg * sigm

		io = ImageIO(options.est_gain, "rw")
		hdr = {"nx": avg.shape[1], "ny": avg.shape[0], "nz": 1}
		io.write_image(0, avg.astype(np.float32), hdr)
		io.write_image(1, sig.astype(np.float32), hdr)
		io.close()
		print("Gain image computed, exiting")
		E3end(pid)
		sys.exit(0)

	# ------------------------------------------------------------------
	# --alignbyccf : sequential CCF alignment
	# ------------------------------------------------------------------
	if options.alignbyccf:
		rsz = 128  # half the size of the CCF regions to correlate
		for mi in range(nmov):
			base = base_name(args[mi])[0]
			f = args[mi]
			nimg = ImageIO(f, "r").nimg
			frames = (options.frames[0], min(options.frames[1], nimg))

			if options.seqavg > 1:
				nblk = (frames[1] - frames[0]) // options.seqavg
				blk = []
				for i in range(nblk):
					a = frames[0] + i * options.seqavg
					b = frames[0] + (i + 1) * options.seqavg
					d, _ = _read_range(f, a, b)
					blk.append(d.sum(axis=0))
				imgs = np.stack(blk)
			else:
				imgs, _ = _read_range(f, frames[0], frames[1])
				imgs = np.asarray(imgs)

			apix = None
			if options.apix is not None:
				apix = options.apix
			if apix is None:
				try:
					apix = _read_frame(f, frames[0])[1].get("apix_x", 1.0)
				except Exception:
					apix = 1.0
			if apix is None:
				apix = 1.0

			if options.align_gain is not None:
				imgs = np.stack([_fixgain_counting(imgs[k], gain, 2, 2) for k in range(imgs.shape[0])])
			# normalization only for alignment
			imgs = np.stack([_edgemean(imgs[k]) for k in range(imgs.shape[0])])
			imgs = _center_crop(imgs, 3072, 3072)
			ffts = jax_fft2d(imgs)

			# CCF between each image and the subsequent image
			ccfs = jax_phaseorigin2d(ffts[:-1] * jnp.conj(ffts[1:]))
			ccfsr = np.asarray(jax_ift2d(ccfs))
			ccfsr = _center_crop(ccfsr, rsz * 2, rsz * 2)
			ccfsr[:, rsz, rsz] = (ccfsr[:, rsz - 1, rsz] + ccfsr[:, rsz + 1, rsz] +
	                             ccfsr[:, rsz, rsz - 1] + ccfsr[:, rsz, rsz + 1]) / 4.0
			for i in range(ccfsr.shape[0]):
				ccfsr[i] -= np.mean(ccfsr[i])
				ccfsr[i] /= np.std(ccfsr[i])

			# gaussian low pass filter to smooth the ccfs after removing the 0 translation
			filt = jax_pointfilt_2d(rsz * 2, 0.125)
			ccfsfilt = jax_fft2d(ccfsr) * filt
			ccfsreal = np.asarray(jax_ift2d(ccfsfilt))

			# extract the peak locations and convert to absolute shifts
			seqoff = _peak_shifts(ccfsreal, rsz)
			seqoff = np.insert(seqoff, 0, np.array([0, 0]), axis=1)  # first image is 0 shift
			for i in range(1, seqoff.shape[1]):
				seqoff[0, i] += seqoff[0, i - 1]
				seqoff[1, i] += seqoff[1, i - 1]
			seqoff[0] -= int(seqoff[0].mean())  # recenter to minimize edge effects
			seqoff[1] -= int(seqoff[1].mean())

			if options.verbose:
				print(seqoff)

			# average the (optionally gain-corrected, translated) frames
			orig = []
			ali = []
			step = max(options.seqavg, 1)
			for i in range(frames[0], frames[1], step):
				if options.seqavg > 1:
					im = _read_range(f, i, i + options.seqavg)[0].sum(axis=0)
				else:
					im = _read_frame(f, i)[0]
				if options.align_gain is not None:
					im = _fixgain_counting(im, gain, 2, 2)
				if clip is not None:
					im = _center_crop(im, clip[0], clip[1])
				orig.append(im)
				# note the x/y order: translate(dx=x-shift, dy=y-shift)
				ali.append(_translate2(im, int(seqoff[1, i - frames[0]]),
				                       int(seqoff[0, i - frames[0]])))
			avgo = np.mean(orig, axis=0)
			avga = np.mean(ali, axis=0)

			if not options.noinvert:
				avgo = -1.0 * avgo
				avga = -1.0 * avga
			if options.shrinkout > 1.0:
				avgo = _resample(avgo, options.shrinkout)
				avga = _resample(avga, options.shrinkout)

			os.makedirs("micrographs", exist_ok=True)
			if options.keepunali:
				_write_image(f"micrographs/{base}__{frames[0]}-{frames[1]}_unali.hdf", avgo, apix)
			_write_image(f"micrographs/{base}__{frames[0]}-{frames[1]}.hdf", avga, apix)

			js = js_open_dict(info_name(f))
			js["movie_frames"] = frames
			js["movie_align_x"] = seqoff[1].tolist()
			js["movie_align_y"] = seqoff[0].tolist()
			js = None

	# ------------------------------------------------------------------
	# --alignbyacfccf : progressive ACF-referenced CCF alignment
	# ------------------------------------------------------------------
	if options.alignbyacfccf:
		rsz = 128
		for mi in range(nmov):
			base = base_name(args[mi])[0]
			f = args[mi]
			nimg = ImageIO(f, "r").nimg
			frames = (options.frames[0], min(options.frames[1], nimg))
			apix = None
			if options.apix is not None:
				apix = options.apix
			if apix is None:
				try:
					apix = _read_frame(f, frames[0])[1].get("apix_x", 1.0)
				except Exception:
					apix = 1.0
			if apix is None:
				apix = 1.0
			imgs, _ = _read_range(f, frames[0], frames[1])
			imgs = np.asarray(imgs)
			if options.align_gain is not None:
				imgs = np.stack([_fixgain_counting(imgs[k], gain, 2, 2) for k in range(imgs.shape[0])])
			imgs = np.stack([_edgemean(imgs[k]) for k in range(imgs.shape[0])])
			imgs = _center_crop(imgs, 3072, 3072)
			ffts = jax_fft2d(imgs)

			# ACF of each image, then average them
			acfs = jax_phaseorigin2d(ffts * jnp.conj(ffts))
			acfsr = np.asarray(jax_ift2d(acfs))
			ny, nx = acfsr.shape[1:]
			acfsr = acfsr[:, ny // 2 - rsz:ny // 2 + rsz, nx // 2 - rsz:nx // 2 + rsz]
			acfav = acfsr.mean(axis=0)

			# zero out / smooth the region near the origin
			acfav[rsz - 1, rsz - 1] = (acfav[rsz - 2, rsz - 1] + acfav[rsz - 1, rsz - 2]) / 2.0
			acfav[rsz, rsz - 1] = acfav[rsz, rsz - 2]
			acfav[rsz + 1, rsz - 1] = (acfav[rsz + 1, rsz - 2] + acfav[rsz + 2, rsz - 1]) / 2.0
			acfav[rsz - 1, rsz] = acfav[rsz - 2, rsz]
			acfav[rsz + 1, rsz] = acfav[rsz + 2, rsz]
			acfav[rsz - 1, rsz + 1] = (acfav[rsz - 2, rsz + 1] + acfav[rsz - 1, rsz + 2]) / 2.0
			acfav[rsz, rsz + 1] = acfav[rsz, rsz + 2]
			acfav[rsz + 1, rsz + 1] = (acfav[rsz + 2, rsz + 1] + acfav[rsz + 1, rsz + 2]) / 2.0
			acfav[rsz, rsz] = (acfav[rsz, rsz - 1] + acfav[rsz - 1, rsz] + acfav[rsz + 1, rsz] + acfav[rsz, rsz + 1]) / 4.0
			acfav -= acfav.mean()
			acfav /= acfav.std()

			acfavf = jax_fft2d(acfav)

			# CCF between each image and the subsequent image, central region
			ccfs = jax_phaseorigin2d(ffts[:-1] * jnp.conj(ffts[1:]))
			ccfsr = np.asarray(jax_ift2d(ccfs))
			ccfsr = ccfsr[:, ny // 2 - rsz:ny // 2 + rsz, nx // 2 - rsz:nx // 2 + rsz].copy()
			ccfsr[:, rsz, rsz] = (ccfsr[:, rsz - 1, rsz] + ccfsr[:, rsz + 1, rsz] +
	                             ccfsr[:, rsz, rsz - 1] + ccfsr[:, rsz, rsz + 1]) / 4.0
			for i in range(ccfsr.shape[0]):
				ccfsr[i] -= np.mean(ccfsr[i])
				ccfsr[i] /= np.std(ccfsr[i])
			ccfsf = jax_fft2d(ccfsr)

			# CCF between the CCFS and the ACF reference
			ccfacf = jax_phaseorigin2d(ccfsf * jnp.conj(acfavf))
			ccfacfr = np.asarray(jax_ift2d(ccfacf))

			seqoff = _peak_shifts(ccfacfr, rsz)
			seqoff = np.insert(seqoff, 0, np.array([0, 0]), axis=1)
			for i in range(1, seqoff.shape[1]):
				seqoff[0, i] += seqoff[0, i - 1]
				seqoff[1, i] += seqoff[1, i - 1]
			seqoff[0] -= int(seqoff[0].mean())
			seqoff[1] -= int(seqoff[1].mean())

			if options.verbose:
				print(seqoff)

			orig = []
			ali = []
			for i in range(frames[0], frames[1]):
				im = _read_frame(f, i)[0]
				if options.align_gain is not None:
					im = _fixgain_counting(im, gain, 2, 2)
				if clip is not None:
					im = _center_crop(im, clip[0], clip[1])
				orig.append(im)
				ali.append(_translate2(im, int(seqoff[1, i - frames[0]]),
				                       int(seqoff[0, i - frames[0]])))
			avgo = np.mean(orig, axis=0)
			avga = np.mean(ali, axis=0)

			if not options.noinvert:
				avgo = -1.0 * avgo
				avga = -1.0 * avga
			if options.shrinkout > 1.0:
				avgo = _resample(avgo, options.shrinkout)
				avga = _resample(avga, options.shrinkout)

			os.makedirs("micrographs", exist_ok=True)
			_write_image(f"micrographs/{base}_{frames[0]}-{frames[1]}_unali.hdf", avgo, apix)
			_write_image(f"micrographs/{base}_{frames[0]}-{frames[1]}.hdf", avga, apix)

			js = js_open_dict(info_name(f))
			js["movie_frames"] = frames
			js["movie_align_x"] = seqoff[1].tolist()
			js["movie_align_y"] = seqoff[0].tolist()
			js = None

	# ------------------------------------------------------------------
	# --ccftest : CCF between each image and the middle image
	# ------------------------------------------------------------------
	if options.ccftest:
		avg = _read_frame("average.hdf", 0)[0]
		avg = avg / avg.mean()
		nimg = ImageIO(args[0], "r").nimg
		imgs = _read_range(args[0], 0, min(50, nimg))[0]
		imgs = imgs / avg
		ffts = jax_fft2d(imgs)
		ccfs = jax_phaseorigin2d(ffts * jnp.conj(ffts[len(imgs) // 2]))
		ccfsr = np.asarray(jax_ift2d(ccfs))
		ny, nx = ccfsr.shape[1:]
		cens = ccfsr[:, ny // 2 - 64:ny // 2 + 64, nx // 2 - 64:nx // 2 + 64]
		_write_stack("ccfs.hdf", cens)

	# ------------------------------------------------------------------
	# --acftest : ACF images for the input stack
	# ------------------------------------------------------------------
	if options.acftest:
		avg = _read_frame("average.hdf", 0)[0]
		avg = avg / avg.mean()
		nimg = ImageIO(args[0], "r").nimg
		imgs = _read_range(args[0], 0, min(50, nimg))[0]
		imgs = imgs / avg
		ffts = jax_fft2d(imgs)
		ccfs = jax_phaseorigin2d(ffts * jnp.conj(ffts))
		ccfsr = np.asarray(jax_ift2d(ccfs))
		ny, nx = ccfsr.shape[1:]
		cens = ccfsr[:, ny // 2 - 64:ny // 2 + 64, nx // 2 - 64:nx // 2 + 64]
		cens = np.stack([_edgemean(cens[k]) for k in range(cens.shape[0])])
		_write_stack("ccfs.hdf", cens)

	# ------------------------------------------------------------------
	# --ccfdtest : CCF between each image and the next, with gain correction
	# ------------------------------------------------------------------
	if options.ccfdtest is not None:
		for db in ("ccfs.hdf", "ccfs2k.hdf", "ccfs1k.hdf", "ccfeo5.hdf", "ccfeo5_imgs.hdf"):
			try:
				os.unlink(db)
			except OSError:
				pass

		avg = _read_frame(options.ccfdtest, 0)[0]
		avg = avg / avg.mean()
		nimg = ImageIO(args[0], "r").nimg

		# first block: sum of 10-frame groups (step 2), 40 images
		print(f"0:{min(50, nimg)}")
		aimgs = _read_range(args[0], 0, min(50, nimg))[0]
		aimgs = np.stack([_fixgain_counting(aimgs[k], avg, 3, 3) for k in range(aimgs.shape[0])])
		imgs = np.stack([aimgs[i:i + 10:2].sum(axis=0) for i in range(40)])
		ffts = jax_fft2d(imgs)
		ccfs = jax_phaseorigin2d(ffts[:-1] * jnp.conj(ffts[1:]))
		ccfsr = np.asarray(jax_ift2d(ccfs))
		cens = _center_crop(ccfsr, 64, 64)
		cens = np.stack([_edgemean(cens[k]) for k in range(cens.shape[0])])
		_write_stack("ccfeo5.hdf", cens)
		_write_stack("ccfeo5_imgs.hdf", imgs)

		for i in range(1, nimg, 25):
			print(f"{i - 1}:{min(i + 25, nimg)}")
			imgs = _read_range(args[0], i - 1, min(i + 25, nimg))[0]
			imgs = np.stack([_fixgain_counting(imgs[k], avg, 3, 3) for k in range(imgs.shape[0])])

			def _ccfblock(im):
				ff = jax_fft2d(im)
				c = jax_phaseorigin2d(ff[:-1] * jnp.conj(ff[1:]))
				cr = np.asarray(jax_ift2d(c))
				cn = _center_crop(cr, 64, 64)
				cn = np.stack([_edgemean(cn[k]) for k in range(cn.shape[0])])
				return cn

			_write_stack("ccfs.hdf", _ccfblock(imgs))

			imgs2k = _center_crop(imgs, 2048, 2048)
			_write_stack("ccfs2k.hdf", _ccfblock(imgs2k))

			imgs1k = _center_crop(imgs, 1024, 1024)
			_write_stack("ccfs1k.hdf", _ccfblock(imgs1k))

	# ------------------------------------------------------------------
	# --ccftiletest : tiled average of the CCF
	# ------------------------------------------------------------------
	if options.ccftiletest:
		avg = _read_frame("average.hdf", 0)[0]
		avg = avg / avg.mean()
		nimg = ImageIO(args[0], "r").nimg
		imgs = _read_range(args[0], 0, min(50, nimg))[0]
		ny, nx = imgs.shape[1:]
		imgs = imgs / avg

		ccfsum = None
		for x in range(0, nx - 1024, 1024):
			for y in range(0, ny - 1024, 1024):
				imgt = imgs[:, y:y + 1024, x:x + 1024]
				ff = jax_fft2d(imgt)
				c = jax_phaseorigin2d(ff[:-1] * jnp.conj(ff[1:]))
				cr = np.asarray(jax_ift2d(c))
				if ccfsum is None:
					ccfsum = cr
				else:
					ccfsum = ccfsum + cr
		ccfsum = np.stack([_edgemean(ccfsum[k]) for k in range(ccfsum.shape[0])])
		_write_stack("ccfs.hdf", ccfsum)

	E3end(pid)


if __name__ == '__main__':
	main()
