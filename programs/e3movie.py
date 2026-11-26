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

from EMAN3.EMAN3 import (EMArgumentParser, E3init, E3end, E3progress, print_progress,
                         error_exit, warning, good_size, base_name, info_name)
from EMAN3.EMAN3jax import (jax_fft2d, jax_ift2d, jax_gaussfilt_2d,
                           jax_downsample_2d, jax_phaseorigin2d, jax_set_device)
from EMAN3.processor import normalize_edgemean_2d_real, xform_translate_int_2d_real
from EMAN3.io.imageio import ImageIO
from EMAN3.EMAN3jsondb import js_open_dict

EMANVERSION = "e3movie (EMAN3)"


# ---------------------------------------------------------------------------
# Low level helpers (numpy real-space image: shape (ny,nx))
# ---------------------------------------------------------------------------

def build_fixgain_map(gain, gainmin=2.0, gainmax=2.0):
	"""One-time pass over a gain image (EMAN2's "math.fixgain.counting" semantics).

	Identifies the bad sensor pixels (gain outside the [gainmin, gainmax] sigma
	window about the gain mean, gainmin floored at 0.1) and, for each, the good
	pixel found by walking the diagonal toward the image center. Returns a
	mapping consumed by apply_fixgain, so the per-pixel bad-pixel search is done
	once per gain image rather than once per image being corrected.
	"""
	gain = np.asarray(gain, dtype=np.float32)
	ny, nx = gain.shape
	gmean = float(gain.mean())
	gsig = float(gain.std())
	hi = gmean + gainmax * gsig
	lo = max(0.1, gmean - gainmin * gsig)

	good = (gain >= lo) & (gain <= hi)
	bad = np.argwhere(~good)   # (M,2), rows [y, x]
	m = bad.shape[0]
	good_y = np.empty(m, dtype=np.intp)
	good_x = np.empty(m, dtype=np.intp)
	fallback = np.zeros(m, dtype=bool)   # no good pixel found on the diagonal
	for k in range(m):
		y, x = bad[k]
		dyy = -1 if y < ny / 2 else 1
		dxx = -1 if x < nx / 2 else 1
		found = False
		for i in range(1, max(nx, ny) // 2 + 1):
			yy, xx = y + dyy * i, x + dxx * i
			if 0 <= yy < ny and 0 <= xx < nx and lo <= gain[yy, xx] <= hi:
				good_y[k], good_x[k] = yy, xx
				found = True
				break
		fallback[k] = not found

	return {"gain": gain, "gmean": gmean,
	        "bad_y": bad[:, 0], "bad_x": bad[:, 1],
	        "good_y": good_y, "good_x": good_x, "fallback": fallback}


def apply_fixgain(img, fmap):
	"""Divide an image or a stack of images by the gain image from a mapping
	built by build_fixgain_map, substituting each bad gain pixel's value with
	the (image/gain) value of its mapped good pixel, or the gain mean if no
	good pixel was found. Fully vectorized.
	"""
	img = np.asarray(img, dtype=np.float32)
	with np.errstate(divide="ignore", invalid="ignore"):
		out = img / fmap["gain"]
	m = fmap["bad_y"].size
	if m:
		remap = ~fmap["fallback"]
		if remap.any():
			out[..., fmap["bad_y"][remap], fmap["bad_x"][remap]] = \
				out[..., fmap["good_y"][remap], fmap["good_x"][remap]]
		fb = fmap["fallback"]
		if fb.any():
			out[..., fmap["bad_y"][fb], fmap["bad_x"][fb]] = fmap["gmean"]
	return out


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


def _peak_shifts(ccfr, half):
	"""Given a real-space CCF/ACF stack {N, 2*half, 2*half}, extract the peak
	shift of each (relative to center) as a (2, N) int32 array (rows: y then x)."""
	flat = jnp.argmax(ccfr.reshape(ccfr.shape[0], -1), axis=1)
	py, px = jnp.unravel_index(flat, (half * 2, half * 2))
	return jnp.array([py, px]) - half  # shape (2, N): row 0 = y, row 1 = x


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

	tlast = 0
	if options.verbose >= 1: tlast = print_progress(tlast, "e3movie", 0, nmov)

	# Load the gain image once if it is used for correction
	gain = None
	if options.align_gain is not None:
		io = ImageIO(options.align_gain, "r")
		gain, _ = io.read_image(0)
		io.close()
		gain = np.asarray(gain)

	if options.verbose>1 : print("Initialized")

	# ------------------------------------------------------------------
	# --est_gain : estimate a gain image across all provided movies
	# ------------------------------------------------------------------
	if options.est_gain is not None:
		sumim = None
		sumsq = None
		sqbuf = None
		n = 0
		nimg_first = 0
		for i in range(nmov):
			if (i % 10 == 0):
				E3progress(pid, i / nmov)
			f = args[i]
			with ImageIO(f, "r") as idio:
				nimg = idio.nimg
				if nimg_first == 0:
					nimg_first = nimg
				print(f"{f}: {nimg}")
				# Read one frame at a time: this path only accumulates sum/sumsq,
				# so materializing a 50-frame stack (read_images -> np.stack of
				# 50x64MB) just to iterate over it wastes a lot of memory traffic.
				for j in range(nimg):
					img, _ = idio.read_image(j)
					img = np.asarray(img, dtype=np.float32)
					if sumim is None:
						sumim = np.zeros_like(img)
						sumsq = np.zeros_like(img)
						sqbuf = np.empty_like(img)
					sumim += img
					# square into a preallocated buffer to avoid a fresh
					# 64MB allocation on every frame
					np.multiply(img, img, out=sqbuf)
					sumsq += sqbuf
					n += 1
					if options.verbose >= 1:
						tlast = print_progress(tlast, "est_gain", i * nimg_first + j, nmov * nimg_first)
		if options.verbose >= 1:
			tlast = print_progress(tlast, "est_gain", nmov * nimg_first, nmov * nimg_first)
		avg = sumim / n
		sig = np.sqrt(sumsq / n - avg ** 2)
		# ensure that on average e- counts are still counts (shared scale factor)
		A = float(avg.mean())
		avg = avg / A
		sig = sig / A  # scale the sigma the same way as the average
		# pixels with very small or very large standard deviations are likely bad
		slow = max(0.1, sig.mean() - 6.0 * sig.std())
		shigh = sig.mean() + 6.0 * sig.std()
		sigm = ((sig >= slow) & (sig <= shigh)).astype(np.float32)
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
			if options.verbose >= 1:
				print(args[mi],end="\t")
				tlast = print_progress(tlast, "alignbyccf", mi, nmov)
				print("")

			base = base_name(args[mi])[0]
			f = args[mi]
			idio = ImageIO(f, "r")
			nimg = idio.nimg
			frames = (options.frames[0], min(options.frames[1], nimg))
			nfr = frames[1] - frames[0]
			neff = nfr // options.seqavg if options.seqavg > 1 else nfr
			if neff > 100:
				print(f"warning: {neff} frames will be aligned, but individual frames of dose-fractionated movies may not have sufficient signal; consider --seqavg")

			if options.seqavg > 1:
				nblk = (frames[1] - frames[0]) // options.seqavg
				blk = []
				for i in range(nblk):
					if options.verbose>1: print(f" reading {i}/{nblk}",end="\r")
					a = frames[0] + i * options.seqavg
					b = frames[0] + (i + 1) * options.seqavg
					d, _ = idio.read_images(list(range(a, b)))
					blk.append(d.sum(axis=0))
				imgs = np.stack(blk)
			else:
				imgs, _ = idio.read_images(list(range(frames[0], frames[1])))
				imgs = np.asarray(imgs)

			apix = None
			if options.apix is not None:
				apix = options.apix
			if apix is None:
				try:
					apix = idio.read_header(frames[0]).get("apix_x", 1.0)
				except Exception:
					apix = 1.0
			if apix is None:
				apix = 1.0

			if options.align_gain is not None:
				fmap = build_fixgain_map(gain, 2, 2)
				imgs = apply_fixgain(imgs, fmap)
			# keep the unmodified (gain-corrected) images for the final aligned
			# average, so the movie does not have to be read from disk a second time
			raw = imgs
			if clip is not None:
				ny, nx = raw.shape[-2:]
				raw = raw[..., (ny - clip[1]) // 2:(ny - clip[1]) // 2 + clip[1],
				           (nx - clip[0]) // 2:(nx - clip[0]) // 2 + clip[0]]
			# normalization only for alignment
			imgs = normalize_edgemean_2d_real(imgs)
			ny, nx = imgs.shape[-2:]
			imgs = imgs[..., (ny - 3072) // 2:(ny - 3072) // 2 + 3072, (nx - 3072) // 2:(nx - 3072) // 2 + 3072]
			ffts = jax_fft2d(imgs)

			if options.verbose>1 : print(" ccfs             ",end="\r")
			# CCF between each image and the subsequent image
			ccfs = jax_phaseorigin2d(ffts[:-1] * jnp.conj(ffts[1:]))
			ccfsr = jax_ift2d(ccfs)
			ny, nx = ccfsr.shape[-2:]
			ccfsr = ccfsr[..., ny // 2 - rsz:ny // 2 + rsz, nx // 2 - rsz:nx // 2 + rsz]
			ccfsr = ccfsr.at[:, rsz, rsz].set((ccfsr[:, rsz - 1, rsz] + ccfsr[:, rsz + 1, rsz] +
			                                    ccfsr[:, rsz, rsz - 1] + ccfsr[:, rsz, rsz + 1]) / 4.0)
			ccfsr = (ccfsr - ccfsr.mean(axis=(1, 2), keepdims=True)) / ccfsr.std(axis=(1, 2), keepdims=True)

			# gaussian low pass filter to smooth the ccfs after removing the 0 translation
			filt = jax_gaussfilt_2d(rsz * 2, 0.125)
			ccfsfilt = jax_fft2d(ccfsr) * filt
			ccfsreal = jax_ift2d(ccfsfilt)

			if options.verbose>1 : print(" peaks             ",end="\r")
			# extract the peak locations and convert to absolute shifts
			seqoff = _peak_shifts(ccfsreal, rsz)
			seqoff = jnp.concatenate([jnp.zeros((2, 1), dtype=seqoff.dtype), seqoff], axis=1)  # first image is 0 shift
			seqoff = jnp.cumsum(seqoff, axis=1)
			seqoff = seqoff - jnp.array([int(seqoff[0].mean()), int(seqoff[1].mean())]).reshape(2, 1)  # recenter to minimize edge effects

			if options.verbose>2:
				print(seqoff)

			# average the frames, reusing the kept (gain-corrected, clip-cropped)
			# copy so the movie does not need a second read from disk; with
			# seqavg>1 the rows are per averaged block, so the shifts index by
			# block number
			orig = raw
			idxs = jnp.arange(orig.shape[0])
			shifts = jnp.stack([seqoff[1, idxs], seqoff[0, idxs]], axis=1)
			ali = xform_translate_int_2d_real(orig, shifts)
			avgo = np.mean(orig, axis=0)
			avga = jnp.mean(ali, axis=0)

			if not options.noinvert:
				avgo = -1.0 * avgo
				avga = -1.0 * avga
			if options.shrinkout > 1.0:
				avgo = _resample(avgo, options.shrinkout)
				avga = _resample(avga, options.shrinkout)

			idio.close()

			if options.verbose>1: print(" saving            ",end="\r")
			os.makedirs("micrographs", exist_ok=True)
			hdr = {"nx": avga.shape[1], "ny": avga.shape[0], "nz": 1}
			if apix is not None:
				hdr["apix_x"] = hdr["apix_y"] = hdr["apix_z"] = apix
			if options.keepunali:
				io = ImageIO(f"micrographs/{base}__{frames[0]}-{frames[1]}_unali.hdf", "rw")
				io.write_image(0, avgo, hdr)
				io.close()
			io = ImageIO(f"micrographs/{base}__{frames[0]}-{frames[1]}.hdf", "rw")
			io.write_image(0, np.asarray(avga), hdr)
			io.close()

			js = js_open_dict(info_name(f))
			js["movie_frames"] = frames
			js["movie_align_x"] = seqoff[1].tolist()
			js["movie_align_y"] = seqoff[0].tolist()
			js = None

		if options.verbose >= 1:
			tlast = print_progress(tlast, "alignbyccf", nmov,nmov)

	# ------------------------------------------------------------------
	# --alignbyacfccf : progressive ACF-referenced CCF alignment
	# ------------------------------------------------------------------
	if options.alignbyacfccf:
		rsz = 128
		for mi in range(nmov):
			if options.verbose >= 1:
				print(args[mi])
				tlast = print_progress(tlast, "alignbyacfccf", mi, nmov)

			base = base_name(args[mi])[0]
			f = args[mi]
			idio = ImageIO(f, "r")
			nimg = idio.nimg
			frames = (options.frames[0], min(options.frames[1], nimg))
			nfr = frames[1] - frames[0]
			if nfr > 100:
				print(f"warning: {nfr} frames will be aligned, but individual frames of dose-fractionated movies may not have sufficient signal; consider using --alignbyccf with --seqavg")
			apix = None
			if options.apix is not None:
				apix = options.apix
			if apix is None:
				try:
					apix = idio.read_header(frames[0]).get("apix_x", 1.0)
				except Exception:
					apix = 1.0
			if apix is None:
				apix = 1.0
			tlast2=0
			blk = []
			for i in range(nfr):
				if options.verbose>1: tlast2=print_progress(tlast2,"reading",i,nfr)
				im, _ = idio.read_image(frames[0] + i)
				blk.append(im)
			imgs = np.stack(blk)
			if options.align_gain is not None:
				fmap = build_fixgain_map(gain, 2, 2)
				imgs = apply_fixgain(imgs, fmap)
			# keep the unmodified (gain-corrected) images for the final aligned
			# average, so the movie does not have to be read from disk a second time
			raw = imgs
			if clip is not None:
				ny, nx = raw.shape[-2:]
				raw = raw[..., (ny - clip[1]) // 2:(ny - clip[1]) // 2 + clip[1],
				           (nx - clip[0]) // 2:(nx - clip[0]) // 2 + clip[0]]
			imgs = normalize_edgemean_2d_real(imgs)
			ny, nx = imgs.shape[-2:]
			imgs = imgs[..., (ny - 3072) // 2:(ny - 3072) // 2 + 3072, (nx - 3072) // 2:(nx - 3072) // 2 + 3072]
			ffts = jax_fft2d(imgs)

			# ACF of each image, then average them
			acfs = jax_phaseorigin2d(ffts * jnp.conj(ffts))
			acfsr = jax_ift2d(acfs)
			ny, nx = acfsr.shape[1:]
			acfsr = acfsr[:, ny // 2 - rsz:ny // 2 + rsz, nx // 2 - rsz:nx // 2 + rsz]
			acfav = acfsr.mean(axis=0)

			# zero out / smooth the region near the origin
			acfav = acfav.at[rsz - 1, rsz - 1].set((acfav[rsz - 2, rsz - 1] + acfav[rsz - 1, rsz - 2]) / 2.0)
			acfav = acfav.at[rsz, rsz - 1].set(acfav[rsz, rsz - 2])
			acfav = acfav.at[rsz + 1, rsz - 1].set((acfav[rsz + 1, rsz - 2] + acfav[rsz + 2, rsz - 1]) / 2.0)
			acfav = acfav.at[rsz - 1, rsz].set(acfav[rsz - 2, rsz])
			acfav = acfav.at[rsz + 1, rsz].set(acfav[rsz + 2, rsz])
			acfav = acfav.at[rsz - 1, rsz + 1].set((acfav[rsz - 2, rsz + 1] + acfav[rsz - 1, rsz + 2]) / 2.0)
			acfav = acfav.at[rsz, rsz + 1].set(acfav[rsz, rsz + 2])
			acfav = acfav.at[rsz + 1, rsz + 1].set((acfav[rsz + 2, rsz + 1] + acfav[rsz + 1, rsz + 2]) / 2.0)
			acfav = acfav.at[rsz, rsz].set((acfav[rsz, rsz - 1] + acfav[rsz - 1, rsz] + acfav[rsz + 1, rsz] + acfav[rsz, rsz + 1]) / 4.0)
			acfav = (acfav - acfav.mean()) / acfav.std()

			acfavf = jax_fft2d(acfav)

			# CCF between each image and the subsequent image, central region
			ccfs = jax_phaseorigin2d(ffts[:-1] * jnp.conj(ffts[1:]))
			ccfsr = jax_ift2d(ccfs)
			ccfsr = ccfsr[..., ny // 2 - rsz:ny // 2 + rsz, nx // 2 - rsz:nx // 2 + rsz]
			ccfsr = ccfsr.at[:, rsz, rsz].set((ccfsr[:, rsz - 1, rsz] + ccfsr[:, rsz + 1, rsz] +
			                                    ccfsr[:, rsz, rsz - 1] + ccfsr[:, rsz, rsz + 1]) / 4.0)
			ccfsr = (ccfsr - ccfsr.mean(axis=(1, 2), keepdims=True)) / ccfsr.std(axis=(1, 2), keepdims=True)
			ccfsf = jax_fft2d(ccfsr)

			# CCF between the CCFS and the ACF reference
			ccfacf = jax_phaseorigin2d(ccfsf * jnp.conj(acfavf))
			ccfacfr = jax_ift2d(ccfacf)

			seqoff = _peak_shifts(ccfacfr, rsz)
			seqoff = jnp.concatenate([jnp.zeros((2, 1), dtype=seqoff.dtype), seqoff], axis=1)
			seqoff = jnp.cumsum(seqoff, axis=1)
			seqoff = seqoff - jnp.array([int(seqoff[0].mean()), int(seqoff[1].mean())]).reshape(2, 1)

			if options.verbose:
				print(seqoff)

			# average the frames, reusing the kept (gain-corrected, clip-cropped)
			# copy so the movie does not need a second read from disk
			orig = raw
			idxs = jnp.arange(orig.shape[0])
			shifts = jnp.stack([seqoff[1, idxs], seqoff[0, idxs]], axis=1)
			ali = xform_translate_int_2d_real(orig, shifts)
			avgo = np.mean(orig, axis=0)
			avga = jnp.mean(ali, axis=0)

			if not options.noinvert:
				avgo = -1.0 * avgo
				avga = -1.0 * avga
			if options.shrinkout > 1.0:
				avgo = _resample(avgo, options.shrinkout)
				avga = _resample(avga, options.shrinkout)

			idio.close()

			os.makedirs("micrographs", exist_ok=True)
			hdr = {"nx": avga.shape[1], "ny": avga.shape[0], "nz": 1}
			if apix is not None:
				hdr["apix_x"] = hdr["apix_y"] = hdr["apix_z"] = apix
			io = ImageIO(f"micrographs/{base}_{frames[0]}-{frames[1]}_unali.hdf", "rw")
			io.write_image(0, avgo, hdr)
			io.close()
			io = ImageIO(f"micrographs/{base}_{frames[0]}-{frames[1]}.hdf", "rw")
			io.write_image(0, np.asarray(avga), hdr)
			io.close()

			js = js_open_dict(info_name(f))
			js["movie_frames"] = frames
			js["movie_align_x"] = seqoff[1].tolist()
			js["movie_align_y"] = seqoff[0].tolist()
			js = None

		if options.verbose >= 1:
			tlast = print_progress(tlast, "alignbyacfccf", nmov,nmov)

	# ------------------------------------------------------------------
	# --ccftest : CCF between each image and the middle image
	# ------------------------------------------------------------------
	if options.ccftest:
		with ImageIO("average.hdf", "r") as ain:
			avg = ain.read_image(0)[0]
		avg = avg / avg.mean()
		with ImageIO(args[0], "r") as idio:
			imgs = idio.read_images(list(range(0, min(50, idio.nimg))))[0]
		imgs = imgs / avg
		if options.verbose >= 1:
			tlast = print_progress(tlast, "ccftest", 1, 2)
		ffts = jax_fft2d(imgs)
		ccfs = jax_phaseorigin2d(ffts * jnp.conj(ffts[len(imgs) // 2]))
		ccfsr = jax_ift2d(ccfs)
		ny, nx = ccfsr.shape[1:]
		cens = ccfsr[..., ny // 2 - 64:ny // 2 + 64, nx // 2 - 64:nx // 2 + 64]
		cens = np.asarray(cens)
		hdr = {"nx": cens.shape[2], "ny": cens.shape[1], "nz": 1}
		io = ImageIO("ccfs.hdf", "rw")
		try:
			for k in range(cens.shape[0]):
				io.write_image(k, cens[k], hdr)
		finally:
			io.close()
		if options.verbose >= 1:
			tlast = print_progress(tlast, "ccftest", 2, 2)

	# ------------------------------------------------------------------
	# --acftest : ACF images for the input stack
	# ------------------------------------------------------------------
	if options.acftest:
		with ImageIO("average.hdf", "r") as ain:
			avg = ain.read_image(0)[0]
		avg = avg / avg.mean()
		with ImageIO(args[0], "r") as idio:
			imgs = idio.read_images(list(range(0, min(50, idio.nimg))))[0]
		imgs = imgs / avg
		if options.verbose >= 1:
			tlast = print_progress(tlast, "acftest", 1, 2)
		ffts = jax_fft2d(imgs)
		ccfs = jax_phaseorigin2d(ffts * jnp.conj(ffts))
		ccfsr = jax_ift2d(ccfs)
		ny, nx = ccfsr.shape[1:]
		cens = ccfsr[..., ny // 2 - 64:ny // 2 + 64, nx // 2 - 64:nx // 2 + 64]
		cens = np.asarray(normalize_edgemean_2d_real(cens))
		hdr = {"nx": cens.shape[2], "ny": cens.shape[1], "nz": 1}
		io = ImageIO("ccfs.hdf", "rw")
		try:
			for k in range(cens.shape[0]):
				io.write_image(k, cens[k], hdr)
		finally:
			io.close()
		if options.verbose >= 1:
			tlast = print_progress(tlast, "acftest", 2, 2)

	# ------------------------------------------------------------------
	# --ccfdtest : CCF between each image and the next, with gain correction
	# ------------------------------------------------------------------
	if options.ccfdtest is not None:
		for db in ("ccfs.hdf", "ccfs2k.hdf", "ccfs1k.hdf", "ccfeo5.hdf", "ccfeo5_imgs.hdf"):
			try:
				os.unlink(db)
			except OSError:
				pass

		with ImageIO(options.ccfdtest, "r") as ain:
			avg = ain.read_image(0)[0]
		avg = avg / avg.mean()
		idio = ImageIO(args[0], "r")
		nimg = idio.nimg
		ntest = 1 + len(range(1, nimg, 25))

		# first block: sum of 10-frame groups (step 2), 40 images
		print(f"0:{min(50, nimg)}")
		aimgs = idio.read_images(list(range(0, min(50, nimg))))[0]
		fmap = build_fixgain_map(avg, 3, 3)
		aimgs = apply_fixgain(aimgs, fmap)
		imgs = np.stack([aimgs[i:i + 10:2].sum(axis=0) for i in range(40)])
		ffts = jax_fft2d(imgs)
		ccfs = jax_phaseorigin2d(ffts[:-1] * jnp.conj(ffts[1:]))
		ccfsr = jax_ift2d(ccfs)
		ny, nx = ccfsr.shape[-2:]
		cens = ccfsr[..., (ny - 64) // 2:(ny - 64) // 2 + 64, (nx - 64) // 2:(nx - 64) // 2 + 64]
		cens = np.asarray(normalize_edgemean_2d_real(cens))
		hdr = {"nx": cens.shape[2], "ny": cens.shape[1], "nz": 1}
		io = ImageIO("ccfeo5.hdf", "rw")
		try:
			for k in range(cens.shape[0]):
				io.write_image(k, cens[k], hdr)
		finally:
			io.close()
		io = ImageIO("ccfeo5_imgs.hdf", "rw")
		try:
			for k in range(imgs.shape[0]):
				io.write_image(k, imgs[k], hdr)
		finally:
			io.close()
		if options.verbose >= 1:
			tlast = print_progress(tlast, "ccfdtest", 1, ntest)

		for i in range(1, nimg, 25):
			print(f"{i - 1}:{min(i + 25, nimg)}")
			if options.verbose >= 1:
				tlast = print_progress(tlast, "ccfdtest", 1 + (i - 1) // 25, ntest)
			imgs = idio.read_images(list(range(i - 1, min(i + 25, nimg))))[0]
			imgs = apply_fixgain(imgs, fmap)

			def _ccfblock(im):
				ff = jax_fft2d(im)
				c = jax_phaseorigin2d(ff[:-1] * jnp.conj(ff[1:]))
				cr = jax_ift2d(c)
				ny, nx = cr.shape[-2:]
				cn = cr[..., (ny - 64) // 2:(ny - 64) // 2 + 64, (nx - 64) // 2:(nx - 64) // 2 + 64]
				return np.asarray(normalize_edgemean_2d_real(cn))

			cn = _ccfblock(imgs)
			hdr = {"nx": cn.shape[2], "ny": cn.shape[1], "nz": 1}
			io = ImageIO("ccfs.hdf", "rw")
			try:
				for k in range(cn.shape[0]):
					io.write_image(k, cn[k], hdr)
			finally:
				io.close()

			ny, nx = imgs.shape[-2:]
			imgs2k = imgs[..., (ny - 2048) // 2:(ny - 2048) // 2 + 2048, (nx - 2048) // 2:(nx - 2048) // 2 + 2048]
			cn = _ccfblock(imgs2k)
			hdr = {"nx": cn.shape[2], "ny": cn.shape[1], "nz": 1}
			io = ImageIO("ccfs2k.hdf", "rw")
			try:
				for k in range(cn.shape[0]):
					io.write_image(k, cn[k], hdr)
			finally:
				io.close()

			imgs1k = imgs[..., (ny - 1024) // 2:(ny - 1024) // 2 + 1024, (nx - 1024) // 2:(nx - 1024) // 2 + 1024]
			cn = _ccfblock(imgs1k)
			hdr = {"nx": cn.shape[2], "ny": cn.shape[1], "nz": 1}
			io = ImageIO("ccfs1k.hdf", "rw")
			try:
				for k in range(cn.shape[0]):
					io.write_image(k, cn[k], hdr)
			finally:
				io.close()

		idio.close()
		if options.verbose >= 1:
			tlast = print_progress(tlast, "ccfdtest", ntest, ntest)

	# ------------------------------------------------------------------
	# --ccftiletest : tiled average of the CCF
	# ------------------------------------------------------------------
	if options.ccftiletest:
		with ImageIO("average.hdf", "r") as ain:
			avg = ain.read_image(0)[0]
		avg = avg / avg.mean()
		with ImageIO(args[0], "r") as idio:
			imgs = idio.read_images(list(range(0, min(50, idio.nimg))))[0]
		ny, nx = imgs.shape[1:]
		imgs = imgs / avg

		ccfsum = None
		ntile = len(range(0, nx - 1024, 1024)) * len(range(0, ny - 1024, 1024))
		nt = 0
		for x in range(0, nx - 1024, 1024):
			for y in range(0, ny - 1024, 1024):
				if options.verbose >= 1:
					tlast = print_progress(tlast, "ccftiletest", nt, ntile)
				imgt = imgs[:, y:y + 1024, x:x + 1024]
				ff = jax_fft2d(imgt)
				c = jax_phaseorigin2d(ff[:-1] * jnp.conj(ff[1:]))
				cr = jax_ift2d(c)
				if ccfsum is None:
					ccfsum = cr
				else:
					ccfsum = ccfsum + cr
				nt += 1
		if options.verbose >= 1:
			tlast = print_progress(tlast, "ccftiletest", ntile, ntile)
		ccfsum = np.asarray(normalize_edgemean_2d_real(ccfsum))
		hdr = {"nx": ccfsum.shape[2], "ny": ccfsum.shape[1], "nz": 1}
		io = ImageIO("ccfs.hdf", "rw")
		try:
			for k in range(ccfsum.shape[0]):
				io.write_image(k, ccfsum[k], hdr)
		finally:
			io.close()

	E3end(pid)


if __name__ == '__main__':
	main()
