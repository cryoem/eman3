#!/usr/bin/env python
#
# Author: Steve Ludtke (sludtke@bcm.edu)
# Copyright (c) 2026- Baylor College of Medicine
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
#
# e3ctf.py
#
# Estimate the CTF (defocus, astigmatism amount, astigmatism direction and
# in-plane beam tilt) of a single micrograph using a sparse 2-D "point" model of
# the object. The approach:
#
#   1. Fit a modest set of 2-D points (x,y,amp; z=0) to the micrograph WITHOUT
#      imposing any CTF, restricted to low resolution (~10% of Nyquist). At low
#      resolution the CTF is smooth and, with a commensurate point count, the
#      fit represents the CTF-free scattering object (the centroids of the
#      dominant scatterers) rather than the CTF itself.
#   2. Project those points (with a Gaussian density PSF, B-factor) and apply a
#      parameterized CTF (defocus, astigmatism, beam tilt) to simulate a
#      CTF-modulated image.
#   3. Optimize the CTF parameters by maximizing the sum of the real-space
#      image correlation and the Fourier power-spectrum correlation between
#      the simulated image and the micrograph (the FRC is intentionally
#      insensitive to radial amplitude variation, hence unsuitable as a
#      defocus metric). The strategy is staged to defeat the cyclical
#      (local-maximum) nature of the CTF:
#         (a) exhaustive defocus search,
#         (b) a gradient fit of the defocus only (astig/tilt held at 0),
#         (c) a simultaneous fit of the points, CTF parameters and the B-factor,
#             with the astigmatism bounded by dfdiffmax / 10% of the defocus,
#         (d) expand the point set (default 5x, new points randomly seeded) and fit
#             the points by maximizing FRC with the CTF held at its estimate,
#         (e) a final simultaneous fit of the points, CTF parameters and the B-factor.
#
# The cropped micrograph is normalize_edgemean'd (codebase processor convention) before any
# fitting, and the same normalization is applied to the simulated point model in the
# comparison outputs, so both share the same edge statistics.
#
# Amplitude contrast is held fixed at a user-provided value (fitting it from the
# data is ill-posed). Voltage and Cs are user-provided (or read from the header
# when present).
#
# Pure Python + JAX, no C++ dependencies.
#

import sys
import os
import math
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from EMAN3.EMAN3 import EMArgumentParser, E3init, E3end, error_exit, warning, good_size_small
from EMAN3.EMAN3jax import Points, fit_points_to_image_2d, point_interpolate_simple, jax_fft2d, jax_frc_jit, rad_img_int
from EMAN3.io.imageio import ImageIO
from EMAN3.processor import normalize_edgemean_2d_real

EMANVERSION = "e3ctf (EMAN3)"



def _dump_points(n, points):
	"""Debug helper: write the current point set (Nx4, x,y,z,amp) to test_<n>.txt."""
	if isinstance(points, Points):
		points.save(f"test_{n}.txt")
	else:
		Points(np.array(points, np.float32)).save(f"test_{n}.txt")


# ---------------------------------------------------------------------------
# Core CTF estimator
# ---------------------------------------------------------------------------
class CtfEstimator():
	"""Estimates CTF parameters for a single (square) micrograph from a point model."""

	def __init__(self, img, apix, wl, cs, ac_phase, npoints, band_ptcl,
	             dfmin, dfmax, dfdiffmax, tilt, seed, bfactor, bmax, expand, verbose=0):
		"""img - (S,S) real micrograph subregion (numpy), S even
		apix - A/pix
		wl - electron wavelength in A
		cs - spherical aberration in mm
		ac_phase - pi/2 - asin(ampcont/100)
		npoints - number of seed points in the low-resolution, CTF-free fit
		band_ptcl - (lo,hi) spatial frequency in 1/A for the CTF-free seed point fit
		dfmin, dfmax - defocus search range in um
		dfdiffmax - max astigmatism (dfdiff) in um
		tilt - fit in-plane beam tilt (tx,ty)?
		bfactor - B-factor (A^2) of the Gaussian density PSF (fixed in the staged CTF
		           fit, fitted in the simultaneous stage)
		bmax - upper bound (A^2) for the fitted B-factor in the simultaneous stage
		expand - stage 3 point expansion factor (total points = expand x current)
		seed - RNG seed"""
		self.S = img.shape[0]
		self.apix = apix
		self.wl = wl
		self.cs = cs
		self.ac_phase = ac_phase
		self.verbose = verbose
		self.tilt = tilt

		self.ny = img.shape[0]

		# normalize_edgemean the cropped micrograph before any fitting (same convention as the
		# normalize_edgemean processor): (image - mean(1-pixel border)) / std(whole image).
		# The correlation/FRC metrics are scale-invariant, so this does not affect the fit; it
		# gives the micrograph and the simulated model comparable edge statistics in the outputs.
		imgj = jnp.asarray(normalize_edgemean_2d_real(jnp.asarray(img, jnp.float32)[None, ...]), jnp.float32)[0]
		self.img = np.asarray(imgj, np.float32)    # normalized cropped micrograph (for the eval image)
		# zero-mean reference image and its norm, for the real-space correlation metric (CTF stages)
		self.img_c = imgj - imgj.mean()
		self.img_norm = jnp.sqrt(jnp.sum(self.img_c * self.img_c))
		# zero-mean micrograph power spectrum and its norm, for the Fourier-space intensity
		# correlation term of the CTF energy (real-space correlation + Fourier power-spectrum correlation)
		pwr = jnp.abs(jnp.fft.rfft2(imgj).astype(jnp.complex64)) ** 2
		self.pwr_c = pwr - pwr.mean()
		self.pwr_norm = jnp.sqrt(jnp.sum(self.pwr_c * self.pwr_c))
		# micrograph FFT for the stage-3 FRC metric ((1, ny, ny//2+1) complex64)
		self.imgf = jax_fft2d(imgj[None, ...])

		# Precompute the (parameter-independent) frequency grid in 1/A
		fx = jnp.arange(self.ny // 2 + 1)
		fy = jnp.arange(self.ny); fy = jnp.where(fy > self.ny // 2, fy - self.ny, fy)
		Y, X = jnp.meshgrid(fy, fx, indexing='ij')          # (ny, ny//2+1)
		self.sx = X / (self.apix * self.ny)
		self.sy = Y / (self.apix * self.ny)
		self.s2 = self.sx * self.sx + self.sy * self.sy
		self.s4 = self.s2 * self.s2
		self.ang = jnp.arctan2(self.sy, self.sx)
		self.g1 = (jnp.pi / 2.0) * self.cs * 1.0e7 * wl ** 3
		self.g2 = jnp.pi * wl * 1.0e4

		self.bfactor0 = bfactor
		self.bmax = bmax
		self.expand = float(expand)
		self.seed = int(seed)

		# Stage 1: fit a 2-D point model to the micrograph WITHOUT CTF, at low resolution
		if self.verbose >= 1:
			print(f"  fitting {npoints} points (band {band_ptcl[0]:.4f}-{band_ptcl[1]:.4f} 1/A) ...")
		t0 = time.time()
		self.points, self.ptcl_frc = fit_points_to_image_2d(self.img, apix, npoints, band=band_ptcl, seed=seed, verbose=verbose, dump_prefix="test_")
		if self.verbose >= 1:
			print(f"    point fit done in {time.time()-t0:.1f}s, low-res FRC={self.ptcl_frc:.3f}")

		# 2-D point interpolation (independent of CTF params); self.points is (N,4) x,y,z,amp, use x,y,amp
		prj = point_interpolate_simple(self.points.jax[:, [0, 1, 3]], self.ny)  # (S,S) real
		self.prjf = jnp.fft.rfft2(prj).astype(jnp.complex64)                    # (S,S//2+1)

		# Search ranges
		self.dfmin = dfmin
		self.dfmax = dfmax
		self.dfdiffmax = dfdiffmax

		# JIT-compiled batched correlation scorer (params: [df,dfdiff,dfang,tx,ty])
		self._score_batch_jit = jax.jit(self._score_batch)

	def ctf_from_params(self, df, dd, da, tx, ty):
		"""CTF amplitude image (ny, ny//2+1) from precomputed frequency grid. Scalar or batched params."""
		dfi = df + (dd / 2.0) * jnp.cos(2.0 * self.ang - (2.0 * jnp.pi / 180.0) * da)
		tilt = 2.0 * jnp.pi * (tx * self.sx + ty * self.sy)
		gam = -self.g1 * self.s4 + self.g2 * dfi * self.s2 + tilt
		return jnp.cos(gam - self.ac_phase)

	def psf(self, B):
		"""Gaussian density PSF (B-factor damping) for a B-factor (A^2), turning the seed-point
		delta functions into a finite-size density model. B in A^2, s in 1/A."""
		return jnp.exp(-B * self.s2 / 4.0)

	def _corr(self, simf):
		"""Pearson (zero-mean) correlation in real space between the CTF-simulated image
		(simf, its rfft2 spectrum) and the micrograph, normalized by both norms."""
		sim = jnp.fft.irfft2(simf, s=(self.ny, self.ny))
		simc = sim - sim.mean()
		return jnp.sum(simc * self.img_c) / (self.img_norm * jnp.sqrt(jnp.sum(simc * simc) + 1.0e-20))

	def _fourier_corr(self, simf):
		"""Pearson (zero-mean) correlation in Fourier space between the power spectrum of the
		CTF-simulated image (simf, its rfft2 spectrum) and the micrograph power spectrum."""
		P = jnp.abs(simf) ** 2
		Pc = P - P.mean()
		return jnp.sum(Pc * self.pwr_c) / (self.pwr_norm * jnp.sqrt(jnp.sum(Pc * Pc) + 1.0e-20))

	def _score(self, simf):
		"""Combined CTF energy to maximize: real-space image correlation plus the correlation of
		the Fourier-space power spectra (both Pearson, zero-mean)."""
		return self._corr(simf) + self._fourier_corr(simf)

	def _score1(self, p, B):
		ctf = self.ctf_from_params(p[0], p[1], p[2], p[3], p[4])
		return self._score(self.prjf * self.psf(B)[None, ...] * ctf[None, ...])

	def _score_batch(self, P, B):
		"""P - (B,5) array of [df,dd,da,tx,ty]. Returns (B,) correlation scores (B-factor B fixed)."""
		return jax.vmap(lambda p: self._score1(p, B))(P)

	def _scan(self, P, B, chunk=8):
		"""Evaluate the correlation score for each row of P (N,5) with B-factor B, in memory-bounded chunks."""
		N = len(P)
		out = np.zeros(N)
		for i in range(0, N, chunk):
			b = P[i:i + chunk]
			out[i:i + chunk] = np.array(self._score_batch_jit(jnp.array(b, jnp.float32), B))
			if self.verbose >= 2:
				print(f"    scored {min(i+chunk,N)}/{N}", end="\r")
		if self.verbose >= 2:
			print()
		return out

	def _rough_defocus(self):
		"""Stage (a): exhaustive defocus search (astig=tilt=0, B-factor fixed at bfactor0)."""
		if self.verbose >= 1:
			print("  stage (a): exhaustive defocus search ...")
		n = 60
		dfs = np.linspace(self.dfmin, self.dfmax, n)
		P = np.zeros((n, 5)); P[:, 0] = dfs
		scores = self._scan(P, self.bfactor0)
		i = int(np.argmax(scores))
		if self.verbose >= 1:
			print(f"    rough defocus = {dfs[i]:.3f} um (corr={scores[i]:.4f})")
		return float(dfs[i])

	def _fit_defocus(self, df_init, niter=200):
		"""Stage 1: refine the defocus only (dfdiff=0, tilt=0). Points and B are held fixed at their
		initial values. Uses the combined real-space + Fourier power-spectrum correlation energy."""
		if self.verbose >= 1:
			print("  stage 1: defocus fit ...")
		scale = jnp.array([1.0], jnp.float32)

		def loss(u):
			df = u * scale
			ctf = self.ctf_from_params(df, 0.0, 0.0, 0.0, 0.0)
			return -self._score(self.prjf * self.psf(self.bfactor0)[None, ...] * ctf[None, ...])

		lv = jax.jit(jax.value_and_grad(loss, argnums=0))
		u = jnp.array([df_init], jnp.float32) / scale
		optim = optax.adam(0.05)
		st = optim.init(u)
		best = (None, float('inf'))
		for it in range(niter):
			val, grad = lv(u)
			upd, st = optim.update(grad, st, u)
			u = optax.apply_updates(u, upd)
			score = -float(np.array(val))
			if self.verbose >= 2 and (it % 30 == 0 or it == niter - 1):
				print(f"    it={it:3d}/{niter}  score={score:.4f}")
			if val < best[1]:
				best = (u, val)
		u = best[0]
		df = float(np.array(u * scale)[0])
		if self.verbose >= 1:
			print(f"    defocus fit: df={df:.3f} um (score={-best[1]:.4f})")
		return df

	def _simultaneous_fit(self, init_ctf, points, niter=200, lr_points=0.002, lr=0.05):
		"""Stage 2: jointly refine the seed points (x,y,amp), the CTF (df,dfdiff,dfang,tx,ty)
		and the B-factor by maximizing the combined energy (real-space image correlation plus
		Fourier power-spectrum correlation) between the CTF-modulated density and the micrograph.
		The B-factor is bounded by bmax (an unbounded B crushes the CTF fringes needed to identify
		df/astig, creating a degenerate fit), and the astigmatism (dfdiff) is bounded by
		min(dfdiffmax, 10% of the starting defocus), as large astigmatism is physically unusual
		and its gradient competes with defocus in the CTF's cyclical phase. Returns
		(ctf_array, points_array, score, sim_density, bfactor)."""
		if self.verbose >= 1:
			print("  stage 2: simultaneous fit (points + CTF + B) ...")
		ny = self.ny
		np_ = 5 if self.tilt else 3
		ctf_scale = jnp.array([1.0, 0.1, 45.0, 10.0, 10.0][:np_], jnp.float32)  # um, um, deg, A, A
		b_scale = 50.0

		def loss(p, u, b):
			ctf = u * ctf_scale
			if np_ == 5:
				ctfimg = self.ctf_from_params(ctf[0], ctf[1], ctf[2], ctf[3], ctf[4])
			else:
				ctfimg = self.ctf_from_params(ctf[0], ctf[1], ctf[2], 0.0, 0.0)
			prj = point_interpolate_simple(p[:, [0, 1, 3]], ny)  # (ny,ny) real; p is (N,4) x,y,z,amp
			simf = jnp.fft.rfft2(prj) * self.psf(b * b_scale)[None, ...] * ctfimg[None, ...]
			return -self._score(simf)

		lv = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))
		p = jnp.array(np.asarray(points, jnp.float32))            # (N,4) x,y,z,amp
		u = jnp.array(np.asarray(init_ctf, jnp.float32)[:np_] / ctf_scale, jnp.float32)
		b = self.bfactor0 / b_scale
		dfdd_max = min(self.dfdiffmax, 0.1 * abs(float(np.asarray(init_ctf, float)[0]))) / ctf_scale[1]
		st_p = optax.adam(lr_points).init(p)
		opt_p = optax.adam(lr_points)
		st_u = optax.adam(lr).init(u)
		opt_u = optax.adam(lr)
		st_b = optax.adam(lr).init(b)
		opt_b = optax.adam(lr)
		best = (p, u, b, float('inf'))
		for it in range(niter):
			val, (gp, gu, gb) = lv(p, u, b)
			up_p, st_p = opt_p.update(gp, st_p, p)
			p = optax.apply_updates(p, up_p)
			p = p.at[:, 2].set(0.0)                                 # keep 2-D (z=0)
			up_u, st_u = opt_u.update(gu, st_u, u)
			u = optax.apply_updates(u, up_u)
			u = u.at[1].set(jnp.clip(u[1], 0.0, dfdd_max))          # bound the astigmatism
			up_b, st_b = opt_b.update(gb, st_b, b)
			b = optax.apply_updates(b, up_b)
			b = jnp.clip(b, 0.0, self.bmax / b_scale)
			score = -float(np.array(val))
			if self.verbose >= 2 and (it % 20 == 0 or it == niter - 1):
				print(f"    it={it:3d}/{niter}  score={score:.4f}  B={b * b_scale:.1f} A^2")
			if val < best[3]:
				best = (p, u, b, val)
		p, u, b, val = best
		ctf = np.array(u * ctf_scale)
		bf = float(b * b_scale)
		if self.verbose >= 1:
			print(f"    simultaneous fit: df={ctf[0]:.3f} um dfdiff={ctf[1]:.3f} um dfang={ctf[2]:.1f} deg "
					f"{'tx=%.2f A ty=%.2f A' % (ctf[3], ctf[4]) if np_ == 5 else ''} B={bf:.1f} A^2 (score={-val:.4f})")
		# Rebuild the final CTF-modulated density (real space) that the final correlation is computed against
		if np_ == 5:
			ctfimg = self.ctf_from_params(ctf[0], ctf[1], ctf[2], ctf[3], ctf[4])
		else:
			ctfimg = self.ctf_from_params(ctf[0], ctf[1], ctf[2], 0.0, 0.0)
		prj = point_interpolate_simple(p[:, [0, 1, 3]], ny)  # (ny,ny) real
		sim = jnp.fft.irfft2(jnp.fft.rfft2(prj) * self.psf(bf)[None, ...] * ctfimg[None, ...], s=(ny, ny))  # (ny,ny) real
		return ctf, np.array(p), float(-val), sim, bf

	def _expand_points(self, points, factor):
		"""Grow the point set so the total count is factor x the current count, keeping the existing
		points and appending randomly seeded new ones (x,y uniform in -0.5..0.5, z=0, amp uniform in
		-1/3..1/3, the same convention as the seed fit). Returns a (M,4) numpy array."""
		base = np.asarray(points, np.float32)
		n = len(base)
		n_new = int(round(n * factor)) - n
		if n_new <= 0:
			return base
		rng = np.random.default_rng(self.seed + 1)
		new = np.stack([rng.uniform(-0.5, 0.5, n_new),
		                rng.uniform(-0.5, 0.5, n_new),
		                np.zeros(n_new),
		                rng.uniform(-1.0 / 3.0, 1.0 / 3.0, n_new)], axis=1).astype(np.float32)
		return np.concatenate([base, new], axis=0)

	def _fit_points_frc(self, points, ctf, bf, niter=200):
		"""Stage 3: optimize the (expanded) point set only, maximizing the full-band FRC between the
		CTF-modulated point image and the micrograph. The CTF parameters are held fixed at their
		estimated values (applied to the point model before the FRC is computed), and the B-factor is
		held at its current value. Returns (points_array, frc)."""
		if self.verbose >= 1:
			print(f"  stage 3: expanded point fit (FRC, CTF/B fixed; {len(points)} points) ...")
		ny = self.ny
		rad_img_int(ny)                                  # required by jax_frc_jit (precomputes FRC_RADS)
		w = jnp.ones(ny // 2 + 1, jnp.float32)           # full band (DC..Nyquist)
		if self.tilt:
			ctfimg = self.ctf_from_params(ctf[0], ctf[1], ctf[2], ctf[3], ctf[4])
		else:
			ctfimg = self.ctf_from_params(ctf[0], ctf[1], ctf[2], 0.0, 0.0)

		def loss(p):
			prj = jnp.expand_dims(point_interpolate_simple(p[:, [0, 1, 3]], ny), 0)   # (1,ny,ny) real
			simf = jnp.fft.rfft2(prj) * self.psf(bf)[None, ...] * ctfimg[None, ...]
			return -jax_frc_jit(simf, self.imgf, w, 0.0)

		lv = jax.jit(jax.value_and_grad(loss, argnums=0))
		p = jnp.array(points, jnp.float32)
		optim = optax.adam(0.003)
		st = optim.init(p)
		best = (p, float('inf'))
		for it in range(niter):
			val, grad = lv(p)
			upd, st = optim.update(grad, st, p)
			p = optax.apply_updates(p, upd)
			p = p.at[:, 2].set(0.0)                         # keep 2-D (z=0)
			if val < best[1]:
				best = (p, val)
		p, val = best
		frc = -float(np.array(val))
		if self.verbose >= 1:
			print(f"    expanded point fit: FRC={frc:.4f}")
		return np.array(p), frc

	def _save_debug_density(self, sim):
		"""Debug helper: write test.hdf — image 0 is the final CTF-modulated point model (real space),
		image 1 is the clipped micrograph region it is compared against. Both are normalize_edgemean'd
		(the micrograph at the start of the fit, the simulation here) so they share edge statistics."""
		sim_arr = np.asarray(sim).reshape(sim.shape[-2], sim.shape[-1]).astype(np.float32)
		sim_arr = np.asarray(normalize_edgemean_2d_real(jnp.asarray(sim_arr, jnp.float32)[None, ...]), np.float32)[0]
		hdr = {"nx": sim_arr.shape[1], "ny": sim_arr.shape[0], "nz": 1,
		       "apix_x": self.apix, "apix_y": self.apix, "apix_z": self.apix}
		io = ImageIO("test.hdf", mode="rw")
		try:
			io.write_image(0, sim_arr, hdr)
			io.write_image(1, self.img, hdr)
		finally:
			io.close()

	def _save_eval_image(self, sim):
		"""Debug helper: write eval.hdf, a side-by-side "amp" (log amplitude) view of the final fit
		arranged to read as one continuous power spectrum: the LEFT half is the real data (the left
		semicircle of the micrograph FFT) and the RIGHT half is the model (the simulation FFT, mirrored
		so it is the right semicircle). Both origins sit at the center of the image, so the two halves
		meet and the Thon-ring shapes can be compared directly, left (real) vs right (model). Each half
		is independently normalized to fill 0-1 (viewer-style amp view, DC spike excluded from the
		scale, which clips to white at the center)."""
		ny = self.ny

		def amp_view(img):
			F = np.fft.fftshift(np.fft.fft2(img))
			la = np.log10(1.0 + np.abs(F))
			# Viewer-style amp display: normalize the log-amplitude to fill 0-1. The range is
			# taken EXCLUDING the central DC spike (which would otherwise eat the whole scale),
			# so the DC pixel clips to white and the Thon rings use the rest of the display.
			m = la.copy()
			m[ny // 2, ny // 2] = 0.0
			lo, hi = float(m.min()), float(m.max())
			if hi <= lo:
				return np.zeros_like(la, dtype=np.float32)
			r = (la - lo) / (hi - lo)
			# Keep only the left semicircle (the right is redundant for a real image); DC is on the
			# inner (right) edge.
			return np.clip(r[:, :ny // 2 + 1], 0.0, 1.0)

		left = amp_view(self.img)                                          # real data: left semicircle
		right = amp_view(np.asarray(sim).reshape(ny, ny))[:, ::-1]         # model: mirror to right semicircle
		comb = np.hstack((left, right)).astype(np.float32)
		io = ImageIO("eval.hdf", mode="rw")
		try:
			io.write_image(0, comb, {"nx": comb.shape[1], "ny": comb.shape[0], "nz": 1,
			                       "apix_x": self.apix, "apix_y": self.apix, "apix_z": self.apix})
		finally:
			io.close()

	def estimate(self):
		"""Run the staged CTF estimation. Returns a dict of CTF parameters.
		Stage 1: fit the defocus only (dfdiff=0, tilt=0), from a robust coarse search.
		Stage 2: refine the defocus, astigmatism, (beam tilt), points and B-factor jointly.
		Stage 3: expand the point set (default 5x, new points randomly seeded) and optimize
		        the points only, by FRC with the CTF held at its estimate.
		Stage 4: a final all-parameter optimization cycle on the expanded point model."""
		df0 = self._rough_defocus()          # robust coarse defocus
		df1 = self._fit_defocus(df0)         # Stage 1: defocus only
		if self.tilt:
			init = [df1, 0.0, 0.0, 0.0, 0.0]
		else:
			init = [df1, 0.0, 0.0]
		# Stage 2: refine points + full CTF + B-factor from the defocus-only solution
		ctf, points, score, sim, bf = self._simultaneous_fit(init, self.points.jax)
		_dump_points(5, points)
		# Stage 3: expand the point set and fit the points by FRC with the CTF held fixed
		points_exp = self._expand_points(points, self.expand)
		_dump_points(6, points_exp)
		points3, frc3 = self._fit_points_frc(points_exp, ctf, bf)
		_dump_points(7, points3)
		# Stage 4: a final all-parameter optimization cycle on the expanded point model
		ctf, points, score, sim, bf = self._simultaneous_fit(ctf, points3)
		_dump_points(8, points)
		self._save_debug_density(sim)
		self._save_eval_image(sim)
		return {"defocus": float(ctf[0]), "dfdiff": float(ctf[1]), "dfang": float(ctf[2]),
		        "tilt_x": float(ctf[3]) if self.tilt else 0.0,
		        "tilt_y": float(ctf[4]) if self.tilt else 0.0,
		        "bfactor": float(bf), "fit_score": float(score),
		        "expand_factor": float(self.expand), "expand_frc": float(frc3),
		        "points": points}


def load_micrograph(path, verbose=0):
	"""Load the first image of a micrograph file, returning (numpy 2-D array, header dict)."""
	io = ImageIO(path, "r")
	try:
		img, hdr = io.read_image(0)
	finally:
		io.close()
	img = np.asarray(img, np.float32)
	if img.ndim != 2:
		raise Exception(f"Expected a 2-D micrograph, got {img.ndim}-D data")
	return img, (hdr if hasattr(hdr, "keys") else {})


def header_get(hdr, *keys, default=None):
	"""Fetch the first present key from a header dict, else default."""
	if hdr:
		for k in keys:
			if k in hdr:
				v = hdr[k]
				if v is not None:
					return v
	return default


def estimate_ctf(img, apix, voltage, cs, ampcont, npoints, dfmin, dfmax, dfdiffmax,
                 tilt, seed, side_frac, bfactor, bmax, expand, verbose=0):
	"""Estimate the CTF of a single micrograph.
	img - (H,W) real micrograph
	Returns a dict with: defocus, dfdiff, dfang, tilt_x, tilt_y, bfactor, apix, voltage, cs, ampcont,
	                     side, npoints, ptcl_frc, fit_score, expand_factor, expand_frc, and the fitted Points."""
	W = img.shape[1]; H = img.shape[0]
	side = good_size_small(int(min(W, H) * side_frac))
	side = max(16, side // 2 * 2)
	x0 = W // 2 - side // 2
	y0 = H // 2 - side // 2
	sub = img[y0:y0 + side, x0:x0 + side]
	if verbose >= 1:
		print(f"  micrograph {W}x{H} -> centered square subregion {side}x{side}")

	wl = 12.2639 / math.sqrt(voltage * 1000.0 + 0.97845 * voltage * voltage)   # A
	ac_phase = math.pi / 2.0 - math.asin(max(-1.0, min(1.0, ampcont / 100.0)))

	nyq = 1.0 / (2.0 * apix)                                                    # 1/A
	band_ptcl = (0.0, 0.1 * nyq)                                                # ~10% of Nyquist (seed point fit only)

	est = CtfEstimator(sub, apix, wl, cs, ac_phase, npoints, band_ptcl,
	                   dfmin, dfmax, dfdiffmax, tilt, seed, bfactor, bmax, expand, verbose)
	params = est.estimate()

	params.update({"apix": apix, "voltage": voltage, "cs": cs, "ampcont": ampcont,
	               "side": side, "npoints": npoints, "ptcl_frc": est.ptcl_frc})
	params["points"] = Points(np.array(params["points"], np.float32))   # refined points (simultaneous fit)
	return params


def main():
	progname = os.path.basename(sys.argv[0])
	usage = """prog [options] <micrograph>

Estimate the CTF (defocus, astigmatism, beam tilt) of a single micrograph using a
sparse 2-D point model of the object fitted at low resolution without CTF, then
optimizing a parameterized CTF (plus the B-factor) by maximizing the real-space
correlation against the micrograph.

Requires voltage, Cs and amplitude contrast (or finds them in the header).


"""
	parser = EMArgumentParser(usage=usage, version=EMANVERSION)
	parser.add_pos_argument(name="micrograph", help="Input micrograph (2-D image file)", default="",
	                        guitype='filebox', row=0, col=0, rowspan=1, colspan=3)
	parser.add_argument("--voltage", type=float, default=None, help="Microscope voltage in kV (default from header if present)")
	parser.add_argument("--cs", type=float, default=None, help="Spherical aberration in mm (default from header if present)")
	parser.add_argument("--ampcont", "-a", type=float, default=None, help="Amplitude contrast as a percentage, eg 7 for 7%% (default from header if present)")
	parser.add_argument("--apix", type=float, default=0.0, help="Angstrom/pixel, if not found in the file")
	parser.add_argument("--npoints", type=int, default=10000, help="Number of seed points in the low-resolution, CTF-free fit (default 10000)")
	parser.add_argument("--expand", type=float, default=5.0, help="Stage 3: factor by which to increase the point count before the expanded-point FRC fit (default 5; the new points are randomly seeded)")
	parser.add_argument("--bfactor", type=float, default=30.0, help="Initial B-factor (A^2) of the Gaussian density PSF (default 30); fixed in the staged CTF fit, refined in the simultaneous fit")
	parser.add_argument("--bmax", type=float, default=100.0, help="Upper bound (A^2) for the fitted B-factor in the simultaneous fit (default 100; use a large value for no bound)")
	parser.add_argument("--dfmin", type=float, default=0.1, help="Minimum defocus to search, in um (default 0.1)")
	parser.add_argument("--dfmax", type=float, default=7.0, help="Maximum defocus to search, in um (default 7.0)")
	parser.add_argument("--dfdiffmax", type=float, default=0.15, help="Maximum astigmatism (defocus difference) to search, in um (default 0.15; also capped at 10%% of the rough defocus)")
	parser.add_argument("--notilt", action="store_true", help="Do not fit in-plane beam tilt")
	parser.add_argument("--seed", type=int, default=0, help="RNG seed for the point fit (default 0)")
	parser.add_argument("--sidefrac", type=float, default=0.9, help="Fraction of the shortest axis used for the centered square working region (default 0.9)")
	parser.add_argument("--savepoints", type=str, default=None, help="If set, save the fitted point model to this text file")
	parser.add_argument("--output", type=str, default=None, help="Output JSON filename (default <micrograph>_ctf.json)")
	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n", type=int, default=0, help="verbose level [0-9]")
	parser.add_argument("--ppid", type=int, default=-1, help="Set the PID of the parent process")

	(options, args) = parser.parse_args()
	logid = E3init(sys.argv, options.ppid)

	micro = args[0]
	if not os.path.isfile(micro):
		error_exit(f"Cannot find input micrograph: {micro}")

	img, hdr = load_micrograph(micro, options.verbose)

	apix = options.apix if options.apix > 0 else float(header_get(hdr, "apix_x", "apix", "pixel_size", default=0.0))
	if apix <= 0:
		error_exit("Could not determine apix. Specify --apix.")
	voltage = options.voltage if options.voltage is not None else header_get(hdr, "voltage", "kV", "voltage_kV")
	cs = options.cs if options.cs is not None else header_get(hdr, "cs", "cs_mm", "spherical_aberration")
	ampcont = options.ampcont if options.ampcont is not None else header_get(hdr, "ampcont", "amp_contrast", "amp_cont")
	if voltage is None or float(voltage) <= 0:
		error_exit("Could not determine voltage (kV). Specify --voltage.")
	voltage = float(voltage)
	if cs is None or float(cs) <= 0:
		warning("Cs not specified/found; defaulting to 2.7 mm. Specify --cs for accuracy.")
		cs = 2.7
	cs = float(cs)
	if ampcont is None:
		warning("Amplitude contrast not specified/found; defaulting to 7%%. Specify --ampcont.")
		ampcont = 7.0
	ampcont = float(ampcont)

	print(f"CTF estimation: {micro}")
	print(f"  apix={apix:.4f} A/px  voltage={voltage} kV  cs={cs} mm  ampcont={ampcont}%")

	t0 = time.time()
	params = estimate_ctf(img, apix, voltage, cs, ampcont, options.npoints,
	                      options.dfmin, options.dfmax, options.dfdiffmax,
	                      not options.notilt, options.seed,
	                      options.sidefrac, options.bfactor, options.bmax, options.expand, options.verbose)
	el = time.time() - t0

	print(f"\n  RESULT (in {el:.1f}s):")
	print(f"    defocus  = {params['defocus']:.4f} um")
	print(f"    dfdiff   = {params['dfdiff']:.4f} um")
	print(f"    dfang    = {params['dfang']:.2f} deg")
	if not options.notilt:
		print(f"    tilt     = ({params['tilt_x']:.2f}, {params['tilt_y']:.2f}) A")
	print(f"    bfactor  = {params['bfactor']:.1f} A^2")
	print(f"    low-res FRC (seed fit) = {params['ptcl_frc']:.4f}   (seed only, expected to be low)")
	print(f"    FRC (stage 3, expanded points) = {params['expand_frc']:.4f}")
	print(f"    score (stage 4 fit) = {params['fit_score']:.4f}")

	out = options.output if options.output else os.path.splitext(micro)[0] + "_ctf.json"
	outdict = {k: params[k] for k in ("defocus", "dfdiff", "dfang", "tilt_x", "tilt_y",
	                                 "bfactor", "apix", "voltage", "cs", "ampcont", "side", "npoints",
	                                 "ptcl_frc", "fit_score", "expand_factor", "expand_frc")}
	import json
	with open(out, "w") as fh:
		json.dump(outdict, fh, indent=2)
	print(f"  wrote {out}")

	if options.savepoints:
		params["points"].save(options.savepoints)
		print(f"  wrote {options.savepoints}")

	E3end(logid)


if __name__ == "__main__":
	main()
