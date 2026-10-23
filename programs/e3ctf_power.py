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
# e3ctf_power.py
#
# Estimate the CTF (defocus, astigmatism amount and direction) of a single
# micrograph in a more traditional way, directly from power spectra:
#
#   1. Extract a grid of non-overlapping 512x512 regions from the micrograph,
#      leaving a border (>128 px) around the edges. Compute the 2-D power
#      spectrum (|FFT|^2, no window) of each region and average them.
#   2. Rotationally average the averaged power spectrum to a radial power
#      spectrum. Fit the mean defocus against CTF^2 (amplitude contrast
#      fixed at a user-provided value) by an exhaustive search followed by
#      a local gradient refinement.
#   3. With the defocus in hand, fit the astigmatism (dfdiff, dfang) — and
#      the B-factor — against the averaged 2-D power spectrum, comparing the
#      CTF^2 wiggle pattern (after removing a smoothly blurred background
#      that represents the object power) in log space.
#   4. Extract a single centered square region of good_size(min(nx,ny)-256)
#      from the micrograph, compute its (higher-detail, higher-noise) power
#      spectrum, and refine all parameters locally (defocus constrained to
#      to +/-10% of the stage-2 value and dfdiff is bounded; the B-factor is then free
#      (bounded below by 0) since the other parameters are approximately correct.
#
# Amplitude contrast is held fixed at a user-provided value (fitting it from
# the data is ill-posed). Voltage and Cs are user-provided (or read from the
# header when present). Fully deterministic (no random components).
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

from EMAN3.EMAN3 import EMArgumentParser, E3init, E3end, error_exit, warning, good_size
from EMAN3.io.imageio import ImageIO

EMANVERSION = "e3ctf_power (EMAN3)"

TILE = 512        # coarse power-spectrum tile size
BORDMIN = 128     # minimum border (px) excluded around the tile grid
FLOOR = 1.0e-4    # relative floor used inside log() of power/CTF^2 terms


def _gauss_kernel_1d(n, sigma):
	"""1-D Gaussian kernel of length n (even), centered, normalized to unit sum."""
	x = np.arange(n) - n / 2.0
	k = np.exp(-0.5 * (x / sigma) ** 2)
	return k / k.sum()


def _gauss_blur_2d(img, sigma):
	"""2-D Gaussian blur (numpy, FFT-based) with a centered, unit-sum kernel, for an even-sized image.
	The kernel is rolled so its peak sits at index 0 before the FFT, giving a zero-phase
	(kernel-anchored) circular convolution."""
	n0, n1 = img.shape
	k0 = np.roll(_gauss_kernel_1d(n0, sigma), -n0 // 2)
	k1 = np.roll(_gauss_kernel_1d(n1, sigma), -n1 // 2)
	K = np.fft.fft2(np.outer(k0, k1))
	return np.real(np.fft.ifft2(np.fft.fft2(img) * K))


def _rpower(img):
	"""|rfft2|^2 of a 2-D real image, shape (n, n//2+1), float32."""
	F = np.fft.rfft2(img)
	return (np.abs(F) ** 2).astype(np.float32)


def _freq_grid(n):
	"""Frequency (cycles/pixel) grid of the rfft half-plane of an n x n image: (n, n//2+1)."""
	fx = np.arange(n // 2 + 1)
	fy = np.arange(n); fy = np.where(fy > n // 2, fy - n, fy)
	Y, X = np.meshgrid(fy, fx, indexing='ij')
	return X, Y


def _pearson(a, b, mask):
	"""Pearson correlation of a and b over the boolean mask (a, b same shape)."""
	am = a[mask]; bm = b[mask]
	ac = am - am.mean(); bc = bm - bm.mean()
	return (ac * bc).sum() / jnp.sqrt((ac * ac).sum() * (bc * bc).sum() + 1.0e-20)


class CtfPowerEstimator():
	"""Estimates CTF parameters for a single micrograph from power spectra (e3ctf_power)."""

	def __init__(self, img, apix, wl, cs, ac_phase, dfmin, dfmax, dfdiffmax,
	             bfactor, bmax, bweight, verbose=0):
		"""img - (ny,nx) real micrograph (numpy)
		apix - A/pix
		wl - electron wavelength in A
		cs - spherical aberration in mm
		ac_phase - pi/2 - asin(ampcont/100)
		dfmin, dfmax - defocus search range in um
		dfdiffmax - max astigmatism (dfdiff) in um
		bfactor - initial B-factor (A^2); bmax - B-factor reference (A^2), capped in the
		astigmatism stage and used as the scale of the soft bias toward lower B in the
		final stage; bweight - strength of that soft bias (0 disables it)
		verbose - 0 (quiet), 1 (per-stage summary), 2 (per-iteration progress)"""
		self.img = np.asarray(img, np.float32)
		self.ny_img, self.nx_img = self.img.shape
		self.apix = apix
		self.wl = wl
		self.cs = cs
		self.ac_phase = ac_phase
		self.dfmin = dfmin
		self.dfmax = dfmax
		self.dfdiffmax = dfdiffmax
		self.bfactor0 = bfactor
		self.bmax = bmax
		self.bweight = bweight
		self.verbose = verbose

		self.g1 = (math.pi / 2.0) * cs * 1.0e7 * wl ** 3
		self.g2 = math.pi * wl * 1.0e4

		# ---- Stage (a): tile grid, averaged power spectrum, radial power spectrum ----
		n_x = (self.nx_img - 2 * BORDMIN) // TILE
		n_y = (self.ny_img - 2 * BORDMIN) // TILE
		if n_x < 1 or n_y < 1:
			raise Exception(f"micrograph {self.nx_img}x{self.ny_img} is too small for a "
			                f"{TILE}-px tile grid with a >{BORDMIN}-px border (need >= {2 * BORDMIN + TILE} on both axes)")
		self.n_tiles_x = n_x
		self.n_tiles_y = n_y
		border_x = (self.nx_img - n_x * TILE) // 2
		border_y = (self.ny_img - n_y * TILE) // 2
		self.border_x = border_x
		self.border_y = border_y

		blur_sigma = TILE / 48.0    # wide enough to erase CTF wiggles, narrow enough to keep the background
		P = np.zeros((TILE, TILE // 2 + 1), np.float32)
		Pbg = np.zeros_like(P)
		for iy in range(n_y):
			for ix in range(n_x):
				tile = self.img[border_y + iy * TILE: border_y + (iy + 1) * TILE,
				               border_x + ix * TILE: border_x + (ix + 1) * TILE]
				P += _rpower(tile)
				Pbg += _rpower(_gauss_blur_2d(tile, blur_sigma))
		ntiles = n_x * n_y
		self.pavg = P / ntiles                       # averaged 2-D power spectrum (kept for later use)
		self.pavg_bg = Pbg / ntiles                  # smoothed background (object power) reference

		# radial power spectrum of the averaged power spectrum (integer-radius bins)
		X, Y = _freq_grid(TILE)
		rad = np.hypot(X, Y).astype(np.int32)
		K = int(rad.max()) + 1
		sums, counts = np.bincount(rad.ravel(), weights=self.pavg.ravel(), minlength=K), np.bincount(rad.ravel(), minlength=K)
		self.rad_pow = (sums / counts).astype(np.float32)
		self.rad_K = K

		if self.verbose >= 1:
			print(f"  tiles: {n_x} x {n_y} of {TILE} px "
			      f"(borders x={border_x}px, y={border_y}px, {ntiles} tiles averaged)")

		# ---- precomputed grids for the 2-D (astig) fits ----
		self._prep_plane(self.pavg, self.pavg_bg, TILE)

	def _prep_plane(self, p, pbg, n):
		"""Precompute the (n, n//2+1) plane terms for a power-spectrum fit: frequency grid,
		radius mask (DC excluded, inside the Nyquist circle), and log-space data/background."""
		X, Y = _freq_grid(n)
		s2 = (X * X + Y * Y) / (self.apix * n) ** 2                    # 1/A^2
		rad = np.hypot(X, Y)
		mask = (rad >= 1.0) & (rad <= 0.95 * (n / 2.0))                # DC excluded, inside Nyquist
		P = jnp.asarray(p, jnp.float32)
		self.s2 = jnp.asarray(s2, jnp.float32)
		self.ang = jnp.asarray(np.arctan2(Y, X), jnp.float32)
		self.s4 = self.s2 * self.s2
		self.mask = jnp.asarray(mask, jnp.bool_)
		self.logP = jnp.log(P / P.max() + FLOOR)
		self.logbg = jnp.log(jnp.asarray(pbg, jnp.float32) / P.max() + FLOOR)

	def _ctf2(self, s2, ang, df, dd, da):
		"""CTF^2 amplitude (no beam tilt) for defocus df (um), astig dd (um), da (deg)."""
		dfi = df + (dd / 2.0) * jnp.cos(2.0 * ang - (2.0 * math.pi / 180.0) * da)
		gam = -self.g1 * s2 * s2 + self.g2 * dfi * s2
		return jnp.cos(gam - self.ac_phase) ** 2

	def _score_plane(self, ctf2, B):
		"""Pearson (log space) between the data power spectrum and the CTF^2 wiggle pattern
		(damped by B) riding on the smoothed background."""
		model = self.logbg + jnp.log(ctf2 * jnp.exp(-B * self.s2 / 4.0) + FLOOR)
		return _pearson(self.logP, model, self.mask)

	def _b_prior(self, B):
		"""Smooth (differentiable) bias on the B-factor toward lower values, with bmax as the
		scale/reference. Unbounded - large values remain reachable if the data strongly
		support them (eg older instruments with heavy high-resolution damping), but each
		step beyond the reference costs correlation in the loss function. The cost is in
		units of the Pearson score, so bweight ~ 0.1 means 'one reference-scale of extra
		B is worth ~1/10 of a correlation point'."""
		t = (B - self.bmax) / self.bmax
		return self.bweight * jnp.square(jax.nn.softplus(t))

	def _boxavg1d(self, x, w):
		"""Centered box average of a 1-D array, 'same' output length (odd w assumed); matches
		np.convolve(x, ones(w)/w, 'same') semantics (fixed /w division, mild edge attenuation)."""
		h = (w - 1) // 2
		c = jnp.concatenate([jnp.zeros(h), x, jnp.zeros(h)])
		P = jnp.concatenate([jnp.zeros(1), jnp.cumsum(c)])
		return (P[w:] - P[:-w]) / w

	def _fit_defocus_radial(self, niter=100):
		"""Stage 1: fit the mean defocus from the radial power spectrum (astig = 0, AC fixed).
		The object power falls off smoothly with radius and is unknown, so both the data and the
		CTF^2 model are first reduced to their wiggle component (deviation from a centered
		11-bin box average, which is larger than the CTF wiggle period at low resolution but
		smaller than the fall-off scale), then Pearson-correlated. Exhaustive scan of [dfmin,
		dfmax], then a local Adam refinement. Returns (df, score)."""
		if self.verbose >= 1:
			print("  stage 1: radial defocus fit ...")
		K = self.rad_K
		s2 = (jnp.arange(K, dtype=jnp.float32) / (self.apix * TILE)) ** 2      # 1/A^2
		kmax = int(0.95 * (TILE / 2.0))
		# Bins 1-7 fall within the AC contrast-reversal dip at DC, whose absolute position is
		# independent of defocus; including them biases the fit, so the band starts at k = 8.
		mask = (jnp.arange(K) >= 8) & (jnp.arange(K) <= kmax)
		W = 11
		logR = jnp.log(jnp.asarray(self.rad_pow, jnp.float32) / self.rad_pow.max() + FLOOR)
		data_w = logR - self._boxavg1d(logR, W)

		def score1(df):
			ctf2 = jnp.cos(-self.g1 * s2 * s2 + self.g2 * df * s2 - self.ac_phase) ** 2
			model_w = jnp.log(ctf2 + FLOOR)
			model_w = model_w - self._boxavg1d(model_w, W)
			return _pearson(data_w, model_w, mask)

		# exhaustive coarse scan
		n = 80
		dfs = jnp.linspace(self.dfmin, self.dfmax, n)
		scores = np.array(jax.jit(jax.vmap(score1))(dfs))
		df0 = float(dfs[int(np.argmax(scores))])
		if self.verbose >= 1:
			print(f"    rough defocus = {df0:.3f} um (corr={scores.max():.4f})")

		# local refinement
		lv = jax.jit(jax.value_and_grad(lambda u: -score1(u[0]), argnums=0))
		u = jnp.array([df0], jnp.float32)
		optim = optax.adam(0.05)
		st = optim.init(u)
		best = (u, float('inf'))
		for it in range(niter):
			val, grad = lv(u)
			upd, st = optim.update(grad, st, u)
			u = optax.apply_updates(u, upd)
			if self.verbose >= 2 and (it % 20 == 0 or it == niter - 1):
				print(f"    it={it:3d}/{niter}  corr={-float(np.array(val)):.4f}")
			if val < best[1]:
				best = (u, val)
		df = float(np.array(best[0])[0])
		score = float(-best[1])
		if self.verbose >= 1:
			print(f"    defocus = {df:.4f} um (corr={score:.4f})")
		return df, score

	def _fit_astig(self, df, niter=200):
		"""Stage 2: fit the astigmatism (dfdiff, dfang) and B-factor against the averaged 2-D
		power spectrum, with the defocus held at the stage-1 estimate. dfdiff is bounded by
		min(dfdiffmax, 10% of the defocus) and B by bmax. Returns (dfdiff, dfang, B, score)."""
		if self.verbose >= 1:
			print("  stage 2: astigmatism fit (averaged power spectrum) ...")
		ddmax = min(self.dfdiffmax, 0.1 * df)
		scale = jnp.array([0.1, 45.0, 50.0], jnp.float32)     # um, deg, A^2

		def loss(u):
			dd, da, B = u * scale
			return -self._score_plane(self._ctf2(self.s2, self.ang, df, dd, da), B)

		lv = jax.jit(jax.value_and_grad(loss, argnums=0))
		u = jnp.zeros(3, jnp.float32)
		u = u.at[2].set(self.bfactor0 / scale[2])
		optim = optax.adam(0.05)
		st = optim.init(u)
		best = (u, float('inf'))
		for it in range(niter):
			val, grad = lv(u)
			upd, st = optim.update(grad, st, u)
			u = optax.apply_updates(u, upd)
			u = u.at[0].set(jnp.clip(u[0], 0.0, ddmax / scale[0]))
			u = u.at[2].set(jnp.clip(u[2], 0.0, self.bmax / scale[2]))
			if self.verbose >= 2 and (it % 20 == 0 or it == niter - 1):
				print(f"    it={it:3d}/{niter}  corr={-float(np.array(val)):.4f}")
			if val < best[1]:
				best = (u, val)
		u, val = best
		dd, da, B = np.array(u * scale)
		if self.verbose >= 1:
			print(f"    dfdiff = {dd:.4f} um  dfang = {da:.2f} deg  B = {B:.1f} A^2  (corr={-val:.4f})")
		return float(dd), float(da), float(B), float(-val)

	def _fit_final(self, df, dd, da, B, niter=300):
		"""Stage 3: refine all parameters against the power spectrum of the centered
		good_size(min(nx,ny)-256) square region. This spectrum has finer CTF oscillation detail
		but is noisier (less averaging), so the refinement is constrained to be local:
		defocus to +/-10% of the stage-1 value and dfdiff to min(dfdiffmax, 10% of the
		defocus). By this stage the CTF parameters are approximately correct, so the B-factor
		is allowed to adjust freely (bounded below by 0) but is softly biased toward lower
		values by _b_prior rather than hard-capped. Returns (df, dfdiff, dfang, B, score, side, sub)."""
		if self.verbose >= 1:
			print("  stage 3: final refinement (centered region) ...")
		side = good_size(min(self.nx_img, self.ny_img) - 2 * BORDMIN)
		side = max(16, side // 2 * 2)
		if side > min(self.nx_img, self.ny_img):
			side = min(self.nx_img, self.ny_img) // 2 * 2
		x0 = self.nx_img // 2 - side // 2
		y0 = self.ny_img // 2 - side // 2
		sub = self.img[y0:y0 + side, x0:x0 + side]
		if self.verbose >= 1:
			print(f"    region {side}x{side} from {self.nx_img}x{self.ny_img}")

		P = _rpower(sub)
		Pbg = _rpower(_gauss_blur_2d(sub, side / 48.0))
		self._prep_plane(P, Pbg, side)

		ddmax = min(self.dfdiffmax, 0.1 * df)
		dflo, dfhi = 0.9 * df, 1.1 * df
		scale = jnp.array([1.0, 0.1, 45.0, 50.0], jnp.float32)     # um, um, deg, A^2

		def loss(u):
			dfi, ddi, dai, Bi = u * scale
			return -self._score_plane(self._ctf2(self.s2, self.ang, dfi, ddi, dai), Bi) + self._b_prior(Bi)

		lv = jax.jit(jax.value_and_grad(loss, argnums=0))
		u = jnp.array([df, dd, da, B], jnp.float32) / scale
		optim = optax.adam(0.05)
		st = optim.init(u)
		best = (u, float('inf'))
		for it in range(niter):
			val, grad = lv(u)
			upd, st = optim.update(grad, st, u)
			u = optax.apply_updates(u, upd)
			u = u.at[0].set(jnp.clip(u[0], dflo / scale[0], dfhi / scale[0]))
			u = u.at[1].set(jnp.clip(u[1], 0.0, ddmax / scale[1]))
			u = u.at[3].set(jnp.maximum(u[3], 0.0))                  # B free (bounded below by 0)
			if self.verbose >= 2 and (it % 30 == 0 or it == niter - 1):
				print(f"    it={it:3d}/{niter}  loss={float(np.array(val)):.4f}  B={float(np.array(u * scale)[3]):.1f}")
			if val < best[1]:
				best = (u, val)
		u, val = best
		dfi, ddi, dai, Bi = np.array(u * scale)
		score = float(self._score_plane(self._ctf2(self.s2, self.ang, dfi, ddi, dai), Bi))
		if self.verbose >= 1:
			print(f"    defocus = {dfi:.4f} um  dfdiff = {ddi:.4f} um  dfang = {dai:.2f} deg  B = {Bi:.1f} A^2  (corr={score:.4f}, prior={float(np.array(val)) + score:.4f})")
		return float(dfi), float(ddi), float(dai), float(Bi), score, side, sub

	def _save_eval_image(self, df, dd, da, B, side, sub):
		"""Write eval.hdf, a side-by-side log-amplitude view arranged to read as one continuous
		power spectrum: the LEFT half is the real data (left semicircle of the centered-region
		micrograph FFT) and the RIGHT half is the model (the predicted CTF^2 modulation with B
		damping, mirrored to the right semicircle). Both origins sit at the center of the image so
		the Thon-ring shapes can be compared directly, left (real) vs right (model). Each half is
		independently normalized to fill 0-1 (DC excluded from the scale)."""
		n = side

		def amp_view(la):
			m = la.copy()
			m[n // 2, n // 2] = 0.0
			lo, hi = float(m.min()), float(m.max())
			if hi <= lo:
				return np.zeros_like(la, dtype=np.float32)
			r = (la - lo) / (hi - lo)
			return np.clip(r[:, :n // 2 + 1], 0.0, 1.0)          # left semicircle, DC at inner edge

		left = amp_view(np.log10(1.0 + np.abs(np.fft.fftshift(np.fft.fft2(sub)))))

		# model: predicted CTF^2 pattern on the full (centered) frequency plane — the grid must be
		# fftshifted so zero frequency sits at index (n//2, n//2), matching the centered data FFT,
		# otherwise the Thon rings end up centered on a corner instead of the origin
		fx = np.fft.fftshift(np.fft.fftfreq(n, d=1.0)) * n
		fy = np.fft.fftshift(np.fft.fftfreq(n, d=1.0)) * n
		Y, X = np.meshgrid(fy, fx, indexing='ij')
		s2 = (X * X + Y * Y) / (self.apix * n) ** 2
		ang = np.arctan2(Y, X)
		ctf2 = np.cos(-self.g1 * s2 * s2 + self.g2 * (df + (dd / 2.0) * np.cos(2.0 * ang - (2.0 * math.pi / 180.0) * da)) * s2
		              - self.ac_phase) ** 2
		# model: the predicted CTF^2 modulation (with B damping), 0..1; displayed as a clipped
		# log amplitude so the Thon-ring pattern fills the half (the background falloff is the
		# data's own object power, already shown in the left half)
		pattern = ctf2 * np.exp(-B * s2 / 4.0)
		right = amp_view(np.clip(np.log10(pattern + FLOOR), -3.0, 0.0))[:, ::-1]   # mirror to right semicircle

		comb = np.hstack((left, right)).astype(np.float32)
		io = ImageIO("eval.hdf", mode="rw")
		try:
			io.write_image(0, comb, {"nx": comb.shape[1], "ny": comb.shape[0], "nz": 1,
			                       "apix_x": self.apix, "apix_y": self.apix, "apix_z": self.apix})
		finally:
			io.close()

	def estimate(self):
		"""Run the staged CTF estimation. Returns a dict of CTF parameters plus debug data."""
		df, s1 = self._fit_defocus_radial()
		dd, da, B, s2 = self._fit_astig(df)
		dfi, ddi, dai, Bi, s3, side, sub = self._fit_final(df, dd, da, B)
		self._save_eval_image(dfi, ddi, dai, Bi, side, sub)
		return {"defocus": dfi, "dfdiff": ddi, "dfang": dai, "bfactor": Bi,
		        "score_radial": s1, "score_astig": s2, "score_final": s3,
		        "defocus_radial": df, "bfactor_astig": B,
		        "side": side, "sub": sub}


def estimate_ctf(img, apix, voltage, cs, ampcont, dfmin, dfmax, dfdiffmax, bfactor, bmax, bweight, verbose=0):
	"""Estimate the CTF of a single micrograph from power spectra.
	img - (H,W) real micrograph
	Returns a dict with: defocus, dfdiff, dfang, bfactor, apix, voltage, cs, ampcont,
	                     tile grid info, side, and the stage scores."""
	wl = 12.2639 / math.sqrt(voltage * 1000.0 + 0.97845 * voltage * voltage)   # A
	ac_phase = math.pi / 2.0 - math.asin(max(-1.0, min(1.0, ampcont / 100.0)))

	est = CtfPowerEstimator(img, apix, wl, cs, ac_phase, dfmin, dfmax, dfdiffmax,
	                        bfactor, bmax, bweight, verbose)
	params = est.estimate()

	params.update({"apix": apix, "voltage": voltage, "cs": cs, "ampcont": ampcont,
	               "tiles_x": est.n_tiles_x, "tiles_y": est.n_tiles_y,
	               "border_x": est.border_x, "border_y": est.border_y, "tile": TILE})
	return params


def main():
	progname = os.path.basename(sys.argv[0])
	usage = """prog [options] <micrograph>

Estimate the CTF (defocus, astigmatism amount and direction, B-factor) of a single
micrograph from power spectra: an averaged 512-px-tile power spectrum gives the mean
defocus (radially) and the astigmatism (2-D), then a centered larger region's power
spectrum is used for a local final refinement.

Requires voltage, Cs and amplitude contrast (or finds them in the header).


"""
	parser = EMArgumentParser(usage=usage, version=EMANVERSION)
	parser.add_pos_argument(name="micrograph", help="Input micrograph (2-D image file)", default="",
	                        guitype='filebox', row=0, col=0, rowspan=1, colspan=3)
	parser.add_argument("--voltage", type=float, default=None, help="Microscope voltage in kV (default from header if present)")
	parser.add_argument("--cs", type=float, default=None, help="Spherical aberration in mm (default from header if present)")
	parser.add_argument("--ampcont", "-a", type=float, default=None, help="Amplitude contrast as a percentage, eg 7 for 7%% (default from header if present)")
	parser.add_argument("--apix", type=float, default=0.0, help="Angstrom/pixel, if not found in the file")
	parser.add_argument("--dfmin", type=float, default=0.1, help="Minimum defocus to search, in um (default 0.1)")
	parser.add_argument("--dfmax", type=float, default=7.0, help="Maximum defocus to search, in um (default 7.0)")
	parser.add_argument("--dfdiffmax", type=float, default=0.15, help="Maximum astigmatism (defocus difference) to fit, in um (default 0.15; also capped at 10%% of the defocus)")
	parser.add_argument("--bfactor", type=float, default=30.0, help="Initial B-factor (A^2) of the Gaussian damping (default 30)")
	parser.add_argument("--bmax", type=float, default=100.0, help="Upper bound (A^2) for the B-factor in the astigmatism stage (default 100); in the final refinement the B-factor adjusts freely but is softly biased toward this reference value")
	parser.add_argument("--bweight", type=float, default=0.1, help="Strength of the soft bias toward lower B-factors in the final refinement, in units of the correlation score (default 0.1; 0 disables it)")
	parser.add_argument("--output", type=str, default=None, help="Output JSON filename (default <micrograph>_ctf.json)")
	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n", type=int, default=0, help="verbose level [0-9]")
	parser.add_argument("--ppid", type=int, default=-1, help="Set the PID of the parent process")

	(options, args) = parser.parse_args()
	logid = E3init(sys.argv, options.ppid)

	micro = args[0]
	if not os.path.isfile(micro):
		error_exit(f"Cannot find input micrograph: {micro}")

	io = ImageIO(micro, "r")
	try:
		img, hdr = io.read_image(0)
	finally:
		io.close()
	img = np.asarray(img, np.float32)
	if img.ndim != 2:
		error_exit(f"Expected a 2-D micrograph, got {img.ndim}-D data")

	def header_get(*keys, default=None):
		for k in keys:
			if hdr and k in hdr and hdr[k] is not None:
				return hdr[k]
		return default

	apix = options.apix if options.apix > 0 else float(header_get("apix_x", "apix", "pixel_size", default=0.0))
	if apix <= 0:
		error_exit("Could not determine apix. Specify --apix.")
	voltage = options.voltage if options.voltage is not None else header_get("voltage", "kV", "voltage_kV")
	cs = options.cs if options.cs is not None else header_get("cs", "cs_mm", "spherical_aberration")
	ampcont = options.ampcont if options.ampcont is not None else header_get("ampcont", "amp_contrast", "amp_cont")
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

	print(f"CTF estimation (power): {micro}")
	print(f"  apix={apix:.4f} A/px  voltage={voltage} kV  cs={cs} mm  ampcont={ampcont}%")

	t0 = time.time()
	params = estimate_ctf(img, apix, voltage, cs, ampcont,
	                      options.dfmin, options.dfmax, options.dfdiffmax,
	                      options.bfactor, options.bmax, options.bweight, options.verbose)
	el = time.time() - t0

	print(f"\n  RESULT (in {el:.1f}s):")
	print(f"    defocus  = {params['defocus']:.4f} um")
	print(f"    dfdiff   = {params['dfdiff']:.4f} um")
	print(f"    dfang    = {params['dfang']:.2f} deg")
	print(f"    bfactor  = {params['bfactor']:.1f} A^2")
	print(f"    corr (radial defocus fit) = {params['score_radial']:.4f}")
	print(f"    corr (astig fit)          = {params['score_astig']:.4f}")
	print(f"    corr (final fit)          = {params['score_final']:.4f}")

	out = options.output if options.output else os.path.splitext(micro)[0] + "_ctf.json"
	outdict = {k: params[k] for k in ("defocus", "dfdiff", "dfang", "bfactor", "apix", "voltage", "cs", "ampcont",
	                                 "tiles_x", "tiles_y", "border_x", "border_y", "tile", "side",
	                                 "defocus_radial", "bfactor_astig",
	                                 "score_radial", "score_astig", "score_final")}
	import json
	with open(out, "w") as fh:
		json.dump(outdict, fh, indent=2)
	print(f"  wrote {out}")
	print("  wrote eval.hdf")

	E3end(logid)


if __name__ == "__main__":
	main()
