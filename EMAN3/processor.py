"""EMAN3 processing functions.

This module reorganizes the EMAN2 Processor class hierarchy (libEM/processor.h/.cpp) into
plain JAX functions operating on EMStack objects:

- Each processor is a single module-level function. Names follow the EMAN2 convention of
  underscore-separated conceptual classification, e.g. "linearfilter_lowpass_gaussian".
- The first parameter of every processor is the input EMStack (EMStack2D or EMStack3D).
- Processors are pure: they return a NEW EMStack and never modify the input.
- Processing is applied to the entire stack in a single (JIT compiled) operation.
- Parameters after the stack carry unenforced type hints, which GUI programs may use to
  generate appropriate widgets. The PROCESSORS registry below (built from the function
  signatures) provides an enumeration of the available processors and their parameters.
- Where an operation makes sense in either real space or Fourier space, a Fourier-space
  input (complex data) produces a Fourier-space output of the same kind. Processors that
  make sense only in real space (e.g. masking) reject Fourier-space (complex) input.
- All image dimensions are assumed even; the image center (and the phase origin in Fourier
  space, when centered) is the pixel index (nx//2, ny//2, nz//2).

New processors should be added as module-level functions and listed in __all__.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints

import jax
import jax.numpy as jnp

from .EMAN3jax import EMStack, EMStack2D, EMStack3D
from .transform import Transform, TransformError, get_symmetry, Symmetry3D

__all__ = [
	"linearfilter_lowpass_gaussian",
	"linearfilter_lowpass_gaussian_2d_real",
	"linearfilter_lowpass_gaussian_2d_fourier",
	"linearfilter_lowpass_gaussian_3d_real",
	"linearfilter_lowpass_gaussian_3d_fourier",
	"linearfilter_lowpass_tophat",
	"linearfilter_lowpass_tophat_2d_real",
	"linearfilter_lowpass_tophat_2d_fourier",
	"linearfilter_lowpass_tophat_3d_real",
	"linearfilter_lowpass_tophat_3d_fourier",
	"linearfilter_highpass_gaussian",
	"linearfilter_highpass_gaussian_2d_real",
	"linearfilter_highpass_gaussian_2d_fourier",
	"linearfilter_highpass_gaussian_3d_real",
	"linearfilter_highpass_gaussian_3d_fourier",
	"linearfilter_highpass_tophat",
	"linearfilter_highpass_tophat_2d_real",
	"linearfilter_highpass_tophat_2d_fourier",
	"linearfilter_highpass_tophat_3d_real",
	"linearfilter_highpass_tophat_3d_fourier",
	"linearfilter_lowpass_butterworth",
	"linearfilter_lowpass_butterworth_2d_real",
	"linearfilter_lowpass_butterworth_2d_fourier",
	"linearfilter_lowpass_butterworth_3d_real",
	"linearfilter_lowpass_butterworth_3d_fourier",
	"linearfilter_highpass_butterworth",
	"linearfilter_highpass_butterworth_2d_real",
	"linearfilter_highpass_butterworth_2d_fourier",
	"linearfilter_highpass_butterworth_3d_real",
	"linearfilter_highpass_butterworth_3d_fourier",
	"mask_tophat",
	"mask_tophat_2d_real",
	"mask_tophat_3d_real",
	"mask_gaussian",
	"mask_gaussian_2d_real",
	"mask_gaussian_3d_real",
	"mask_edgefill",
	"mask_edgefill_2d_real_flat",
	"mask_edgefill_2d_real_apod",
	"mask_edgefill_3d_real_flat",
	"mask_edgefill_3d_real_apod",
	"normalize_standard",
	"normalize_standard_2d_real",
	"normalize_standard_3d_real",
	"normalize_edgemean",
	"normalize_edgemean_2d_real",
	"normalize_edgemean_3d_real",
	"normalize_circlemean",
	"normalize_circlemean_2d_real",
	"normalize_circlemean_3d_real",
	"normalize_mask",
	"normalize_mask_2d_real",
	"normalize_mask_3d_real",
	"xform",
	"xform_2d_real",
	"xform_3d_real",
	"xform_applysym",
	"xform_center",
	"xform_center_2d_real",
	"xform_center_3d_real",
	"xform_centeracf",
	"xform_centeracf_2d_real",
	"xform_centeracf_3d_real",
	"xform_centerofmass",
	"xform_centerofmass_2d_real",
	"xform_centerofmass_3d_real",
	"xform_flip",
	"xform_flip_x_2d_real",
	"xform_flip_y_2d_real",
	"xform_flip_x_3d_real",
	"xform_flip_y_3d_real",
	"xform_flip_z_3d_real",
	"xform_mirror",
	"xform_mirror_x_2d_real",
	"xform_mirror_y_2d_real",
	"xform_mirror_x_3d_real",
	"xform_mirror_y_3d_real",
	"xform_mirror_z_3d_real",
	"xform_phaseorigin",
	"xform_phaseorigin_2d_real",
	"xform_phaseorigin_2d_fourier",
	"xform_phaseorigin_3d_real",
	"xform_phaseorigin_3d_fourier",
	"PROCESSORS",
]


# ----------------------------------------------------------------------------------
# helper machinery
# ----------------------------------------------------------------------------------

def _resolve_cutoff(stack: EMStack, cutoff: float, cutoff_pix: float, cutoff_freq: float) -> float:
	"""Convert whichever of the three cutoff specifications was provided (all defaulting to
	None, exactly one must be non-None) into the canonical cutoff scale in which 0.5 = Nyquist
	(i.e. spatial frequency in cycles/pixel). Intended for use by any frequency-domain
	processor, not just the Gaussian lowpass."""

	provided = [n for n, v in (("cutoff", cutoff), ("cutoff_pix", cutoff_pix), ("cutoff_freq", cutoff_freq)) if v is not None]
	if len(provided) == 0: raise ValueError("one of cutoff, cutoff_pix or cutoff_freq must be provided")
	if len(provided) > 1: raise ValueError(f"only one of cutoff, cutoff_pix, cutoff_freq may be provided, got {provided}")

	if cutoff is not None:
		cut = float(cutoff)
	elif cutoff_pix is not None:
		## cutoff radius in Fourier pixels, normalized against the (largest) image dimension so that 0.5*Nyquist = N/2 pixels
		nmax = int(max(stack.shape[1:]))
		cut = float(cutoff_pix) / nmax
	else:
		## cutoff frequency in reciprocal Angstroms; 1 cycle/pixel = 1/apix 1/Angstrom, so multiply by apix to get cycles/pixel
		apix = stack.apix if stack.apix is not None else 1.0
		cut = float(cutoff_freq) * float(apix)

	if cut <= 0.0: raise ValueError(f"cutoff must be positive, got {cut} (0.5=Nyquist scale)")
	return cut


def _freq_axis_full(n: int) -> jnp.ndarray:
	"""Spatial frequency (cycles/pixel, 0.5 = Nyquist) of each position on a full (complex FFT) axis
	of length n. Positions at index i >= n/2 hold negative frequencies, so the index is folded
	(min(i, n-i)) to recover the magnitude."""
	idx = jnp.arange(n, dtype=jnp.float32)
	return jnp.minimum(idx, n - idx) / n

def _freq_axis_half(n: int) -> jnp.ndarray:
	"""Spatial frequency (cycles/pixel, 0.5 = Nyquist) of each position on a half (rfft) axis
	of length n = m/2+1. The frequencies are simply i/m."""
	idx = jnp.arange(n, dtype=jnp.float32)
	return idx / (2 * (n - 1))

def _gauss_halfwidth_filter_2d_rfft(ny: int, nxh: int, hw: float) -> jnp.ndarray:
	"""Gaussian lowpass kernel in Fourier space for an rfft2 array of shape (ny, nxh), where nxh = nx/2+1.
	Values = exp(-(r/hw)^2), where r is the spatial frequency in cycles/pixel (0.5 = Nyquist)."""
	r2 = _freq_axis_full(ny)[:, None]**2 + _freq_axis_half(nxh)[None, :]**2
	return jnp.exp(-r2 / hw**2)

def _gauss_halfwidth_filter_3d_rfft(nz: int, ny: int, nxh: int, hw: float) -> jnp.ndarray:
	"""Gaussian lowpass kernel in Fourier space for a rfftn(axes=(-3,-2,-1)) array of shape (nz, ny, nxh),
	where only the last axis is halved (nxh = nx/2+1).
	Values = exp(-(r/hw)^2), where r is the spatial frequency in cycles/pixel (0.5 = Nyquist)."""
	r2 = _freq_axis_full(nz)[:, None, None]**2 + _freq_axis_full(ny)[None, :, None]**2 + _freq_axis_half(nxh)[None, None, :]**2
	return jnp.exp(-r2 / hw**2)

def _toptohat_filter_2d_rfft(ny: int, nxh: int, cutoff: float) -> jnp.ndarray:
	"""Tophat (sharp cutoff) lowpass kernel in Fourier space for an rfft2 array of shape (ny, nxh), where nxh = nx/2+1.
	Value is 1 at or below the cutoff and 0 above, where the cutoff is on the 0.5=Nyquist scale."""
	r2 = _freq_axis_full(ny)[:, None]**2 + _freq_axis_half(nxh)[None, :]**2
	return (r2 <= cutoff**2).astype(jnp.float32)

def _toptohat_filter_3d_rfft(nz: int, ny: int, nxh: int, cutoff: float) -> jnp.ndarray:
	"""Tophat (sharp cutoff) lowpass kernel in Fourier space for a rfftn(axes=(-3,-2,-1)) array of shape (nz, ny, nxh),
	where only the last axis is halved (nxh = nx/2+1).
	Value is 1 at or below the cutoff and 0 above, where the cutoff is on the 0.5=Nyquist scale."""
	r2 = _freq_axis_full(nz)[:, None, None]**2 + _freq_axis_full(ny)[None, :, None]**2 + _freq_axis_half(nxh)[None, None, :]**2
	return (r2 <= cutoff**2).astype(jnp.float32)

def _butterworth_filter_2d_rfft(ny: int, nxh: int, cutoff: float, order: int) -> jnp.ndarray:
	"""Butterworth lowpass kernel in Fourier space for an rfft2 array of shape (ny, nxh), where nxh = nx/2+1.
	Values = 1 / sqrt(1 + (f/cutoff)^(2*order)), where f is the spatial frequency in cycles/pixel (0.5 = Nyquist).
	The filter value at the cutoff is 1/sqrt(2) for any order."""
	r2 = _freq_axis_full(ny)[:, None]**2 + _freq_axis_half(nxh)[None, :]**2
	return 1.0 / jnp.sqrt(1.0 + (r2 / cutoff**2) ** order)

def _butterworth_filter_3d_rfft(nz: int, ny: int, nxh: int, cutoff: float, order: int) -> jnp.ndarray:
	"""Butterworth lowpass kernel in Fourier space for a rfftn(axes=(-3,-2,-1)) array of shape (nz, ny, nxh),
	where only the last axis is halved (nxh = nx/2+1).
	Values = 1 / sqrt(1 + (f/cutoff)^(2*order)), where f is the spatial frequency in cycles/pixel (0.5 = Nyquist).
	The filter value at the cutoff is 1/sqrt(2) for any order."""
	r2 = _freq_axis_full(nz)[:, None, None]**2 + _freq_axis_full(ny)[None, :, None]**2 + _freq_axis_half(nxh)[None, None, :]**2
	return 1.0 / jnp.sqrt(1.0 + (r2 / cutoff**2) ** order)

def _toptohat_mask_2d(ny: int, nx: int, r_in: float, r_out: float, tx: float, ty: float) -> jnp.ndarray:
	"""Tophat mask of shape (ny, nx): 1 where radius_inner <= dist < radius_outer (in pixels), else 0.
	The mask center is at pixel index (ny//2 + ty, nx//2 + tx)."""
	y = jnp.arange(ny, dtype=jnp.float32)[:, None] - (ny // 2 + ty)
	x = jnp.arange(nx, dtype=jnp.float32)[None, :] - (nx // 2 + tx)
	d2 = x**2 + y**2
	return ((d2 >= r_in**2) & (d2 < r_out**2)).astype(jnp.float32)

def _toptohat_mask_3d(nz: int, ny: int, nx: int, r_in: float, r_out: float, tx: float, ty: float, tz: float) -> jnp.ndarray:
	"""Tophat mask of shape (nz, ny, nx): 1 where radius_inner <= dist < radius_outer (in pixels), else 0.
	The mask center is at pixel index (nz//2 + tz, ny//2 + ty, nx//2 + tx)."""
	z = jnp.arange(nz, dtype=jnp.float32)[:, None, None] - (nz // 2 + tz)
	y = jnp.arange(ny, dtype=jnp.float32)[None, :, None] - (ny // 2 + ty)
	x = jnp.arange(nx, dtype=jnp.float32)[None, None, :] - (nx // 2 + tx)
	d2 = x**2 + y**2 + z**2
	return ((d2 >= r_in**2) & (d2 < r_out**2)).astype(jnp.float32)

def _gauss_mask_2d(ny: int, nx: int, r_in: float, r_out: float, width: float, tx: float, ty: float) -> jnp.ndarray:
	"""Gaussian (soft) tophat mask of shape (ny, nx): 1.0 over the plateau radius_inner <= r <= radius_outer,
	with a Gaussian decay to 0 (1/e half-width = width, in pixels) beginning at each edge - inward at
	radius_inner, outward at radius_outer. radius_inner <= 0 gives no inner masking. Centered at pixel
	index (ny//2 + ty, nx//2 + tx)."""
	y = jnp.arange(ny, dtype=jnp.float32)[:, None] - (ny // 2 + ty)
	x = jnp.arange(nx, dtype=jnp.float32)[None, :] - (nx // 2 + tx)
	r = jnp.sqrt(x**2 + y**2)
	d_in = jnp.maximum(0.0, r_in - r)
	d_out = jnp.maximum(0.0, r - r_out)
	return jnp.exp(-(d_in**2 + d_out**2) / width**2).astype(jnp.float32)

def _gauss_mask_3d(nz: int, ny: int, nx: int, r_in: float, r_out: float, width: float, tx: float, ty: float, tz: float) -> jnp.ndarray:
	"""Gaussian (soft) tophat mask of shape (nz, ny, nx): 1.0 over the plateau radius_inner <= r <= radius_outer,
	with a Gaussian decay to 0 (1/e half-width = width, in pixels) beginning at each edge - inward at
	radius_inner, outward at radius_outer. radius_inner <= 0 gives no inner masking. Centered at pixel
	index (nz//2 + tz, ny//2 + ty, nx//2 + tx)."""
	z = jnp.arange(nz, dtype=jnp.float32)[:, None, None] - (nz // 2 + tz)
	y = jnp.arange(ny, dtype=jnp.float32)[None, :, None] - (ny // 2 + ty)
	x = jnp.arange(nx, dtype=jnp.float32)[None, None, :] - (nx // 2 + tx)
	r = jnp.sqrt(x**2 + y**2 + z**2)
	d_in = jnp.maximum(0.0, r_in - r)
	d_out = jnp.maximum(0.0, r - r_out)
	return jnp.exp(-(d_in**2 + d_out**2) / width**2).astype(jnp.float32)

def _edge_weight_flat(n: int, width: float) -> jnp.ndarray:
	"""1-D edge weight of length n for mask_edgefill, flat mode: 1.0 in the interior
	(width <= i < n-width) and 0.0 in the width-pixel edge bands at either end.
	Assumes 0 < width < n/2."""
	i = jnp.arange(n, dtype=jnp.float32)
	return ((i >= width) & (i < n - width)).astype(jnp.float32)

def _edge_weight_apod(n: int, width: float) -> jnp.ndarray:
	"""1-D edge weight of length n for mask_edgefill, apodized mode: 1.0 in the interior,
	decaying linearly to 0.0 over the width-pixel edge band at each end:
	weight(i) = min(i, n-1-i, width) / width.
	Assumes 0 < width < n/2."""
	i = jnp.arange(n, dtype=jnp.float32)
	return jnp.minimum(jnp.minimum(i / width, (n - 1 - i) / width), 1.0)


def _normalize(data: jnp.ndarray, region_mask: jnp.ndarray) -> jnp.ndarray:
	"""Shared body for the normalize_* processors. For each image in the stack it computes
	    result = (image - mean_region) / std_whole
	where mean_region = sum(image * region_mask) / count(region_mask > 0) is the mean of the image
	over the region, and std_whole is the population standard deviation of the image over all of its
	pixels. region_mask may be a spatial mask (shape data.shape[1:]) applied to every image, or a
	per-image mask (shape data.shape); it may be binary (0/1) or weighted. Branch-free: the
	data-dependent statistics use only sums, products and the sqrt of a non-negative quantity.
	"""
	S = data.ndim - 1                                                # number of spatial dims
	d_axes = tuple(range(1, data.ndim))                             # spatial axes of data
	rm_axes = tuple(range(region_mask.ndim - S, region_mask.ndim))  # spatial axes of region_mask
	n = data.size // data.shape[0]                                  # pixels per image (static)
	image_mask_sum = jnp.sum(data * region_mask, axis=d_axes)       # (N,)
	region_count = jnp.sum(region_mask > 0, axis=rm_axes)           # (N,) or ()
	mean_region = image_mask_sum / region_count                     # (N,)
	sum_all = jnp.sum(data, axis=d_axes)                            # (N,)
	mean_all = sum_all / n                                          # (N,)
	sumsq_all = jnp.sum(data * data, axis=d_axes)                   # (N,)
	std_all = jnp.sqrt(jnp.maximum(sumsq_all / n - mean_all ** 2, 0.0))  # (N,)
	out_shape = (data.shape[0],) + (1,) * S
	return (data - mean_region.reshape(out_shape)) / std_all.reshape(out_shape)


def _interior_weight(n: int) -> jnp.ndarray:
	"""1-D interior weight of length n: 1.0 at the interior indices (1..n-2) and 0.0 at the two
	edge positions (0 and n-1). Used to build the 1-pixel edge mask for normalize_edgemean."""
	i = jnp.arange(n, dtype=jnp.float32)
	return ((i >= 1.0) & (i < n - 1.0)).astype(jnp.float32)


def _ring_mask_2d(ny: int, nx: int, radius: float, width: float) -> jnp.ndarray:
	"""2-D ring (annulus) mask of shape (ny, nx): 1 where radius <= dist < radius+width (pixels),
	0 otherwise, centered at (ny//2, nx//2). radius >= 0 and width > 0 are assumed pre-validated."""
	y = jnp.arange(ny, dtype=jnp.float32)[:, None] - (ny // 2)
	x = jnp.arange(nx, dtype=jnp.float32)[None, :] - (nx // 2)
	dist = jnp.sqrt(x ** 2 + y ** 2)
	return ((dist >= radius) & (dist < radius + width)).astype(jnp.float32)


def _ring_mask_3d(nz: int, ny: int, nx: int, radius: float, width: float) -> jnp.ndarray:
	"""3-D ring (spherical shell) mask of shape (nz, ny, nx): 1 where radius <= dist < radius+width
	(pixels), 0 otherwise, centered at (nz//2, ny//2, nx//2). radius >= 0 and width > 0 are assumed
	pre-validated."""
	z = jnp.arange(nz, dtype=jnp.float32)[:, None, None] - (nz // 2)
	y = jnp.arange(ny, dtype=jnp.float32)[None, :, None] - (ny // 2)
	x = jnp.arange(nx, dtype=jnp.float32)[None, None, :] - (nx // 2)
	dist = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)
	return ((dist >= radius) & (dist < radius + width)).astype(jnp.float32)


# ----------------------------------------------------------------------------------
# dispatch helper
# ----------------------------------------------------------------------------------

def _process_stack(stack: EMStack, v_2d_real, v_2d_fourier, v_3d_real, v_3d_fourier, *extra) -> EMStack:
	"""Apply the appropriate JIT variant (2D/3D x real/Fourier) to a stack and return the result
	as a new stack of the same kind with inherited metadata.
	Any arguments after the four variants (e.g. the resolved cutoff, the Butterworth order)
	are forwarded positionally to the JIT variant."""
	data = stack.jax
	in_fourier = jnp.issubdtype(data.dtype, jnp.complexfloating)
	if isinstance(stack, EMStack2D):
		newdata = v_2d_fourier(data, *extra) if in_fourier else v_2d_real(data, *extra)
	else:
		newdata = v_3d_fourier(data, *extra) if in_fourier else v_3d_real(data, *extra)
	return type(stack)(newdata, parent=stack)

def _process_stack_real(stack: EMStack, v_2d_real, v_3d_real, *extra) -> EMStack:
	"""Real-space-only counterpart of _process_stack, for processors (e.g. masking) that make sense
	only on real data: dispatches on the stack's dimensionality, rejects Fourier-space (complex)
	input, and returns the result as a new stack of the same kind with inherited metadata.
	Arguments after the two variants are forwarded to the JIT variant."""
	data = stack.jax
	if jnp.issubdtype(data.dtype, jnp.complexfloating):
		raise ValueError("this processor operates on real-space data only; a Fourier-space (complex) stack was provided")
	if isinstance(stack, EMStack2D):
		newdata = v_2d_real(data, *extra)
	else:
		newdata = v_3d_real(data, *extra)
	return type(stack)(newdata, parent=stack)


# ----------------------------------------------------------------------------------
# linearfilter_lowpass_gaussian
# ----------------------------------------------------------------------------------
# four JIT compiled subfunctions, covering {2D,3D} x {real space, Fourier space}.
# Each takes a stack-shaped jax array {N,...} plus the half-width in the 0.5=Nyquist scale
# as its only additional parameter, and is free of conditionals.

@jax.jit
def linearfilter_lowpass_gaussian_2d_real(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a Gaussian lowpass to a stack of 2-D images in real space (data: float {N,Y,X}).
	cutoff is the half-width of the Gaussian (1/e point) on the 0.5=Nyquist scale."""
	filt = _gauss_halfwidth_filter_2d_rfft(data.shape[-2], data.shape[-1] // 2 + 1, cutoff)
	return jnp.fft.irfft2(jnp.fft.rfft2(data) * filt)

@jax.jit
def linearfilter_lowpass_gaussian_2d_fourier(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a Gaussian lowpass to a stack of 2-D images in Fourier space (data: complex rfft2 {N,Y,X/2+1},
	i.e. the FFT of real data). cutoff is the half-width of the Gaussian (1/e point) on the 0.5=Nyquist scale."""
	filt = _gauss_halfwidth_filter_2d_rfft(data.shape[-2], data.shape[-1], cutoff)
	return data * filt

@jax.jit
def linearfilter_lowpass_gaussian_3d_real(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a Gaussian lowpass to a stack of 3-D volumes in real space (data: float {N,Z,Y,X}).
	cutoff is the half-width of the Gaussian (1/e point) on the 0.5=Nyquist scale."""
	filt = _gauss_halfwidth_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1] // 2 + 1, cutoff)
	return jnp.fft.irfftn(jnp.fft.rfftn(data, axes=(-3, -2, -1)) * filt, axes=(-3, -2, -1))

@jax.jit
def linearfilter_lowpass_gaussian_3d_fourier(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a Gaussian lowpass to a stack of 3-D volumes in Fourier space (data: complex rfftn {N,Z,Y,X/2+1},
	i.e. the FFT of real data). cutoff is the half-width of the Gaussian (1/e point) on the 0.5=Nyquist scale."""
	filt = _gauss_halfwidth_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1], cutoff)
	return data * filt


def linearfilter_lowpass_gaussian(stack: EMStack,
								  cutoff: float = None,
								  cutoff_pix: float = None,
								  cutoff_freq: float = None) -> EMStack:
	"""Gaussian lowpass filter a stack of 2-D images or 3-D volumes, applied to the entire stack at once.
	Returns a new EMStack of the same kind; the input is not modified.

	The input may be in real space or Fourier space (the FFT of real data). A Fourier-space input
	returns a Fourier-space result, so it can be chained with other Fourier-space processors between
	a single FFT and IFT; a real-space input returns a real-space result (the FFT/IFT are done
	internally for convenience).

	The half-width of the Gaussian (where the filter value is 1/e) may be specified in any ONE of three ways:
		cutoff       - the half-width in terms of Nyquist = 0.5 (cycles/pixel, as used by jax_pointfilt_2d)
		cutoff_pix   - the same half-width expressed as a radius in Fourier pixels (relative to the image dimension)
		cutoff_freq  - the same half-width expressed as a spatial frequency in reciprocal Angstroms
					    (uses the stack's apix; requires stack.apix to be set, otherwise apix = 1.0 is assumed)

	Example: linearfilter_lowpass_gaussian(my_stack, cutoff=0.1)
	"""
	return _process_stack(stack,
						    linearfilter_lowpass_gaussian_2d_real, linearfilter_lowpass_gaussian_2d_fourier,
					    linearfilter_lowpass_gaussian_3d_real, linearfilter_lowpass_gaussian_3d_fourier,
					    _resolve_cutoff(stack, cutoff, cutoff_pix, cutoff_freq))


# ----------------------------------------------------------------------------------
# linearfilter_lowpass_tophat
# ----------------------------------------------------------------------------------

@jax.jit
def linearfilter_lowpass_tophat_2d_real(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a tophat (sharp cutoff) lowpass to a stack of 2-D images in real space (data: float {N,Y,X}).
	cutoff is on the 0.5=Nyquist scale; the filter is 1 at or below the cutoff and 0 above."""
	filt = _toptohat_filter_2d_rfft(data.shape[-2], data.shape[-1] // 2 + 1, cutoff)
	return jnp.fft.irfft2(jnp.fft.rfft2(data) * filt)

@jax.jit
def linearfilter_lowpass_tophat_2d_fourier(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a tophat (sharp cutoff) lowpass to a stack of 2-D images in Fourier space (data: complex rfft2 {N,Y,X/2+1},
	i.e. the FFT of real data). cutoff is on the 0.5=Nyquist scale; the filter is 1 at or below the cutoff and 0 above."""
	filt = _toptohat_filter_2d_rfft(data.shape[-2], data.shape[-1], cutoff)
	return data * filt

@jax.jit
def linearfilter_lowpass_tophat_3d_real(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a tophat (sharp cutoff) lowpass to a stack of 3-D volumes in real space (data: float {N,Z,Y,X}).
	cutoff is on the 0.5=Nyquist scale; the filter is 1 at or below the cutoff and 0 above."""
	filt = _toptohat_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1] // 2 + 1, cutoff)
	return jnp.fft.irfftn(jnp.fft.rfftn(data, axes=(-3, -2, -1)) * filt, axes=(-3, -2, -1))

@jax.jit
def linearfilter_lowpass_tophat_3d_fourier(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a tophat (sharp cutoff) lowpass to a stack of 3-D volumes in Fourier space (data: complex rfftn {N,Z,Y,X/2+1},
	i.e. the FFT of real data). cutoff is on the 0.5=Nyquist scale; the filter is 1 at or below the cutoff and 0 above."""
	filt = _toptohat_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1], cutoff)
	return data * filt


def linearfilter_lowpass_tophat(stack: EMStack,
							     cutoff: float = None,
							     cutoff_pix: float = None,
							     cutoff_freq: float = None) -> EMStack:
	"""Tophat (sharp cutoff) lowpass filter a stack of 2-D images or 3-D volumes, applied to the entire stack at once:
	frequencies at or below the cutoff pass unchanged (filter value 1), all others are removed (filter value 0).
	Returns a new EMStack of the same kind; the input is not modified.
	A Fourier-space input (the FFT of real data) returns a Fourier-space result; a real-space input returns a
	real-space result (the FFT/IFT are done internally for convenience).

	The cutoff may be specified in any ONE of three ways:
		cutoff       - in terms of Nyquist = 0.5 (cycles/pixel)
		cutoff_pix   - as a radius in Fourier pixels (relative to the image dimension)
		cutoff_freq  - as a spatial frequency in reciprocal Angstroms (uses the stack's apix,
					    defaulting to 1.0 if unset)

	Example: linearfilter_lowpass_tophat(my_stack, cutoff=0.2)
	"""
	return _process_stack(stack,
					    linearfilter_lowpass_tophat_2d_real, linearfilter_lowpass_tophat_2d_fourier,
					    linearfilter_lowpass_tophat_3d_real, linearfilter_lowpass_tophat_3d_fourier,
					    _resolve_cutoff(stack, cutoff, cutoff_pix, cutoff_freq))


# ----------------------------------------------------------------------------------
# highpass variants (each is 1.0 - the corresponding lowpass filter)
# ----------------------------------------------------------------------------------

@jax.jit
def linearfilter_highpass_gaussian_2d_real(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a Gaussian highpass to a stack of 2-D images in real space (data: float {N,Y,X}).
	The filter is 1.0 minus the Gaussian lowpass; cutoff is its half-width (1/e point) on the 0.5=Nyquist scale."""
	filt = 1.0 - _gauss_halfwidth_filter_2d_rfft(data.shape[-2], data.shape[-1] // 2 + 1, cutoff)
	return jnp.fft.irfft2(jnp.fft.rfft2(data) * filt)

@jax.jit
def linearfilter_highpass_gaussian_2d_fourier(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a Gaussian highpass to a stack of 2-D images in Fourier space (data: complex rfft2 {N,Y,X/2+1},
	i.e. the FFT of real data). The filter is 1.0 minus the Gaussian lowpass; cutoff is on the 0.5=Nyquist scale."""
	filt = 1.0 - _gauss_halfwidth_filter_2d_rfft(data.shape[-2], data.shape[-1], cutoff)
	return data * filt

@jax.jit
def linearfilter_highpass_gaussian_3d_real(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a Gaussian highpass to a stack of 3-D volumes in real space (data: float {N,Z,Y,X}).
	The filter is 1.0 minus the Gaussian lowpass; cutoff is its half-width (1/e point) on the 0.5=Nyquist scale."""
	filt = 1.0 - _gauss_halfwidth_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1] // 2 + 1, cutoff)
	return jnp.fft.irfftn(jnp.fft.rfftn(data, axes=(-3, -2, -1)) * filt, axes=(-3, -2, -1))

@jax.jit
def linearfilter_highpass_gaussian_3d_fourier(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a Gaussian highpass to a stack of 3-D volumes in Fourier space (data: complex rfftn {N,Z,Y,X/2+1},
	i.e. the FFT of real data). The filter is 1.0 minus the Gaussian lowpass; cutoff is on the 0.5=Nyquist scale."""
	filt = 1.0 - _gauss_halfwidth_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1], cutoff)
	return data * filt


def linearfilter_highpass_gaussian(stack: EMStack,
							       cutoff: float = None,
							       cutoff_pix: float = None,
							       cutoff_freq: float = None) -> EMStack:
	"""Gaussian highpass filter a stack of 2-D images or 3-D volumes, applied to the entire stack at once.
	The filter is 1.0 minus the Gaussian lowpass of the same half-width: it removes low frequencies
	(generating a DC-free, edge-enhanced result) while passing high frequencies.
	Returns a new EMStack of the same kind; the input is not modified.
	A Fourier-space input (the FFT of real data) returns a Fourier-space result; a real-space input returns a
	real-space result (the FFT/IFT are done internally for convenience).

	The half-width of the Gaussian (where the LOWPASS value is 1/e) may be specified in any ONE of three ways:
		cutoff       - in terms of Nyquist = 0.5 (cycles/pixel)
		cutoff_pix   - as a radius in Fourier pixels (relative to the image dimension)
		cutoff_freq  - as a spatial frequency in reciprocal Angstroms (uses the stack's apix,
					    defaulting to 1.0 if unset)

	Example: linearfilter_highpass_gaussian(my_stack, cutoff=0.1)
	"""
	return _process_stack(stack,
					    linearfilter_highpass_gaussian_2d_real, linearfilter_highpass_gaussian_2d_fourier,
					    linearfilter_highpass_gaussian_3d_real, linearfilter_highpass_gaussian_3d_fourier,
					    _resolve_cutoff(stack, cutoff, cutoff_pix, cutoff_freq))


@jax.jit
def linearfilter_highpass_tophat_2d_real(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a tophat (sharp cutoff) highpass to a stack of 2-D images in real space (data: float {N,Y,X}).
	The filter is 1.0 minus the tophat lowpass: 0 at or below the cutoff, 1 above. cutoff is on the 0.5=Nyquist scale."""
	filt = 1.0 - _toptohat_filter_2d_rfft(data.shape[-2], data.shape[-1] // 2 + 1, cutoff)
	return jnp.fft.irfft2(jnp.fft.rfft2(data) * filt)

@jax.jit
def linearfilter_highpass_tophat_2d_fourier(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a tophat (sharp cutoff) highpass to a stack of 2-D images in Fourier space (data: complex rfft2 {N,Y,X/2+1},
	i.e. the FFT of real data). The filter is 1.0 minus the tophat lowpass: 0 at or below the cutoff, 1 above."""
	filt = 1.0 - _toptohat_filter_2d_rfft(data.shape[-2], data.shape[-1], cutoff)
	return data * filt

@jax.jit
def linearfilter_highpass_tophat_3d_real(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a tophat (sharp cutoff) highpass to a stack of 3-D volumes in real space (data: float {N,Z,Y,X}).
	The filter is 1.0 minus the tophat lowpass: 0 at or below the cutoff, 1 above. cutoff is on the 0.5=Nyquist scale."""
	filt = 1.0 - _toptohat_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1] // 2 + 1, cutoff)
	return jnp.fft.irfftn(jnp.fft.rfftn(data, axes=(-3, -2, -1)) * filt, axes=(-3, -2, -1))

@jax.jit
def linearfilter_highpass_tophat_3d_fourier(data: jnp.ndarray, cutoff: float) -> jnp.ndarray:
	"""Apply a tophat (sharp cutoff) highpass to a stack of 3-D volumes in Fourier space (data: complex rfftn {N,Z,Y,X/2+1},
	i.e. the FFT of real data). The filter is 1.0 minus the tophat lowpass: 0 at or below the cutoff, 1 above."""
	filt = 1.0 - _toptohat_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1], cutoff)
	return data * filt


def linearfilter_highpass_tophat(stack: EMStack,
							     cutoff: float = None,
							     cutoff_pix: float = None,
							     cutoff_freq: float = None) -> EMStack:
	"""Tophat (sharp cutoff) highpass filter a stack of 2-D images or 3-D volumes, applied to the entire stack at once:
	frequencies above the cutoff pass unchanged (filter value 1), all at or below are removed (filter value 0).
	It is 1.0 minus the corresponding tophat lowpass.
	Returns a new EMStack of the same kind; the input is not modified.
	A Fourier-space input (the FFT of real data) returns a Fourier-space result; a real-space input returns a
	real-space result (the FFT/IFT are done internally for convenience).

	The cutoff may be specified in any ONE of three ways:
		cutoff       - in terms of Nyquist = 0.5 (cycles/pixel)
		cutoff_pix   - as a radius in Fourier pixels (relative to the image dimension)
		cutoff_freq  - as a spatial frequency in reciprocal Angstroms (uses the stack's apix,
					    defaulting to 1.0 if unset)

	Example: linearfilter_highpass_tophat(my_stack, cutoff=0.2)
	"""
	return _process_stack(stack,
					    linearfilter_highpass_tophat_2d_real, linearfilter_highpass_tophat_2d_fourier,
					    linearfilter_highpass_tophat_3d_real, linearfilter_highpass_tophat_3d_fourier,
					    _resolve_cutoff(stack, cutoff, cutoff_pix, cutoff_freq))


# ----------------------------------------------------------------------------------
# linearfilter_lowpass_butterworth / linearfilter_highpass_butterworth
# ----------------------------------------------------------------------------------
# Classical single-cutoff Butterworth pair. The lowpass has (amplitude) transfer function
# H_lp(f) = 1 / sqrt(1 + (f/cutoff)^(2*order)) and the highpass is its exact complement,
# H_hp(f) = 1 - H_lp(f), so the two sum to 1 at every frequency. The value at the
# cutoff is 1/sqrt(2) for any order; the order controls the steepness of the transition.

@jax.jit
def linearfilter_lowpass_butterworth_2d_real(data: jnp.ndarray, cutoff: float, order: int) -> jnp.ndarray:
	"""Apply a Butterworth lowpass to a stack of 2-D images in real space (data: float {N,Y,X}).
	cutoff is on the 0.5=Nyquist scale; order is the Butterworth order."""
	filt = _butterworth_filter_2d_rfft(data.shape[-2], data.shape[-1] // 2 + 1, cutoff, order)
	return jnp.fft.irfft2(jnp.fft.rfft2(data) * filt)

@jax.jit
def linearfilter_lowpass_butterworth_2d_fourier(data: jnp.ndarray, cutoff: float, order: int) -> jnp.ndarray:
	"""Apply a Butterworth lowpass to a stack of 2-D images in Fourier space (data: complex rfft2 {N,Y,X/2+1},
	i.e. the FFT of real data). cutoff is on the 0.5=Nyquist scale; order is the Butterworth order."""
	filt = _butterworth_filter_2d_rfft(data.shape[-2], data.shape[-1], cutoff, order)
	return data * filt

@jax.jit
def linearfilter_lowpass_butterworth_3d_real(data: jnp.ndarray, cutoff: float, order: int) -> jnp.ndarray:
	"""Apply a Butterworth lowpass to a stack of 3-D volumes in real space (data: float {N,Z,Y,X}).
	cutoff is on the 0.5=Nyquist scale; order is the Butterworth order."""
	filt = _butterworth_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1] // 2 + 1, cutoff, order)
	return jnp.fft.irfftn(jnp.fft.rfftn(data, axes=(-3, -2, -1)) * filt, axes=(-3, -2, -1))

@jax.jit
def linearfilter_lowpass_butterworth_3d_fourier(data: jnp.ndarray, cutoff: float, order: int) -> jnp.ndarray:
	"""Apply a Butterworth lowpass to a stack of 3-D volumes in Fourier space (data: complex rfftn {N,Z,Y,X/2+1},
	i.e. the FFT of real data). cutoff is on the 0.5=Nyquist scale; order is the Butterworth order."""
	filt = _butterworth_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1], cutoff, order)
	return data * filt


def linearfilter_lowpass_butterworth(stack: EMStack,
								  cutoff: float = None,
								  cutoff_pix: float = None,
								  cutoff_freq: float = None,
								  order: int = 2) -> EMStack:
	"""Butterworth lowpass filter a stack of 2-D images or 3-D volumes, applied to the entire stack at once.
	The (amplitude) transfer function is H(f) = 1 / sqrt(1 + (f/cutoff)^(2*order)): the value at the cutoff is
	1/sqrt(2) (-3 dB) and higher frequencies are progressively attenuated, more steeply the higher the order.
	Returns a new EMStack of the same kind; the input is not modified.
	A Fourier-space input (the FFT of real data) returns a Fourier-space result; a real-space input returns a
	real-space result (the FFT/IFT are done internally for convenience).

	The cutoff may be specified in any ONE of three ways:
		cutoff       - in terms of Nyquist = 0.5 (cycles/pixel)
		cutoff_pix   - as a radius in Fourier pixels (relative to the image dimension)
		cutoff_freq  - as a spatial frequency in reciprocal Angstroms (uses the stack's apix,
					    defaulting to 1.0 if unset)
		order        - the Butterworth order (positive integer, default 2)

	Example: linearfilter_lowpass_butterworth(my_stack, cutoff=0.1, order=2)
	"""
	order = int(order)
	if order < 1: raise ValueError(f"Butterworth order must be a positive integer, got {order}")
	return _process_stack(stack,
					    linearfilter_lowpass_butterworth_2d_real, linearfilter_lowpass_butterworth_2d_fourier,
					    linearfilter_lowpass_butterworth_3d_real, linearfilter_lowpass_butterworth_3d_fourier,
					    _resolve_cutoff(stack, cutoff, cutoff_pix, cutoff_freq),
					    order)


@jax.jit
def linearfilter_highpass_butterworth_2d_real(data: jnp.ndarray, cutoff: float, order: int) -> jnp.ndarray:
	"""Apply a Butterworth highpass to a stack of 2-D images in real space (data: float {N,Y,X}).
	The filter is 1.0 minus the Butterworth lowpass of the same cutoff and order.
	cutoff is on the 0.5=Nyquist scale; order is the Butterworth order."""
	filt = 1.0 - _butterworth_filter_2d_rfft(data.shape[-2], data.shape[-1] // 2 + 1, cutoff, order)
	return jnp.fft.irfft2(jnp.fft.rfft2(data) * filt)

@jax.jit
def linearfilter_highpass_butterworth_2d_fourier(data: jnp.ndarray, cutoff: float, order: int) -> jnp.ndarray:
	"""Apply a Butterworth highpass to a stack of 2-D images in Fourier space (data: complex rfft2 {N,Y,X/2+1},
	i.e. the FFT of real data). The filter is 1.0 minus the Butterworth lowpass of the same cutoff and order.
	cutoff is on the 0.5=Nyquist scale; order is the Butterworth order."""
	filt = 1.0 - _butterworth_filter_2d_rfft(data.shape[-2], data.shape[-1], cutoff, order)
	return data * filt

@jax.jit
def linearfilter_highpass_butterworth_3d_real(data: jnp.ndarray, cutoff: float, order: int) -> jnp.ndarray:
	"""Apply a Butterworth highpass to a stack of 3-D volumes in real space (data: float {N,Z,Y,X}).
	The filter is 1.0 minus the Butterworth lowpass of the same cutoff and order.
	cutoff is on the 0.5=Nyquist scale; order is the Butterworth order."""
	filt = 1.0 - _butterworth_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1] // 2 + 1, cutoff, order)
	return jnp.fft.irfftn(jnp.fft.rfftn(data, axes=(-3, -2, -1)) * filt, axes=(-3, -2, -1))

@jax.jit
def linearfilter_highpass_butterworth_3d_fourier(data: jnp.ndarray, cutoff: float, order: int) -> jnp.ndarray:
	"""Apply a Butterworth highpass to a stack of 3-D volumes in Fourier space (data: complex rfftn {N,Z,Y,X/2+1},
	i.e. the FFT of real data). The filter is 1.0 minus the Butterworth lowpass of the same cutoff and order.
	cutoff is on the 0.5=Nyquist scale; order is the Butterworth order."""
	filt = 1.0 - _butterworth_filter_3d_rfft(data.shape[-3], data.shape[-2], data.shape[-1], cutoff, order)
	return data * filt


def linearfilter_highpass_butterworth(stack: EMStack,
								   cutoff: float = None,
								   cutoff_pix: float = None,
								   cutoff_freq: float = None,
								   order: int = 2) -> EMStack:
	"""Butterworth highpass filter a stack of 2-D images or 3-D volumes, applied to the entire stack at once.
	It is 1.0 minus the Butterworth lowpass of the same cutoff and order:
	H(f) = 1 - 1/sqrt(1 + (f/cutoff)^(2*order)),
	so it removes low frequencies (generating a DC-free, edge-enhanced result) while passing high frequencies.
	Returns a new EMStack of the same kind; the input is not modified.
	A Fourier-space input (the FFT of real data) returns a Fourier-space result; a real-space input returns a
	real-space result (the FFT/IFT are done internally for convenience).

	The cutoff may be specified in any ONE of three ways:
		cutoff       - in terms of Nyquist = 0.5 (cycles/pixel)
		cutoff_pix   - as a radius in Fourier pixels (relative to the image dimension)
		cutoff_freq  - as a spatial frequency in reciprocal Angstroms (uses the stack's apix,
					    defaulting to 1.0 if unset)
		order        - the Butterworth order (positive integer, default 2)

	Example: linearfilter_highpass_butterworth(my_stack, cutoff=0.1, order=2)
	"""
	order = int(order)
	if order < 1: raise ValueError(f"Butterworth order must be a positive integer, got {order}")
	return _process_stack(stack,
					    linearfilter_highpass_butterworth_2d_real, linearfilter_highpass_butterworth_2d_fourier,
					    linearfilter_highpass_butterworth_3d_real, linearfilter_highpass_butterworth_3d_fourier,
					    _resolve_cutoff(stack, cutoff, cutoff_pix, cutoff_freq),
					    order)


# ----------------------------------------------------------------------------------
# mask_tophat
# ----------------------------------------------------------------------------------
# Masking is a real-space operation (a 1/0 mask is multiplied into the data); complex
# (Fourier-space) input is rejected by _process_stack_real.

@jax.jit
def mask_tophat_2d_real(data: jnp.ndarray, radius_inner: float, radius_outer: float, tx: float, ty: float, tz: float) -> jnp.ndarray:
	"""Multiply a stack of 2-D images in real space (data: float {N,Y,X}) by a tophat (circular) mask:
	1 where radius_inner <= dist < radius_outer (in pixels), 0 otherwise, centered at (Y//2 + ty, X//2 + tx).
	tz is accepted for signature uniformity with the 3-D variant but ignored in 2-D."""
	mask = _toptohat_mask_2d(data.shape[-2], data.shape[-1], radius_inner, radius_outer, tx, ty)
	return data * mask[None, :, :]

@jax.jit
def mask_tophat_3d_real(data: jnp.ndarray, radius_inner: float, radius_outer: float, tx: float, ty: float, tz: float) -> jnp.ndarray:
	"""Multiply a stack of 3-D volumes in real space (data: float {N,Z,Y,X}) by a tophat (spherical) mask:
	1 where radius_inner <= dist < radius_outer (in pixels), 0 otherwise, centered at (Z//2 + tz, Y//2 + ty, X//2 + tx)."""
	mask = _toptohat_mask_3d(data.shape[-3], data.shape[-2], data.shape[-1], radius_inner, radius_outer, tx, ty, tz)
	return data * mask[None, :, :, :]


def mask_tophat(stack: EMStack,
				      radius_inner: float = None,
				      radius_outer: float = None,
				      tx: float = 0.0,
				      ty: float = 0.0,
				      tz: float = 0.0) -> EMStack:
	"""Tophat (sharp circular / spherical) mask a stack of 2-D images or 3-D volumes, applied to the
	entire stack at once. Pixels at distance radius_inner <= dist < radius_outer (in pixels) from the
	mask center are kept, all others are set to zero.
	Returns a new EMStack of the same kind; the input is not modified.
	Masking is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even; the image center is the pixel (nx//2, ny//2, nz//2).

	radius_inner - inner radius in pixels (default 0, i.e. a solid disc/sphere; pixels strictly
				   inside it are set to zero)
	radius_outer - outer radius in pixels (required; pixels at or beyond it are set to zero)
	tx, ty, tz   - offset of the mask center from the image center, in pixels (default 0);
				   tz is ignored for 2-D stacks

	Example: mask_tophat(my_stack, radius_outer=100)
	"""
	r_in = 0.0 if radius_inner is None else float(radius_inner)
	if radius_outer is None:
		raise ValueError("radius_outer (in pixels) must be provided")
	r_out = float(radius_outer)
	if r_out <= 0.0:
		raise ValueError(f"radius_outer must be positive, got {r_out}")
	if r_in < 0.0:
		raise ValueError(f"radius_inner must be non-negative, got {r_in}")
	if r_in > r_out:
		raise ValueError(f"radius_inner ({r_in}) must not exceed radius_outer ({r_out})")
	return _process_stack_real(stack,
					       mask_tophat_2d_real, mask_tophat_3d_real,
					       r_in, r_out,
					       float(tx), float(ty), float(tz))


# ----------------------------------------------------------------------------------
# mask_gaussian
# ----------------------------------------------------------------------------------
# Same geometry and centering as mask_tophat, but with a Gaussian (1/e half-width = width)
# decay to 0 beginning at each edge instead of a sharp cutoff.

@jax.jit
def mask_gaussian_2d_real(data: jnp.ndarray, radius_inner: float, radius_outer: float, width: float, tx: float, ty: float, tz: float) -> jnp.ndarray:
	"""Multiply a stack of 2-D images in real space (data: float {N,Y,X}) by a Gaussian (soft) tophat mask:
	1.0 over the plateau radius_inner <= dist <= radius_outer, with a Gaussian decay to 0 (1/e half-width =
	width, in pixels) beginning at each edge, centered at (Y//2 + ty, X//2 + tx).
	tz is accepted for signature uniformity with the 3-D variant but ignored in 2-D."""
	mask = _gauss_mask_2d(data.shape[-2], data.shape[-1], radius_inner, radius_outer, width, tx, ty)
	return data * mask[None, :, :]

@jax.jit
def mask_gaussian_3d_real(data: jnp.ndarray, radius_inner: float, radius_outer: float, width: float, tx: float, ty: float, tz: float) -> jnp.ndarray:
	"""Multiply a stack of 3-D volumes in real space (data: float {N,Z,Y,X}) by a Gaussian (soft) tophat mask:
	1.0 over the plateau radius_inner <= dist <= radius_outer, with a Gaussian decay to 0 (1/e half-width =
	width, in pixels) beginning at each edge, centered at (Z//2 + tz, Y//2 + ty, X//2 + tx)."""
	mask = _gauss_mask_3d(data.shape[-3], data.shape[-2], data.shape[-1], radius_inner, radius_outer, width, tx, ty, tz)
	return data * mask[None, :, :, :]


def mask_gaussian(stack: EMStack,
				     radius_inner: float = None,
				     radius_outer: float = None,
				     width: float = None,
				     tx: float = 0.0,
				     ty: float = 0.0,
				     tz: float = 0.0) -> EMStack:
	"""Gaussian (soft) tophat mask a stack of 2-D images or 3-D volumes, applied to the entire stack at
	once. Like mask_tophat, radius_inner and radius_outer (in pixels) specify the edges of the tophat
	plateau (value 1.0); rather than a sharp cutoff to zero, each edge begins a Gaussian decay to zero,
	inward at radius_inner and outward at radius_outer, with 1/e half-width = width (pixels):
	the mask value is 1/e at one width past each edge, falling to 0 a few widths beyond.
	If radius_inner <= 0 there is no inner masking (the plateau starts at the center).
	Returns a new EMStack of the same kind; the input is not modified.
	Masking is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even; the image center is the pixel (nx//2, ny//2, nz//2).

	radius_inner - inner edge of the plateau in pixels (default 0, i.e. no inner masking;
				   values <= 0 both mean no inner masking)
	radius_outer - outer edge of the plateau in pixels (required)
	width        - 1/e half-width of the Gaussian decay at each edge, in pixels (required, positive)
	tx, ty, tz   - offset of the mask center from the image center, in pixels (default 0);
				   tz is ignored for 2-D stacks

	Example: mask_gaussian(my_stack, radius_outer=100, width=10)
	"""
	if radius_outer is None:
		raise ValueError("radius_outer (in pixels) must be provided")
	r_out = float(radius_outer)
	if r_out <= 0.0:
		raise ValueError(f"radius_outer must be positive, got {r_out}")
	r_in = 0.0 if radius_inner is None else float(radius_inner)
	if r_in > r_out:
		raise ValueError(f"radius_inner ({r_in}) must not exceed radius_outer ({r_out})")
	# radius_inner <= 0 means no inner masking (the kernel's max(0, r_in - r) then vanishes)
	if width is None:
		raise ValueError("width (in pixels) must be provided")
	w = float(width)
	if w <= 0.0:
		raise ValueError(f"width must be positive, got {w}")
	return _process_stack_real(stack,
					       mask_gaussian_2d_real, mask_gaussian_3d_real,
					       r_in, r_out, w,
					       float(tx), float(ty), float(tz))


# ----------------------------------------------------------------------------------
# mask_edgefill
# ----------------------------------------------------------------------------------
# Edge-filling is a real-space operation on the image borders (not a radial/centered mask),
# so it takes no centering offsets and no Fourier variants. The mask is the product of
# independent per-axis edge weights (broadcast), which is exactly equivalent to applying
# the per-axis decays sequentially (the x decay last, on the already y/z-decayed values).
# All width assumptions (0 < width < n/2 on every spatial axis) are validated in the parent
# function, so the JIT variants are branch-free.

@jax.jit
def mask_edgefill_2d_real_flat(data: jnp.ndarray, width: float, value: float) -> jnp.ndarray:
	"""Edge-fill a stack of 2-D images in real space (data: float {N,Y,X}), flat mode: the width pixels
	at each x/y edge (indices 0..width-1 and n-width..n-1) are set to value, the interior is unchanged.
	Assumes 0 < width < n/2 on both axes."""
	nx, ny = data.shape[-1], data.shape[-2]
	mask = _edge_weight_flat(ny, width)[:, None] * _edge_weight_flat(nx, width)[None, :]
	return value + mask[None, :, :] * (data - value)

@jax.jit
def mask_edgefill_2d_real_apod(data: jnp.ndarray, width: float, value: float) -> jnp.ndarray:
	"""Edge-fill a stack of 2-D images in real space (data: float {N,Y,X}), apodized mode: the value
	decays linearly from the original (just inside the width border) to value at each x/y edge.
	In corners the decays compose as the product of the two per-axis linear ramps.
	Assumes 0 < width < n/2 on both axes."""
	nx, ny = data.shape[-1], data.shape[-2]
	mask = _edge_weight_apod(ny, width)[:, None] * _edge_weight_apod(nx, width)[None, :]
	return value + mask[None, :, :] * (data - value)

@jax.jit
def mask_edgefill_3d_real_flat(data: jnp.ndarray, width: float, value: float) -> jnp.ndarray:
	"""Edge-fill a stack of 3-D volumes in real space (data: float {N,Z,Y,X}), flat mode: the width pixels
	at each x/y/z edge (indices 0..width-1 and n-width..n-1) are set to value, the interior is unchanged.
	Assumes 0 < width < n/2 on all three axes."""
	nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
	mask = (_edge_weight_flat(nz, width)[:, None, None]
			* _edge_weight_flat(ny, width)[None, :, None]
			* _edge_weight_flat(nx, width)[None, None, :])
	return value + mask[None, :, :, :] * (data - value)

@jax.jit
def mask_edgefill_3d_real_apod(data: jnp.ndarray, width: float, value: float) -> jnp.ndarray:
	"""Edge-fill a stack of 3-D volumes in real space (data: float {N,Z,Y,X}), apodized mode: the value
	decays linearly from the original (just inside the width border) to value at each x/y/z edge.
	In corners the decays compose as the product of the three per-axis linear ramps.
	Assumes 0 < width < n/2 on all three axes."""
	nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
	mask = (_edge_weight_apod(nz, width)[:, None, None]
			* _edge_weight_apod(ny, width)[None, :, None]
			* _edge_weight_apod(nx, width)[None, None, :])
	return value + mask[None, :, :, :] * (data - value)


def mask_edgefill(stack: EMStack,
				      width: float,
				      value: float = 0.0,
				      apodize: bool = False) -> EMStack:
	"""Edge-fill a stack of 2-D images or 3-D volumes, applied to the entire stack at once:
	the width pixels at every edge (indices 0..width-1 and n-width..n-1 on each axis) are replaced.
	Returns a new EMStack of the same kind; the input is not modified.
	Edge-filling is a real-space operation: Fourier-space (complex) input is rejected.

	apodize = False: edge pixels are set to value.
	apodize = True: the value decays linearly from the original pixel value (just inside the width
		border) down to value at the image edge. In corners, where the decay direction is ambiguous,
		the decays are applied with the x decay last, using the already y/z-decayed values as its
		starting point (equivalent to the product of the per-axis linear ramps).

	width    - number of pixels in from each edge to affect (required; integer, 0 < width < min(n//2)
			   over the image dimensions)
	value    - the value that edge pixels take at the image edge (default 0.0)
	apodize  - if True, linearly apodize the edge band instead of setting it (default False)

	Example: mask_edgefill(my_stack, width=5, apodize=True)
	"""
	width = int(width)
	if width <= 0:
		raise ValueError(f"width must be a positive integer number of pixels, got {width}")
	dims = [int(d) for d in stack.shape[1:]]
	halfmin = min(d // 2 for d in dims)
	if width >= halfmin:
		raise ValueError(f"width ({width}) must be less than min(n//2) = {halfmin} for the image dimensions {dims}")
	value = float(value)
	if not isinstance(apodize, bool):
		raise ValueError(f"apodize must be a boolean, got {apodize!r}")
	if apodize:
		v2d, v3d = mask_edgefill_2d_real_apod, mask_edgefill_3d_real_apod
	else:
		v2d, v3d = mask_edgefill_2d_real_flat, mask_edgefill_3d_real_flat
	return _process_stack_real(stack, v2d, v3d, width, value)


# ----------------------------------------------------------------------------------
# normalize_*  (real-space linear transforms: zero the region mean, unit whole-image std)
# ----------------------------------------------------------------------------------
# Each processor applies, independently to every image in the stack, the linear transform
#     result = (image - mean_region) / std_whole
# where mean_region is the mean of the image over a region (defined below, per processor) and
# std_whole is the standard deviation of the image over all of its pixels. The four processors
# differ only in the region used to compute the mean to be subtracted. Normalization is a
# real-space operation, so it is real-only (Fourier-space input is rejected) and has only
# {2D,3D} variants, like the masks.

@jax.jit
def normalize_standard_2d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Normalize a stack of 2-D images in real space (data: float {N,Y,X}): subtract the image mean
	(the region is the entire image) and divide by the image's standard deviation, so each image has
	zero mean and unit standard deviation."""
	return _normalize(data, jnp.ones((data.shape[-2], data.shape[-1]), dtype=jnp.float32))

@jax.jit
def normalize_standard_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Normalize a stack of 3-D volumes in real space (data: float {N,Z,Y,X}): subtract the volume mean
	(the region is the entire volume) and divide by the volume's standard deviation, so each volume has
	zero mean and unit standard deviation."""
	return _normalize(data, jnp.ones((data.shape[-3], data.shape[-2], data.shape[-1]), dtype=jnp.float32))


def normalize_standard(stack: EMStack) -> EMStack:
	"""Normalize a stack of 2-D images or 3-D volumes to zero mean and unit standard deviation, applied
	independently to every image/volume in the stack and to the entire stack at once. Each image is
	shifted so its overall mean is zero, then scaled so its overall standard deviation is one.
	Returns a new EMStack of the same kind; the input is not modified.
	Normalization is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even.

	Example: normalize_standard(my_stack)
	"""
	return _process_stack_real(stack, normalize_standard_2d_real, normalize_standard_3d_real)


@jax.jit
def normalize_edgemean_2d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Normalize a stack of 2-D images in real space (data: float {N,Y,X}): subtract the mean of the
	1-pixel border (edge) of each image, then divide by the image's standard deviation over all pixels."""
	ny, nx = data.shape[-2], data.shape[-1]
	edge = 1.0 - _interior_weight(ny)[:, None] * _interior_weight(nx)[None, :]
	return _normalize(data, edge)

@jax.jit
def normalize_edgemean_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Normalize a stack of 3-D volumes in real space (data: float {N,Z,Y,X}): subtract the mean of the
	1-pixel surface (the pixels on any outer face) of each volume, then divide by the volume's standard
	deviation over all pixels."""
	nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
	surface = 1.0 - (_interior_weight(nz)[:, None, None]
		         * _interior_weight(ny)[None, :, None]
		         * _interior_weight(nx)[None, None, :])
	return _normalize(data, surface)


def normalize_edgemean(stack: EMStack) -> EMStack:
	"""Normalize a stack of 2-D images or 3-D volumes by subtracting the mean of the image edge
	(the 1-pixel border in 2-D, the 1-pixel outer surface in 3-D) and dividing by the standard
	deviation over all of the image's pixels, applied independently to every image/volume in the
	stack and to the entire stack at once. This zeroes the background/edge level of each image while
	unit-scaling the whole image.
	Returns a new EMStack of the same kind; the input is not modified.
	Normalization is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even.

	Example: normalize_edgemean(my_stack)
	"""
	return _process_stack_real(stack, normalize_edgemean_2d_real, normalize_edgemean_3d_real)


@jax.jit
def normalize_circlemean_2d_real(data: jnp.ndarray, radius: float, width: float) -> jnp.ndarray:
	"""Normalize a stack of 2-D images in real space (data: float {N,Y,X}): subtract the mean of the
	circular ring (radius <= dist < radius+width, centered on the image) of each image, then divide by
	the image's standard deviation over all pixels. radius (>=0) and width (>0) are in pixels and are
	assumed pre-validated."""
	ny, nx = data.shape[-2], data.shape[-1]
	return _normalize(data, _ring_mask_2d(ny, nx, radius, width))

@jax.jit
def normalize_circlemean_3d_real(data: jnp.ndarray, radius: float, width: float) -> jnp.ndarray:
	"""Normalize a stack of 3-D volumes in real space (data: float {N,Z,Y,X}): subtract the mean of the
	spherical shell (radius <= dist < radius+width, centered on the volume) of each volume, then divide
	by the volume's standard deviation over all pixels. radius (>=0) and width (>0) are in pixels and
	are assumed pre-validated."""
	nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
	return _normalize(data, _ring_mask_3d(nz, ny, nx, radius, width))


def normalize_circlemean(stack: EMStack,
				 radius: float = None,
				 width: float = 2.0) -> EMStack:
	"""Normalize a stack of 2-D images or 3-D volumes by subtracting the mean of a circular ring
	(a spherical shell in 3-D) and dividing by the standard deviation over all of the image's pixels,
	applied independently to every image/volume in the stack and to the entire stack at once. The ring
	is centered on the image center (pixel (nx//2, ny//2[, nz//2])). This zeroes the mean of a
	circular background ring of each image while unit-scaling the whole image.
	Returns a new EMStack of the same kind; the input is not modified.
	Normalization is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even.

	radius - inner radius of the ring, in pixels (default ny//2 - 2, near the image edge; a negative
		   value is measured from the outer edge, i.e. radius = ny/2 + radius)
	width  - width of the ring, in pixels (default 2.0, must be positive); the ring spans
		   radius <= dist < radius + width

	Example: normalize_circlemean(my_stack, radius=50, width=5)
	"""
	dims = [int(d) for d in stack.shape[1:]]
	ny = dims[-2]
	if width is None:
		raise ValueError("width (in pixels) must be provided")
	w = float(width)
	if w <= 0.0:
		raise ValueError(f"width must be positive, got {w}")
	if radius is None:
		r = ny / 2.0 - 2.0
	else:
		r = float(radius)
		if r < 0.0:
			r = ny / 2.0 + r
	if r < 0.0:
		raise ValueError(f"radius ({r}) places the ring entirely outside the image")
	max_dist = sum((d / 2.0) ** 2 for d in dims) ** 0.5
	if r > max_dist:
		raise ValueError(f"radius ({r}) places the ring entirely outside the image (farthest pixel is at ~{max_dist:.1f} px from the center)")
	return _process_stack_real(stack, normalize_circlemean_2d_real, normalize_circlemean_3d_real, r, w)


@jax.jit
def normalize_mask_2d_real(data: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
	"""Normalize a stack of 2-D images in real space (data: float {N,Y,X}) using a mask: subtract the
	mean of each image over the masked region (sum(image*mask)/count(mask>0)), then divide by the
	image's standard deviation over all pixels. mask is a real array broadcastable to data.shape whose
	non-zero pixels mark the region."""
	return _normalize(data, mask)

@jax.jit
def normalize_mask_3d_real(data: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
	"""Normalize a stack of 3-D volumes in real space (data: float {N,Z,Y,X}) using a mask: subtract the
	mean of each volume over the masked region (sum(volume*mask)/count(mask>0)), then divide by the
	volume's standard deviation over all pixels. mask is a real array broadcastable to data.shape whose
	non-zero voxels mark the region."""
	return _normalize(data, mask)


def normalize_mask(stack: EMStack,
			       mask: EMStack) -> EMStack:
	"""Normalize a stack of 2-D images or 3-D volumes by subtracting the mean of each image over the
	region marked by a provided mask, and dividing by the standard deviation over all of the image's
	pixels, applied independently to every image/volume in the stack and to the entire stack at once.
	Each image's masked region is zeroed (its mean subtracted) while the whole image is unit-scaled.
	Returns a new EMStack of the same kind; the input is not modified.
	Normalization is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even.

	mask - an EMStack of real data whose non-zero pixels define the region used to compute the
	       subtracted mean. Its spatial shape must match the stack's; it may have 1 image (applied to
	       every image in the stack) or the same number of images as the stack (one region per image).
	       The mean is sum(image*mask)/count(mask>0); the standard deviation is over the whole image.

	Example: normalize_mask(my_stack, mask)
	"""
	if mask is None:
		raise ValueError("a mask EMStack must be provided")
	mdata = mask.jax
	if jnp.issubdtype(mdata.dtype, jnp.complexfloating):
		raise ValueError("the mask must be real-valued (not Fourier-space/complex)")
	spatial = [int(d) for d in stack.shape[1:]]
	msk = [int(d) for d in mdata.shape[1:]]
	if msk != spatial:
		raise ValueError(f"mask spatial shape {msk} must match the stack's {spatial}")
	n_data = int(stack.shape[0])
	n_mask = int(mdata.shape[0])
	if n_mask not in (1, n_data):
		raise ValueError(f"mask has {n_mask} images; it must have 1 (applied to all) or {n_data} (one per image)")
	if int(jnp.sum(mdata > 0)) == 0:
		raise ValueError("the mask contains no non-zero pixels")
	return _process_stack_real(stack, normalize_mask_2d_real, normalize_mask_3d_real, mdata)


# ----------------------------------------------------------------------------------
# xform_*  (affine transforms, symmetry averaging, and centering - real space only)
# ----------------------------------------------------------------------------------
# All affine transforms here follow the EMAN2 convention: the rotation/scale part is applied
# about the image center (pixel index (nx//2, ny//2, [nz//2])) and the translation part shifts
# the image content by its (pixel) values along each axis. I.e. a source point p, expressed
# relative to the center, maps to
#     dest = matrix @ p + translation
# (relative to the center). xform applies the inverse map dest -> src = M^-1 @ dest - M^-1 @ t
# and bilinearly samples the input; destination pixels whose source falls outside the input
# are zero. Centering processors compute, per image, an integer translation that moves the
# object toward the image center and apply it with zero fill (content shifted out is lost,
# vacated pixels are zero).
# The JIT variants are branch-free; validation and dispatch happen in the public wrappers.

def _transform_matrix(transform: Transform) -> jnp.ndarray:
	"""Validate and convert a Transform to its 3x4 matrix (row-major: rotation + translation) as a
	float32 jnp array. Raises ValueError if not a Transform."""
	if not isinstance(transform, Transform):
		raise ValueError(f"transform must be a Transform, got {type(transform).__name__}")
	return jnp.asarray(transform.get_matrix(), dtype=jnp.float32)


def _parse_symmetry(sym: str) -> Symmetry3D:
	"""Parse an EMAN2-style symmetry specification string into a Symmetry3D object.
	Accepted (case-insensitive): cN, dN, hN[:nstart:daz:tz:maxtilt] (':' or ',' separators),
	tet/t, oct/o, icos/i (also icos5), icos2/i2."""
	if not isinstance(sym, str):
		raise ValueError(f"sym must be a symmetry specification string, got {type(sym).__name__}")
	s = sym.strip().lower()
	if not s:
		raise ValueError("sym must be a non-empty symmetry specification string")
	c = s[0]
	if c in ("c", "d"):
		try:
			nsym = int(s[1:])
		except ValueError:
			raise ValueError(f"unrecognized symmetry specification {sym!r}; expected cN, dN, hN[:nstart:daz:tz:maxtilt], tet/t, oct/o, icos/i or icos2/i2")
		if nsym <= 0:
			raise ValueError(f"symmetry order must be positive, got {nsym}")
		return get_symmetry(c, nsym=nsym)
	if c == "h":
		rest = s[1:]
		nsym, nstart, daz, tz, maxtilt = 1, 1, 0.0, 0.0, 90.0
		if rest:
			parts = rest.replace(",", ":").split(":")
			try:
				nsym = int(parts[0])
				if len(parts) > 1: nstart = int(parts[1])
				if len(parts) > 2: daz = float(parts[2])
				if len(parts) > 3: tz = float(parts[3])
				if len(parts) > 4: maxtilt = float(parts[4])
			except ValueError:
				raise ValueError(f"unrecognized helical symmetry specification {sym!r}; expected hN[:nstart:daz:tz:maxtilt]")
		if nsym <= 0:
			raise ValueError(f"symmetry order must be positive, got {nsym}")
		return get_symmetry("h", nsym=nsym, nstart=nstart, daz=daz, tz=tz, maxtilt=maxtilt)
	if s in ("t", "tet"):
		return get_symmetry("tet")
	if s in ("o", "oct"):
		return get_symmetry("oct")
	if s == "i2":
		return get_symmetry("icos2")
	if s in ("i", "icos", "icos5"):
		return get_symmetry("icos")
	raise ValueError(f"unrecognized symmetry specification {sym!r}; expected cN, dN, hN[:nstart:daz:tz:maxtilt], tet/t, oct/o, icos/i or icos2/i2")


def _translate_2d(data: jnp.ndarray, dx: jnp.ndarray, dy: jnp.ndarray) -> jnp.ndarray:
	"""Translate each image of a 2-D stack (data: float {N,Y,X}) by per-image shifts dx, dy
	(each shape {N}), moving the content by +dx in x and +dy in y (out[p] = in[p - shift]).
	Sampled bilinearly (exact when the shifts are integers); pixels whose source falls outside
	the input are zero. Branch-free."""
	ny, nx = data.shape[-2], data.shape[-1]
	sx = jnp.arange(nx, dtype=jnp.float32)[None, :] - dx[:, None]    # {N,X}
	sy = jnp.arange(ny, dtype=jnp.float32)[None, :] - dy[:, None]    # {N,Y}
	valid = ((sy >= 0) & (sy < ny))[:, :, None] & ((sx >= 0) & (sx < nx))[:, None, :]   # {N,Y,X}
	xi = jnp.floor(sx).astype(jnp.int32).clip(0, nx - 1)
	yi = jnp.floor(sy).astype(jnp.int32).clip(0, ny - 1)
	wx = sx - jnp.floor(sx)
	wy = sy - jnp.floor(sy)
	xi_b = xi[:, None, :];  xi1_b = jnp.minimum(xi + 1, nx - 1)[:, None, :]
	yi_b = yi[:, :, None];  yi1_b = jnp.minimum(yi + 1, ny - 1)[:, :, None]
	b = jnp.arange(data.shape[0])[:, None, None]      # explicit batch index so all index arrays broadcast together
	c00 = data[b, yi_b, xi_b]
	c10 = data[b, yi_b, xi1_b]
	c01 = data[b, yi1_b, xi_b]
	c11 = data[b, yi1_b, xi1_b]
	out = (c00 * (1.0 - wx)[:, None, :] * (1.0 - wy)[:, :, None]
		   + c10 * wx[:, None, :] * (1.0 - wy)[:, :, None]
		   + c01 * (1.0 - wx)[:, None, :] * wy[:, :, None]
		   + c11 * wx[:, None, :] * wy[:, :, None])
	return jnp.where(valid, out, 0.0)


def _translate_3d(data: jnp.ndarray, dx: jnp.ndarray, dy: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
	"""Translate each volume of a 3-D stack (data: float {N,Z,Y,X}) by per-image shifts dx, dy, dz
	(each shape {N}), moving the content by +dx/+dy/+dz along x/y/z (out[p] = in[p - shift]).
	Sampled trilinearly (exact when the shifts are integers); pixels whose source falls outside
	the input are zero. Branch-free."""
	nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
	sz = jnp.arange(nz, dtype=jnp.float32)[None, :] - dz[:, None]    # {N,Z}
	sy = jnp.arange(ny, dtype=jnp.float32)[None, :] - dy[:, None]    # {N,Y}
	sx = jnp.arange(nx, dtype=jnp.float32)[None, :] - dx[:, None]    # {N,X}
	valid = (((sz >= 0) & (sz < nz))[:, :, None, None]
			& ((sy >= 0) & (sy < ny))[:, None, :, None]
			& ((sx >= 0) & (sx < nx))[:, None, None, :])             # {N,Z,Y,X}
	zi = jnp.floor(sz).astype(jnp.int32).clip(0, nz - 1)
	yi = jnp.floor(sy).astype(jnp.int32).clip(0, ny - 1)
	xi = jnp.floor(sx).astype(jnp.int32).clip(0, nx - 1)
	wz = sz - jnp.floor(sz)
	wy = sy - jnp.floor(sy)
	wx = sx - jnp.floor(sx)
	zi_b = zi[:, :, None, None];  zi1_b = jnp.minimum(zi + 1, nz - 1)[:, :, None, None]
	yi_b = yi[:, None, :, None];  yi1_b = jnp.minimum(yi + 1, ny - 1)[:, None, :, None]
	xi_b = xi[:, None, None, :];  xi1_b = jnp.minimum(xi + 1, nx - 1)[:, None, None, :]
	wz_b = wz[:, :, None, None]; wy_b = wy[:, None, :, None]; wx_b = wx[:, None, None, :]
	b = jnp.arange(data.shape[0])[:, None, None, None]   # explicit batch index so all index arrays broadcast together
	out = (data[b, zi_b, yi_b, xi_b] * (1 - wz_b) * (1 - wy_b) * (1 - wx_b)
	   + data[b, zi1_b, yi_b, xi_b] * wz_b * (1 - wy_b) * (1 - wx_b)
	   + data[b, zi_b, yi1_b, xi_b] * (1 - wz_b) * wy_b * (1 - wx_b)
	   + data[b, zi1_b, yi1_b, xi_b] * wz_b * wy_b * (1 - wx_b)
	   + data[b, zi_b, yi_b, xi1_b] * (1 - wz_b) * (1 - wy_b) * wx_b
	   + data[b, zi1_b, yi_b, xi1_b] * wz_b * (1 - wy_b) * wx_b
	   + data[b, zi_b, yi1_b, xi1_b] * (1 - wz_b) * wy_b * wx_b
	   + data[b, zi1_b, yi1_b, xi1_b] * wz_b * wy_b * wx_b)
	return jnp.where(valid, out, 0.0)


def _com_2d(data: jnp.ndarray, threshold: float, powercenter: bool) -> jnp.ndarray:
	"""Per-image center of mass of a 2-D stack (data: float {N,Y,X}), as an array of shape {N,2}
	(pixel indices x,y). Only pixels with value >= threshold contribute, weighted by their value
	(the image is squared first if powercenter; the threshold then applies to the squared values).
	Images with no pixels above threshold yield (0,0)."""
	nx, ny = data.shape[-1], data.shape[-2]
	d = jnp.where(powercenter, data * data, data)
	w = jnp.where(d >= threshold, d, 0.0)                      # {N,Y,X}
	count = w.sum(axis=(1, 2))                                 # {N,}
	safe = jnp.where(count > 0.0, count, 1.0)                   # {N,}
	cx = (w * jnp.arange(nx, dtype=jnp.float32)[None, None, :]).sum(axis=(1, 2)) / safe
	cy = (w * jnp.arange(ny, dtype=jnp.float32)[None, :, None]).sum(axis=(1, 2)) / safe
	return jnp.where((count > 0.0)[:, None], jnp.stack([cx, cy], axis=1), 0.0)


def _com_3d(data: jnp.ndarray, threshold: float, powercenter: bool) -> jnp.ndarray:
	"""Per-image center of mass of a 3-D stack (data: float {N,Z,Y,X}), as an array of shape {N,3}
	(pixel indices x,y,z). Only pixels with value >= threshold contribute, weighted by their value
	(the volume is squared first if powercenter; the threshold then applies to the squared values).
	Volumes with no pixels above threshold yield (0,0,0)."""
	nx, ny, nz = data.shape[-1], data.shape[-2], data.shape[-3]
	d = jnp.where(powercenter, data * data, data)
	w = jnp.where(d >= threshold, d, 0.0)                      # {N,Z,Y,X}
	count = w.sum(axis=(1, 2, 3))                              # {N,}
	safe = jnp.where(count > 0.0, count, 1.0)                   # {N,}
	cx = (w * jnp.arange(nx, dtype=jnp.float32)[None, None, None, :]).sum(axis=(1, 2, 3)) / safe
	cy = (w * jnp.arange(ny, dtype=jnp.float32)[None, None, :, None]).sum(axis=(1, 2, 3)) / safe
	cz = (w * jnp.arange(nz, dtype=jnp.float32)[None, :, None, None]).sum(axis=(1, 2, 3)) / safe
	return jnp.where((count > 0.0)[:, None], jnp.stack([cx, cy, cz], axis=1), 0.0)


def _acf_peak_2d(data: jnp.ndarray) -> jnp.ndarray:
	"""Per-image peak of the circular self-convolution (autocorrelation function) of a 2-D stack
	(data: float {N,Y,X}), returned as signed pixel offsets (shape {N,2}, x,y) from the image
	center, searched within +/- maxshift on each axis where maxshift = n//4 clamped to n//2-1.
	Ties are broken toward the more negative offset (matching EMAN2's search order). Branch-free.
	"""
	ny, nx = data.shape[-2], data.shape[-1]
	F = jnp.fft.rfft2(data)
	acf = jnp.fft.irfft2(F * F)                               # circular self-convolution {N,Y,X}
	mx = min(nx // 4, nx // 2 - 1)
	my = min(ny // 4, ny // 2 - 1)
	ix = (jnp.arange(-mx, mx + 1)) % nx                        # wrapped indices, {2mx+1}
	iy = (jnp.arange(-my, my + 1)) % ny                        # {2my+1}
	win = acf[:, iy[:, None], ix[None, :]]                     # {N,2my+1,2mx+1}
	flat = win.reshape(win.shape[0], -1).argmax(axis=1)        # {N,} (x index is fastest)
	px = (flat % (2 * mx + 1)).astype(jnp.float32) - mx
	py = (flat // (2 * mx + 1)).astype(jnp.float32) - my
	return jnp.stack([px, py], axis=1)


def _acf_peak_3d(data: jnp.ndarray) -> jnp.ndarray:
	"""Per-image peak of the circular self-convolution of a 3-D stack (data: float {N,Z,Y,X}),
	returned as signed pixel offsets (shape {N,3}, x,y,z) from the volume center, searched within
	+/- maxshift on each axis where maxshift = n//4 clamped to n//2-1. Ties are broken toward the
	more negative offsets (matching EMAN2's search order). Branch-free."""
	nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
	F = jnp.fft.rfftn(data, axes=(-3, -2, -1))
	acf = jnp.fft.irfftn(F * F, axes=(-3, -2, -1))            # circular self-convolution
	mx = min(nx // 4, nx // 2 - 1)
	my = min(ny // 4, ny // 2 - 1)
	mz = min(nz // 4, nz // 2 - 1)
	ix = (jnp.arange(-mx, mx + 1)) % nx                        # {2mx+1}
	iy = (jnp.arange(-my, my + 1)) % ny                        # {2my+1}
	iz = (jnp.arange(-mz, mz + 1)) % nz                        # {2mz+1}
	win = acf[:, iz[:, None, None], iy[None, :, None], ix[None, None, :]]   # {N,2mz+1,2my+1,2mx+1}
	flat = win.reshape(win.shape[0], -1).argmax(axis=1)     # {N,} (x index is fastest)
	px = (flat % (2 * mx + 1)).astype(jnp.float32) - mx
	py = ((flat // (2 * mx + 1)) % (2 * my + 1)).astype(jnp.float32) - my
	pz = (flat // ((2 * mx + 1) * (2 * my + 1))).astype(jnp.float32) - mz
	return jnp.stack([px, py, pz], axis=1)


@jax.jit
def xform_2d_real(data: jnp.ndarray, mat34: jnp.ndarray) -> jnp.ndarray:
	"""Apply an affine transform (mat34: 3x4 row-major rotation+translation, EMAN2 convention) to
	each image of a 2-D stack (data: float {N,Y,X}): the rotation/scale part acts about the image
	center (nx//2, ny//2), plus the translation in pixels; the inverse map is sampled bilinearly
	and out-of-range pixels are zero. Branch-free: assumes mat34 is a valid (invertible) affine
	transform; use assert_valid_2d on the Transform before calling this for 2-D data."""
	ny, nx = data.shape[-2], data.shape[-1]
	rot_inv = jnp.linalg.inv(mat34[:, :3])                     # {3,3}
	tr = rot_inv @ mat34[:, 3]                                 # {3,}
	y = jnp.arange(ny, dtype=jnp.float32)[:, None] - (ny // 2)  # {ny,1}
	x = jnp.arange(nx, dtype=jnp.float32)[None, :] - (nx // 2)  # {1,nx}
	sx = rot_inv[0, 0] * x + rot_inv[0, 1] * y + (nx // 2 - tr[0])
	sy = rot_inv[1, 0] * x + rot_inv[1, 1] * y + (ny // 2 - tr[1])
	valid = (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)      # {ny,nx}
	xi = jnp.floor(sx).astype(jnp.int32).clip(0, nx - 1)
	yi = jnp.floor(sy).astype(jnp.int32).clip(0, ny - 1)
	wx = sx - jnp.floor(sx)
	wy = sy - jnp.floor(sy)
	c00 = data[:, yi, xi]
	c10 = data[:, yi, jnp.minimum(xi + 1, nx - 1)]
	c01 = data[:, jnp.minimum(yi + 1, ny - 1), xi]
	c11 = data[:, jnp.minimum(yi + 1, ny - 1), jnp.minimum(xi + 1, nx - 1)]
	out = (c00 * (1.0 - wx) * (1.0 - wy)
		   + c10 * wx * (1.0 - wy)
		   + c01 * (1.0 - wx) * wy
		   + c11 * wx * wy)
	return jnp.where(valid[None, :, :], out, 0.0)


@jax.jit
def xform_3d_real(data: jnp.ndarray, mat34: jnp.ndarray) -> jnp.ndarray:
	"""Apply an affine transform (mat34: 3x4 row-major rotation+translation, EMAN2 convention) to
	each volume of a 3-D stack (data: float {N,Z,Y,X}): the rotation/scale part acts about the
	volume center (nx//2, ny//2, nz//2), plus the translation in pixels; the inverse map is sampled
	trilinearly and out-of-range pixels are zero. Branch-free: assumes mat34 is a valid (invertible)
	affine transform."""
	nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
	rot_inv = jnp.linalg.inv(mat34[:, :3])                     # {3,3}
	tr = rot_inv @ mat34[:, 3]                                 # {3,}
	z = jnp.arange(nz, dtype=jnp.float32)[:, None, None] - (nz // 2)   # {nz,1,1}
	y = jnp.arange(ny, dtype=jnp.float32)[None, :, None] - (ny // 2)   # {1,ny,1}
	x = jnp.arange(nx, dtype=jnp.float32)[None, None, :] - (nx // 2)   # {1,1,nx}
	sx = rot_inv[0, 0] * x + rot_inv[0, 1] * y + rot_inv[0, 2] * z + (nx // 2 - tr[0])
	sy = rot_inv[1, 0] * x + rot_inv[1, 1] * y + rot_inv[1, 2] * z + (ny // 2 - tr[1])
	sz = rot_inv[2, 0] * x + rot_inv[2, 1] * y + rot_inv[2, 2] * z + (nz // 2 - tr[2])
	valid = ((sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny) & (sz >= 0) & (sz < nz))
	xi = jnp.floor(sx).astype(jnp.int32).clip(0, nx - 1)
	yi = jnp.floor(sy).astype(jnp.int32).clip(0, ny - 1)
	zi = jnp.floor(sz).astype(jnp.int32).clip(0, nz - 1)
	wx = sx - jnp.floor(sx)
	wy = sy - jnp.floor(sy)
	wz = sz - jnp.floor(sz)
	out = (data[:, zi, yi, xi] * (1 - wz) * (1 - wy) * (1 - wx)
		   + data[:, zi, yi, jnp.minimum(xi + 1, nx - 1)] * (1 - wz) * (1 - wy) * wx
		   + data[:, zi, jnp.minimum(yi + 1, ny - 1), xi] * (1 - wz) * wy * (1 - wx)
		   + data[:, zi, jnp.minimum(yi + 1, ny - 1), jnp.minimum(xi + 1, nx - 1)] * (1 - wz) * wy * wx
		   + data[:, jnp.minimum(zi + 1, nz - 1), yi, xi] * wz * (1 - wy) * (1 - wx)
		   + data[:, jnp.minimum(zi + 1, nz - 1), yi, jnp.minimum(xi + 1, nx - 1)] * wz * (1 - wy) * wx
		   + data[:, jnp.minimum(zi + 1, nz - 1), jnp.minimum(yi + 1, ny - 1), xi] * wz * wy * (1 - wx)
		   + data[:, jnp.minimum(zi + 1, nz - 1), jnp.minimum(yi + 1, ny - 1), jnp.minimum(xi + 1, nx - 1)] * wz * wy * wx)
	return jnp.where(valid[None, :, :, :], out, 0.0)


@jax.jit
def xform_flip_x_2d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Reverse the order of the pixels along the x axis of each 2-D image (data: float {N,Y,X}):
	out[... , i, ...] = in[..., nx-1-i, ...]."""
	return jnp.flip(data, axis=-1)


@jax.jit
def xform_flip_y_2d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Reverse the order of the pixels along the y axis of each 2-D image (data: float {N,Y,X}):
	out[... , j, ...] = in[..., ny-1-j, ...]."""
	return jnp.flip(data, axis=-2)


@jax.jit
def xform_flip_x_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Reverse the order of the pixels along the x axis of each 3-D volume (data: float {N,Z,Y,X}):
	out[..., i, ...] = in[..., nx-1-i, ...]."""
	return jnp.flip(data, axis=-1)


@jax.jit
def xform_flip_y_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Reverse the order of the pixels along the y axis of each 3-D volume (data: float {N,Z,Y,X}):
	out[..., j, ...] = in[..., ny-1-j, ...]."""
	return jnp.flip(data, axis=-2)


@jax.jit
def xform_flip_z_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Reverse the order of the pixels along the z axis of each 3-D volume (data: float {N,Z,Y,X}):
	out[..., k, ...] = in[..., nz-1-k, ...]."""
	return jnp.flip(data, axis=-3)


def _mirror_axis(data: jnp.ndarray, axis: int) -> jnp.ndarray:
	"""Mirror a stack along one axis about its center pixel (n//2): for index i >= 1, out[i] =
	in[n-i] (so the center pixel n//2 stays stationary), and out[0] = 0. Assumes even n.
	Branch-free."""
	n = data.shape[axis]
	i = jnp.arange(n, dtype=jnp.int32)
	src = jnp.where(i == 0, 0, n - i)                          # i=0 maps to 0 (safe gather); i>=1 -> n-i
	mask = (i != 0).astype(jnp.float32)
	shp = [1] * data.ndim
	shp[axis] = n
	out = jnp.take(data, src, axis=axis)                       # 1-D index: axis dim replaced by n
	return out * mask.reshape(shp)


@jax.jit
def xform_mirror_x_2d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Mirror each 2-D image (data: float {N,Y,X}) about the center pixel of its x axis (nx//2):
	for i >= 1, out[..., i, ...] = in[..., nx-i, ...] (the center pixel nx//2 is stationary) and
	out[..., 0, ...] = 0. Assumes even nx."""
	return _mirror_axis(data, -1)


@jax.jit
def xform_mirror_y_2d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Mirror each 2-D image (data: float {N,Y,X}) about the center pixel of its y axis (ny//2):
	for j >= 1, out[..., j, ...] = in[..., ny-j, ...] (the center pixel ny//2 is stationary) and
	out[..., 0, ...] = 0. Assumes even ny."""
	return _mirror_axis(data, -2)


@jax.jit
def xform_mirror_x_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Mirror each 3-D volume (data: float {N,Z,Y,X}) about the center pixel of its x axis (nx//2):
	for i >= 1, out[..., i, ...] = in[..., nx-i, ...] (the center pixel nx//2 is stationary) and
	out[..., 0, ...] = 0. Assumes even nx."""
	return _mirror_axis(data, -1)


@jax.jit
def xform_mirror_y_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Mirror each 3-D volume (data: float {N,Z,Y,X}) about the center pixel of its y axis (ny//2):
	for j >= 1, out[..., j, ...] = in[..., ny-j, ...] (the center pixel ny//2 is stationary) and
	out[..., 0, ...] = 0. Assumes even ny."""
	return _mirror_axis(data, -2)


@jax.jit
def xform_mirror_z_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Mirror each 3-D volume (data: float {N,Z,Y,X}) about the center pixel of its z axis (nz//2):
	for k >= 1, out[..., k, ...] = in[..., nz-k, ...] (the center pixel nz//2 is stationary) and
	out[..., 0, ...] = 0. Assumes even nz."""
	return _mirror_axis(data, -3)


@jax.jit
def xform_centerofmass_2d_real(data: jnp.ndarray, threshold: float, int_shift_only: bool, powercenter: bool) -> jnp.ndarray:
	"""Center each 2-D image of a stack (data: float {N,Y,X}) at its center of mass: the center of
	mass (weighted by pixel value, over pixels >= threshold; the image is squared first if
	powercenter, in which case threshold applies to the squared values) is found per image, and the
	image is translated (zero fill) by the (rounded, if int_shift_only) integer/float shift that
	places it at (nx//2, ny//2). Branch-free; images with no pixels above threshold are unchanged.
	"""
	nx, ny = data.shape[-1], data.shape[-2]
	com = _com_2d(data, threshold, powercenter)               # {N,2}
	shift = jnp.array([nx / 2.0, ny / 2.0], dtype=jnp.float32) - com
	shift = jnp.where(int_shift_only, jnp.round(shift), shift)
	return _translate_2d(data, shift[:, 0], shift[:, 1])


@jax.jit
def xform_centerofmass_3d_real(data: jnp.ndarray, threshold: float, int_shift_only: bool, powercenter: bool) -> jnp.ndarray:
	"""Center each 3-D volume of a stack (data: float {N,Z,Y,X}) at its center of mass: the center
	of mass (weighted by pixel value, over pixels >= threshold; the volume is squared first if
	powercenter, in which case threshold applies to the squared values) is found per volume, and the
	volume is translated (zero fill) by the (rounded, if int_shift_only) shift that places it at
	(nx//2, ny//2, nz//2). Branch-free; volumes with no pixels above threshold are unchanged."""
	nx, ny, nz = data.shape[-1], data.shape[-2], data.shape[-3]
	com = _com_3d(data, threshold, powercenter)               # {N,3}
	shift = jnp.array([nx / 2.0, ny / 2.0, nz / 2.0], dtype=jnp.float32) - com
	shift = jnp.where(int_shift_only, jnp.round(shift), shift)
	return _translate_3d(data, shift[:, 0], shift[:, 1], shift[:, 2])


@jax.jit
def xform_centeracf_2d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Center each 2-D image of a stack (data: float {N,Y,X}) using its autocorrelation function:
	the peak of the circular self-convolution is searched within +/- (n//4, clamped to n//2-1)
	pixels of the center on each axis, and the image is translated (zero fill) by -peak/2 rounded
	to the nearest pixel (EMAN2's xform.centeracf always uses integer shifts). Branch-free."""
	peak = _acf_peak_2d(data)                                 # {N,2}
	shift = jnp.round(-peak / 2.0)
	return _translate_2d(data, shift[:, 0], shift[:, 1])


@jax.jit
def xform_centeracf_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Center each 3-D volume of a stack (data: float {N,Z,Y,X}) using its autocorrelation
	function: the peak of the circular self-convolution is searched within +/- (n//4, clamped to
	n//2-1) pixels of the center on each axis, and the volume is translated (zero fill) by -peak/2
	rounded to the nearest pixel (EMAN2's xform.centeracf always uses integer shifts). Branch-free."""
	peak = _acf_peak_3d(data)                                 # {N,3}
	shift = jnp.round(-peak / 2.0)
	return _translate_3d(data, shift[:, 0], shift[:, 1], shift[:, 2])


@jax.jit
def xform_center_2d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Center each 2-D image of a stack (data: float {N,Y,X}) by identifying the object silhouette
	and centering it (EMAN2's xform.center), approximated with the available EMAN3 processors:
	highpass (removes low-frequency gradient) -> circle-mean normalize -> peripheral soft mask ->
	square (emphasizes strong density) -> lowpass (0.05) -> circle-mean normalize -> threshold
	enclosing the top ~1/10 of the pixels (searched within [mean, mean+4*sigma]) -> center of mass
	of the resulting silhouette -> integer translation (zero fill) to the image center. Constant
	(sigma == 0) images are left unchanged. Branch-free."""
	ny, nx = data.shape[-2], data.shape[-1]
	sigma0 = jnp.sqrt(jnp.maximum((data * data).mean(axis=(1, 2)) - data.mean(axis=(1, 2)) ** 2, 0.0))
	active = (sigma0 > 0.0).astype(jnp.float32)               # {N,} (1.0 unless the image is constant)
	gmw = max(nx // 16, 5)
	cut_hp = (float(nx // 10) if nx < 50 else 5.0) / max(nx, ny)
	d = linearfilter_highpass_gaussian_2d_real(data, cut_hp)
	d = normalize_circlemean_2d_real(d, float(ny // 2 - 4), 2.0)
	peripheral = _gauss_mask_2d(ny, nx, 0.0, float(nx // 2 - gmw), float(gmw / 1.3), 0.0, 0.0)
	d = d * peripheral[None, :, :]
	d = d * d
	d = linearfilter_lowpass_gaussian_2d_real(d, 0.05)
	d = normalize_circlemean_2d_real(d, float(ny // 2 - 6), 2.0)
	mean = d.mean(axis=(1, 2))                                # {N,}
	sigma = jnp.sqrt(jnp.maximum((d * d).mean(axis=(1, 2)) - mean * mean, 0.0))
	rank = (nx * ny) // 10
	v = jnp.clip(d, mean[:, None, None], (mean + 4.0 * sigma)[:, None, None])
	thr = jnp.sort(v.reshape(v.shape[0], -1), axis=1, descending=True)[:, rank]   # {N,}
	b = (d >= thr[:, None, None]).astype(jnp.float32)         # binary silhouette {N,Y,X}
	com = _com_2d(b, 0.5, False)                              # {N,2}
	shift = jnp.round(jnp.array([nx / 2.0, ny / 2.0], dtype=jnp.float32) - com)
	shift = jnp.where((active > 0.0)[:, None], shift, 0.0)
	return _translate_2d(data, shift[:, 0], shift[:, 1])


@jax.jit
def xform_center_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Center each 3-D volume of a stack (data: float {N,Z,Y,X}) by identifying the object
	silhouette and centering it (EMAN2's xform.center), approximated with the available EMAN3
	processors: highpass -> circle-mean (surface) normalize -> peripheral soft mask -> square ->
	lowpass (0.05) -> normalize -> threshold enclosing the top ~1/10 of the pixels (searched within
	[mean, mean+4*sigma]) -> center of mass of the resulting silhouette -> integer translation
	(zero fill) to the volume center. Constant (sigma == 0) volumes are left unchanged. Branch-free.
	"""
	nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
	sigma0 = jnp.sqrt(jnp.maximum((data * data).mean(axis=(1, 2, 3)) - data.mean(axis=(1, 2, 3)) ** 2, 0.0))
	active = (sigma0 > 0.0).astype(jnp.float32)               # {N,}
	gmw = max(nx // 16, 5)
	cut_hp = (float(nx // 10) if nx < 50 else 5.0) / max(nx, ny, nz)
	d = linearfilter_highpass_gaussian_3d_real(data, cut_hp)
	d = normalize_circlemean_3d_real(d, float(ny // 2 - 4), 2.0)
	peripheral = _gauss_mask_3d(nz, ny, nx, 0.0, float(nx // 2 - gmw), float(gmw / 1.3), 0.0, 0.0, 0.0)
	d = d * peripheral[None, :, :, :]
	d = d * d
	d = linearfilter_lowpass_gaussian_3d_real(d, 0.05)
	d = normalize_circlemean_3d_real(d, float(ny // 2 - 6), 2.0)
	mean = d.mean(axis=(1, 2, 3))                             # {N,}
	sigma = jnp.sqrt(jnp.maximum((d * d).mean(axis=(1, 2, 3)) - mean * mean, 0.0))
	rank = (nx * ny * nz) // 10
	v = jnp.clip(d, mean[:, None, None, None], (mean + 4.0 * sigma)[:, None, None, None])
	thr = jnp.sort(v.reshape(v.shape[0], -1), axis=1, descending=True)[:, rank]   # {N,}
	b = (d >= thr[:, None, None, None]).astype(jnp.float32)   # binary silhouette {N,Z,Y,X}
	com = _com_3d(b, 0.5, False)                              # {N,3}
	shift = jnp.round(jnp.array([nx / 2.0, ny / 2.0, nz / 2.0], dtype=jnp.float32) - com)
	shift = jnp.where((active > 0.0)[:, None], shift, 0.0)
	return _translate_3d(data, shift[:, 0], shift[:, 1], shift[:, 2])


def xform(stack: EMStack,
	       transform: Transform) -> EMStack:
	"""Apply the same affine Transform to every image/volume in the stack (bilinear interpolation,
	out-of-range pixels zeroed), applied to the entire stack at once.
	Returns a new EMStack of the same kind; the input is not modified.
	xform is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even.

	The rotation/scale part of the transform acts about the image center (pixel index
	nx//2, ny//2, [nz//2]); the translation part shifts the content by its (pixel) values, i.e.
	a source point p (relative to the center) maps to dest = matrix @ p + translation.
	
	transform - a Transform (2-D or 3-D); for a 2-D stack it must be valid for 2-D (a
	            TransformError is raised otherwise)

	Example: xform(my_stack, t)
	"""
	mat = _transform_matrix(transform)
	if isinstance(stack, EMStack2D):
		try:
			transform.assert_valid_2d()
		except TransformError as e:
			raise ValueError(str(e))
	return _process_stack_real(stack, xform_2d_real, xform_3d_real, mat)


def xform_applysym(stack: EMStack,
		       sym: str) -> EMStack:
	"""Symmetrize: apply each symmetry operation in the group (the same xform as xform) to every
	image/volume in the stack and average the results, applied to the entire stack at once.
	Returns a new EMStack of the same kind; the input is not modified.
	xform_applysym is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even. 2-D stacks require Cn symmetry (as in EMAN2).

	sym - an EMAN2-style symmetry specification string: cN, dN, hN[:nstart:daz:tz:maxtilt],
	      tet/t, oct/o, icos/i (also icos5), or icos2/i2 (case-insensitive)

	Example: xform_applysym(my_stack, \"c4\")
	"""
	s = _parse_symmetry(sym)
	nsym = int(s.get_nsym())
	if nsym <= 0:
		raise ValueError(f"symmetry {sym!r} has a non-positive number of operations")
	is2d = isinstance(stack, EMStack2D)
	if is2d and s.get_name() != "c":
		raise ValueError(f"xform_applysym: Cn symmetry is required for 2-D symmetrization, got {sym!r}")
	data = stack.jax
	if jnp.issubdtype(data.dtype, jnp.complexfloating):
		raise ValueError("this processor operates on real-space data only; a Fourier-space (complex) stack was provided")
	acc = jnp.zeros_like(data)
	for t in s.get_syms():
		if is2d:
			try:
				t.assert_valid_2d()
			except TransformError as e:
				raise ValueError(str(e))
		mat = jnp.asarray(t.get_matrix(), dtype=jnp.float32)
		acc += xform_2d_real(data, mat) if is2d else xform_3d_real(data, mat)
	acc = acc / nsym
	return type(stack)(acc, parent=stack)


def xform_flip(stack: EMStack,
		   axis: str) -> EMStack:
	"""Reverse the order of the pixels along the given axis of every image/volume in the stack
	(out[..., i, ...] = in[..., n-1-i, ...] along that axis), applied to the entire stack at once.
	Returns a new EMStack of the same kind; the input is not modified.
	xform_flip is a real-space operation: Fourier-space (complex) input is rejected.

	axis - the axis to flip: "x", "y" or "z" ("z" is not valid for a 2-D stack); case-insensitive

	Example: xform_flip(my_stack, "x")
	"""
	a = axis.lower() if isinstance(axis, str) else None
	if a not in ("x", "y", "z"):
		raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
	if a == "z" and isinstance(stack, EMStack2D):
		raise ValueError("axis 'z' is not valid for a 2-D stack")
	if a == "x":
		v2d, v3d = xform_flip_x_2d_real, xform_flip_x_3d_real
	elif a == "y":
		v2d, v3d = xform_flip_y_2d_real, xform_flip_y_3d_real
	else:
		v2d, v3d = xform_flip_z_3d_real, xform_flip_z_3d_real
	return _process_stack_real(stack, v2d, v3d)


def xform_mirror(stack: EMStack,
			axis: str) -> EMStack:
	"""Mirror every image/volume in the stack about the center pixel (n//2) of the given axis,
	applied to the entire stack at once: for index i >= 1, out[..., i, ...] = in[..., n-i, ...]
	(the center pixel n//2 stays stationary), and out[..., 0, ...] = 0.
	Returns a new EMStack of the same kind; the input is not modified.
	xform_mirror is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even.

	axis - the axis to mirror: "x", "y" or "z" ("z" is not valid for a 2-D stack); case-insensitive

	Example: xform_mirror(my_stack, "y")
	"""
	a = axis.lower() if isinstance(axis, str) else None
	if a not in ("x", "y", "z"):
		raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
	if a == "z" and isinstance(stack, EMStack2D):
		raise ValueError("axis 'z' is not valid for a 2-D stack")
	if a == "x":
		v2d, v3d = xform_mirror_x_2d_real, xform_mirror_x_3d_real
	elif a == "y":
		v2d, v3d = xform_mirror_y_2d_real, xform_mirror_y_3d_real
	else:
		v2d, v3d = xform_mirror_z_3d_real, xform_mirror_z_3d_real
	return _process_stack_real(stack, v2d, v3d)


def xform_center(stack: EMStack) -> EMStack:
	"""Center every image/volume in the stack by identifying the object silhouette and centering
	it, as in EMAN2's xform.center, applied to the entire stack at once: highpass (removes
	low-frequency gradient) -> circle-mean normalize -> peripheral soft mask -> square (emphasizes
	strong density) -> lowpass (0.05) -> circle-mean normalize -> threshold enclosing the top ~1/10
	of the pixels (searched within [mean, mean+4*sigma]) -> center of mass of the resulting
	silhouette -> integer translation (zero fill) to the image center. Constant (sigma == 0) images
	are left unchanged.
	Returns a new EMStack of the same kind; the input is not modified.
	xform_center is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even.

	Note: EMAN2 additionally refines the silhouette with auto-masking (mask.auto2d/3d); that
	step is approximated here by the thresholded silhouette (which is EMAN2's fallback when the
	auto-mask fails).

	Example: xform_center(my_stack)
	"""
	return _process_stack_real(stack, xform_center_2d_real, xform_center_3d_real)


def xform_centeracf(stack: EMStack) -> EMStack:
	"""Center every image/volume in the stack using its autocorrelation function, as in EMAN2's
	xform.centeracf, applied to the entire stack at once: the peak of the circular self-convolution
	is searched within +/- (n//4, clamped to n//2-1) pixels of the center on each axis, and the
	image is translated (zero fill) by -peak/2 rounded to the nearest pixel.
	Returns a new EMStack of the same kind; the input is not modified.
	xform_centeracf is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even.

	Example: xform_centeracf(my_stack)
	"""
	return _process_stack_real(stack, xform_centeracf_2d_real, xform_centeracf_3d_real)


def xform_centerofmass(stack: EMStack,
			    threshold: float = 0.0,
			    int_shift_only: bool = True,
			    powercenter: bool = False) -> EMStack:
	"""Center every image/volume in the stack at its center of mass, as in EMAN2's
	xform.centerofmass, applied to the entire stack at once: the center of mass (weighted by pixel
	value, over the pixels >= threshold) is found per image, and the image is translated (zero
	fill) by the shift that places it at (nx//2, ny//2, [nz//2]).
	Returns a new EMStack of the same kind; the input is not modified.
	xform_centerofmass is a real-space operation: Fourier-space (complex) input is rejected.
	All image dimensions are assumed even.

	threshold    - only pixels with value >= threshold are included in the center of mass (default 0.0)
	int_shift_only - if True the shift is rounded to whole pixels (default True, as in EMAN2);
	                 if False the (possibly fractional) shift is applied with bilinear interpolation
	powercenter  - if True the pixel values are squared before computing the center (default False);
	                 the threshold then applies to the squared values

	Example: xform_centerofmass(my_stack, threshold=0.5)
	"""
	if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
		raise ValueError(f"threshold must be a number, got {threshold!r}")
	if not isinstance(int_shift_only, bool):
		raise ValueError(f"int_shift_only must be a boolean, got {int_shift_only!r}")
	if not isinstance(powercenter, bool):
		raise ValueError(f"powercenter must be a boolean, got {powercenter!r}")
	return _process_stack_real(stack,
				    xform_centerofmass_2d_real, xform_centerofmass_3d_real,
				    float(threshold), bool(int_shift_only), bool(powercenter))


# ----------------------------------------------------------------------------------
# xform_phaseorigin  (shift the phase origin between the corner (0,0[,0]) and the center)
# ----------------------------------------------------------------------------------
# With even-sized images this shift is its own inverse - applying it twice returns the
# input - so one processor covers both EMAN2's xform.phaseorigin.tocenter and
# xform.phaseorigin.tocorner. It is valid in both spaces:
#   - real space: a circular pixel shift (roll) of n//2 on each axis;
#   - Fourier space (rfft of real data): by the DFT shift theorem, equivalent to
#     multiplying by the phase ramp exp(-i 2 pi k n//2 / n) = (-1)^k, where k is the
#     physical frequency. For even n this equals (-1)^(pixel index) on every axis:
#     the full axes (k = i, or k = i-n with n even) and the half (rfft) axis
#     (k = j, j = 0..n/2). The factors square to 1, so the Fourier shift is also an
#     involution.

def _parity_signs(n: int) -> jnp.ndarray:
	"""Per-axis multiplier of length n: (+1, -1, +1, ...) = (-1)^(pixel index).
	Implements the n//2 phase-origin shift on a full or half (rfft) FFT axis."""
	return jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)


@jax.jit
def xform_phaseorigin_2d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Shift the phase origin of each 2-D image (data: float {N,Y,X}) by (ny//2, nx//2)
	pixels (circular): out[i, j] = in[(i - ny//2) % ny, (j - nx//2) % nx].
	Assumes even dimensions; the shift is its own inverse."""
	ny, nx = data.shape[1], data.shape[2]
	return jnp.roll(data, (ny // 2, nx // 2), axis=(1, 2))


@jax.jit
def xform_phaseorigin_3d_real(data: jnp.ndarray) -> jnp.ndarray:
	"""Shift the phase origin of each 3-D volume (data: float {N,Z,Y,X}) by
	(nz//2, ny//2, nx//2) pixels (circular):
	out[k, i, j] = in[(k - nz//2) % nz, (i - ny//2) % ny, (j - nx//2) % nx].
	Assumes even dimensions; the shift is its own inverse."""
	nz, ny, nx = data.shape[1], data.shape[2], data.shape[3]
	return jnp.roll(data, (nz // 2, ny // 2, nx // 2), axis=(1, 2, 3))


@jax.jit
def xform_phaseorigin_2d_fourier(data: jnp.ndarray) -> jnp.ndarray:
	"""Shift the phase origin of each 2-D Fourier image (data: complex64 rfft2 {N,Y,X/2+1})
	between the corner (0,0) and the center (ny//2, nx//2), by multiplying with the n//2
	DFT-shift factor (-1)^(pixel index) on each axis (see section note).
	Assumes even dimensions; the shift is its own inverse (the factors square to 1)."""
	ny, nxh = data.shape[1], data.shape[2]
	return data * _parity_signs(ny)[None, :, None] * _parity_signs(nxh)[None, None, :]


@jax.jit
def xform_phaseorigin_3d_fourier(data: jnp.ndarray) -> jnp.ndarray:
	"""Shift the phase origin of each 3-D Fourier volume (data: complex64 rfftn {N,Z,Y,X/2+1})
	between the corner (0,0,0) and the center (nz//2, ny//2, nx//2), by multiplying with the
	n//2 DFT-shift factor (-1)^(pixel index) on each axis (see section note).
	Assumes even dimensions; the shift is its own inverse (the factors square to 1)."""
	nz, ny, nxh = data.shape[1], data.shape[2], data.shape[3]
	return data * _parity_signs(nz)[None, :, None, None] * _parity_signs(ny)[None, None, :, None] * _parity_signs(nxh)[None, None, None, :]


def xform_phaseorigin(stack: EMStack) -> EMStack:
	"""Shift the phase origin of every image/volume in the stack between the image corner
	(0,0[, 0]) and the image center (nx//2, ny//2, [nz//2]), applied to the entire stack at
	once. For even-sized images the shift is symmetric: applying it twice returns the input,
	so this single processor covers both EMAN2's xform.phaseorigin.tocenter and
	xform.phaseorigin.tocorner.
	Works in both real space (a circular pixel shift of n//2 on each axis) and Fourier
	space (the equivalent phase-ramp multiplication on the rfft of real data; a Fourier
	input gives a Fourier output of the same kind).
	Returns a new EMStack of the same kind; the input is not modified.
	All image dimensions are assumed even.

	Example: xform_phaseorigin(my_stack)
	"""
	return _process_stack(stack,
				    xform_phaseorigin_2d_real, xform_phaseorigin_2d_fourier,
				    xform_phaseorigin_3d_real, xform_phaseorigin_3d_fourier)


# ----------------------------------------------------------------------------------
# registry (enumeration of processors and their parameters, for GUIs)
# ----------------------------------------------------------------------------------

def _build_processors() -> dict:
	ret = {}
	for name, fn in globals().items():
		if not name.startswith("_") and inspect.isfunction(fn) and fn.__module__ == __name__:
			try:
				sig = inspect.signature(fn)
				hints = get_type_hints(fn)
			except (TypeError, ValueError):
				continue
			params = []
			for pname, p in sig.parameters.items():
				ann = hints.get(pname)
				if ann is EMStack or (isinstance(ann, type) and issubclass(ann, EMStack)):
					continue  # the stack argument is implicit in this framework
				params.append((pname, ann if ann is not None else None, p.default if p.default is not inspect.Parameter.empty else None))
			ret[name] = {"function": fn, "parameters": params, "doc": (fn.__doc__ or "").strip()}
	return ret

PROCESSORS = _build_processors()
