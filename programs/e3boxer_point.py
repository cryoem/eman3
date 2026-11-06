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
# e3_boxer_point.py
#
# Locate particles within a micrograph by fitting a set of 2-D points (x, y, amplitude)
# to the image with the known CTF, maximizing FRC between the CTF-modulated point
# model and the micrograph (fit_points_to_image_ctf_2d in EMAN3jax).
#
# After the fit, only positive-amplitude points are considered: the micrograph is
# assumed to have been contrast-inverted on import so that higher density yields
# larger (positive) amplitude in the raw image. The positive points are clustered
# into groups spanning approximately the user-specified particle size (in angstroms),
# and each accepted cluster is stored as a putative box location ([x_px, y_px,
# "auto_point"] under the "boxes" key) in the micrograph's info file.
#
# CTF parameters come from the micrograph's info file ("ctf_frame3", falling back to
# "ctf_frame") and may be overridden/supplied via the command-line CTF options.
# Beam tilt is assumed to be zero for the CTF calculations in this program.
# A detailed pick record is also written as a JSON file.
#
# Pure Python + JAX, no C++ dependencies.
#

import sys
import os
import math
import time
import json

import numpy as np
import jax.numpy as jnp

from EMAN3.EMAN3 import EMArgumentParser, E3init, E3end, error_exit, warning, info_name
from EMAN3.EMAN3jax import Points, fit_points_to_image_ctf_2d, fit_points_to_image_dot_2d, point_interpolate_simple, ctf_image
from EMAN3.io.imageio import ImageIO

EMANVERSION = "e3_boxer_point (EMAN3)"


def _stem(micro):
	"""Basename of a file with all extensions stripped (used for batch-mode naming)."""
	b = os.path.basename(micro)
	while "." in b:
		b = b.rsplit(".", 1)[0]
	return b


def _perfile_name(path, micro, batch):
	"""For a batch run, derive a unique output filename from the micrograph name so results
	don't clobber each other, e.g. results.json + micrographs/set001.hdf -> results_set001.json.
	Single-file runs keep the given name unchanged."""
	if not batch:
		return path
	root, ext = os.path.splitext(path)
	s = _stem(micro)
	if root.endswith(f"_{s}"):
		return path
	return f"{root}_{s}{ext}" if ext else f"{root}_{s}"


def _info_file_for(micro):
	"""Info file (info/<base>_info.json) for a micrograph, following the EMAN2 project
	convention (info/ next to micrographs/, programs run from the project folder).
	Prefers an existing info/ directory: CWD-relative (the project folder), within the
	micrograph's own folder, or as a sibling of it; otherwise CWD-relative (created)."""
	base = info_name(micro)
	for cand in (base,
	             os.path.join(os.path.dirname(micro), base),
	             os.path.join(os.path.dirname(os.path.dirname(micro)), base)):
		if os.path.isdir(os.path.dirname(cand)):
			return cand
	return base


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


def ctf_from_info(micro):
	"""Read the CTF parameters from the micrograph's info file, if present ("ctf_frame3" preferred,
	"ctf_frame" as fallback; a list-wrapped entry is unwrapped), returning them as a plain dict.
	Plain JSON is used (no object hook) so that other, unregistered object types in the file
	(eg EMData boxes) do not prevent reading the CTF."""
	infoname = _info_file_for(micro)
	if not os.path.isfile(infoname):
		return None
	try:
		with open(infoname) as fh:
			data = json.load(fh)
	except Exception:
		return None
	for key in ("ctf_frame3", "ctf_frame"):
		if key in data:
			o = data[key]
			if isinstance(o, (list, tuple)) and len(o) > 1:
				o = o[1]
			if isinstance(o, dict) and "defocus" in o:
				return o
	return None


METHOD = "auto_point"


def save_boxes(micro, parts):
	"""Store the putative box locations in the micrograph's info file under the "boxes" key,
	following the EMAN2 convention (a list of [x_px, y_px, method, quality] entries, where
	quality is a relative indicator of the likelihood that the location corresponds to a
	particle), with the method named "auto_point". Existing entries written by this program
	are replaced on rerun; entries from other methods (eg swarm) are preserved. Plain JSON
	is used so that other object types in the file round-trip untouched. Returns the info
	filename."""
	infoname = _info_file_for(micro)
	os.makedirs(os.path.dirname(infoname), exist_ok=True)
	data = {}
	if os.path.isfile(infoname):
		try:
			with open(infoname) as fh:
				data = json.load(fh)
		except Exception:
			data = {}
	boxes = data.get("boxes", [])
	kept = [b for b in boxes if not (isinstance(b, (list, tuple)) and len(b) >= 3 and b[2] == METHOD)]
	kept.extend([p["x_px"], p["y_px"], METHOD, p["quality"]] for p in parts)
	data["boxes"] = kept
	with open(infoname, "w") as fh:
		json.dump(data, fh, indent=1)
	return infoname


def cluster_points(pos, apix, side, psize, minpts=2, maxextent=1.0):
	"""Cluster positive-amplitude points into groups spanning approximately psize (A).

	pos - (M,4) points, x,y in -0.5..0.5 (box-relative), amplitude in column 3
	Returns a list of candidate particle dicts (center in A from the region's lower-left
	corner and in pixels, cluster point count, summed amplitude, extent in A), sorted by
	(y, x). A cluster is accepted if it has >= minpts points and its maximum pairwise
	point distance is <= psize*maxextent."""
	M = len(pos)
	if M == 0:
		return []
	xy = (pos[:, [0, 1]] + 0.5) * (side * apix)      # A from region lower-left
	amp = pos[:, 3]
	eps = psize / 2.0
	limit = psize * maxextent

	parent = list(range(M))
	def find(a):
		while parent[a] != a:
			parent[a] = parent[parent[a]]
			a = parent[a]
		return a
	def union(a, b):
		ra, rb = find(a), find(b)
		if ra != rb:
			parent[max(ra, rb)] = min(ra, rb)

	# spatial hashing: candidate neighbors lie in the 3x3 block of cells of size eps
	cellmap = {}
	for i in range(M):
		cellmap.setdefault((int(xy[i, 0] // eps), int(xy[i, 1] // eps)), []).append(i)

	PAIRCAP = 5_000_000
	pairs = 0
	crowded = False
	for i in range(M):
		if crowded:
			break
		cx, cy = int(xy[i, 0] // eps), int(xy[i, 1] // eps)
		for gx in range(cx - 1, cx + 2):
			for gy in range(cy - 1, cy + 2):
				for j in cellmap.get((gx, gy), ()):
					if j <= i:
						continue
					pairs += 1
					if pairs > PAIRCAP:
						warning(f"clustering: points are too crowded ({pairs} candidate pairs) "
						        f"to enumerate; clustering is approximate. Try fewer --npoints or a larger --psize.")
						crowded = True
						break
					dx = xy[i, 0] - xy[j, 0]; dy = xy[i, 1] - xy[j, 1]
					if dx * dx + dy * dy <= eps * eps:
						union(i, j)
				if crowded:
					break

	comps = {}
	for i in range(M):
		comps.setdefault(find(i), []).append(i)

	out = []
	for members in comps.values():
		n = len(members)
		if n < minpts:
			continue
		mxy = xy[members]
		# cheap reject first: the bounding-box diagonal bounds the exact extent
		diag = float(np.hypot(mxy[:, 0].max() - mxy[:, 0].min(), mxy[:, 1].max() - mxy[:, 1].min()))
		if diag > limit:
			continue
		ext = 0.0
		for a in range(n):
			ext = max(ext, float(np.hypot(mxy[a, 0] - mxy[:, 0], mxy[a, 1] - mxy[:, 1]).max()))
		if ext > limit:
			continue
		cx = float(mxy[:, 0].mean())
		cy = float(mxy[:, 1].mean())
		out.append({"x_A": cx, "y_A": cy, "x_px": cx / apix, "y_px": cy / apix,
		            "npoints": n, "amp": float(amp[members].sum()), "extent_A": ext})
	out.sort(key=lambda p: (p["y_A"], p["x_A"]))
	return out


def _ctf_at_size(side, apix, ctf):
	"""CTF amplitude (with B damping) on the rfft grid of a size-`side` image — the same operator
	the FRC loss uses in fit_points_to_image_ctf_2d. Beam tilt is assumed to be zero."""
	wl = 12.2639 / math.sqrt(ctf["voltage"] * 1000.0 + 0.97845 * ctf["voltage"] * ctf["voltage"])
	ac_phase = math.pi / 2.0 - math.asin(max(-1.0, min(1.0, ctf["ampcont"] / 100.0)))
	c = ctf_image(side, apix, wl, ctf["cs"], ac_phase, ctf["defocus"], ctf["dfdiff"], ctf["dfang"], 0.0, 0.0)
	if ctf.get("bfactor", 0.0) > 0:
		fx = jnp.arange(side // 2 + 1); fy = jnp.arange(side); fy = jnp.where(fy > side // 2, fy - side, fy)
		Y, X = jnp.meshgrid(fy, fx, indexing='ij')
		s2 = (X * X + Y * Y) / (apix * side) ** 2
		c = c * jnp.exp(-ctf["bfactor"] * s2 / 4.0)
	return c


def _render_debug(pos, side, apix, ctf, img, outname):
	"""Debug helper: write outname — image 0 is the positive points rendered with the CTF
	(plus B damping) applied, image 1 is the micrograph region they were fit to."""
	c = _ctf_at_size(side, apix, ctf)
	prj = point_interpolate_simple(jnp.asarray(pos[:, [0, 1, 3]], jnp.float32), side)
	sim = np.asarray(jnp.fft.irfft2(jnp.fft.rfft2(prj) * c, s=(side, side)), np.float32)
	hdr = {"nx": side, "ny": side, "nz": 1, "apix_x": apix, "apix_y": apix, "apix_z": apix}
	io = ImageIO(outname, mode="rw")
	try:
		io.write_image(0, sim, hdr)
		io.write_image(1, img, hdr)
	finally:
		io.close()


def _render_points_image(points, side, apix, ctf, img, outname):
	"""Debug helper: write outname — image 0 is the rendering of ALL fitted points with the CTF
	(plus B damping) applied, the model image actually used for the fit, scaled so its
	own std matches that of the micrograph (sim/std(sim)*std(img)) so the amplitudes are directly
	comparable; image 1 is the micrograph region being compared against, for visual comparison;
	image 2 is the same point rendering WITHOUT the CTF (plain bilinear interpolation),
	std-matched to the micrograph like image 0."""
	c = _ctf_at_size(side, apix, ctf)
	prj = point_interpolate_simple(jnp.asarray(points[:, [0, 1, 3]], jnp.float32), side)
	sim = np.asarray(jnp.fft.irfft2(jnp.fft.rfft2(prj) * c, s=(side, side)), np.float32)
	sstd = float(np.std(sim))
	if sstd > 0:
		sim = (sim / sstd * float(np.std(img))).astype(np.float32)
	pstd = float(np.std(prj))
	if pstd > 0:
		prj = (np.asarray(prj, np.float32) / pstd * float(np.std(img))).astype(np.float32)
	hdr = {"nx": side, "ny": side, "nz": 1, "apix_x": apix, "apix_y": apix, "apix_z": apix}
	io = ImageIO(outname, mode="rw")
	try:
		io.write_image(0, sim, hdr)
		io.write_image(1, img, hdr)
		io.write_image(2, prj, hdr)
	finally:
		io.close()


def main():
	progname = os.path.basename(sys.argv[0])
	usage = """prog [options] <micrograph> [<micrograph> ...]

Locate particles within a micrograph: fit a set of 2-D points (x, y, amplitude) to the
image using the known CTF (maximizing FRC between the CTF-modulated point model and
the micrograph), keep the positive-amplitude points, and cluster them into groups
spanning approximately the specified particle size. Each accepted cluster is stored
as a putative box location (method "auto_point") in the micrograph's info file.

The micrograph is assumed to be contrast-inverted (higher density -> higher amplitude).
The CTF is read from the micrograph's info file ("ctf_frame3", falling back to
"ctf_frame") and may be overridden/supplied via the CTF options below. Each file
is processed independently with the same command-line parameters.
"""
	parser = EMArgumentParser(usage=usage, version=EMANVERSION)
	parser.add_pos_argument(name="micrograph", help="Input micrograph(s) (2-D image file(s))", default="",
	                        guitype='filebox', row=0, col=0, rowspan=1, colspan=3)
	parser.add_argument("--npoints", type=int, default=10000, help="Number of points to fit to the micrograph (default 10000)")
	parser.add_argument("--psize", type=float, default=0.0, help="Approximate particle size to cluster to, in angstroms (required)")
	parser.add_argument("--minpts", type=int, default=2, help="Minimum number of points for a cluster to count as a particle (default 2)")
	parser.add_argument("--maxextent", type=float, default=1.0, help="Maximum cluster extent (max pairwise point distance) as a fraction of --psize (default 1.0)")
	parser.add_argument("--apix", type=float, default=0.0, help="Angstrom/pixel, if not found in the file")
	parser.add_argument("--voltage", type=float, default=None, help="Microscope voltage in kV (default from the info-file CTF if present)")
	parser.add_argument("--cs", type=float, default=None, help="Spherical aberration in mm (default from the info-file CTF if present)")
	parser.add_argument("--ampcont", "-a", type=float, default=None, help="Amplitude contrast as a percentage, eg 7 for 7%% (default from the info-file CTF, otherwise 7)")
	parser.add_argument("--defocus", type=float, default=None, help="Defocus in um (default from the info-file CTF)")
	parser.add_argument("--dfdiff", type=float, default=None, help="Astigmatism in um (default from the info-file CTF, otherwise 0)")
	parser.add_argument("--dfang", type=float, default=None, help="Astigmatism angle in degrees (default from the info-file CTF, otherwise 0)")
	parser.add_argument("--bfactor", type=float, default=None, help="B-factor in A^2 (default from the info-file CTF, otherwise 0)")
	parser.add_argument("--seed", type=int, default=0, help="RNG seed for the point fit (default 0)")
	parser.add_argument("--iters", type=int, default=100, help="Adam iterations per coarse-to-fine working size (default 100)")
	parser.add_argument("--dot", action="store_true", help="Use the Fourier-space inner product metric (fit_points_to_image_dot_2d) instead of FRC (fit_points_to_image_ctf_2d) for the point fit")
	parser.add_argument("--output", type=str, default=None, help="Output basename for the pick JSON (default <micrograph>_boxer_point)")
	parser.add_argument("--debug", action="store_true", help="Write debug outputs: points.txt (final fit points, x y amp), points_image.hdf (image 0: fit points with CTF, image 1: micrograph region, image 2: fit points without CTF), per-stage point dumps, and <output>_sim.hdf (CTF-modulated positive points + the micrograph region)")
	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n", type=int, default=0, help="verbose level [0-9]")
	parser.add_argument("--ppid", type=int, default=-1, help="Set the PID of the parent process")

	(options, args) = parser.parse_args()
	logid = E3init(sys.argv, options.ppid)

	if not args:
		error_exit("No micrograph(s) specified")
	if options.psize <= 0:
		error_exit("Specify --psize (approximate particle size, in angstroms)")
	for micro in args:
		if not os.path.isfile(micro):
			error_exit(f"Cannot find input micrograph: {micro}")

	batch = len(args) > 1
	results = []
	t0batch = time.time()
	for micro in args:
		t0 = time.time()
		try:
			img, hdr = load_micrograph(micro, options.verbose)
			nx, ny = img.shape[1], img.shape[0]
			apix = options.apix if options.apix > 0 else float(header_get(hdr, "apix_x", "apix", "pixel_size", default=0.0))
			if apix <= 0:
				raise Exception("Could not determine apix. Specify --apix.")

			side = min(nx, ny) // 2 * 2
			if side < 16:
				raise Exception(f"micrograph {nx}x{ny} is too small")
			if nx != ny:
				warning(f"non-square micrograph {nx}x{ny}; using the centered {side}x{side} region")
			x0 = nx // 2 - side // 2
			y0 = ny // 2 - side // 2
			sub = img[y0:y0 + side, x0:x0 + side]

			# CTF resolution: command line takes precedence, then the info file
			infoctf = ctf_from_info(micro)

			def _v(cli, attr, dflt=0.0):
				v = getattr(options, cli)
				if v is None and infoctf is not None:
					v = infoctf.get(attr)
				if v is None:
					v = dflt
				return float(v)

			ctf = {"voltage": _v("voltage", "voltage"),
			       "cs": _v("cs", "cs"),
			       "ampcont": _v("ampcont", "ampcont", 7.0),
			       "defocus": _v("defocus", "defocus"),
			       "dfdiff": _v("dfdiff", "dfdiff"),
			       "dfang": _v("dfang", "dfang"),
			       "bfactor": _v("bfactor", "bfactor")}
			# beam tilt is assumed to be zero (fit_points_to_image_ctf_2d defaults it to 0)
			if ctf["voltage"] <= 0 or ctf["cs"] <= 0 or ctf["defocus"] <= 0:
				raise Exception("Could not determine voltage/cs/defocus. Provide --voltage --cs --defocus, "
				                "or a ctf_frame3/ctf_frame entry in the micrograph's info file.")
			if infoctf is None and options.ampcont is None:
				warning("No CTF found in the info file; assuming ampcont=7%. Specify --ampcont for accuracy.")

			outbase = options.output if options.output else _stem(micro) + "_boxer_point"
			outbase = _perfile_name(outbase, micro, batch)

			print(f"Boxer point: {micro}")
			print(f"  apix={apix:.4f} A/px  region {side}x{side} (offset {x0},{y0})  npoints={options.npoints}")
			print(f"  CTF: defocus={ctf['defocus']:.4f} um  dfdiff={ctf['dfdiff']:.4f} um  dfang={ctf['dfang']:.2f} deg  "
			      f"B={ctf['bfactor']:.1f}  V={ctf['voltage']} kV  Cs={ctf['cs']} mm  AC={ctf['ampcont']}%")

			fitfn = fit_points_to_image_dot_2d if options.dot else fit_points_to_image_ctf_2d
			metric = "DOT" if options.dot else "FRC"
			points, score = fitfn(sub, apix, options.npoints, ctf,
		                                          seed=options.seed, iters=options.iters, verbose=options.verbose,
		                                          dump_prefix=(outbase + "_") if options.debug else None)
			pn = points.numpy
			pos = pn[pn[:, 3] > 0.0]
			print(f"  fit done in {time.time() - t0:.1f}s: {metric}={score:.4f};  "
			      f"{len(pos)} positive / {len(pn) - len(pos)} negative-amplitude points (positives used for clustering)")

			parts = cluster_points(pos, apix, side, options.psize, options.minpts, options.maxextent)
			# relative quality: the summed cluster amplitude, normalized so the strongest
			# cluster in this micrograph is 1.0 (a relative likelihood indicator)
			maxamp = max((p["amp"] for p in parts), default=0.0)
			for p in parts:
				p["quality"] = (p["amp"] / maxamp) if maxamp > 0.0 else 0.0
			print(f"  {len(parts)} particle clusters (npts >= {options.minpts}, extent <= {options.maxextent:.2f} x {options.psize:.1f} A)")
			for i, p in enumerate(parts[:50]):
				print(f"    {i + 1:5d}  x={p['x_px']:8.1f} y={p['y_px']:8.1f} px  ({p['x_A']:8.1f},{p['y_A']:8.1f} A)  "
				      f"npts={p['npoints']:4d}  amp={p['amp']:8.3f}  extent={p['extent_A']:6.1f} A  q={p['quality']:.3f}")
			if len(parts) > 50:
				print(f"    ... and {len(parts) - 50} more")

			rec = {"micrograph": micro, "apix": apix, "region_px": [x0, y0, side, side],
			       "psize_A": options.psize, "minpts": options.minpts, "maxextent": options.maxextent,
			       "npoints_fitted": int(len(pn)), "npoints_positive": int(len(pos)),
			       "fit_metric": "dot" if options.dot else "frc", "fit_score": float(score),
			       "ctf": ctf, "nparticles": len(parts),
			       "particles": parts}
			outjson = outbase + ".json"
			with open(outjson, "w") as fh:
				json.dump(rec, fh, indent=2)
			print(f"  wrote {outjson}")

			# Store the putative box locations in the micrograph's info file (EMAN2 "boxes" convention)
			infoname = save_boxes(micro, parts)
			print(f"  wrote {len(parts)} boxes (method {METHOD}) -> {infoname}")

			if options.debug:
				pointsname = "points.txt" if not batch else _perfile_name("points.txt", micro, batch)
				np.savetxt(pointsname, pn[:, [0, 1, 3]], delimiter="\t")
				pimgname = "points_image.hdf" if not batch else _perfile_name("points_image.hdf", micro, batch)
				_render_points_image(pn, side, apix, ctf, sub, pimgname)
				simname = outbase + "_sim.hdf"
				_render_debug(pos, side, apix, ctf, sub, simname)
				print(f"  wrote {pointsname} (final fit points, x\ty\tamp)")
				print(f"  wrote {pimgname} (image 0: fit points with CTF, image 1: micrograph region, image 2: fit points without CTF)")
				print(f"  wrote {simname} (and {outbase}_<stage>.txt point dumps)")

			results.append((micro, True, None))
		except Exception as e:
			results.append((micro, False, str(e)))
			print(f"\n  ERROR on {micro}: {e}")

	if batch:
		print(f"\nSUMMARY (batch, {time.time() - t0batch:.1f}s total): {sum(1 for _, ok, _ in results if ok)}/{len(results)} succeeded")
		for micro, ok, err in results:
			if ok:
				print(f"  OK    {micro}")
			else:
				print(f"  FAIL  {micro}  ({err})")

	E3end(logid)


if __name__ == "__main__":
	main()
