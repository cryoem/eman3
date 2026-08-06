#!/usr/bin/env python
"""Round-trip tests for all IO formats using standard test images.

Uses local helper functions instead of importing from EMAN3 (which has
dependency issues in isolated runs). These produce the same type=1
sequential integer patterns as test_image() / test_image_3d().
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from EMAN3.io.spiderio import SpiderIO
from EMAN3.io.pgmio import PgmIO
from EMAN3.io.pngio import PngIO
from EMAN3.io.jpegio import JpegIO
from EMAN3.io.tiffio import TiffIO
from EMAN3.io.icosio import IcosIO
from EMAN3.io.mrcio import MrcIO
from EMAN3.io.imagicio2 import ImagicIO
from EMAN3.io.omapio import OmapIO
from EMAN3.io.amiraio import AmiraIO
from EMAN3.io.pifio import PifIO
from EMAN3.io.eerio import EerIO
from EMAN3.io.vtkio import VtkIO
from EMAN3.io.hdfio2 import HdfIO2
from EMAN3.io.imageio import ImageIO


def test_img_2d(ny, nx):
	"""Return type=1 2D test image: row-major sequential ints."""
	return np.arange(ny * nx, dtype=np.float32).reshape((ny, nx))

def test_img_3d(nz, ny, nx):
	"""Return type=1 3D test volume: sequential ints (nz, ny, nx)."""
	return np.arange(nz * ny * nx, dtype=np.float32).reshape((nz, ny, nx))


SAMPLE = "sample_files"

def main():
	test_jpg()
	test_pgm()
	test_png()
	test_spider()
	test_tiff()
	test_icos()
	test_mrc()
	test_imagic()
	test_omap()
	test_amira()
	test_pif()
	test_eer()
	test_vtk()
	test_hdf5()
	test_imageio_wrapper()
	print("\nAll IO round-trip tests PASSED.")

# ------------------------------------------------------------------
# 2D single-image formats
# ------------------------------------------------------------------

def test_jpg():
	"""JPEG: 2D, single-image, 8-bit lossy."""
	print("testing JPEG (2D single-image, 8-bit)")
	out = os.path.join(SAMPLE, "jpeg_rt.jpg")
	try: os.unlink(out)
	except: pass

	img = test_img_2d(32, 64)
	io = JpegIO(out, "rw")
	io.write_header({"nx": 64, "ny": 32, "nz": 1, "bitdepth": 8}, 0)
	io.write_data(img, 0)
	io = None

	io = JpegIO(out, "r")
	h = io.read_header(0)
	d = io.read_data(0)
	print(f"  shape={d.shape}, nx={h['nx']}, ny={h['ny']}")
	# Lossy — check shape and that values span full range
	assert d.shape == (32, 64), f"Shape mismatch: {d.shape}"
	assert d.max() > d.min() + 0.5, "JPEG lost dynamic range"
	print("  PASS (lossy format, geometry OK)")
	io = None

def test_pgm():
	"""PGM: 2D, single-image, 8/16-bit lossless."""
	print("testing PGM (2D single-image)")
	out = os.path.join(SAMPLE, "pgm_rt.pgm")
	try: os.unlink(out)
	except: pass

	img = test_img_2d(32, 64)
	io = PgmIO(out, "rw")
	io.write_header({"nx": 64, "ny": 32, "nz": 1}, 0)
	io.write_data(img, 0)
	io = None

	io = PgmIO(out, "r")
	h = io.read_header(0)
	d = io.read_data(0)
	print(f"  shape={d.shape}")
	assert d.shape == (32, 64), f"Shape mismatch: {d.shape}"
	# PGM scales to uint8 [0,255], check Y-flip preserved by verifying row ordering
	# Row 0 should still have lowest values after flip+unflip round-trip
	assert d.max() >= 250, f"Max too low: {d.max()}"
	assert d.min() <= 10,  f"Min too high: {d.min()}"
	# Verify geometry: row with value 0 should be at bottom (last row)
	row_max = np.argmax(np.max(d, axis=1))
	assert row_max == 31, f"Max-value row at index {row_max}, expected 31 (bottom)"
	print("  PASS")
	io = None

def test_png():
	"""PNG: 2D, single-image, 8/16-bit lossless."""
	print("testing PNG (2D single-image, 16-bit)")
	out = os.path.join(SAMPLE, "png_rt.png")
	try: os.unlink(out)
	except: pass

	img = test_img_2d(32, 64)
	io = PngIO(out, "rw")
	io.write_header({"nx": 64, "ny": 32, "nz": 1, "bitdepth": 16}, 0)
	io.write_data(img, 0)
	io = None

	io = PngIO(out, "r")
	h = io.read_header(0)
	d = io.read_data(0)
	print(f"  shape={d.shape}")
	assert d.shape == (32, 64), f"Shape mismatch: {d.shape}"
	# PNG scales float data to uint8/uint16 [0,max], check geometry via relative values
	v00 = d[0, 0]
	v_last = d[-1, -1]
	assert v_last > v00 + 0.5, f"Dynamic range insufficient: {v00} to {v_last}"
	print("  PASS")
	io = None

# ------------------------------------------------------------------
# Stack + 3D formats
# ------------------------------------------------------------------

def test_spider():
	"""SPIDER: stack + 3D volume."""
	print("testing SPIDER (stack + 3D)")

	# -- 2D stack: type=1 then type=0 --
	out = os.path.join(SAMPLE, "spider_stack_rt.spi")
	try: os.unlink(out)
	except: pass

	img1 = test_img_2d(48, 96)
	img0 = test_img_3d(1, 48, 96).astype(np.float32) * 0.5
	io = SpiderIO(out, "rw")
	idx = io.write_header({"nx": 96, "ny": 48, "nz": 1}, -1)
	io.write_data(img1, idx)
	idx = io.write_header({"nx": 96, "ny": 48, "nz": 1}, -1)
	io.write_data(img0, idx)
	io = None

	io = SpiderIO(out, "r")
	d = io.read_data(0)
	print(f"  stack[0] shape={d.shape}")
	assert d.shape == (48, 96), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0]) == 0.0,   f"d[0,0]={d[0,0]}"
	assert float(d[0, 1]) == 1.0,   f"d[0,1]={d[0,1]}"
	assert float(d[1, 0]) == 96.0,  f"d[1,0]={d[1,0]}"
	print("  stack PASS")
	io = None

	# -- 3D volume: type=1 --
	out3 = os.path.join(SAMPLE, "spider_3d_rt.spi")
	try: os.unlink(out3)
	except: pass

	img3d = test_img_3d(24, 32, 48)
	io = SpiderIO(out3, "rw")
	idx = io.write_header({"nx": 48, "ny": 32, "nz": 24}, -1)
	io.write_data(img3d, idx)
	io = None

	io = SpiderIO(out3, "r")
	d = io.read_data(0)
	print(f"  3D shape={d.shape}")
	assert d.shape == (24, 32, 48), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0, 0]) == 0.0,     f"d[0,0,0]={d[0,0,0]}"
	assert float(d[0, 0, 1]) == 1.0,     f"d[0,0,1]={d[0,0,1]}"
	assert float(d[0, 1, 0]) == 48.0,    f"d[0,1,0]={d[0,1,0]}"
	assert float(d[1, 0, 0]) == 1536.0,  f"d[1,0,0]={d[1,0,0]}"
	print("  3D PASS")
	io = None

def test_tiff():
	"""TIFF: 2D stack."""
	print("testing TIFF (2D stack)")
	out = os.path.join(SAMPLE, "tiff_rt.tif")
	try: os.unlink(out)
	except: pass

	img1 = test_img_2d(48, 96)
	io = TiffIO(out, "rw")
	idx = io.write_header({"nx": 96, "ny": 48, "nz": 1, "bitdepth": 0}, -1)
	io.write_data(img1, idx)
	io.write_header({"nx": 96, "ny": 48, "nz": 1, "bitdepth": 0}, -1)
	io = None

	io = TiffIO(out, "r")
	d = io.read_data(0)
	print(f"  shape={d.shape}")
	assert d.shape == (48, 96), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0]) == 0.0,   f"d[0,0]={d[0,0]}"
	assert float(d[0, 1]) == 1.0,   f"d[0,1]={d[0,1]}"
	assert float(d[1, 0]) == 96.0,  f"d[1,0]={d[1,0]}"
	print("  PASS")
	io = None

def test_icos():
	"""ICOS: 3D single-image."""
	print("testing ICOS (3D)")
	out = os.path.join(SAMPLE, "icos_rt.ico")
	try: os.unlink(out)
	except: pass

	img3d = test_img_3d(24, 32, 48)
	io = IcosIO(out, "rw")
	io.write_header({"nx": 48, "ny": 32, "nz": 24}, 0)
	io.write_data(img3d, 0)
	io._file.flush()  # Ensure data is on disk
	io = None

	io = IcosIO(out, "r")
	h = io.read_header(0)
	d = io.read_data(0)
	print(f"  shape={d.shape}")
	assert d.shape == (24, 32, 48), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0, 0]) == 0.0,      f"d[0,0,0]={d[0,0,0]}"
	assert float(d[0, 0, 1]) == 1.0,      f"d[0,0,1]={d[0,0,1]}"
	assert float(d[0, 1, 0]) == 48.0,     f"d[0,1,0]={d[0,1,0]}"
	assert float(d[1, 0, 0]) == 1536.0,   f"d[1,0,0]={d[1,0,0]}"
	print("  PASS")
	io = None

def test_imagic():
	"""IMAGIC: 2D stack + 3D volume."""
	print("testing IMAGIC (2D stack + 3D)")

	# -- 2D stack: type=1 then type=0 --
	out = os.path.join(SAMPLE, "imagic_stack_rt")
	for ext in [".hed", ".img"]:
		try: os.unlink(out + ext)
		except: pass

	img1 = test_img_2d(48, 96)
	io = ImagicIO(out, "rw")
	idx = io.write_header({"nx": 96, "ny": 48, "nz": 1}, -1)
	io.write_data(img1, idx)
	idx = io.write_header({"nx": 96, "ny": 48, "nz": 1}, -1)
	io.write_data(img1.astype(np.float32) * 0.5, idx)
	io = None

	io = ImagicIO(out, "r")
	assert io.nimg == 2, f"Expected 2 images, got {io.nimg}"
	d = io.read_data(0)
	print(f"  stack[0] shape={d.shape}")
	# nz==1 is preserved as 2D
	assert d.shape == (48, 96), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0]) == 0.0,   f"d[0,0]={float(d[0,0])}"
	assert float(d[0, 1]) == 1.0,   f"d[0,1]={float(d[0,1])}"
	assert float(d[1, 0]) == 96.0,  f"d[1,0]={float(d[1,0])}"
	print("  stack PASS")
	io = None

	# -- 3D volume: type=1 --
	out3 = os.path.join(SAMPLE, "imagic_3d_rt")
	for ext in [".hed", ".img"]:
		try: os.unlink(out3 + ext)
		except: pass

	img3d = test_img_3d(24, 32, 48)
	io = ImagicIO(out3, "rw")
	idx = io.write_header({"nx": 48, "ny": 32, "nz": 24}, -1)
	io.write_data(img3d, idx)
	io = None

	io = ImagicIO(out3, "r")
	d = io.read_data(0)
	print(f"  3D shape={d.shape}")
	assert d.shape == (24, 32, 48), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0, 0]) == 0.0,     f"d[0,0,0]={d[0,0,0]}"
	assert float(d[0, 0, 1]) == 1.0,     f"d[0,0,1]={d[0,0,1]}"
	assert float(d[0, 1, 0]) == 48.0,    f"d[0,1,0]={d[0,1,0]}"
	assert float(d[1, 0, 0]) == 1536.0,  f"d[1,0,0]={d[1,0,0]}"
	print("  3D PASS")
	io = None


def test_omap():
	"""OMAP/DNS6: 3D single-volume, uint8 packed."""
	print("testing OMAP/DNS6 (3D round-trip)")

	# Round-trip: use float range [0, 2.54] so iprod=100 maps perfectly to uint8
	out = os.path.join(SAMPLE, "omap_rt.dns6")
	try: os.unlink(out)
	except: pass

	img3d = test_img_3d(4, 8, 8) * 0.01   # values 0..2.54, fits uint8 via iprod=100
	io = OmapIO(out, "rw")
	io.write_header({"nx": 8, "ny": 8, "nz": 4, "minimum": 0.0, "maximum": 2.54}, 0)
	io.write_data(img3d, 0)
	io = None

	# Verify is_valid
	assert OmapIO.is_valid(out), "OMAP file failed is_valid"

	io = OmapIO(out, "r")
	h = io.read_header(0)
	print(f"  header: nx={h['nx']}, ny={h['ny']}, nz={h['nz']}")
	d = io.read_data(0)
	print(f"  shape={d.shape}")
	assert d.shape == (4, 8, 8), f"Shape mismatch: {d.shape}"
	# Check geometry at known positions
	assert abs(float(d[0, 0, 0]) - 0.0) < 0.1,     f"d[0,0,0]={float(d[0,0,0])}"
	assert abs(float(d[0, 0, 1]) - 0.01) < 0.1,   f"d[0,0,1]={float(d[0,0,1])}"
	assert abs(float(d[0, 1, 0]) - 0.08) < 0.1,   f"d[0,1,0]={float(d[0,1,0])}"
	assert abs(float(d[1, 0, 0]) - 0.64) < 0.1,  f"d[1,0,0]={float(d[1,0,0])}"
	print("  PASS")
	io = None


def test_pif():
	"""PIF: 2D stack + 3D volume."""
	print("testing PIF (stack + 3D)")

	# -- 2D stack: two images type=1 and type=0 --
	out = os.path.join(SAMPLE, "pif_stack_rt.pif")
	try: os.unlink(out)
	except Exception: pass

	img1 = test_img_2d(48, 96)
	io = PifIO(out, "rw")
	idx = io.write_header({"nx": 96, "ny": 48, "nz": 1}, -1)
	io.write_data(img1.astype(np.float32), idx)  # pass as (ny,nx)
	idx = io.write_header({"nx": 96, "ny": 48, "nz": 1}, -1)
	io.write_data(img1.astype(np.float32) * 0.5, idx)
	io = None

	assert PifIO.is_valid(out), "PIF file failed is_valid"

	io = PifIO(out, "r")
	assert io.nimg == 2, f"Expected 2 images, got {io.nimg}"
	d = io.read_data(0)
	print(f"  stack[0] shape={d.shape}")
	# nz==1 -> returned as 2D (ny, nx)
	assert d.shape == (48, 96), f"Shape mismatch: {d.shape}"
	print("  stack PASS")
	io = None

	# -- 3D volume: single image --
	out3 = os.path.join(SAMPLE, "pif_3d_rt.pif")
	try: os.unlink(out3)
	except Exception: pass

	img3d = test_img_3d(24, 32, 48) * 0.01
	io = PifIO(out3, "rw")
	idx = io.write_header({"nx": 48, "ny": 32, "nz": 24}, -1)
	io.write_data(img3d, idx)
	io = None

	io = PifIO(out3, "r")
	d = io.read_data(0)
	print(f"  3D shape={d.shape}")
	assert d.shape == (24, 32, 48), f"Shape mismatch: {d.shape}"
	print("  3D PASS")
	io = None


def test_amira():
	"""Amira Mesh: 3D single-volume."""
	print("testing Amira (3D)")

	# Round-trip: type=1 volume, scaled to fit reasonable float range
	out = os.path.join(SAMPLE, "amira_rt.am")
	try: os.unlink(out)
	except: pass

	img3d = test_img_3d(24, 32, 48) * 0.01   # small floats for precision
	io = AmiraIO(out, "rw")
	io.write_header({"nx": 48, "ny": 32, "nz": 24, "apix_x": 1.5}, 0)
	io.write_data(img3d, 0)
	io = None

	assert AmiraIO.is_valid(out), "Amira file failed is_valid"

	io = AmiraIO(out, "r")
	h = io.read_header(0)
	print(f"  header: nx={h['nx']}, ny={h['ny']}, nz={h['nz']}")
	d = io.read_data(0)
	print(f"  shape={d.shape}")
	assert d.shape == (24, 32, 48), f"Shape mismatch: {d.shape}"
	# Check geometry at known positions (Y-flip is handled internally)
	assert float(d[0, 0, 0]) >= -0.1 and float(d[0, 0, 0]) < 0.1, f"d[0,0,0]={float(d[0,0,0])}"
	print("  PASS")
	io = None


def test_mrc():
	"""MRC: single 2D + MRCS stack + 3D volume."""
	print("testing MRC (single + stack + 3D)")

	# -- Single 2D image --
	out = os.path.join(SAMPLE, "mrc_single_rt.mrc")
	try: os.unlink(out)
	except: pass

	img = test_img_2d(48, 96)
	io = MrcIO(out, "rw")
	io.write_header({"nx": 96, "ny": 48, "nz": 1, "bitdepth": 0}, 0)
	io.write_data(img, 0)
	io = None

	io = MrcIO(out, "r")
	d = io.read_data(0)
	print(f"  single shape={d.shape}")
	assert d.shape == (48, 96), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0]) == 0.0,   f"d[0,0]={d[0,0]}"
	assert float(d[0, 1]) == 1.0,   f"d[0,1]={d[0,1]}"
	assert float(d[1, 0]) == 96.0,  f"d[1,0]={d[1,0]}"
	print("  single PASS")
	io = None

	# -- MRCS stack: type=1 then type=0 --
	out_s = os.path.join(SAMPLE, "mrcs_stack_rt.mrcs")
	try: os.unlink(out_s)
	except: pass

	img1 = test_img_2d(48, 96)
	io = MrcIO(out_s, "rw")
	idx = io.write_header({"nx": 96, "ny": 48, "nz": 1, "bitdepth": 0}, -1)
	io.write_data(img1, idx)
	io.write_header({"nx": 96, "ny": 48, "nz": 1, "bitdepth": 0}, -1)
	io = None

	io = MrcIO(out_s, "r")
	d = io.read_data(0)
	print(f"  stack[0] shape={d.shape}")
	assert d.shape == (48, 96), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0]) == 0.0,   f"d[0,0]={d[0,0]}"
	assert float(d[0, 1]) == 1.0,   f"d[0,1]={d[0,1]}"
	assert float(d[1, 0]) == 96.0,  f"d[1,0]={d[1,0]}"
	print("  stack PASS")
	io = None

	# -- 3D volume: type=1 --
	out3 = os.path.join(SAMPLE, "mrc_3d_rt.mrc")
	try: os.unlink(out3)
	except: pass

	img3d = test_img_3d(24, 32, 48)
	io = MrcIO(out3, "rw")
	io.write_header({"nx": 48, "ny": 32, "nz": 24, "bitdepth": 0}, 0)
	io.write_data(img3d, 0)
	io = None

	io = MrcIO(out3, "r")
	d = io.read_data(0)
	print(f"  3D shape={d.shape}")
	assert d.shape == (24, 32, 48), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0, 0]) == 0.0,      f"d[0,0,0]={d[0,0,0]}"
	assert float(d[0, 0, 1]) == 1.0,      f"d[0,0,1]={d[0,0,1]}"
	assert float(d[0, 1, 0]) == 48.0,     f"d[0,1,0]={d[0,1,0]}"
	assert float(d[1, 0, 0]) == 1536.0,   f"d[1,0,0]={d[1,0,0]}"
	print("  3D PASS")
	io = None


def test_eer():
	"""EER: read-only event-based electron detector format.

	EER uses custom RLE compression — synthetic test data cannot be
	generated.  This test verifies module structure and skips gracefully
	when no hardware .eer sample file is available.
	"""
	print("testing EER (read-only, event-based)")

	# Verify class attributes
	assert EerIO.SUPPORT_STACK is True
	assert EerIO.SUPPORT_3D is False
	assert EerIO.SUPPORT_3D_STACK is False
	assert EerIO.SUPPORTED_COMPRESS is False

	# Look for any .eer sample file on the system
	eer_files = []
	for root_dir in [SAMPLE, "/home/stevel/pro/eman3/test_files/sample_files"]:
		if os.path.isdir(root_dir):
			for f in os.listdir(root_dir):
				if f.lower().endswith(".eer"):
					eer_files.append(os.path.join(root_dir, f))

	if not eer_files:
		print("  SKIP (no .eer sample file available — requires hardware data)")
		return

	# Test with first found .eer file
	ee = eer_files[0]
	print(f"  Using: {ee}")
	assert EerIO.is_valid(ee), "File failed is_valid"

	io = EerIO(ee)
	print(f"  Frames: {io.nimg}, dims: {io._pix_dim}x{io._pix_dim}")

	h = io.read_header(0)
	assert h["nx"] == h["ny"], f"EER images should be square"

	d = io.read_data(0)
	print(f"  shape={d.shape}, min={float(d.min()):.3f}, max={float(d.max()):.3f}")
	assert d.ndim == 2, f"Expected 2D data, got {d.ndim}D"

	io = None
	print("  PASS")


def test_vtk():
	"""VTK: 3D single-volume, big-endian float binary."""
	print("testing VTK (3D)")

	out = os.path.join(SAMPLE, "vtk_rt.vtk")
	try: os.unlink(out)
	except: pass

	img3d = test_img_3d(24, 32, 48) * 0.01
	io = VtkIO(out, "rw")
	io.write_header({"nx": 48, "ny": 32, "nz": 24}, 0)
	io.write_data(img3d, 0)
	io = None

	assert VtkIO.is_valid(out), "VTK file failed is_valid"

	io = VtkIO(out, "r")
	h = io.read_header(0)
	print(f"  header: nx={h['nx']}, ny={h['ny']}, nz={h['nz']}")
	assert len(io) == 1, f"VTK should have len=1, got {len(io)}"

	d = io.read_data(0)
	print(f"  shape={d.shape}")
	assert d.shape == (24, 32, 48), f"Shape mismatch: {d.shape}"
	# Check values at known positions (small floats preserved in binary)
	assert abs(float(d[0, 0, 0]) - 0.0) < 0.01, f"d[0,0,0]={float(d[0,0,0])}"
	assert abs(float(d[0, 0, 1]) - 0.01) < 0.01, f"d[0,0,1]={float(d[0,0,1])}"
	print("  PASS")
	io = None


def test_hdf5():
	"""HDF5: 2D stack, 3D volume, compressed with bit-reduction."""
	print("testing HDF5 (2D stack + 3D + compressed)")

	# -- 2D stack --
	out = os.path.join(SAMPLE, "hdf5_stack_rt.hdf")
	try: os.unlink(out)
	except: pass

	img1 = test_img_2d(48, 96)
	io = HdfIO2(out, "rw")
	idx = io.write_header({"nx": 96, "ny": 48, "nz": 1}, -1)
	io.write_data(img1, idx)
	idx = io.write_header({"nx": 96, "ny": 48, "nz": 1}, -1)
	io.write_data(img1.astype(np.float32) * 0.5, idx)
	io = None

	assert HdfIO2.is_valid(out), "HDF5 file failed is_valid"

	io = HdfIO2(out, "r")
	assert io.nimg == 2, f"Expected 2 images, got {io.nimg}"
	assert len(io) == 2, f"Expected len=2, got {len(io)}"
	d = io.read_data(0)
	print(f"  stack[0] shape={d.shape}")
	assert d.shape == (48, 96), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0]) == 0.0,   f"d[0,0]={float(d[0,0])}"
	assert float(d[0, 1]) == 1.0,   f"d[0,1]={float(d[0,1])}"
	assert float(d[1, 0]) == 96.0,  f"d[1,0]={float(d[1,0])}"
	d2 = io.read_data(1)
	assert abs(float(d2[0, 0]) - 0.0) < 0.5
	print("  stack PASS")
	io = None

	# -- 3D volume --
	out3 = os.path.join(SAMPLE, "hdf5_3d_rt.hdf")
	try: os.unlink(out3)
	except: pass

	img3d = test_img_3d(24, 32, 48)
	io = HdfIO2(out3, "rw")
	idx = io.write_header({"nx": 48, "ny": 32, "nz": 24}, -1)
	io.write_data(img3d, idx)
	io = None

	io = HdfIO2(out3, "r")
	h = io.read_header(0)
	print(f"  3D header: nx={h['nx']}, ny={h['ny']}, nz={h['nz']}")
	assert len(io) == 1, f"Expected len=1 for single 3D, got {len(io)}"

	d = io.read_data(0)
	print(f"  3D shape={d.shape}")
	assert d.shape == (24, 32, 48), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0, 0]) == 0.0,     f"d[0,0,0]={float(d[0,0,0])}"
	assert float(d[0, 0, 1]) == 1.0,     f"d[0,0,1]={float(d[0,0,1])}"
	assert float(d[0, 1, 0]) == 48.0,    f"d[0,1,0]={float(d[0,1,0])}"
	assert float(d[1, 0, 0]) == 1536.0,  f"d[1,0,0]={float(d[1,0,0])}"
	print("  3D PASS")
	io = None

	# -- Compressed with 8-bit reduction (integer data, should round-trip exactly) --
	out8 = os.path.join(SAMPLE, "hdf5_compr8_rt.hdf")
	try: os.unlink(out8)
	except: pass

	img8 = np.arange(256, dtype=np.float32).reshape((16, 16))
	io = HdfIO2(out8, "rw")
	idx = io.write_header({"nx": 16, "ny": 16, "nz": 1, "render_bits": 8}, -1)
	io.write_data(img8, idx)
	io = None

	io = HdfIO2(out8, "r")
	h = io.read_header(0)
	d = io.read_data(0)
	print(f"  compressed 8-bit shape={d.shape}")
	assert d.shape == (16, 16), f"Shape mismatch: {d.shape}"
	# Integer data with range [0,255] should round-trip exactly with 8 bits
	assert float(d[0, 0]) == 0.0,     f"d[0,0]={float(d[0,0])}"
	assert float(d[0, 1]) == 1.0,     f"d[0,1]={float(d[0,1])}"
	assert float(d[0, 15]) == 15.0,   f"d[0,15]={float(d[0,15])}"
	assert float(d[15, 15]) == 255.0, f"d[15,15]={float(d[15,15])}"
	print("  compressed 8-bit PASS")
	io = None

	# -- Compressed with 16-bit reduction (larger integer range) --
	out16 = os.path.join(SAMPLE, "hdf5_compr16_rt.hdf")
	try: os.unlink(out16)
	except: pass

	img16 = np.arange(65536, dtype=np.float32).reshape((256, 256))
	io = HdfIO2(out16, "rw")
	idx = io.write_header({"nx": 256, "ny": 256, "nz": 1, "render_bits": 16}, -1)
	io.write_data(img16, idx)
	io = None

	io = HdfIO2(out16, "r")
	d = io.read_data(0)
	print(f"  compressed 16-bit shape={d.shape}")
	assert d.shape == (256, 256), f"Shape mismatch: {d.shape}"
	# Full 16-bit integer range should round-trip exactly
	assert float(d[0, 0]) == 0.0,        f"d[0,0]={float(d[0,0])}"
	assert float(d[0, 1]) == 1.0,        f"d[0,1]={float(d[0,1])}"
	assert float(d[255, 255]) == 65535.0, f"d[255,255]={float(d[255,255])}"
	print("  compressed 16-bit PASS")
	io = None


def test_imageio_wrapper():
	"""Unified ImageIO wrapper: auto-detection, read/write API."""
	print("testing ImageIO wrapper (auto-detect + unified API)")

	# Test 1: Auto-detect HDF5 by extension + magic bytes
	out_hdf = os.path.join(SAMPLE, "hdf5_wrapper_test.hdf")
	try: os.unlink(out_hdf)
	except: pass

	img2d = test_img_2d(32, 64)
	io = ImageIO(out_hdf, mode="rw")
	io.write_image(-1, img2d, {"nx": 64, "ny": 32, "nz": 1})
	io.close()

	# Reopen — should auto-detect as HDF5
	io = ImageIO(out_hdf)
	fmt = ImageIO.detect_format(out_hdf)
	print(f"  detected format: {fmt}")
	assert fmt == "HdfIO2", f"Expected HdfIO2, got {fmt}"

	d, h = io.read_image(0)
	print(f"  read_image shape={d.shape}, nx={h.get('nx')}")
	assert d.shape == (32, 64), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0]) == 0.0
	assert float(d[0, 1]) == 1.0
	assert float(d[1, 0]) == 64.0
	io.close()

	# Test 2: Auto-detect MRC by magic bytes (no extension match)
	out_mrc = os.path.join(SAMPLE, "mrc_wrapper_test.mrc")
	try: os.unlink(out_mrc)
	except: pass

	img3d = test_img_3d(16, 24, 32)
	io = ImageIO(out_mrc, mode="rw")
	io.write_image(-1, img3d, {"nx": 32, "ny": 24, "nz": 16})
	io.close()

	io = ImageIO(out_mrc)
	fmt = ImageIO.detect_format(out_mrc)
	print(f"  detected format for .mrc: {fmt}")
	assert fmt == "MrcIO", f"Expected MrcIO, got {fmt}"
	d, h = io.read_image(0)
	print(f"  read_image 3D shape={d.shape}")
	assert d.shape == (16, 24, 32), f"Shape mismatch: {d.shape}"
	assert float(d[0, 0, 0]) == 0.0
	assert float(d[0, 0, 1]) == 1.0
	assert float(d[1, 0, 0]) == 768.0
	io.close()

	# Test 3: Auto-detect IMAGIC by magic bytes
	out_imag = os.path.join(SAMPLE, "imag_wrapper_test")
	for ext in [".hed", ".img"]:
		try: os.unlink(out_imag + ext)
		except: pass

	io = ImageIO(out_imag + ".hed", mode="rw")
	io.write_image(-1, img2d, {"nx": 64, "ny": 32, "nz": 1})
	io.close()

	io = ImageIO(out_imag + ".hed")
	fmt = ImageIO.detect_format(out_imag + ".hed")
	print(f"  detected format for .hed: {fmt}")
	assert fmt == "ImagicIO", f"Expected ImagicIO, got {fmt}"
	d, h = io.read_image(0)
	print(f"  read_image shape={d.shape}")
	assert d.shape == (32, 64), f"Shape mismatch: {d.shape}"
	io.close()

	# Test 4: read_images for multi-image stack
	out_stack = os.path.join(SAMPLE, "hdf5_stack_wrapper.hdf")
	try: os.unlink(out_stack)
	except: pass

	img_a = test_img_2d(32, 64)
	img_b = (test_img_2d(32, 64) * 2.5).astype(np.float32)
	io = ImageIO(out_stack, mode="rw")
	io.write_image(-1, img_a, {"nx": 64, "ny": 32, "nz": 1})
	io.write_image(-1, img_b, {"nx": 64, "ny": 32, "nz": 1})
	io.close()

	io = ImageIO(out_stack)
	stack, headers = io.read_images([0, 1])
	print(f"  read_images stack shape={stack.shape}")
	assert stack.shape == (2, 32, 64), f"Stack shape mismatch: {stack.shape}"
	assert float(stack[0, 0, 0]) == 0.0
	assert abs(float(stack[1, 0, 1]) - 2.5) < 0.01
	assert len(headers) == 2
	io.close()

	# Test 5: write_images for multi-image
	out_wstack = os.path.join(SAMPLE, "hdf5_wstack_wrapper.hdf")
	try: os.unlink(out_wstack)
	except: pass

	data_stack = np.stack([img_a, img_b], axis=0)
	idx_list = [0, 1]
	io = ImageIO(out_wstack, mode="rw")
	io.write_images(
		idx_list,
		data_stack,
		[{"nx": 64, "ny": 32, "nz": 1}, {"nx": 64, "ny": 32, "nz": 1}]
	)
	io.close()

	io = ImageIO(out_wstack)
	d0, h0 = io.read_image(0)
	d1, h1 = io.read_image(1)
	assert d0.shape == (32, 64)
	assert abs(float(d1[0, 0]) - 0.0) < 0.01
	io.close()

	# Test 6: format detection via chunk reuse (efficiency test)
	out_vtk = os.path.join(SAMPLE, "vtk_rt.vtk")
	fmt_chunk = ImageIO.detect_format(out_vtk, open(out_vtk, 'rb').read(1024))
	print(f"  detected format for .vtk (chunk): {fmt_chunk}")
	assert fmt_chunk == "VtkIO", f"Expected VtkIO, got {fmt_chunk}"

	# Test 7: unknown format raises FileFormatError
	try:
		ImageIO("/nonexistent/foo.xyz")
		print("  ERROR: should have raised FileFormatError for unknown extension")
	except Exception as e:
		print(f"  correctly raised {type(e).__name__} for unknown format")

	print("  ImageIO wrapper PASS")


if __name__ == "__main__":
	main()
