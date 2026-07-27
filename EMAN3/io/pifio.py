#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ PifIO)
# Copyright (c) 2000-2026 Baylor College of Medicine
#
# BSD/GPL license. See top-level for details.

"""PIF (Portable Image Format) reader/writer for EMAN.

File layout:
  PifFileHeader (512 bytes, overall file metadata)
  + for each image:
    PifImageHeader (512 bytes)
    + pixel data (nx*ny*nz * mode_size bytes)

All images share same dimensions from the file header.
Floats stored as scaled integers with file-level scalefactor string.

Supported extensions: .pif, .stk
"""

import os
import sys
import time
import numpy as np


class FileFormatError(Exception):
	pass


class FileIOError(Exception):
	pass


class InvalidDimensions(Exception):
	pass


PIF_MAGIC_NUM = 8

# Data modes from C++ enum PifDataMode
PIF_CHAR = 0
PIF_SHORT = 1
PIF_FLOAT_INT = 2
PIF_SHORT_COMPLEX = 3
PIF_FLOAT_INT_COMPLEX = 4
PIF_BOXED_DATA = 6
PIF_SHORT_FLOAT = 7
PIF_SHORT_FLOAT_COMPLEX = 8
PIF_FLOAT = 9
PIF_FLOAT_COMPLEX = 10
PIF_MAP_FLOAT_SHORT = 20
PIF_MAP_FLOAT_INT = 21
PIF_MAP_FLOAT_INT_2 = 40
PIF_BOXED_FLOAT_INT = 46

# Mode byte sizes
MODE_SIZE = {
	0: 1, 6: 1,       # char types
	1: 2, 3: 2, 7: 2, 8: 2, 20: 2,   # short types
	2: 4, 4: 4, 9: 4, 10: 4, 21: 4, 40: 4, 46: 4,  # int/float types
}

FLOAT_INT_MODES = {7, 8, 2, 4, 20, 21, 40, 46}
COMPLEX_MODES = {3, 8, 10, 4}


def _be_int(val):
	"""Return big-endian int32 bytes."""
	return np.array([int(val)], dtype='<i4').tobytes()


class PifIO:
	"""PIF (Portable Image Format) reader/writer.

	Supports multi-image stacks with per-image headers.
	Homogeneous dimensions only (htype == 1).
	Default write mode: big-endian int32 scaled floats.

	Modes: "r" (read), "rw" (write — appends to existing or creates new).
	"""

	SUPPORT_STACK = True
	SUPPORT_3D = True
	SUPPORT_3D_STACK = False
	SUPPORTED_COMPRESS = False

	def __init__(self, filename, mode="r"):
		self.filename = os.path.expanduser(filename)
		self.mode = mode

		known_exts = (".pif", ".stk")
		_, ext = os.path.splitext(self.filename)
		if ext.lower() not in known_exts:
			raise FileFormatError(f"Unknown extension '{ext}' for PIF format.")

		self.nimg = 0
		self._mode_size = 4
		self._is_big_endian = True
		self._scale_factor = 1.0
		self._is_new_file = False
		self._fh_nx = 0
		self._fh_ny = 0
		self._fh_nz = 0

		try:
			if mode == "r":
				self._file = open(self.filename, "rb")
				self._read_file_header()
			else:
				exists = os.path.exists(self.filename) and os.path.getsize(
					self.filename) > 0
				mode_str = "r+b" if exists else "wb+"
				self._file = open(self.filename, mode_str)
				self._is_new_file = not exists
				if not self._is_new_file:
					self._read_file_header()
		except OSError as e:
			raise FileIOError(f"Cannot open file '{self.filename}': {e}")

	def __del__(self):
		if hasattr(self, '_file') and self._file is not None:
			try:
				self._file.close()
			except Exception:
				pass

	def _read_file_header(self):
		"""Parse the overall PIF file header (512 bytes at start)."""
		self._file.seek(0)
		raw = self._file.read(512)
		if len(raw) < 8:
			raise FileFormatError("Incomplete PIF file header")

		magic = np.frombuffer(raw[:8], dtype='<i4')
		if magic[0] != PIF_MAGIC_NUM or magic[1] != PIF_MAGIC_NUM:
			raise FileFormatError("Invalid PIF magic number")

		nimg  = int(np.frombuffer(raw[24:28], dtype='<i4')[0])
		endian = bool(int(np.frombuffer(raw[28:32], dtype='<i4')[0]))
		htype = int(np.frombuffer(raw[64:68], dtype='<i4')[0])
		nx    = int(np.frombuffer(raw[68:72], dtype='<i4')[0])
		ny    = int(np.frombuffer(raw[72:76], dtype='<i4')[0])
		nz    = int(np.frombuffer(raw[76:80], dtype='<i4')[0])
		mode  = int(np.frombuffer(raw[80:84], dtype='<i4')[0])

		if htype != 1:
			raise FileFormatError(
				"PIF with varying dimensions (htype=0) not supported")

		self._is_big_endian = endian
		self.nimg = nimg
		self._fh_nx = nx
		self._fh_ny = ny
		self._fh_nz = nz
		self._mode_size = MODE_SIZE.get(mode, 4)

		if mode in FLOAT_INT_MODES:
			sf_raw = raw[8:24]
			try:
				self._scale_factor = float(sf_raw.split(b'\x00')[0].decode())
			except (ValueError, UnicodeDecodeError):
				self._scale_factor = 1.0

	def __len__(self):
		"""Return number of images in the file."""
		return self.nimg

	@staticmethod
	def bit_depths():
		"""Return list of supported bit depths: 8, 16, and 32."""
		return [8, 16, 32]

	@staticmethod
	def is_valid(filepath, chunk=None):
		"""Check whether a file has a valid PIF header (magic numbers).
		If chunk is provided, checks the chunk instead of reopening the file.
		"""
		try:
			if chunk is not None:
				raw = chunk[:8]
			else:
				with open(os.path.expanduser(filepath), "rb") as f:
					raw = f.read(8)
			if len(raw) < 8:
				return False
			vals = np.frombuffer(raw, dtype='<i4')
			return int(vals[0]) == PIF_MAGIC_NUM and \
			       int(vals[1]) == PIF_MAGIC_NUM
		except Exception:
			return False

	def _fseek_to(self, image_index):
		"""Seek to start of an image block."""
		data_sz = self._fh_nx * self._fh_ny * self._fh_nz * self._mode_size
		offset = 512 + (512 + data_sz) * image_index
		self._file.seek(offset)

	def read_header(self, index=0):
		if index < 0:
			index = 0

		self._fseek_to(index)
		raw = self._file.read(512)
		if len(raw) < 512:
			raise FileIOError("Incomplete PIF image header")

		scale = self._scale_factor

		def _fi(offset):
			return int(np.frombuffer(raw[offset:offset+4], dtype='<i4')[0])

		nx    = _fi(0)
		ny    = _fi(4)
		nz    = _fi(8)
		mode  = _fi(12)
		xlen  = _fi(48)
		ylen  = _fi(52)
		zlen  = _fi(56)
		minv  = _fi(88)
		maxv  = _fi(92)
		meanv = _fi(96)
		sigv  = _fi(100)
		xorg  = _fi(116)
		yorg  = _fi(120)

		return {
			"nx": nx, "ny": ny, "nz": nz,
			"datatype": mode,
			"apix_x": float(xlen) * scale,
			"apiz_y": float(ylen) * scale,
			"apiz_z": float(zlen) * scale,
			"minimum": float(minv) * scale,
			"maximum": float(maxv) * scale,
			"mean": float(meanv) * scale,
			"sigma": float(sigv) * scale,
			"origin_x": float(xorg) * scale,
			"origin_y": float(yorg) * scale,
		}

	def read_data(self, index=0):
		if index < 0:
			index = 0

		self._fseek_to(index)
		self._file.read(512)  # skip image header

		nx = self._fh_nx
		ny = self._fh_ny
		nz = self._fh_nz
		total = nx * ny * nz

		# We write as BE float32 (mode=PIF_FLOAT), read the same way
		data = np.fromfile(self._file, dtype="<f4", count=total)

		data = data.reshape((nz, ny, nx))
		# Squeeze singleton z dimension: nz==1 means 2D image
		if nz == 1:
			data = data.reshape((ny, nx))

		return data

	def _write_int_to_file(self, val):
		return np.array([int(val)], dtype='<i4').tobytes()

	def write_header(self, d, index=0):
		if index < 0:
			index = self.nimg if self.nimg else 0

		nx = int(d["nx"])
		ny = int(d["ny"])
		nz = int(d["nz"])
		now = time.localtime()

		if not self._is_new_file:
			if nx != self._fh_nx or ny != self._fh_ny or nz != self._fh_nz:
				raise FileIOError("PIF write dimension mismatch")
			self._fseek_to(index)
		else:
			# Write file header as raw big-endian bytes
			fh = bytearray(512)
			o = 0
			fh[o:o+4] = self._write_int_to_file(PIF_MAGIC_NUM); o += 4
			fh[o:o+4] = self._write_int_to_file(PIF_MAGIC_NUM); o += 4
			fh[o:o+16] = b'1.0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'; o += 16
			fh[o:o+4] = self._write_int_to_file(0); o += 4    # nimg=0
			fh[o:o+4] = self._write_int_to_file(0); o += 4    # little-endian flag
			prog = f'EMAN {now.tm_mon:02d}/{now.tm_mday:02d}/{now.tm_year}'
			fh[o:o+32] = prog.ljust(32).encode(); o += 32     # program
			fh[o:o+4] = self._write_int_to_file(1); o += 4    # htype=1
			fh[o:o+4] = self._write_int_to_file(nx); o += 4   # nx
			fh[o:o+4] = self._write_int_to_file(ny); o += 4   # ny
			fh[o:o+4] = self._write_int_to_file(nz); o += 4   # nz
			fh[o:o+4] = self._write_int_to_file(2); o += 4    # mode=PIF_FLOAT_INT

			self._is_big_endian = True
			self._mode_size = 4
			self._fh_nx, self._fh_ny, self._fh_nz = nx, ny, nz
			self.nimg = 0

			self._file.seek(0)
			self._file.write(bytes(fh))
			self._is_new_file = False  # only write file header once

		# Build per-image header (512 bytes BE ints + padding)
		now = time.localtime()
		ts = f'{now.tm_mon:02d}/{now.tm_mday:02d}/{now.tm_year} {now.tm_hour:02d}:{now.tm_min:02d}'

		def _pi(v):
			return self._write_int_to_file(int(v))

		hdr = bytearray(512)
		i = 0
		hdr[i:i+4] = _pi(nx); i += 4
		hdr[i:i+4] = _pi(ny); i += 4
		hdr[i:i+4] = _pi(nz); i += 4
		hdr[i:i+4] = _pi(9); i += 4       # mode=PIF_FLOAT (per-image, always float)

		# bkg, radius, xstart, ystart, zstart -> zeros already
		i += 20  # 5 ints
		# mx, my, mz -> zeros
		i += 12
		hdr[i:i+4] = _pi(d.get("apix_x", 1.0)); i += 4   # xlen
		hdr[i:i+4] = _pi(d.get("apiz_y", 1.0)); i += 4   # ylen
		hdr[i:i+4] = _pi(d.get("apiz_z", 1.0)); i += 4   # zlen

		sc = int(90 * self._scale_factor)
		hdr[i:i+4] = _pi(sc); i += 4    # alpha
		hdr[i:i+4] = _pi(sc); i += 4    # beta
		hdr[i:i+4] = _pi(sc); i += 4    # gamma
		hdr[i:i+4] = _pi(1); i += 4     # mapc
		hdr[i:i+4] = _pi(2); i += 4     # mapr
		hdr[i:i+4] = _pi(3); i += 4     # maps
		hdr[i:i+4] = _pi(d.get("minimum", 0)); i += 4    # min
		hdr[i:i+4] = _pi(d.get("maximum", 1)); i += 4    # max
		hdr[i:i+4] = _pi(d.get("mean", 0)); i += 4       # mean
		hdr[i:i+4] = _pi(d.get("sigma", 0)); i += 4      # sigma
		i += 8           # ispg, nsymbt -> zeros
		hdr[i:i+4] = _pi(d.get("origin_x", 0.0)); i += 4
		hdr[i:i+4] = _pi(d.get("origin_y", 0.0)); i += 4
		hdr[i:i+80] = ts.ljust(80).encode()[:80]; i += 80   # title
		hdr[i:i+32] = ts.ljust(32).encode()[:32]; i += 32   # time
		i += 24  # imagenum, scannum -> zeros
		i += 8   # aoverb, mapabang -> zeros
		i += 252  # pad (63 ints) -> already zeroed

		self._file.write(bytes(hdr))
		self.nimg += 1

		# Update nimg in file header
		self._file.seek(24)
		self._file.write(self._write_int_to_file(self.nimg))

		# Restore file position to after image header for write_data
		self._fseek_to(index)
		self._file.seek(512, 1)  # skip past image header to data

		return index

	def write_data(self, data, index=0):
		data = np.asarray(data, dtype=np.float32)

		nx = self._fh_nx
		ny = self._fh_ny
		nz = self._fh_nz

		# Accept (ny,nx) for 2D or (nz,ny,nx) for 3D
		if data.ndim == 2 and nz == 1:
			data = data.reshape((1, ny, nx))
		if data.shape != (nz, ny, nx):
			raise InvalidDimensions(
				f"Data shape {data.shape} doesn't match ({nz},{ny},{nx})")

		# Write as big-endian float32 directly (mode=PIF_FLOAT in image header)
		data.tofile(self._file)

	def flush(self):
		if self._file is not None:
			self._file.flush()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self._file is not None:
			try:
				self._file.close()
			except Exception:
				pass
		self._file = None
		return False
