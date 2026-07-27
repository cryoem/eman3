#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ HdfIO2)
# Copyright (c) 2000-2006 Baylor College of Medicine
#
# This software is issued under a joint BSD/GPL license. You may use the
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
# Foundation, Inc., 59 Temple Place, Suite 330, Boston MA 02111-1307 USA
#

"""HDF5 image format reader/writer for EMAN.

HDF5 stores images in a hierarchical structure:
  /MDF/images/<N>/image	  — dataset with pixel data
  /MDF/images/<N>/		  — group with EMAN-prefixed metadata attributes
  /MDF/images/			  — parent group with imageid_max attribute + meta attrs

Supports float32, uint16, and uint8 storage. When render_bits > 0, data is
resampled to the specified bit-depth before storage. Compression (zlib) is
available when render_compress_level > 0. Data dimensionality is preserved:
nz==1 produces 2-D datasets (ny,nx), not 3-D (1,ny,nx).

References: HDF5 format specification, EMAN2 HdfIO2 implementation.
"""

import os
import numpy as np
import h5py


class FileFormatError(Exception):
	"""Exception for problems with file formats."""
	pass


class FileIOError(Exception):
	"""Exception for problems with file IO."""
	pass


class InvalidDimensions(Exception):
	"""Indicates a mismatch between data size and header indicated size."""
	pass


class InvalidIndex(Exception):
	"""Indicates a request for an unsupported or missing image index."""
	pass


# ---------------------------------------------------------------------------
# HDF5 constants — match C++ HdfIO2
# ---------------------------------------------------------------------------

_HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'


def _get_render_limits(d):
	"""Read rendermin/rendermax/renderbits from a header dict.

	Replicates EMUtil::getRenderLimits(). Returns (rendermin, rendermax, renderbits).
	When render_max <= render_min, automatic mode is triggered in
	_get_render_min_max().
	"""
	rendermin = 0.0
	rendermax = 0.0
	renderbits = 0

	if "render_bits" in d:
		renderbits = int(d["render_bits"])
	if "render_min" in d:
		rendermin = float(d["render_min"])
	if "render_max" in d:
		rendermax = float(d["render_max"])

	return rendermin, rendermax, renderbits


def _get_render_min_max(data, nx, ny, nz, rendermin, rendermax, renderbits):
	"""Compute data-driven render limits if auto mode is active.

	Replicates EMUtil::getRenderMinMax(). Only modifies limits when
	renderbits > 0 and (rendermax <= rendermin or values are out-of-range).

	Returns (rendermin, rendermax, renderbits).
	"""
	if renderbits <= 0:
		return rendermin, rendermax, renderbits
	if renderbits > 16:
		renderbits = 16

	flag_base = 1.0e30
	use_num_std_devs = flag_base / 100.0

	# Auto mode triggered when limits are invalid or too large
	haslim = True
	if (rendermax <= rendermin or
			np.isnan(rendermin) or np.isnan(rendermax) or
			abs(rendermin) > use_num_std_devs or
			abs(rendermax) > use_num_std_devs):
		haslim = False

	data_flat = data.ravel()
	size = len(data_flat)
	bitval = 1 << renderbits

	min_val = float(np.min(data_flat))
	max_val = float(np.max(data_flat))

	if haslim:
		clipped = data_flat.copy()
		np.clip(clipped, rendermin, rendermax, out=clipped)
		m = float(np.mean(clipped))
		s = float(np.std(clipped)) + 1e-30
	else:
		m = float(np.mean(data_flat))
		s = float(np.std(data_flat)) + 1e-30

	n0 = np.sum(data_flat == 0.0)
	nint = np.sum(data_flat == np.floor(data_flat))

	if s <= 0 or np.isnan(s):
		s = 1.0

	# Binary / normalised to [0,1] range
	if abs(max_val - min_val) <= 1.0 + 1e-6:
		rendermin = min_val
		rendermax = max_val
	# All integer values
	elif nint == size:
		if (max_val - min_val) < bitval:
			rendermin = min_val
			rendermax = min_val + bitval - 1
		else:
			import math
			neededbits = math.ceil(math.log2(max_val - min_val + 1))
			step = 1 << (neededbits - renderbits)
			if min_val < 0 < max_val:
				rendermin = round(min_val / step) * step
			else:
				rendermin = min_val
			rendermax = rendermin + step * (bitval - 1)
	else:
		# General case — try to preserve zero if data spans negative and positive
		if min_val < 0 < max_val:
			if min_val == 0:
				rendermin = 0.0
				rendermax = max_val
			elif max_val == 0:
				rendermax = 0.0
				rendermin = min_val
			else:
				step = (max_val - min_val) / (bitval - 1)
				rounded = (min_val / step)
				if min_val == rounded * step:
					rendermin = min_val
					rendermax = max_val
				else:
					step = (max_val - min_val) / (bitval - 2)
					rendermin = float(np.floor(min_val / step) * step)
					rendermax = rendermin + step * (bitval - 1)

	return rendermin, rendermax, renderbits


def _render_data_to_type(data, dtype_np, rendermin, rendermax, renderbits):
	"""Scale float32 data to the target integer type.

	Replicates Renderer::getRenderedDataAndRendertrunc().

	Returns (rendered_array, trunc_count).	For float32 output returns
	a copy with trunc_count=0.
	"""
	if dtype_np == np.float32:
		return data.copy(), 0

	rendered = np.empty(data.shape, dtype=dtype_np)
	is_unsigned = np.issubdtype(dtype_np, np.unsignedinteger)

	if is_unsigned:
		rmin = 0.0
		rmax = float((1 << renderbits) - 1)
	else:
		rmin = float(-(1 << (renderbits - 1)))
		rmax = float((1 << (renderbits - 1)) - 1)

	scale = (rmax - rmin) / (rendermax - rendermin) if (rendermax - rendermin) != 0 else 1.0

	data_flat = data.ravel()
	out_flat = rendered.ravel()
	count = 0
	for i in range(len(data_flat)):
		val = data_flat[i]
		if val < rendermin:
			out_flat[i] = dtype_np(rmin)
			count += 1
		elif val > rendermax:
			out_flat[i] = dtype_np(rmax)
			count += 1
		else:
			out_flat[i] = dtype_np(np.round((val - rendermin) * scale + rmin))

	return rendered, count


def _em_to_hdf5_dtype(renderbits):
	"""Map renderbits to an HDF5 storage dtype.

	renderbits <= 0	 → float32 (lossless compression possible)
	renderbits <= 8	 → uint8
	renderbits <= 16 → uint16
	"""
	if renderbits <= 0:
		return np.float32
	elif renderbits <= 8:
		return np.uint8
	else:
		return np.uint16


def _hdf5_dtype_to_em(dtype):
	"""Map HDF5 dtype to EM data type string for header dict."""
	if dtype == np.float32:
		return "float"
	elif dtype in (np.uint16, np.int16):
		return "ushort"
	elif dtype in (np.uint8, np.int8):
		return "uchar"
	else:
		return "unknown"


def _read_scalar_attr(attrs_dict, key):
	"""Read an HDF5 attribute value that is always a scalar.

	Works around h5py quirks where attrs[key][()] may return 0-D or 1-D
	numpy arrays instead of Python scalars. Uses numpy to guarantee a
	clean conversion.
	"""
	val = attrs_dict[key][()]
	import numpy as np
	return int(np.asarray(val).item())


def _read_hdf5_attr(attr):
	"""Read an HDF5 attribute value and return a Python-native type.

	Matches C++ HdfIO2::read_attr() behaviour.
	Accepts either an h5py Attribute object or the value directly from
	attrs[key] (which is how h5py works in practice).
	"""
	# Normalize: always get raw value as numpy scalar/array or Python native
	dtype = None
	if hasattr(attr, 'dtype'):
		dtype = attr.dtype

	# Try to extract the value — handle various h5py return types
	if isinstance(attr, (bytes, str)):
		return _decode_string(attr)
	elif hasattr(attr, 'item'):
		# numpy scalar or similar — try extracting cleanly
		try:
			val = attr.item()
		except ValueError:
			# multi-element array stored as attribute
			val = np.asarray(attr).tolist()
		dtype = dtype if dtype is not None else getattr(attr, 'dtype', None)
	elif isinstance(attr, np.ndarray):
		val = attr
		dtype = dtype if dtype is not None else attr.dtype
	else:
		# Fallback: try [()] or just use directly
		try:
			val = attr[()]
		except (TypeError, IndexError):
			val = attr
		dtype = dtype if dtype is not None else getattr(attr, 'dtype', None)

	if isinstance(val, (bytes, str)):
		return _decode_string(val)
	elif isinstance(val, bool):
		return val
	elif isinstance(val, (int, np.integer)):
		return int(val)
	elif isinstance(val, (float, np.floating)):
		return float(val)
	elif isinstance(val, np.ndarray):
		return val.tolist()

	# If we somehow got an unrecognised type, convert to string
	try:
		return val.tolist()
	except Exception:
		return str(val) if val is not None else None


def _decode_string(val):
	"""Decode a string value from HDF5."""
	if isinstance(val, bytes):
		return val.decode('utf-8', errors='replace')
	return str(val)


# ---------------------------------------------------------------------------
# HdfIO2 class
# ---------------------------------------------------------------------------

class HdfIO2:
	"""HDF5 image I/O for EMAN3.

	Parameters
	----------
	filename : str
		Path to the HDF5 file (.hdf / .h5).
	mode : str
		"r" (read-only) or "rw" (read/write, creates if absent).
	"""

	def __init__(self, filename, mode="r"):
		self._filename = os.path.abspath(filename)
		self._mode = mode
		self._file = None
		self._initialized = False
		self._meta_attr_dict = {}  # group-level metadata on /MDF/images
		self._nimg = 1

	# ----- property helpers -----------------------------------------------------

	@property
	def filename(self):
		return self._filename

	@property
	def nimg(self):
		self._init()
		return self._nimg

	def __len__(self):
		return self.nimg

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	# ----- lifecycle -----------------------------------------------------------

	def _init(self):
		if self._initialized:
			return

		if self._mode == "r":
			if not os.path.exists(self._filename):
				raise FileIOError(f"HDF5 file not found: {self._filename}")
			self._file = h5py.File(self._filename, "r")
		else:
			if os.path.exists(self._filename):
				self._file = h5py.File(self._filename, "a")
			else:
				self._file = h5py.File(self._filename, "w")

		# Ensure /MDF/images group exists
		if "/MDF" not in self._file:
			if self._mode == "r":
				raise FileFormatError("HDF5 file has no /MDF group (not an EMAN image file)")
			self._file.create_group("/MDF")

		mdff = self._file["/MDF"]
		if "images" not in mdff:
			if self._mode == "r":
				raise FileFormatError("HDF5 file has no /MDF/images group")
			images_grp = mdff.create_group("images")
			images_grp.attrs["imageid_max"] = -1
		else:
			images_grp = mdff["images"]

		# Read group-level meta attributes (prefixed "Group.")
		for key in images_grp.attrs.keys():
			if key == "imageid_max":
				continue
			val = _read_hdf5_attr(images_grp.attrs[key])
			self._meta_attr_dict["Group." + key] = val

		# Read nimg from imageid_max
		if "imageid_max" in images_grp.attrs:
			self._nimg = _read_scalar_attr(images_grp.attrs, "imageid_max") + 1
		else:
			self._nimg = 0

		self._initialized = True

	def close(self):
		if self._file and self._file.id.valid:
			self._file.flush()
			self._file.close()

	def __del__(self):
		try:
			self.close()
		except Exception:
			pass

	# ----- public API ----------------------------------------------------------

	@staticmethod
	def is_valid(filepath, chunk=None):
		"""Check if a file is a valid HDF5 EMAN image.

		If chunk is provided, checks the HDF5 magic bytes in the chunk
		instead of reopening the file.
		"""
		try:
			if chunk is not None:
				return chunk[:8] == _HDF5_MAGIC
			with open(filepath, "rb") as f:
				magic = f.read(8)
			return magic == _HDF5_MAGIC
		except Exception:
			return False

	@staticmethod
	def is_image_big_endian():
		"""HDF5 data is always stored in big-endian by convention."""
		return True

	@staticmethod
	def is_complex_mode():
		return False

	def read_header(self, index=0):
		"""Read the header dict for image at *index*.

		Returns a dict with nx, ny, nz, datatype, apix_x/y/z, origin_x/y/z and
		any other EMAN-prefixed attributes stored on the image group.
		"""
		self._init()

		d = {}
		# Copy group-level meta attributes
		for k, v in self._meta_attr_dict.items():
			d[k] = v

		images_grp = self._file["/MDF/images"]

		# Open the per-image group
		grp_name = str(index)
		if grp_name not in images_grp:
			raise InvalidIndex(f"Image {index} does not exist in HDF5 file")

		igrp = images_grp[grp_name]

		# Read all EMAN.* attributes on the image group, stripping prefix
		for key in igrp.attrs.keys():
			val = _read_hdf5_attr(igrp.attrs[key])
			if key.startswith("EMAN."):
				d[key[5:]] = val  # strip "EMAN."
			else:
				d[key] = val

		# Handle legacy IMOD pixel spacing key
		if "Group.IMOD.PixelSpacing" in d:
			apix = float(d["Group.IMOD.PixelSpacing"])
			del d["Group.IMOD.PixelSpacing"]
			d["apix_x"] = apix
			d["apix_y"] = apix
			d["apix_z"] = apix

		# Auto-detect CTF strings starting with E followed by digit
		if "ctf" in d and isinstance(d["ctf"], str):
			ctf_str = d["ctf"].strip()
			if len(ctf_str) > 1 and ctf_str[0] == 'E' and (ctf_str[1].isdigit() or ctf_str[1] in ('-', '+', '.')):
				try:
					from EMAN3.ctf import EMAN2Ctf
					ctf_obj = EMAN2Ctf()
					ctf_obj.from_string(ctf_str)
					d["ctf"] = ctf_obj
				except Exception:
					pass  # leave as string if parse fails

		# Auto-detect serialized Transform objects in any key
		# Transform matrices are stored as strings of 12 floats in tuple/list format
		for key in list(d.keys()):
			val = d[key]
			if isinstance(val, str):
				try:
					stripped = val.strip().strip('()[]')
					parts = [x.strip() for x in stripped.split(',')]
					if len(parts) == 12:
						vals = [float(x) for x in parts]
						from EMAN3.transform import Transform
						d[key] = Transform()
						d[key].set_matrix(vals)
				except (ValueError, TypeError):
					pass  # Not a serialized Transform, leave as string

		if "TiltAngle" in d and "tilt_angle" not in d:
			d["tilt_angle"] = float(d["TiltAngle"])

		if "PriorRecordDose" in d and "tilt_dose_begin" not in d:
			d["tilt_dose_begin"] = float(d["PriorRecordDose"])

		# Get dimensions and dtype from the dataset
		ds_name = "image"
		if ds_name not in igrp:
			raise FileFormatError(f"Image {index} has no 'image' dataset")

		ds = igrp[ds_name]
		rank = ds.ndim

		if rank == 1:
			nx, ny, nz = ds.shape[0], 1, 1
		elif rank == 2:
			nx, ny, nz = ds.shape[1], ds.shape[0], 1
		else:
			nx, ny, nz = ds.shape[2], ds.shape[1], ds.shape[0]

		if "nx" not in d:
			d["nx"] = nx
		if "ny" not in d:
			d["ny"] = ny
		if "nz" not in d:
			d["nz"] = nz

		# Datatype from dataset dtype size
		itemsize = ds.dtype.itemsize
		if itemsize == 4:
			d["datatype"] = "float"
		elif itemsize == 2:
			d["datatype"] = "ushort"
		elif itemsize == 1:
			d["datatype"] = "uchar"
		else:
			raise FileFormatError(f"HDF5 data type size {itemsize} not supported")

		return d

	def read_data(self, index=0):
		"""Read image data and return as float32 numpy array.

		If the data was stored with bit-reduction (render_bits > 0), it is
		automatically rescaled back to the original float range using the
		stored rendermin/rendermax metadata.

		Dimensionality is preserved: nz==1 returns a 2-D array (ny, nx).
		"""
		self._init()

		images_grp = self._file["/MDF/images"]
		grp_name = str(index)
		if grp_name not in images_grp:
			raise InvalidIndex(f"Image {index} does not exist")

		igrp = images_grp[grp_name]
		ds = igrp["image"]

		# Read raw data as float32 for rescaling (h5py auto-converts)
		data = ds[()].astype(np.float32)
		rank = ds.ndim

		# Rescale if bit-reduction was used during write
		scaled = False
		rendermin = 0.0
		rendermax = 0.0
		renderbits = 0

		if "EMAN.stored_renderbits" in igrp.attrs:
			rb = int(_read_hdf5_attr(igrp.attrs["EMAN.stored_renderbits"]))
			if rb > 0:
				scaled = True
				renderbits = rb
				if "EMAN.stored_rendermax" in igrp.attrs:
					rendermax = float(_read_hdf5_attr(igrp.attrs["EMAN.stored_rendermax"]))
				if "EMAN.stored_rendermin" in igrp.attrs:
					rendermin = float(_read_hdf5_attr(igrp.attrs["EMAN.stored_rendermin"]))

		if scaled and (rendermax - rendermin) != 0:
			rumax = float((1 << renderbits) - 1)
			data = (data / rumax) * (rendermax - rendermin) + rendermin

		# Ensure correct dimensionality
		if rank == 2:
			# Already (ny, nx) — keep as 2D
			pass
		elif rank == 3 and data.shape[0] == 1:
			# Squeeze the singleton z-dimension
			data = np.squeeze(data, axis=0)

		return data

	def write_header(self, d, index=-1):
		"""Write a header dict for an image.

		Parameters
		----------
		d : dict
			Must contain at minimum nx, ny, nz.	 Optional keys include
			apix_x/y/z, origin_x/y/z, render_bits, render_min, render_max,
			render_compress_level, and any other metadata.
		index : int
			Image index. -1 to append.

		Returns the assigned image index.
		"""
		self._init()

		nx = int(d["nx"])
		ny = int(d["ny"])
		nz = int(d["nz"])

		images_grp = self._file["/MDF/images"]

		# Determine target index
		if index < 0:
			current_max = _read_scalar_attr(images_grp.attrs, "imageid_max")
			index = current_max + 1

		# Update imageid_max if appending
		current_max = _read_scalar_attr(images_grp.attrs, "imageid_max")
		if index > current_max:
			images_grp.attrs["imageid_max"] = index
			self._nimg = index + 1

		# Open or create the per-image group
		grp_name = str(index)
		is_new = grp_name not in images_grp

		if is_new:
			igrp = images_grp.create_group(grp_name)
		else:
			igrp = images_grp[grp_name]
			# If existing, erase old attributes (but keep the group/dataset for region writes)
			existing_attrs = list(igrp.attrs.keys())
			for key in existing_attrs:
				del igrp.attrs[key]

			# Check if we need to unlink existing dataset (size/type/compression mismatch)
			ds_name = "image"
			if ds_name in igrp:
				existing_ds = igrp[ds_name]
				existing_shape = existing_ds.shape
				old_vol = nx * ny * nz
				new_vol = 1
				if len(existing_shape) == 1:
					new_vol = existing_shape[0]
				elif len(existing_shape) == 2:
					new_vol = existing_shape[0] * existing_shape[1]
				else:
					new_vol = existing_shape[0] * existing_shape[1] * existing_shape[2]

				render_compress_level = d.get("render_compress_level", None)
				need_unlink = (old_vol != new_vol or
							  render_compress_level is not None or
							  d.get("datatype") == "compressed")

				if need_unlink:
					del igrp[ds_name]

		# Write attributes (prefixed with "EMAN."), skipping special keys
		skip_keys = {"stored_rendermin", "stored_rendermax", "stored_renderbits",
					 "render_min", "render_max", "render_bits"}

		for key, val in d.items():
			if key in skip_keys:
				continue
			attr_name = "EMAN." + key
			_write_attribute(igrp, attr_name, val)

		# Store render limits for later use in write_data
		self._rendermin, self._rendermax, self._renderbits = _get_render_limits(d)
		if "render_compress_level" in d:
			self._renderlevel = float(d["render_compress_level"])
		else:
			self._renderlevel = 1

		return index

	def write_data(self, data, index=0):
		"""Write image data.

		Parameters
		----------
		data : numpy array
			Float32 data. Shape (ny, nx) for 2-D or (nz, ny, nx) for 3-D.
		index : int
			Image index. -1 to append to the last written header.
		"""
		self._init()

		if data.dtype != np.float32:
			data = data.astype(np.float32)

		images_grp = self._file["/MDF/images"]

		if index < 0:
			current_max = _read_scalar_attr(images_grp.attrs, "imageid_max")
			index = current_max + 1

		grp_name = str(index)
		if grp_name not in images_grp:
			raise InvalidIndex(f"Image group {index} does not exist — call write_header first")

		igrp = images_grp[grp_name]

		# Determine dimensions from data shape
		if data.ndim == 2:
			ny, nx = data.shape
			nz = 1
			dims = (ny, nx)
			rank = 2
		elif data.ndim == 3:
			nz, ny, nx = data.shape
			dims = (nz, ny, nx)
			rank = 3
		else:
			raise InvalidDimensions(f"Data must be 2-D or 3-D, got {data.ndim}-D")

		size = int(np.prod(data.shape))

		# Compute render limits from data if auto-mode
		rendermin = getattr(self, '_rendermin', 0.0)
		rendermax = getattr(self, '_rendermax', 0.0)
		renderbits = getattr(self, '_renderbits', 0)
		renderlevel = getattr(self, '_renderlevel', 1)

		rendermin, rendermax, renderbits = _get_render_min_max(
			data, nx, ny, nz, rendermin, rendermax, renderbits
		)

		# Determine storage dtype
		store_dtype = _em_to_hdf5_dtype(renderbits)

		# Setup compression properties if needed
		if renderbits > 0:
			compressed = True
			comp_level = max(1, min(9, int(renderlevel)))
			if rank == 2:
				chunks = (8, nx)
			else:
				chunks = (1, 8, nx)
		else:
			compressed = False
			comp_level = None
			chunks = None

		# Render data to target dtype
		rendered, trunc_count = _render_data_to_type(data, store_dtype, rendermin, rendermax, renderbits)

		ds_name = "image"
		if ds_name in igrp:
			del igrp[ds_name]

		kw = {}
		if compressed:
			kw["compression"] = "gzip"
			kw["compression_opts"] = comp_level
			kw["chunks"] = chunks

		igrp.create_dataset(ds_name, data=rendered, dtype=store_dtype, **kw)

		# Write scaling metadata if data was rescaled (i.e., not float32)
		if store_dtype != np.float32:
			_write_attribute(igrp, "EMAN.stored_rendermin", rendermin)
			_write_attribute(igrp, "EMAN.stored_rendermax", rendermax)
			_write_attribute(igrp, "EMAN.stored_renderbits", renderbits)
			_write_attribute(igrp, "EMAN.stored_truncated", trunc_count)

	def flush(self):
		if self._file and self._file.id.valid:
			self._file.flush()


def _write_attribute(loc, name, val):
	"""Write a Python value as an HDF5 attribute.

	Matches C++ HdfIO2::write_attr() type mapping:
	  bool      -> single-byte 'T'/'F'
	  int       -> int32
	  float     -> float32
	  str       -> variable-length string
	  list      -> array of float32 or int32 (auto-detected)
	"""
	if isinstance(val, bool):
		loc.attrs[name] = np.int8(ord('T') if val else ord('F'))
	elif isinstance(val, (int, np.integer)):
		loc.attrs[name] = np.int32(val)
	elif isinstance(val, (float, np.floating)):
		loc.attrs[name] = np.float32(val)
	elif isinstance(val, str):
		loc.attrs[name] = val
	else:
		# Try to serialize Transform objects (have get_matrix_string method)
		if hasattr(val, 'get_matrix_string'):
			loc.attrs[name] = val.get_matrix_string()
		# Try to serialize CTF-like objects (have to_string/from_string methods)
		elif hasattr(val, 'to_string') and hasattr(val, 'from_string'):
			loc.attrs[name] = val.to_string()
		elif isinstance(val, (list, tuple)):
			arr = np.array(val)
			if arr.dtype.kind == 'f':
				loc.attrs[name] = arr.astype(np.float32)
			else:
				loc.attrs[name] = arr.astype(np.int32)
		elif isinstance(val, np.ndarray):
			if val.dtype.kind == 'f':
				loc.attrs[name] = val.astype(np.float32)
			else:
				loc.attrs[name] = val.astype(np.int32)
		else:
			# Fallback - convert to string
			loc.attrs[name] = str(val)
