#!/usr/bin/env python
#
# Author: Steven Ludtke (unified IO wrapper)
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

"""Unified ImageIO wrapper for all EMAN image formats.

Provides a single class that auto-detects file type and delegates to the
appropriate format-specific IO class. Reads the first ~1K of the file once
and reuses it for type detection across all registered formats.

Usage::

	from EMAN3.io.imageio import ImageIO

	# Read — auto-detects format from file contents
	io = ImageIO("mydata.mrcs")
	images, headers = io.read_images([0, 1, 5])		  # list of indices
	img, hdr = io.read_image(2)						   # single image

	# Write — infers format from extension
	io = ImageIO("output.hdf", mode="rw")
	io.write_image(0, data_array, {"nx": 256, "ny": 256, "nz": 1})
"""

import os
import numpy as np


class FileFormatError(Exception):
	"""Raised when the file format cannot be determined or is unsupported."""
	pass


class FileIOError(Exception):
	"""Raised on IO errors during read/write operations."""
	pass


class InvalidDimensions(Exception):
	"""Raised when data dimensions don't match header expectations."""
	pass


class InvalidIndex(Exception):
	"""Raised when an image index is out of range."""
	pass


# ---------------------------------------------------------------------------
# Format registry: extension -> (IO class, constructor kwargs)
# ---------------------------------------------------------------------------
# Lazy imports to avoid pulling in heavy dependencies until needed.

def _get_io_class_map():
	"""Return a dict mapping lowercase extensions to IO classes.

	The actual import is deferred so that the module loads quickly even if
	some format libraries are not installed.
	"""
	return {
		# 2D single-image formats
		".jpg":	  ("jpegio", "JpegIO"),
		".jpeg":  ("jpegio", "JpegIO"),
		".pgm":	  ("pgmio", "PgmIO"),
		".png":	  ("pngio", "PngIO"),

		# Stack formats
		".tif":	   ("tiffio", "TiffIO"),
		".tiff":   ("tiffio", "TiffIO"),
		".mrcs":   ("mrcio", "MrcIO"),
		".mrsc":   ("mrcio", "MrcIO"),
		".raw":	   ("mrcio", "MrcIO"),
		".hed":	   ("imagicio2", "ImagicIO"),
		".img":	   ("imagicio2", "ImagicIO"),
		".hdf":	   ("hdfio2", "HdfIO2"),
		".h5":	   ("hdfio2", "HdfIO2"),

		# 3D single-image formats
		".ico":	   ("icosio", "IcosIO"),
		".icos":   ("icosio", "IcosIO"),
		".mrc":	   ("mrcio", "MrcIO"),
		".hdr":	   ("mrcio", "MrcIO"),
		".omap":   ("omapio", "OmapIO"),
		".dns6":   ("omapio", "OmapIO"),
		".brix":   ("omapio", "OmapIO"),
		".am":	   ("amiraio", "AmiraIO"),
		".amira":  ("amiraio", "AmiraIO"),
		".vtk":	   ("vtkio", "VtkIO"),

		# Special formats (no common extension or detected by magic only)
		".pif":	   ("pifio", "PifIO"),
		".eer":	   ("eerio", "EerIO"),
		".spi":	   ("spiderio", "SpiderIO"),
	}


def _import_io(module_name, class_name):
	"""Import an IO class by module and class name (lazy import)."""
	import importlib
	mod = importlib.import_module(f"EMAN3.io.{module_name}")
	return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# All known format classes for magic-byte probing (order matters — more
# specific detectors first to avoid false positives).
# ---------------------------------------------------------------------------

def _get_all_io_classes():
	"""Return a list of (module, classname) tuples for every supported format.

	This is the order in which formats are probed when extension-based
	detection fails.  More distinctive magic bytes come first.
	"""
	return [
		("hdfio2", "HdfIO2"),		# HDF5 magic: \x89HDF\r\n\x1a\n (very distinctive)
		("jpegio", "JpegIO"),		# JPEG SOI: ff d8 ff
		("pngio", "PngIO"),			# PNG signature: 89 50 4e 47 ...
		("pgmio", "PgmIO"),			# PGM: P5 or P2-P6 ASCII
		("vtkio", "VtkIO"),			# VTK: "# vtk DataFile Version"
		("amiraio", "AmiraIO"),		# Amira: "# AmiraMesh"
		("mrcio", "MrcIO"),			 # MRC: machine stamp + mode validation
			("tiffio", "TiffIO"),		 # TIFF: II/ MM magic at offset 0,42
		("pifio", "PifIO"),			# PIF: dual magic int32
		("icosio", "IcosIO"),		# ICOS: stamp fields in header
		("spiderio", "SpiderIO"),	# Spider: type float at offset 16
		("omapio", "OmapIO"),		# OMAP: scale2==100 check
		("imagicio2", "ImagicIO"),	# IMAGIC: realtype machine stamp
		("eerio", "EerIO"),			# EER: TIFF-based (last, expensive)
	]


# ---------------------------------------------------------------------------
# ImageIO — unified wrapper
# ---------------------------------------------------------------------------

class ImageIO:
	"""Unified image I/O that auto-detects file format.

	Parameters
	----------
	filename : str
		Path to the image file.	 The extension is used as a first hint for
		format detection; if the file doesn't exist the extension alone
		determines the format class.
	mode : str, optional
		"r" (read-only) or "rw" (read-write).  Defaults to "r".
	**kwargs : extra arguments forwarded to the underlying IO constructor.

	Examples
	--------
	Read a single image::

		io = ImageIO("data.mrcs")
		img, hdr = io.read_image(0)			 # numpy array + dict

	Read multiple images::

		data_stack, headers = io.read_images([0, 1, 2])

	Write a new file (extension drives format selection)::

		io = ImageIO("output.hdf", mode="rw")
		io.write_image(0, my_array, {"nx": 64, "ny": 64, "nz": 1})
	"""

	def __init__(self, filename, mode="r", **kwargs):
		self._filename = os.path.abspath(filename)
		self._mode = mode
		self._io = None			   # underlying IO instance
		self._raw_header = None	   # first 1K of the file (bytes)

		_, ext = os.path.splitext(self._filename)
		self._ext = ext.lower()

		if not os.path.exists(self._filename):
			# File doesn't exist — infer format from extension only
			io_cls = self._class_for_extension(ext)
			if io_cls is None:
				raise FileFormatError(
					f"Cannot determine image format for extension '{ext}'")
			init_kwargs = self._filter_kwargs(io_cls, {"mode": mode, **kwargs})
			self._io = io_cls(self._filename, **init_kwargs)
			return

		# File exists — read first 1K once for magic-byte probing
		with open(self._filename, "rb") as f:
			self._raw_header = f.read(1024)

		# Try extension-based match first
		io_cls = self._class_for_extension(ext)
		if io_cls is not None and io_cls.is_valid(self._filename, self._raw_header):
			init_kwargs = self._filter_kwargs(io_cls, {"mode": mode, **kwargs})
			self._io = io_cls(self._filename, **init_kwargs)
			return

		# Fall back to probing all formats with the cached header chunk
		for mod_name, cls_name in _get_all_io_classes():
			try:
				candidate = _import_io(mod_name, cls_name)
			except (ImportError, AttributeError):
				continue
			try:
				if candidate.is_valid(self._filename, self._raw_header):
					init_kwargs = self._filter_kwargs(candidate, {"mode": mode, **kwargs})
					self._io = candidate(self._filename, **init_kwargs)
					return
			except TypeError:
				# is_valid may not accept chunk arg in older implementations
				try:
					if candidate.is_valid(self._filename):
						init_kwargs = self._filter_kwargs(candidate, {"mode": mode, **kwargs})
						self._io = candidate(self._filename, **init_kwargs)
						return
				except Exception:
					pass

		raise FileFormatError(
			f"Cannot determine image format for '{self._filename}'")

	# ----- helpers --------------------------------------------------------

	@staticmethod
	def _class_for_extension(ext):
		"""Look up the IO class for a file extension (lazy import)."""
		cmap = _get_io_class_map()
		entry = cmap.get(ext)
		if entry is None:
			return None
		mod_name, cls_name = entry
		try:
			return _import_io(mod_name, cls_name)
		except (ImportError, AttributeError):
			return None

	@staticmethod
	def _filter_kwargs(io_cls, kwargs):
		"""Remove unsupported constructor kwargs for the target IO class.

		EerIO, for example, has no ``mode`` argument but uses
		``frame_averaging`` and ``oversample``.	 This method inspects
		the constructor signature and drops anything it can't consume.
		"""
		import inspect
		sig = inspect.signature(io_cls.__init__)
		filtered = {}
		for k, v in kwargs.items():
			if k in sig.parameters:
				filtered[k] = v
		return filtered

	@property
	def filename(self):
		return self._filename

	@property
	def mode(self):
		return self._mode

	@property
	def nimg(self):
		"""Number of images in the file."""
		if self._io is not None:
			# Trigger lazy init on classes that support it (e.g. HdfIO2)
			return getattr(self._io, 'nimg', len(self._io))
		return 0

	def __len__(self):
		return self.nimg

	def close(self):
		"""Close the underlying file handle."""
		if self._io is not None:
			# Try explicit close() method first
			try:
				self._io.close()
			except AttributeError:
				# Fall back to manually closing _file if available
				if hasattr(self._io, '_file') and self._io._file is not None:
					try:
						self._io._file.close()
					except Exception:
						pass
			self._io = None

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	# ----- read API -------------------------------------------------------

	def read_image(self, index):
		"""Read a single image and return (data_array, header_dict).

		Parameters
		----------
		index : int
			Image index within the file.

		Returns
		-------
		data : numpy.ndarray of float32
			Shape ``(ny, nx)`` for 2D images or ``(nz, ny, nx)`` for 3D.
			Dimensionality is preserved — ``nz==1`` does NOT produce a
			singleton z-axis.
		header : dict
			Metadata dictionary from the image header (nx, ny, nz,
			apix_x/y/z, origin_x/y/z, datatype, etc.).
		"""
		if self._io is None:
			raise FileIOError("IO class not initialized")

		header = self._io.read_header(index)
		data = self._io.read_data(index)

		if data.dtype != np.float32:
			data = data.astype(np.float32)

		return data, header

	def read_images(self, indices=None):
		"""Read multiple images in the given order.

		If called with no arguments, reads all images in the file.

		Parameters
		----------
		indices : list of int, optional
			Image indices to read. Order is preserved. If None, reads
			all images (0 through nimg-1).

		Returns
		-------
		stack : numpy.ndarray of float32
			Shape ``(N, ny, nx)`` for 2D stacks or ``(N, nz, ny, nx)``
			for 3D stacks where N = len(indices).
		headers : list of dict
			Per-image metadata dictionaries in the same order as indices.
		"""
		if self._io is None:
			raise FileIOError("IO class not initialized")

		if indices is None:
			indices = list(range(self.nimg))

		results = []
		headers = []
		for idx in indices:
			data, header = self.read_image(idx)
			results.append(data)
			headers.append(header)

		if len(results) == 0:
			return np.array([]), []

		stack = np.stack(results, axis=0)

		return stack, headers

	def read_header(self, index):
		"""Read only the header for a single image."""
		if self._io is None:
			raise FileIOError("IO class not initialized")
		return self._io.read_header(index)

	def read_headers(self, indices=None):
		"""Read headers for multiple images.

		If called with no arguments, reads all image headers (0 through nimg-1).

		Parameters
		----------
		indices : list of int, optional
			Image indices to read. Order is preserved.

		Returns
		-------
		headers : list of dict
			Per-image metadata dictionaries in the same order as indices.
		"""
		if self._io is None:
			raise FileIOError("IO class not initialized")
		if indices is None:
			indices = list(range(self.nimg))
		return [self._io.read_header(idx) for idx in indices]

	# ----- write API ------------------------------------------------------

	def write_image(self, index, data, header=None):
		"""Write a single image.

		Parameters
		----------
		index : int
			Target image index (-1 to append).
		data : numpy.ndarray
			Float32 array of shape ``(ny, nx)`` or ``(nz, ny, nx)``.
		header : dict, optional
			Metadata dictionary.  Must at minimum contain ``nx``, ``ny``,
			``nz``.	 If not provided these are inferred from data shape.
		"""
		if self._io is None:
			raise FileIOError("IO class not initialized")

		if data.dtype != np.float32:
			data = data.astype(np.float32)

		if header is None:
			if data.ndim == 2:
				ny, nx = data.shape
				header = {"nx": nx, "ny": ny, "nz": 1}
			elif data.ndim == 3:
				nz, ny, nx = data.shape
				header = {"nx": nx, "ny": ny, "nz": nz}
			else:
				raise InvalidDimensions(
					f"Data must be 2-D or 3-D, got {data.ndim}-D")

		# write_header may resolve -1 to the actual index; use that
		resolved = self._io.write_header(header, index)
		if resolved is None:
			resolved = index
		self._io.write_data(data, resolved)

	def write_images(self, indices, data, headers=None):
		"""Write multiple images.

		Parameters
		----------
		indices : list of int
			Target image indices in order.
		data : numpy.ndarray
			Float32 stack of shape ``(N, ny, nx)`` or ``(N, nz, ny, nx)``.
		headers : list of dict, optional
			Per-image metadata dictionaries.  If not provided, dimensions
			are inferred from each image's shape.
		"""
		if self._io is None:
			raise FileIOError("IO class not initialized")

		if data.ndim < 3:
			raise InvalidDimensions(
				f"Stack data must be at least 3-D, got {data.ndim}-D")

		for i, idx in enumerate(indices):
			img_data = data[i]
			hdr = headers[i] if headers else None
			self.write_image(idx, img_data, hdr)

	# ----- utility --------------------------------------------------------

	@staticmethod
	def detect_format(filepath, chunk=None):
		"""Detect the format of a file without opening it for IO.

		Parameters
		----------
		filepath : str
			Path to the file.
		chunk : bytes, optional
			First ~1K of the file.	If not provided, the file is read
			once just for detection.

		Returns
		-------
		name : str
			The class name (e.g. "MrcIO", "HdfIO2") or ``None`` if no
			format matched.
		"""
		if chunk is None:
			with open(filepath, "rb") as f:
				chunk = f.read(1024)

		_, ext = os.path.splitext(filepath)
		io_cls = ImageIO._class_for_extension(ext.lower())
		if io_cls is not None:
			try:
				if io_cls.is_valid(filepath, chunk):
					return io_cls.__name__
			except TypeError:
				if io_cls.is_valid(filepath):
					return io_cls.__name__

		for mod_name, cls_name in _get_all_io_classes():
			try:
				candidate = _import_io(mod_name, cls_name)
			except (ImportError, AttributeError):
				continue
			try:
				if candidate.is_valid(filepath, chunk):
					return candidate.__name__
			except TypeError:
				try:
					if candidate.is_valid(filepath):
						return candidate.__name__
				except Exception:
					pass

		return None
