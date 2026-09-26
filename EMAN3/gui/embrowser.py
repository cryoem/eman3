#!/usr/bin/env python
#
# Author: Steven Ludtke (sludtke@bcm.edu)
# Copyright (c) 2011- Baylor College of Medicine
#
# e3display file browser (EMAN3).
#
# Ported from EMAN2 qtgui/embrowser.py to the pure-Python EMAN3 GUI. The column
# metadata (Type/Dim/N/Size) is no longer computed in-process (which was slow and
# GIL-bound); it is precomputed by the standalone e3browsercache program into a
# browser cache file (see EMAN3.EMAN3.browser_cache_*). The browser keeps a live
# list of files, polls that cache file's timestamp, and refreshes the metadata
# columns whenever a newer cache appears -- so the UI stays fully responsive.
#
# Action buttons whose target display widgets are not yet ported to EMAN3/gui
# keep their full event wiring but print "unimplemented" until those widgets are
# available.
#
import os
import re
import sys
import time
import random
import shutil
import subprocess
import traceback

import numpy as np

import EMAN3
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer

from EMAN3.EMAN3 import local_datetime, browser_cache_read, BROWSER_CACHE_VERSION
from EMAN3.EMAN3jsondb import js_open_dict
from EMAN3.io.imageio import ImageIO

# Repository root (EMAN3 is a subpackage of the repo), used to launch the
# standalone updater module when the console script is not on PATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(EMAN3.__file__)))


def display_error(msg):
	"""Displays an error message, in gui and on terminal."""
	print(msg)
	sys.stdout.flush()
	QtWidgets.QMessageBox.warning(None, "Error", msg)


# We need to sort ints and floats as themselves, not string
def safe_int(v):
	try:
		return int(v)
	except (ValueError, TypeError):
		return -sys.maxsize - 1


def safe_float(v):
	try:
		return float(v)
	except Exception:
		try:
			return float(v.split()[0])
		except Exception:
			return sys.float_info.min


def size_sortable(s):
	try:
		return int(s)
	except Exception:
		return 0


def nonone(val):
	"""Returns '-' for None, otherwise the string representation of the passed value"""
	try:
		if val != None:
			return str(val)
		return "-"
	except Exception:
		return "X"


def humansize(val):
	"""Representation of an integer in readable form"""
	try:
		val = int(val)
	except Exception:
		return val
	if val > 2 ** 30:
		return "%d g" % (val // (2 ** 30))
	elif val > 2 ** 20:
		return "%d m" % (val // (2 ** 20))
	elif val > 2 ** 10:
		return "%d k" % (val // (2 ** 10))
	return str(val)


def get_platform():
	return {"linux": "Linux", "darwin": "Darwin"}.get(sys.platform, "Windows")


def home_dir():
	return os.path.expanduser("~")


# Small module-level memo of folder metadata caches, keyed by absolute folder
# path -> (cache timestamp, entries dict). Refreshed automatically when the
# updater writes a newer cache (different timestamp).
_folder_meta_memo = {}


def _load_folder_meta(root):
	"""Return (timestamp, entries_dict) for a folder's browser cache (memoized by timestamp)."""
	ts, entries, _path, _version = browser_cache_read(root)
	if ts is None:
		return None, {}
	key = os.path.abspath(root)
	cur = _folder_meta_memo.get(key)
	if cur is not None and cur[0] == ts:
		return ts, cur[1]
	_folder_meta_memo[key] = (ts, entries)
	return ts, entries


# ---------------------------------------------------------------------------
# Restricted orthogonal (XYZ) projections for the Rng XYZ / All XYZ actions
# (EMAN2 embrowser.py makeOrthoProj equivalent, in pure numpy).
# ---------------------------------------------------------------------------

def _gauss_filter_2d(img, apix, lowpass, highpass):
	"""Gaussian low- and/or high-pass filtering of a 2-D image in the
	frequency domain; cutoffs are in Angstroms, <= 0 disables. Uses
	H_lp(f) = exp(-(f/fo)^2) and H_hp(f) = 1 - exp(-(f/fo)^2), where
	fo = 1/cutoff (f in cycles/A, cutoffs in A)."""
	if lowpass <= 0 and highpass <= 0:
		return img
	ny, nx = img.shape
	fx = np.fft.fftfreq(nx, d=apix)
	fy = np.fft.fftfreq(ny, d=apix)
	F = np.sqrt(fx[np.newaxis, :] ** 2 + fy[:, np.newaxis] ** 2)
	H = None
	if lowpass > 0:
		H = np.exp(-(F * lowpass) ** 2)
	if highpass > 0:
		Ph = 1.0 - np.exp(-(F * highpass) ** 2)
		H = Ph if H is None else H * Ph
	return np.fft.ifft2(np.fft.fft2(img) * H).real


def _normalize_2d(img):
	"""Scale to zero mean and unit std (unchanged when std is 0)."""
	img = img - img.mean()
	std = img.std()
	if std > 0:
		img = img / std
	return img


def make_ortho_proj(vol, layers=-1, center=0, lowpass=-1, highpass=-1,
	apix=1.0, stack_out=False):
	"""Restricted orthogonal projections of a 3-D volume (nz, ny, nx).

	Sums slices along each axis over the range center±layers (layers < 0
	means the full axis), optionally Gaussian-filters (cutoffs in A), and
	normalizes each projection (zero mean, unit std) to balance the three
	directions, as in EMAN2. With stack_out=False the three projections are
	packed side by side into one 2-D image in (x, y, z) order; otherwise a
	list of the three separate images is returned.
	"""
	vol = np.asarray(vol, dtype=np.float32)
	if vol.ndim == 2:
		vol = vol[np.newaxis, :, :]
	_nz, _ny, nx = vol.shape

	def proj(axis, size):
		if layers >= 0:
			first = size // 2 + center - layers
			last = size // 2 + center + layers + 1
			first = max(0, min(first, size))
			last = max(first, min(last, size))
			img = np.sum(np.take(vol, range(first, last), axis=axis), axis=axis)
		else:
			img = np.sum(vol, axis=axis)
		img = _gauss_filter_2d(img, apix, lowpass, highpass)
		return _normalize_2d(img).astype(np.float32)

	# numpy axes: 2 = x, 1 = y, 0 = z
	px = proj(2, nx)
	py = proj(1, _ny)
	pz = proj(0, _nz)
	if stack_out:
		return [px, py, pz]
	# pack side by side (x, y, z); pad to the max height for non-cubic boxes
	height = max(px.shape[0], py.shape[0], pz.shape[0])
	width = px.shape[1] + py.shape[1] + pz.shape[1]
	packed = np.zeros((height, width), dtype=np.float32)
	c = 0
	for p in (px, py, pz):
		packed[:p.shape[0], c:c + p.shape[1]] = p
		c += p.shape[1]
	return packed


class EMSliceParamDialog(QtWidgets.QDialog):
	"""Modal parameters dialog for restricted XYZ projections (Rng XYZ /
	All XYZ actions), as in EMAN2's EMSliceParamDialog. Recent values are
	remembered between invocations."""
	dlayers = -1
	dcenter = 0
	dlp = "-1"
	dhp = "-1"
	dmask = ""
	dstkout = 0
	doldwin = 0

	def __init__(self, parent=None, nimg=1):
		QtWidgets.QDialog.__init__(self, parent)
		self.nimg = int(nimg)
		self.setWindowTitle("Slice Parameters")

		self.vbl = QtWidgets.QVBoxLayout(self)
		self.fol = QtWidgets.QFormLayout()
		self.vbl.addLayout(self.fol)

		if self.nimg > 1:
			self.wspinmin = QtWidgets.QSpinBox()
			self.wspinmin.setRange(-1, self.nimg - 1)
			self.wspinmin.setValue(-1)
			self.wspinmin.setToolTip("Start of range of volumes to display, -1 for all")
			self.fol.addRow("First Image #:", self.wspinmin)

			self.wspinmax = QtWidgets.QSpinBox()
			self.wspinmax.setRange(-1, self.nimg)
			self.wspinmax.setValue(-1)
			self.wspinmax.setToolTip("End (exclusive) of range of volumes to display, -1 for all")
			self.fol.addRow("Last Image #:", self.wspinmax)

			self.wspinstep = QtWidgets.QSpinBox()
			self.wspinstep.setRange(1, max(self.nimg // 2, 1))
			self.wspinstep.setValue(1)
			self.wspinstep.setToolTip("Step for range of volumes to display (partial display of file)")
			self.fol.addRow("Step:", self.wspinstep)

		self.wspinlayers = QtWidgets.QSpinBox()
		self.wspinlayers.setRange(-1, 1024)
		self.wspinlayers.setValue(self.dlayers)
		self.wspinlayers.setToolTip("Sum about center over +- selected number of layers, eg 0 -> central section only, -1 full projection")
		self.fol.addRow("Sum Layers (0->1, 1->3, 2->5, ...):", self.wspinlayers)

		self.wspincenter = QtWidgets.QSpinBox()
		self.wspincenter.setRange(-1024, 1024)
		self.wspincenter.setValue(self.dcenter)
		self.wspincenter.setToolTip("Center about which sum is generated, 0 = center of volume")
		self.fol.addRow("Center for sum:", self.wspincenter)

		self.wlelp = QtWidgets.QLineEdit(self.dlp)
		self.wlelp.setToolTip("If >0 applies a 2-D Gaussian low-pass filter. Specify in A.")
		self.fol.addRow("Lowpass (A):", self.wlelp)

		self.wlehp = QtWidgets.QLineEdit(self.dhp)
		self.wlehp.setToolTip("If >0 applies a 2-D Gaussian high-pass filter. Specify in A.")
		self.fol.addRow("Highpass (A):", self.wlehp)

		self.wlemask = QtWidgets.QLineEdit(self.dmask)
		self.wlemask.setToolTip("Optional filename of a mask volume (same dimensions as the volumes)")
		self.fol.addRow("Mask volume:", self.wlemask)

		self.wleref = QtWidgets.QLineEdit("")
		self.wleref.setToolTip("Optional filename of a reference volume (same dimensions); its projections are appended last")
		self.fol.addRow("Reference volume:", self.wleref)

		self.wcheckstk = QtWidgets.QCheckBox("enable")
		self.wcheckstk.setChecked(bool(self.dstkout))
		self.wcheckstk.setToolTip("If set, makes 3 separate images instead of a single rectangular image. Good for FFTs.")
		self.fol.addRow("Stack output:", self.wcheckstk)

		self.wcheckoldwin = QtWidgets.QCheckBox("enable")
		self.wcheckoldwin.setChecked(bool(self.doldwin))
		self.wcheckoldwin.setToolTip("If set, uses the last existing 2-D image window (replacing its contents)")
		self.fol.addRow("Same window:", self.wcheckoldwin)

		self.bhb = QtWidgets.QHBoxLayout()
		self.vbl.addLayout(self.bhb)
		self.wbutok = QtWidgets.QPushButton("OK")
		self.bhb.addWidget(self.wbutok)
		self.wbutcancel = QtWidgets.QPushButton("Cancel")
		self.bhb.addWidget(self.wbutcancel)

		self.wbutok.clicked.connect(self.okpress)
		self.wbutcancel.clicked.connect(self.reject)
		self.wbutok.setDefault(True)

	def okpress(self, state=None):
		EMSliceParamDialog.dlayers = self.wspinlayers.value()
		EMSliceParamDialog.dcenter = self.wspincenter.value()
		EMSliceParamDialog.dlp = self.wlelp.text()
		EMSliceParamDialog.dhp = self.wlehp.text()
		EMSliceParamDialog.dmask = self.wlemask.text()
		EMSliceParamDialog.dstkout = int(self.wcheckstk.isChecked())
		EMSliceParamDialog.doldwin = int(self.wcheckoldwin.isChecked())
		self.accept()

	def parameters(self):
		"""The dialog values as a dict (call after a successful exec())."""
		return {
			"first": self.wspinmin.value() if self.nimg > 1 else 0,
			"last": self.wspinmax.value() if self.nimg > 1 else 1,
			"step": self.wspinstep.value() if self.nimg > 1 else 1,
			"layers": self.wspinlayers.value(),
			"center": self.wspincenter.value(),
			"lowpass": float(self.wlelp.text() or -1),
			"highpass": float(self.wlehp.text() or -1),
			"mask": self.wlemask.text().strip(),
			"reference": self.wleref.text().strip(),
			"stack_out": self.wcheckstk.isChecked(),
			"same_window": self.wcheckoldwin.isChecked(),
		}


def _reduce_bits(data, bits):
	"""Reduce image data to at most 'bits' bits (8/16/32), rescaling to the
	full range of the target depth. Data already at or below the requested
	depth is returned unchanged (no expansion); bits <= 0 means no reduction."""
	if bits is None or bits <= 0:
		return data
	a = np.asarray(data)
	cur = 64
	if a.dtype == np.uint8:
		cur = 8
	elif a.dtype == np.uint16:
		cur = 16
	elif a.dtype in (np.int32, np.float32):
		cur = 32
	if cur <= bits:
		return a
	if bits < 16:
		top, outdt = 255.0, np.uint8
	elif bits < 32:
		top, outdt = 65535.0, np.uint16
	else:
		if a.dtype == np.float64:
			return a.astype(np.float32)
		return a
	lo, hi = float(a.min()), float(a.max())
	if hi <= lo:
		return np.zeros(a.shape, dtype=outdt)
	return np.clip((a - lo) / (hi - lo) * top, 0, top).astype(outdt)


class EMSaveAsDialog(QtWidgets.QDialog):
	"""'Save As' dialog: output filename, an explicit format (otherwise the
	format is determined by the output extension), and a First/Last image
	range for input files with more than one image."""

	def __init__(self, parent=None, default_name="", nimg=1):
		QtWidgets.QDialog.__init__(self, parent)
		self.setWindowTitle("Save As")
		self.nimg = int(nimg)

		self.vbl = QtWidgets.QVBoxLayout(self)
		self.fol = QtWidgets.QFormLayout()
		self.vbl.addLayout(self.fol)

		self.wlefile = QtWidgets.QLineEdit(default_name)
		self.fol.addRow("Output file:", self.wlefile)

		self.wcombo = QtWidgets.QComboBox()
		self.wcombo.addItem("(determined by output extension)", None)
		for label, ext in [
			("JPEG (.jpg)", ".jpg"),
			("PGM (.pgm)", ".pgm"),
			("PNG (.png)", ".png"),
			("MRC (.mrc)", ".mrc"),
			("MRC stack (.mrcs)", ".mrcs"),
			("TIFF (.tif)", ".tif"),
			("IMAGIC (.img)", ".img"),
			("HDF5 (.hdf)", ".hdf"),
		]:
			self.wcombo.addItem(label, ext)
		self.fol.addRow("Format:", self.wcombo)

		self.wspinbits = QtWidgets.QSpinBox()
		self.wspinbits.setRange(-1, 32)
		self.wspinbits.setValue(-1)
		self.wspinbits.setToolTip("Bit depth of output data: -1 = no bit reduction (as far as the format allows); 8 or 16 rescales the data to that depth")
		self.fol.addRow("Bits:", self.wspinbits)

		if self.nimg > 1:
			self.wspinfirst = QtWidgets.QSpinBox()
			self.wspinfirst.setRange(0, self.nimg - 1)
			self.wspinfirst.setValue(0)
			self.wspinfirst.setToolTip("First image to save (0-based)")
			self.fol.addRow("First Image #:", self.wspinfirst)

			self.wspinlast = QtWidgets.QSpinBox()
			self.wspinlast.setRange(0, self.nimg - 1)
			self.wspinlast.setValue(self.nimg - 1)
			self.wspinlast.setToolTip("Last image to save (0-based, inclusive)")
			self.fol.addRow("Last Image #:", self.wspinlast)

		self.bhb = QtWidgets.QHBoxLayout()
		self.vbl.addLayout(self.bhb)
		self.wbutok = QtWidgets.QPushButton("OK")
		self.bhb.addWidget(self.wbutok)
		self.wbutcancel = QtWidgets.QPushButton("Cancel")
		self.bhb.addWidget(self.wbutcancel)
		self.wbutok.clicked.connect(self.accept)
		self.wbutcancel.clicked.connect(self.reject)
		self.wbutok.setDefault(True)

	def parameters(self):
		"""The dialog values as a dict (call after a successful exec())."""
		return {
			"file": self.wlefile.text().strip(),
			"format": self.wcombo.currentData(),
			"bits": self.wspinbits.value(),
			"first": self.wspinfirst.value() if self.nimg > 1 else 0,
			"last": self.wspinlast.value() if self.nimg > 1 else 0,
		}


# ===========================================================================
# Info panes (right-hand detail display). The heavy, widget-dependent ones are
# stubbed to show basic metadata and mark their interactive actions
# "unimplemented" until the corresponding EMAN3/gui display widgets exist.
# ===========================================================================

class EMInfoPane(QtWidgets.QWidget):
	"""Base info pane."""

	def __init__(self, parent=None):
		QtWidgets.QWidget.__init__(self, parent)
		self.vbl = QtWidgets.QVBoxLayout(self)
		self.wtitle = QtWidgets.QLabel("No information available")
		self.vbl.addWidget(self.wtitle)
		self.wtext = QtWidgets.QTextEdit()
		self.wtext.setReadOnly(True)
		self.vbl.addWidget(self.wtext)

	def display(self, target):
		self.wtitle.setText("No information available")
		self.wtext.clear()

	def busy(self):
		QtWidgets.QApplication.instance().setOverrideCursor(Qt.BusyCursor)

	def notbusy(self):
		QtWidgets.QApplication.instance().setOverrideCursor(Qt.ArrowCursor)


class EMBasicInfoPane(EMInfoPane):
	"""Shows the standard metadata fields for any file (from its EMDirEntry)."""

	def display(self, target):
		self.wtitle.setText(target.name)
		lines = [
			"Path:  %s" % target.path(),
			"Type:  %s" % nonone(target.filetype),
			"Dim:   %s" % nonone(target.dim),
			"N:     %s" % nonone(target.nimg),
			"Size:  %s" % humansize(target.size),
			"Date:  %s" % nonone(target.date),
			"",
			"(Display actions for this type are not yet ported to EMAN3.)"
		]
		self.wtext.setPlainText("\n".join(lines))


class EMFolderInfoPane(EMInfoPane):
	def display(self, target):
		self.wtitle.setText("Folder: %s" % target.name)
		self.wtext.setPlainText("Path: %s\n\nDouble-click to open this folder." % target.path())


class EMTextInfoPane(EMInfoPane):
	def __init__(self, parent=None):
		EMInfoPane.__init__(self, parent)
		self.wtitle.setText("Text file")

	def display(self, target):
		self.wtitle.setText(target.name)
		try:
			with open(target.path(), "r", errors="replace") as f:
				self.wtext.setPlainText(f.read())
		except Exception as e:
			self.wtext.setPlainText("Cannot read file: %s" % e)


class EMHTMLInfoPane(EMInfoPane):
	def __init__(self, parent=None):
		EMInfoPane.__init__(self, parent)
		self.wtext.setReadOnly(False)
		self.wtitle.setText("HTML file")

	def display(self, target):
		self.wtitle.setText(target.name)
		try:
			with open(target.path(), "r", errors="replace") as f:
				self.wtext.setHtml(f.read())
		except Exception as e:
			self.wtext.setPlainText("Cannot read file: %s" % e)


class EMJSONInfoPane(EMBasicInfoPane):
	def display(self, target):
		EMBasicInfoPane.display(self, target)
		try:
			with open(target.path(), "r", errors="replace") as f:
				self.wtext.setPlainText(f.read())
		except Exception:
			pass


class EMImageInfoPane(EMBasicInfoPane):
	pass


class EMStackInfoPane(EMBasicInfoPane):
	pass


class _DataColumnsModel(QtCore.QAbstractTableModel):
	"""Virtual table model for a Data file: four summary rows (Mean/Std/Min/Max)
	followed by the full data rows. Backed by numpy arrays so large files stay
	responsive (the view virtualizes rows instead of creating per-cell widgets).

	Row labels (Mean/Std/Min/Max, 0, 1, ...) are a leading data column rather
	than the vertical header: in this PySide6 build the C++->Python dispatch of
	QAbstractTableModel virtuals is context-dependent -- data() dispatches
	everywhere, but headerData() calls for the vertical header silently fail in
	some contexts (labels vanished). The view's vertical header is hidden.

	NOTE: the overridden C++ virtuals are defined exactly like EMFileItemModel's
	(no default arguments, headerData(sec, orient, role) instead of the
	horizontal/vertical variants) -- that is the style this PySide6 build
	dispatches reliably from the C++ side."""

	STATS = ("Mean", "Std", "Min", "Max")

	def __init__(self, parent=None):
		QtCore.QAbstractTableModel.__init__(self)
		self._arr = None
		self._stats = None
		self._labels = None

	def set_data(self, arr, labels=None):
		self.beginResetModel()
		self._arr = arr
		self._labels = labels
		if arr is not None and arr.size:
			self._stats = np.array([arr.mean(axis=0), arr.std(axis=0),
				arr.min(axis=0), arr.max(axis=0)])
		else:
			self._stats = None
		self.endResetModel()

	def _rowlabel(self, r):
		if r < len(self.STATS):
			return self.STATS[r]
		return str(r - len(self.STATS))

	def rowCount(self, parent):
		if parent is not None and parent.isValid():
			return 0
		if self._arr is None:
			return 0
		return len(self.STATS) + self._arr.shape[0]

	def columnCount(self, parent):
		if parent is not None and parent.isValid():
			return 0
		if self._arr is None:
			return 0
		return self._arr.shape[1] + 1  # leading row-label column

	def data(self, index, role):
		if not index.isValid() or self._arr is None:
			return None
		r, c = index.row(), index.column()
		if role == Qt.DisplayRole:
			if c == 0:
				return self._rowlabel(r)
			if r < len(self.STATS):
				v = self._stats[r, c - 1]
			else:
				v = self._arr[r - len(self.STATS), c - 1]
			return "%.6g" % v
		if role == Qt.FontRole and (c == 0 or r < len(self.STATS)):
			f = QtGui.QFont()
			f.setBold(True)
			return f
		return None

	def headerData(self, sec, orient, role):
		if self._arr is None:
			return None
		if orient != Qt.Horizontal:
			return None
		if role != Qt.DisplayRole:
			return None
		if sec == 0:
			return "Row"
		d = sec - 1
		if self._labels is not None and d < len(self._labels) and self._labels[d]:
			return str(self._labels[d])
		return "Col %d" % d


class EMPlotInfoPane(EMInfoPane):
	"""Info pane for columnar Data files (plot/sphere sources): the path and
	file size up top, and a table of the full file contents with per-column
	Mean/Std/Min/Max rows above the data."""

	def __init__(self, parent=None):
		EMInfoPane.__init__(self, parent)
		self.wtext.setFixedHeight(60)
		self.model = _DataColumnsModel(self)
		self.table = QtWidgets.QTableView()
		self.table.setModel(self.model)
		self.table.setAlternatingRowColors(True)
		self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
		self.table.verticalHeader().setDefaultSectionSize(18)
		self.table.verticalHeader().hide()  # row labels are a leading data column
		self.table.horizontalHeader().setStretchLastSection(True)
		self.table.setColumnWidth(0, 50)
		self.vbl.addWidget(self.table)

	def display(self, target):
		self.wtitle.setText(target.name)
		try:
			size = os.path.getsize(target.path())
		except OSError:
			size = 0
		prefix = "Path:  %s\nFile size:  %d bytes" % (target.path(), size)
		self.wtext.setPlainText(prefix)
		self.busy()
		try:
			columns, labels = EMPlotFileType(target.path()).plot_data()
			if columns is None:
				arr = None
			else:
				arr = np.array([np.asarray(c, dtype=np.float64) for c in columns]).T
				if labels and len(arr) and arr.shape[1] == len(labels) + 1:
					# A leading 0,1,2,... index column is unlabeled: right-align the labels
					c0 = arr[:, 0]
					if all(abs(c0[i] - i) < 1e-12 for i in range(min(8, len(c0)))):
						labels = [None] + list(labels)
		except Exception as e:
			arr = None
			self.wtext.setPlainText(prefix + "\n\nCannot parse data: %s" % e)
		self.notbusy()
		self.model.set_data(arr, labels)


class EMPDBInfoPane(EMBasicInfoPane):
	pass


class EMBDBInfoPane(EMBasicInfoPane):
	pass


# ===========================================================================
# File type classes. `name()` must match the "type" strings produced by the
# e3browsercache updater so EMDirEntry.fileTypeClass() can resolve them.
# Action slots that would open a not-yet-ported display widget print
# "unimplemented".
# ===========================================================================

class EMFileType(object):
	"""Base class for handling interaction with files of different types."""

	typesbyft = {}          # filetype string -> subclass (filled after definitions)
	typesbyext = {}         # extension -> subclass(es)
	alltocheck = ()

	setsmode = False

	def __init__(self, path):
		if path[:2] == "./":
			path = path[2:]
		self.path = path
		self.n = -1

	def setFile(self, path):
		if path[:2] == "./":
			path = path[2:]
		self.path = path

	def setN(self, n):
		self.n = n

	@staticmethod
	def name():
		return None

	@staticmethod
	def isValid(path, header):
		return False

	@staticmethod
	def infoClass():
		return EMInfoPane

	def actions(self):
		"""Returns a list of (name, help, callback) tuples."""
		return []

	def _unimpl(self, what):
		print("unimplemented: %s  (%s)" % (what, self.path))

	_SPLITS = re.compile(r"[\s,;]+")

	def plot_data(self):
		"""Parse this file for plotting the same way the 2-D/3-D plot widgets do:
		skip blank and '#' lines, strip '#' comments, split fields on
		whitespace/comma/semicolon, all fields numeric. Column labels may be
		supplied by an optional first header line, or by a leading '#' comment
		line of ';' separated labels (e.g. "#az0; az1; alt0; ...").
		Returns (columns, labels) where columns is a list of per-column value
		lists, or (None, labels) if no data lines."""
		rows = []
		labels = None
		with open(self.path, "r") as f:
			for line in f:
				if len(line) < 2:
					continue
				if line[0] == "#":
					if labels is None and not rows and ";" in line:
						toks = [t.strip() for t in line.lstrip("#").split(";")]
						toks = [t for t in toks if t]
						if len(toks) >= 2:
							try:
								[float(t) for t in toks]
							except ValueError:
								labels = toks
					continue
				fields = [x for x in EMFileType._SPLITS.split(line.split("#")[0]) if x]
				if not fields:
					continue
				try:
					rows.append([float(x) for x in fields])
				except ValueError:
					if not rows and labels is None:
						labels = [x.strip() for x in fields]
						continue
					raise
		if not rows:
			return None, labels
		ncols = len(rows[0])
		if any(len(r) != ncols for r in rows[1:]):
			raise ValueError("inconsistent column counts in %s" % self.path)
		return [list(c) for c in zip(*rows)], labels

	def plot2dApp(self, brws):
		"""Add this file's data to the last opened 2-D plot window (new window if none)."""
		brws.busy()
		try:
			columns, labels = self.plot_data()
			brws._show_plot2d(columns, self.path, new=False, column_labels=labels)
		finally:
			brws.notbusy()

	def plot2dNew(self, brws):
		"""Open a new 2-D plot window with this file's data."""
		brws.busy()
		try:
			columns, labels = self.plot_data()
			brws._show_plot2d(columns, self.path, new=True, column_labels=labels)
		finally:
			brws.notbusy()

	def show2dAvg(self, brws):
		"""Avg All: unaligned average of the entire stack, shown in the last
		opened 2-D image window (or a new one if none)."""
		brws.busy()
		try:
			io = ImageIO(self.path, "r")
			n = io.nimg
			acc = None
			for i in range(n):
				img, _h = io.read_image(i)
				img = np.asarray(img, dtype=np.float64)
				acc = img if acc is None else acc + img
			io.close()
		except Exception as e:
			brws.notbusy()
			display_error("could not read %s: %s" % (self.path, e))
			return
		if acc is None:
			brws.notbusy()
			display_error("no images to average")
			return
		brws.notbusy()
		brws._show_image2d([acc / n], self.path, new=False)

	def show2dAvgRnd(self, brws):
		"""Avg Rnd Subset: 10 averages, each over a random subset of
		min(1/4 of the images, 1000) images, shown as a set of 2-D images
		in the last opened 2-D image window (or a new one if none)."""
		if getattr(self, "nimg", 1) < 8:
			print("need at least 8 images in the stack to average random subsets")
			return
		brws.busy()
		try:
			io = ImageIO(self.path, "r")
			n = io.nimg
			nsub = min(n // 4, 1000)
			avgs = []
			for _ in range(10):
				idx = sorted(random.randrange(n) for _ in range(nsub))
				acc = None
				for i in idx:
					img, _h = io.read_image(i)
					img = np.asarray(img, dtype=np.float64)
					acc = img if acc is None else acc + img
				avgs.append(acc / nsub)
			io.close()
		except Exception as e:
			brws.notbusy()
			display_error("could not compute random averages of %s: %s" % (self.path, e))
			return
		brws.notbusy()
		brws._show_image2d(avgs, self.path, new=False)

	def plot3dApp(self, brws):
		"""Add this file's data to the last opened 3-D plot window (new window if none)."""
		brws.busy()
		try:
			columns, labels = self.plot_data()
			brws._show_plot3d(columns, self.path, new=False, column_labels=labels)
		finally:
			brws.notbusy()

	def plot3dNew(self, brws):
		"""Open a new 3-D plot window with this file's data."""
		brws.busy()
		try:
			columns, labels = self.plot_data()
			brws._show_plot3d(columns, self.path, new=True, column_labels=labels)
		finally:
			brws.notbusy()

	def histApp(self, brws):
		"""Add this file's distributions to the last opened histogram window (new window if none)."""
		brws.busy()
		try:
			columns, labels = self.plot_data()
			brws._show_hist(columns, self.path, new=False, column_labels=labels)
		finally:
			brws.notbusy()

	def histNew(self, brws):
		"""Open a new histogram window with this file's distributions."""
		brws.busy()
		try:
			columns, labels = self.plot_data()
			brws._show_hist(columns, self.path, new=True, column_labels=labels)
		finally:
			brws.notbusy()

	def saveAs(self, brws):
		"""Save this file's image(s) in a new file format (dialog in the browser)."""
		brws._save_as(self.path, getattr(self, "nimg", 1))


class EMFolderFileType(EMFileType):
	@staticmethod
	def name():
		return "Folder"

	@staticmethod
	def infoClass():
		return EMFolderInfoPane

	def actions(self):
		return []


class EMTextFileType(EMFileType):
	@staticmethod
	def name():
		return "Text"

	@staticmethod
	def infoClass():
		return EMTextInfoPane

	def actions(self):
		return []


class EMHTMLFileType(EMFileType):
	@staticmethod
	def name():
		return "HTML"

	@staticmethod
	def infoClass():
		return EMHTMLInfoPane

	def actions(self):
		return [("Firefox", "Open in Firefox", self.showFirefox)]

	def showFirefox(self, brws):
		p = os.path.abspath(self.path)
		plat = get_platform()
		if plat == "Darwin":
			os.system("open %s" % p)
		elif plat == "Linux":
			os.system("xdg-open %s" % p)
		else:
			self._unimpl("open HTML in browser")


class EMPDFFileType(EMFileType):
	@staticmethod
	def name():
		return "PDF"

	@staticmethod
	def infoClass():
		return EMBasicInfoPane

	def actions(self):
		"""Order matches the EMAN2 browser (top-left button = double-click)."""
		plat = get_platform()
		if plat == "Linux":
			return [("gv", "Open using gv", self.showPlatform),
				("Firefox", "Open in Firefox", self.showFirefox)]
		elif plat == "Darwin":
			return [("Open", "Default Open", self.showPlatform)]
		return []

	def showPlatform(self, brws):
		p = os.path.abspath(self.path)
		plat = get_platform()
		if plat == "Linux":
			os.system("gv %s" % p)
		elif plat == "Darwin":
			os.system("open %s" % p)
		else:
			self._unimpl("open PDF (platform default)")

	def showFirefox(self, brws):
		p = os.path.abspath(self.path)
		plat = get_platform()
		if plat == "Linux":
			os.system("firefox -new-tab file://%s" % p)
		elif plat == "Darwin":
			os.system("open %s" % p)
		else:
			self._unimpl("showFirefox")


class EMJSONFileType(EMFileType):
	@staticmethod
	def name():
		return "JSON"

	@staticmethod
	def infoClass():
		return EMJSONInfoPane

	def __init__(self, path):
		EMFileType.__init__(self, path)
		self.xform = False
		self.first = None
		try:
			js = js_open_dict(path, create=False)
			try:
				keys = list(js.keys())
				if keys:
					self.first = keys[0]
					v = js[keys[0]]
					if isinstance(v, dict) and "xform.align3d" in v:
						self.xform = True
			finally:
				js.close()
		except Exception:
			pass

	def actions(self):
		"""Order matches the EMAN2 browser (top-left button = double-click)."""
		if not self.xform:
			return []
		ret = [("Plot 2D", "plot xform", self.plot2dApp),
			("Plot 2D+", "plot xform", self.plot2dNew),
			("Histogram", "histogram xform", self.histApp),
			("Histogram+", "histogram xform", self.histNew)]
		try:
			if self.first is not None:
				io = ImageIO(self.first, "r")
				io.close()
				ret.append(("All XYZ", "Show restricted XYZ projections of all", self.show2dStack3sec))
		except Exception:
			pass
		return ret

	def plot2dApp(self, brws): self._unimpl("plot2dApp")
	def plot2dNew(self, brws): self._unimpl("plot2dNew")
	def histApp(self, brws): self._unimpl("histApp")
	def histNew(self, brws): self._unimpl("histNew")
	def show2dStack3sec(self, brws): self._unimpl("show2dStack3sec")


class EMPlotFileType(EMFileType):
	@staticmethod
	def name():
		return "Plot"

	@staticmethod
	def infoClass():
		return EMPlotInfoPane

	def __init__(self, path):
		EMFileType.__init__(self, path)
		# Column count for choosing the action set (from the browser cache dim "Ncol x Nrow")
		self.ncol = 0
		try:
			_ts, entries, _p, _v = browser_cache_read(os.path.dirname(path))
			dim = entries.get(os.path.basename(path), {}).get("dim")
			if dim and " x " in dim:
				self.ncol = int(dim.split(" x ")[0])
		except Exception:
			pass

	def actions(self):
		"""Order matches the EMAN2 browser (top-left button = double-click)."""
		if self.ncol > 2:
			rtr = [("Plot 2D", "Add to current plot", self.plot2dApp),
				("Plot 2D+", "Make new plot", self.plot2dNew),
				("Hist", "Add to current histogram", self.histApp),
				("Hist +", "Make new histogram", self.histNew),
				("Plot 3D", "Add to current 3-D plot", self.plot3dApp),
				("Plot 3D+", "Make new 3-D plot", self.plot3dNew)]
		else:
			rtr = [("Plot 2D", "Add to current plot", self.plot2dApp),
				("Plot 2D+", "Make new plot", self.plot2dNew),
				("Hist", "Add to current histogram", self.histApp),
				("Hist +", "Make new histogram", self.histNew)]
		if 3 <= self.ncol <= 5:
			rtr.append(("Spheres", "Each X line is X-Y-Z[-A[-S]]. Add to current 3-D window", self.showSpheres))
			rtr.append(("Spheres +", "Each X line is X-Y-Z[-A[-S]]. New 3-D window", self.showSpheresNew))
		return rtr

	def _spheres_data(self):
		"""Rows of the data file as (N, ncols) points: X-Y-Z[-amplitude[-S]]."""
		columns, _labels = self.plot_data()
		if columns is None:
			raise ValueError("no data lines in %s" % self.path)
		pts = np.array([np.asarray(c, dtype=np.float64) for c in columns]).T
		if pts.shape[1] < 3:
			raise ValueError("need at least 3 columns (X-Y-Z) in %s" % self.path)
		return pts

	def showSpheres(self, brws):
		"""Show each data row as a sphere at X-Y-Z (4th column = brightness,
		5th ignored) in the last opened 3-D window (new window if none)."""
		brws.busy()
		try:
			brws._show_spheres([self._spheres_data()], self.path, new=False)
		finally:
			brws.notbusy()

	def showSpheresNew(self, brws):
		"""Open a new 3-D window with each data row as a sphere at X-Y-Z."""
		brws.busy()
		try:
			brws._show_spheres([self._spheres_data()], self.path, new=True)
		finally:
			brws.notbusy()


class EMBdbFileType(EMFileType):
	@staticmethod
	def name():
		return "BDB"

	@staticmethod
	def infoClass():
		return EMBDBInfoPane

	def __init__(self, path):
		EMFileType.__init__(self, path)
		# bdb files are single 3-D volumes
		self.nimg = 1
		self.dim = (0, 0, 2)
		try:
			io = ImageIO(path, "r")
			self.nimg = io.nimg or 1
			h = io.read_header(0)
			self.dim = (int(h.get("nx", 0)), int(h.get("ny", 0)), int(h.get("nz", 0)))
			io.close()
		except Exception:
			pass

	def actions(self):
		"""Order matches the EMAN2 browser (top-left button = double-click)."""
		if self.nimg == 1 and self.dim[2] > 1:
			return [("Show 3D", "Add to 3D window", self.show3dApp),
				("Show 3D+", "New 3D Window", self.show3DNew),
				("Show Stack", "Show as set of 2-D Z slices", self.show2dStack),
				("Show Stack+", "Show all images together in a new window", self.show2dStackNew),
				("Show 2D", "Show in a scrollable 2D image window", self.show2dSingle),
				("Show 2D+", "Show all images, one at a time in a new window", self.show2dSingleNew),
				("FilterTool", "Open in filter tool", self.showFilterTool),
				("ProjXYZ", "Make projections along Z,Y,X", self.showProjXYZ),
				("Save As", "Saves images in new file format", self.saveAs)]
		elif self.nimg == 1 and self.dim[1] > 1:
			return [("Show 2D", "Show in a 2D single image display", self.show2dSingle),
				("Show 2D+", "Show in new 2D single image display", self.show2dSingleNew),
				("FilterTool", "Open in filter tool", self.showFilterTool),
				("Save As", "Saves images in new file format", self.saveAs)]
		elif self.nimg == 1:
			return [("Plot 2D", "Add to current plot", self.plot2dApp),
				("Plot 2D+", "Make new plot", self.plot2dNew),
				("Show 2D", "Replace in 2D single image display", self.show2dSingle),
				("Show 2D+", "New 2D single image display", self.show2dSingleNew),
				("Save As", "Saves images in new file format", self.saveAs)]
		elif self.nimg > 1 and self.dim[2] > 1:
			return [("Show 3D", "Show all in a single 3D window", self.show3DNew),
				("Save As", "Saves images in new file format", self.saveAs)]
		elif self.nimg > 1 and self.dim[1] > 1:
			return [("Show Stack", "Show all images together in one window", self.show2dStack),
				("Show Stack+", "Show all images together in a new window", self.show2dStackNew),
				("Show 2D", "Show all images, one at a time in current window", self.show2dSingle),
				("Show 2D+", "Show all images, one at a time in a new window", self.show2dSingleNew),
				("Avg All", "Unaligned average of entire stack", self.show2dAvg),
				("Avg Sample", "Averages random min(1/4 of images,1000) multiple times", self.show2dAvgRnd),
				("FilterTool", "Open in filter tool", self.showFilterTool),
				("Save As", "Saves images in new file format", self.saveAs)]
		elif self.nimg > 0:
			return [("Plot 2D", "Plot all on a single 2-D plot", self.plot2dNew),
				("Save As", "Saves images in new file format", self.saveAs)]
		return []

	def show3dApp(self, brws): self._unimpl("show3dApp")
	def show3DNew(self, brws): self._unimpl("show3DNew")
	def show2dStack(self, brws): self._unimpl("show2dStack")
	def show2dStackNew(self, brws): self._unimpl("show2dStackNew")
	def show2dSingle(self, brws): self._unimpl("show2dSingle")
	def show2dSingleNew(self, brws): self._unimpl("show2dSingleNew")
	def showFilterTool(self, brws): self._unimpl("showFilterTool")
	def show2dAvg(self, brws): self._unimpl("show2dAvg")
	def show2dAvgRnd(self, brws): self._unimpl("show2dAvgRnd")
	def plot2dApp(self, brws): self._unimpl("plot2dApp")
	def plot2dNew(self, brws): self._unimpl("plot2dNew")
	def showProjXYZ(self, brws): self._unimpl("showProjXYZ")


class EMImageFileType(EMFileType):
	"""A single 1-3D image (or a 1-image file)."""

	def __init__(self, path):
		EMFileType.__init__(self, path)
		self.nimg = 1
		self.dim = (0, 0, 0)
		try:
			io = ImageIO(path, "r")
			self.nimg = io.nimg or 1
			h = io.read_header(0)
			self.dim = (int(h.get("nx", 0)), int(h.get("ny", 0)), int(h.get("nz", 0)))
			io.close()
		except Exception:
			pass

	@staticmethod
	def name():
		return "Image"

	@staticmethod
	def infoClass():
		return EMImageInfoPane

	def actions(self):
		if self.dim[2] > 1:
			return [("Show 3D", "Add to 3D window", self.show3dApp),
				("Show 3D+", "New 3D Window", self.show3DNew),
				("Show Stack", "Show as set of 2-D Z slices", self.show2dStack),
				("Show Stack+", "New window", self.show2dStackNew),
				("Show 2D", "Show in 2D image window", self.show2dSingle),
				("Show 2D+", "New 2D window", self.show2dSingleNew),
				("FilterTool", "Open in filter tool", self.showFilterTool),
				("Rng XYZ", "Show restricted XYZ projection", self.show2dStack3sec),
				("Save As", "Save in new file format", self.saveAs)]
		elif self.dim[1] > 1:
			return [("Show 2D", "Show in 2D single image display", self.show2dSingle),
				("Show 2D+", "New 2D display", self.show2dSingleNew),
				("FilterTool", "Open in filter tool", self.showFilterTool),
				("Save As", "Save in new file format", self.saveAs)]
		else:
			return [("Plot 2D", "Add to current plot", self.plot2dApp),
				("Plot 2D+", "Make new plot", self.plot2dNew),
				("Show 2D", "Show in 2D display", self.show2dSingle),
				("Show 2D+", "New 2D display", self.show2dSingleNew),
				("Save As", "Save in new file format", self.saveAs)]

	def show3dApp(self, brws):
		brws.busy()
		try:
			vols = brws._read_volumes(self.path)
			brws._show_scene3d(vols, self.path, new=False)
		finally:
			brws.notbusy()
	def show3DNew(self, brws):
		brws.busy()
		try:
			vols = brws._read_volumes(self.path)
			brws._show_scene3d(vols, self.path, new=True)
		finally:
			brws.notbusy()
	def show2dStack(self, brws):
		brws.busy()
		try:
			brws._show_imagemx(brws._read_2d_list(self.path), self.path, new=False)
		finally:
			brws.notbusy()
	def show2dStackNew(self, brws):
		brws.busy()
		try:
			brws._show_imagemx(brws._read_2d_list(self.path), self.path, new=True)
		finally:
			brws.notbusy()
	def show2dSingle(self, brws):
		brws.busy()
		try:
			brws._show_image2d(brws._read_2d_list(self.path), self.path, new=False)
		finally:
			brws.notbusy()
	def show2dSingleNew(self, brws):
		brws.busy()
		try:
			brws._show_image2d(brws._read_2d_list(self.path), self.path, new=True)
		finally:
			brws.notbusy()
	def showFilterTool(self, brws): self._unimpl("showFilterTool")
	def show2dStack3sec(self, brws):
		brws._show_xyz_projections(self.path, self.nimg)

	def plot_data(self):
		"""Images plot as intensity vs pixel index (all pixels, row-major)."""
		io = ImageIO(self.path, "r")
		try:
			img, _header = io.read_image(0)
		finally:
			io.close()
		if isinstance(img, (list, tuple)):
			flat = [v for row in img for v in (row if isinstance(row, (list, tuple)) else [row])]
		else:
			flat = img.ravel().tolist()
		return [flat], None


class EMStackFileType(EMFileType):
	"""A stack (or .lst) of 1-3D images."""

	def __init__(self, path):
		EMFileType.__init__(self, path)
		self.nimg = 1
		self.dim = (0, 0, 0)
		self.xfparms = False
		try:
			io = ImageIO(path, "r")
			self.nimg = io.nimg or 1
			h = io.read_header(0)
			self.dim = (int(h.get("nx", 0)), int(h.get("ny", 0)), int(h.get("nz", 0)))
			io.close()
		except Exception:
			pass

	@staticmethod
	def name():
		return "Image Stack"

	@staticmethod
	def infoClass():
		return EMStackInfoPane

	def actions(self):
		"""Order matches the EMAN2 browser (top-left button = double-click)."""
		if self.nimg > 1 and self.dim[2] > 1:
			rtr = [("Show all 3D", "Show all in a single 3D window", self.show3DAll),
				("Show 1st 3D", "Show only the first volume", self.show3DNew),
				("All XYZ", "Show restricted XYZ projections of all", self.show2dStack3sec),
				("Save As", "Save in new file format", self.saveAs)]
		elif self.nimg > 1 and self.dim[1] > 1:
			rtr = [("Show Stack", "Show all images together in one window", self.show2dStack),
				("Show Stack+", "Show all images together in a new window", self.show2dStackNew),
				("Show 2D", "Show all images, one at a time in current window", self.show2dSingle),
				("Show 2D+", "Show all images, one at a time in a new window", self.show2dSingleNew),
				("Avg All", "Unaligned average of entire stack", self.show2dAvg),
				("Avg Rnd Subset", "Averages random min(1/4 of images,1000) multiple times", self.show2dAvgRnd),
				("FilterTool", "Open in filter tool", self.showFilterTool),
				("Save As", "Save in new file format", self.saveAs)]
			if 3 <= self.dim[0] <= 5:
				rtr.append(("Spheres", "Each X line is X-Y-Z[-A[-S]]. Add to current 3-D window", self.showSpheres))
				rtr.append(("Spheres +", "Each X line is X-Y-Z[-A[-S]]. New 3-D window", self.showSpheresNew))
		elif self.nimg > 1:
			rtr = [("Plot 2D", "Plot all on a single 2-D plot", self.plot2dNew),
				("Save As", "Save in new file format", self.saveAs)]
		else:
			rtr = []

		if self.xfparms:
			rtr.extend([("Plot 2D", "Plot xform", self.plot2dLstApp),
				("Plot 2D+", "New plot window", self.plot2dLstNew)])

		return rtr

	def show3DAll(self, brws):
		brws.busy()
		try:
			vols = brws._read_volumes(self.path)
			brws._show_scene3d(vols, self.path, new=True)
		finally:
			brws.notbusy()
	def show3DNew(self, brws):
		brws.busy()
		try:
			vols = brws._read_volumes(self.path, first_only=True)
			brws._show_scene3d(vols, self.path, new=True)
		finally:
			brws.notbusy()
	def show2dStack3sec(self, brws):
		brws._show_xyz_projections(self.path, self.nimg)
	def show2dStack(self, brws):
		brws.busy()
		try:
			brws._show_imagemx(brws._read_2d_list(self.path), self.path, new=False)
		finally:
			brws.notbusy()
	def show2dStackNew(self, brws):
		brws.busy()
		try:
			brws._show_imagemx(brws._read_2d_list(self.path), self.path, new=True)
		finally:
			brws.notbusy()
	def show2dSingle(self, brws):
		brws.busy()
		try:
			brws._show_image2d(brws._read_2d_list(self.path), self.path, new=False)
		finally:
			brws.notbusy()
	def show2dSingleNew(self, brws):
		brws.busy()
		try:
			brws._show_image2d(brws._read_2d_list(self.path), self.path, new=True)
		finally:
			brws.notbusy()
	def showSpheres(self, brws):
		"""Show each image's rows as spheres (X-Y-Z[-A[-S]]) in the last opened
		3-D window (new window if none); each image is its own scatter node."""
		brws.busy()
		try:
			brws._show_spheres(self._spheres_data(), self.path, new=False)
		finally:
			brws.notbusy()

	def showSpheresNew(self, brws):
		"""Open a new 3-D window with each image's rows as spheres."""
		brws.busy()
		try:
			brws._show_spheres(self._spheres_data(), self.path, new=True)
		finally:
			brws.notbusy()

	def _spheres_data(self):
		"""A 2-D image stack with 3-5 columns: the rows of each image are its
		points (X-Y-Z[-A[-S]]); each image is one point set (as in EMAN2)."""
		io = ImageIO(self.path, "r")
		sets = []
		try:
			for i in range(io.nimg):
				img, _h = io.read_image(i)
				arr = np.asarray(img, dtype=np.float64)
				if arr.ndim == 3:
					arr = arr.reshape(arr.shape[0], -1)
				if arr.ndim != 2 or arr.shape[1] < 3:
					raise ValueError("need at least 3 columns (X-Y-Z) in %s" % self.path)
				sets.append(arr)  # each row of the image -> a point
		finally:
			io.close()
		if not sets:
			raise ValueError("no images in %s" % self.path)
		return sets

	def showFilterTool(self, brws): self._unimpl("showFilterTool")
	def plot2dLstApp(self, brws): self._unimpl("plot2dLstApp")
	def plot2dLstNew(self, brws): self._unimpl("plot2dLstNew")

	def plot_data(self):
		"""A 1-D image stack plots as one dataset per image (the pixels of the
		image as the series values), as in EMAN2."""
		io = ImageIO(self.path, "r")
		try:
			cols = []
			for i in range(io.nimg):
				img, _h = io.read_image(i)
				cols.append(np.asarray(img).ravel().tolist())
		finally:
			io.close()
		if not cols:
			return None, None
		return cols, ["image %d" % i for i in range(len(cols))]


class EMPDBFileType(EMFileType):
	@staticmethod
	def name():
		return "PDB"

	@staticmethod
	def infoClass():
		return EMPDBInfoPane

	def actions(self):
		"""Order matches the EMAN2 browser (top-left button = double-click)."""
		return [("Show Ball and Stick", "Show ball and stick representation of this PDB model in a new 3D window", self.showBallStick3dNew),
			("Show Ball and Stick +", "Show ball and stick representation of this PDB model in the current 3D window", self.showBallStick3dApp),
			("Show Spheres", "Show spheres representation of this PDB model in a new 3D window", self.showSpheres3dNew),
			("Show Spheres +", "Show spheres representation of this PDB model in the current 3D window", self.showSpheres3dApp),
			("Save As", "Saves a copy of the selected PDB file", self.saveAs)]

	def showSpheres3dApp(self, brws): self._unimpl("showSpheres3dApp")
	def showSpheres3dNew(self, brws): self._unimpl("showSpheres3dNew")
	def showBallStick3dApp(self, brws): self._unimpl("showBallStick3dApp")
	def showBallStick3dNew(self, brws): self._unimpl("showBallStick3dNew")
	def saveAs(self, brws):
		brws._copy_file_as(self.path)


# Register file types (keys must match the "type" strings from the updater)
EMFileType.typesbyft = {
	"Folder": EMFolderFileType,
	"Image": EMImageFileType,
	"Image Stack": EMStackFileType,
	"Text": EMTextFileType,
	"Data": EMPlotFileType,
	"HTML": EMHTMLFileType,
	"PDF": EMPDFFileType,
	"JSON": EMJSONFileType,
	"PDB": EMPDBFileType,
	"BDB": EMBdbFileType,
	"Plot": EMPlotFileType,
}


# ===========================================================================
# Directory entry + item model
# ===========================================================================

class EMDirEntry(object):
	"""Represents a directory entry in the filesystem. Column metadata is drawn
	from the precomputed browser cache (see e3browsercache), not from in-process
	file probing."""

	col = (lambda x: int(x.index), lambda x: x.name, lambda x: x.filetype if x.filetype != None else "",
		lambda x: size_sortable(x.size), lambda x: str(x.dim), lambda x: safe_int(x.nimg), lambda x: x.date)

	def __init__(self, root, name, index, parent=None, hidedot=True, dirregex=None):
		self.__parent = parent
		self.__children = None
		self.dirregex = dirregex
		self.root = str(root)
		self.name = str(name)
		self.index = str(index)
		self.hidedot = hidedot

		if self.root[-1] == "/" or self.root[-1] == "\\":
			self.root = self.root[:-1]

		self.filepath = os.path.join(self.root, self.name)

		try:
			stat = os.stat(self.filepath)
		except Exception:
			stat = (0, 0, 0, 0, 0, 0, 0, 0, 0)

		self.size = stat[6]
		self.date = local_datetime(stat[8])

		self.dim = None
		self.filetype = None
		self.nimg = None

		if os.path.isdir(self.filepath):
			self.filetype = "Folder"
			self.dim = ""
			self.nimg = ""
			self.size = ""

	def __repr__(self):
		return "<%s %s>" % (self.__class__.__name__, self.path())

	def path(self):
		return os.path.join(self.root, self.name).replace("\\", "/")

	def truepath(self):
		return os.path.join(self.root, self.name).replace("\\", "/")

	def fileTypeClass(self):
		return EMFileType.typesbyft.get(self.filetype)

	def sort(self, column, order):
		if self.__children == None or len(self.__children) == 0 or isinstance(self.__children[0], str):
			return
		self.__children.sort(key=self.__class__.col[column], reverse=order)
		for i, child in enumerate(self.__children):
			child.index = i

	def findSelected(self, ret):
		if self.__children == None or len(self.__children) == 0 or isinstance(self.__children[0], str):
			return
		for i, child in enumerate(self.__children):
			try:
				if child.sel:
					child.sel = False
					ret.append((self, i))
			except Exception:
				pass
			child.findSelected(ret)

	def parent(self):
		return self.__parent

	def child(self, n):
		self.fillChildEntries()
		try:
			return self.__children[n]
		except Exception:
			print("Request for child %d of children %s (%d)" % (n, self.__children, len(self.__children)))
			traceback.print_stack()
			raise

	def nChildren(self):
		self.fillChildNames()
		return len(self.__children)

	def fillChildNames(self):
		if self.__children == None:
			if not os.path.isdir(self.filepath):
				self.__children = []
				return
			if self.hidedot:
				filelist = [i for i in os.listdir(self.filepath) if i[0] != '.']
			else:
				filelist = os.listdir(self.filepath)

			self.__children = []
			if self.dirregex != None:
				for child in filelist:
					ctt = self.filepath + "/" + child
					have_dir = os.path.isdir(ctt)
					if not (have_dir or os.path.isfile(ctt) or os.path.islink(ctt)):
						continue
					if isinstance(self.dirregex, str):
						chl = (child + ".dir") if have_dir else child
						try:
							matching = bool(re.search(self.dirregex, chl))
						except Exception:
							matching = False
						have_dir = False
					else:
						matching = self.dirregex.match(child) != None
					if have_dir or matching:
						self.__children.append(child)
			else:
				self.__children = filelist

			if "EMAN2DB" in self.__children:
				self.__children.remove("EMAN2DB")

			self.__children.sort()

	def fillChildEntries(self):
		if self.__children == None:
			self.fillChildNames()
		if len(self.__children) == 0:
			return
		if not isinstance(self.__children[0], str):
			return
		for i, n in enumerate(self.__children):
			self.__children[i] = self.__class__(self.filepath, n, i, self)

	def reload_meta(self):
		"""Reset and re-fill the metadata for this (file) entry from the cache."""
		if self.filetype == "Folder":
			return False
		self.filetype = None
		return bool(self.fillDetails())

	def refresh_meta_tree(self):
		"""Refresh metadata for already-instantiated file children, recursing into
		already-instantiated (expanded) subfolders. Returns True if anything
		changed; children that are not yet instantiated (collapsed folders) are
		left alone."""
		if self.__children is None or len(self.__children) == 0 or isinstance(self.__children[0], str):
			return False
		changed = False
		for c in self.__children:
			if c.filetype == "Folder":
				if c.refresh_meta_tree():
					changed = True
			elif c.reload_meta():
				changed = True
		return changed

	def file_children(self):
		"""The already-instantiated non-folder children of this entry."""
		self.fillChildEntries()
		return [c for c in self.__children if isinstance(c, self.__class__) and c.filetype != "Folder"]

	def fillDetails(self):
		"""Fill in the (precomputed) metadata for this entry from the browser cache.
		Returns 1 if metadata was applied, 0 if nothing was done/found."""
		if self.filetype != None:
			return 0
		if os.path.isdir(self.filepath):
			self.filetype = "Folder"
			self.dim = ""
			self.nimg = ""
			self.size = ""
			return 1
		if not (os.path.isfile(self.filepath) or os.path.islink(self.filepath)):
			self.filetype = "SPECIAL"
			return 0

		_ts, entries = _load_folder_meta(self.root)
		meta = entries.get(self.name)
		if meta is None:
			self.filetype = "-"
			self.dim = "-"
			self.nimg = "-"
			return 1
		self.filetype = meta.get("type") or "-"
		self.dim = meta.get("dim")
		self.nimg = meta.get("nimg")
		if meta.get("size") is not None:
			self.size = meta.get("size")
		return 1


class EMFileItemModel(QtCore.QAbstractItemModel):
	"""Item model representing the local filesystem, with the extra image
	columns supplied from the precomputed browser cache."""

	headers = ("Row", "Name", "Type", "Size", "Dim", "N", "Date")

	def __init__(self, startpath=None, direntryclass=EMDirEntry, dirregex=None):
		QtCore.QAbstractItemModel.__init__(self)
		if startpath[:2] == "./":
			startpath = startpath[2:]
		self.root = direntryclass(startpath, "", 0, dirregex=dirregex)
		self.rootpath = startpath
		self.last = (0, 0)

	def canFetchMore(self, idx):
		return False

	def columnCount(self, parent):
		return 7

	def rowCount(self, parent):
		if parent != None and parent.isValid():
			return parent.internalPointer().nChildren()
		return self.root.nChildren()

	def data(self, index, role):
		if not index.isValid():
			return None
		if role != Qt.DisplayRole:
			return None
		data = index.internalPointer()
		if data == None:
			return "XXX"
		col = index.column()
		if col == 0:
			return nonone(data.index)
		elif col == 1:
			return nonone(data.name)
		elif col == 2:
			return nonone(data.filetype)
		elif col == 3:
			return humansize(data.size)
		elif col == 4:
			if data.dim == 0:
				return "-"
			return nonone(data.dim)
		elif col == 5:
			return nonone(data.nimg)
		elif col == 6:
			return nonone(data.date)
		return None

	def headerData(self, sec, orient, role):
		if orient == Qt.Horizontal:
			if role == Qt.DisplayRole:
				return self.__class__.headers[sec]
		return None

	def hasChildren(self, parent):
		try:
			if parent != None and parent.isValid():
				return parent.internalPointer().nChildren() > 0
			return True
		except Exception:
			return False

	def hasIndex(self, row, col, parent):
		try:
			if parent != None and parent.isValid():
				data = parent.internalPointer().child(row)
			else:
				data = self.root.child(row)
		except Exception:
			traceback.print_exc()
			return False
		return True

	def index(self, row, column, parent):
		try:
			if parent != None and parent.isValid():
				data = parent.internalPointer().child(row)
			else:
				data = self.root.child(row)
		except Exception:
			traceback.print_exc()
			return QtCore.QModelIndex()
		return self.createIndex(row, column, data)

	def parent(self, index):
		if index.isValid():
			try:
				data = index.internalPointer().parent()
			except Exception:
				data = None
		else:
			return QtCore.QModelIndex()
		if data == None:
			return QtCore.QModelIndex()
		return self.createIndex(index.row(), 0, data)

	def sort(self, column, order):
		if column < 0:
			return
		self.root.sort(column, order)
		self.layoutChanged.emit()

	def findSelected(self, toplevel=True):
		sel = []
		self.root.findSelected(sel)
		if toplevel:
			return [self.createIndex(i[1], 0, i[0]) for i in sel if i[0] == self.root]
		return [self.createIndex(i[1], 0, i[0]) for i in sel]

	def details(self, index):
		if not index.isValid():
			return
		if index.internalPointer().fillDetails():
			self.dataChanged.emit(index, self.createIndex(index.row(), 5, index.internalPointer()))


class SortSelTree(QtWidgets.QTreeView):
	"""QTreeView with selection-preserving sort-by-column support."""

	def __init__(self, parent=None):
		QtWidgets.QTreeView.__init__(self, parent)
		self.header().setSectionsClickable(True)
		self.header().sectionClicked.connect(self.colclick)
		self.scol = -1
		self.sdir = 1

	def setSortingEnabled(self, enable):
		return  # always enabled

	def colclick(self, col):
		if col == self.scol:
			self.sdir ^= 1
		else:
			self.scol = col
			self.sdir = 1
		self.header().setSortIndicator(self.scol, self.sdir)
		self.header().setSortIndicatorShown(True)
		self.sortByColumn(self.scol, self.sdir)

	def sortByColumn(self, col, ascend):
		if col == -1:
			return
		try:
			for s in self.selectedIndexes():
				if s.column() == 0:
					s.internalPointer().sel = True
		except Exception:
			pass

		QtWidgets.QTreeView.sortByColumn(self, col, ascend)

		sel = self.model().findSelected()
		if len(sel) == 0:
			return
		qis = QtCore.QItemSelection()
		for i in sel:
			qis.select(i, i)
		self.selectionModel().select(qis, QtCore.QItemSelectionModel.ClearAndSelect | QtCore.QItemSelectionModel.Rows)


# ===========================================================================
# Info window
# ===========================================================================

class EMInfoWin(QtWidgets.QWidget):
	winclosed = QtCore.Signal()

	def __init__(self, parent=None):
		QtWidgets.QWidget.__init__(self, parent)
		self.target = None
		self.stack = QtWidgets.QStackedLayout(self)
		self.stack.addWidget(EMInfoPane())

	def set_target(self, target, ftype):
		self.target = target
		if target == None:
			self.stack.setCurrentIndex(0)
			return
		if hasattr(ftype, "infoClass"):
			cls = ftype.infoClass()
		else:
			return
		for i in range(self.stack.count()):
			w = self.stack.widget(i)
			if isinstance(w, cls):
				self.stack.setCurrentIndex(i)
				w.display(target)
				break
		else:
			if cls == None:
				print("No class ! (%s)" % str(ftype))
				return
			pane = cls()
			i = self.stack.addWidget(pane)
			pane.display(target)
			self.stack.setCurrentIndex(i)

	def closeEvent(self, event):
		super().closeEvent(event)
		self.winclosed.emit()


# ===========================================================================
# The browser widget
# ===========================================================================

class EMBrowserWidget(QtWidgets.QWidget):
	"""A file browser for EMAN3. In addition to being a regular file browser it
	- shows precomputed metadata (Type/Dim/N/Size) from the browser cache
	- stays fully interactive while that metadata is (re)computed in a separate
	  process
	- provides an 'Update' button that launches the standalone updater"""
	ok = QtCore.Signal()
	cancel = QtCore.Signal()
	module_closed = QtCore.Signal()

	def __init__(self, parent=None, withmodal=False, multiselect=False, startpath=".", setsmode=None, dirregex=""):
		QtWidgets.QWidget.__init__(self, parent)

		cwd = os.path.basename(os.getcwd())
		self.setWindowTitle("%s - e3display" % cwd)

		self.withmodal = withmodal
		self.multiselect = multiselect

		self.resize(780, 580)
		self.gbl = QtWidgets.QGridLayout(self)

		# Top toolbar
		self.wtoolhbl = QtWidgets.QHBoxLayout()
		self.wtoolhbl.setContentsMargins(0, 0, 0, 0)

		self.wbutback = QtWidgets.QPushButton(chr(0x2190))
		self.wbutback.setMaximumWidth(36)
		self.wbutback.setEnabled(False)
		self.wtoolhbl.addWidget(self.wbutback, 0)

		self.wbutfwd = QtWidgets.QPushButton(chr(0x2192))
		self.wbutfwd.setMaximumWidth(36)
		self.wbutfwd.setEnabled(False)
		self.wtoolhbl.addWidget(self.wbutfwd, 0)

		self.lpath = QtWidgets.QLabel("  Path:")
		self.wtoolhbl.addWidget(self.lpath)

		self.wpath = QtWidgets.QLineEdit()
		self.wtoolhbl.addWidget(self.wpath, 5)

		self.wbutinfo = QtWidgets.QPushButton("Info")
		self.wbutinfo.setCheckable(True)
		self.wtoolhbl.addWidget(self.wbutinfo, 1)

		self.gbl.addLayout(self.wtoolhbl, 0, 0, 1, 2)

		# 2nd toolbar
		self.wtoolhbl2 = QtWidgets.QHBoxLayout()
		self.wtoolhbl2.setContentsMargins(0, 0, 0, 0)

		self.wbutup = QtWidgets.QPushButton(chr(0x2191))
		self.wbutup.setMaximumWidth(36)
		self.wtoolhbl2.addWidget(self.wbutup, 0)

		self.wbutrefresh = QtWidgets.QPushButton(chr(0x21ba))
		self.wbutrefresh.setMaximumWidth(36)
		self.wbutrefresh.setToolTip("Update: re-scan this folder and refresh file metadata")
		self.wtoolhbl2.addWidget(self.wbutrefresh, 0)

		self.lfilter = QtWidgets.QLabel("Filter:")
		self.wtoolhbl2.addWidget(self.lfilter)

		self.wfilter = QtWidgets.QComboBox()
		self.wfilter.setEditable(True)
		self.wfilter.setInsertPolicy(QtWidgets.QComboBox.InsertAtBottom)
		self.wfilter.addItem("")
		self.wfilter.addItem(r"(.(?!_ctf))*$")
		self.wfilter.addItem(r".*\.img")
		self.wfilter.addItem(r".*\.hdf")
		self.wfilter.addItem(r".*_ptcls$")
		self.wfilter.addItem(r".*\.mrc")
		self.wfilter.addItem(r".*\.tif")
		self.wfilter.addItem(r".*\.pdb")
		self.wfilter.addItem("help")
		self.wtoolhbl2.addWidget(self.wfilter, 5)
		if dirregex != "":
			self.wfilter.setEditText(dirregex)

		self.selectall = QtWidgets.QPushButton("Sel All")
		self.wtoolhbl2.addWidget(self.selectall, 1)
		self.selectall.setEnabled(withmodal)

		self.gbl.addLayout(self.wtoolhbl2, 1, 0, 1, 2)

		# Bookmarks
		self.wbookmarkfr = QtWidgets.QFrame()
		self.wbookmarkfr.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.wbookmarkfr.setFrameShadow(QtWidgets.QFrame.Raised)
		self.wbmfrbl = QtWidgets.QVBoxLayout(self.wbookmarkfr)
		self.wbookmarks = QtWidgets.QToolBar()
		self.wbookmarks.setOrientation(Qt.Orientation.Vertical)
		self.addBookmark("Root", "/")
		self.addBookmark("Current", os.getcwd())
		self.addBookmark("Home", home_dir())
		self.wbmfrbl.addWidget(self.wbookmarks)
		self.gbl.addWidget(self.wbookmarkfr, 2, 0)

		# Main tree
		self.wtree = SortSelTree()
		if multiselect:
			self.wtree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
		else:
			self.wtree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
		self.wtree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
		self.wtree.setAllColumnsShowFocus(True)
		self.wtree.sortByColumn(-1, 0)
		self.gbl.addWidget(self.wtree, 2, 1)

		# Lower action-button region (10 programmable slots)
		self.hbl2 = QtWidgets.QGridLayout()
		self.wbutmisc = []
		self.hbl2.setRowStretch(0, 1)
		self.hbl2.setRowStretch(1, 1)
		for i in range(5):
			self.hbl2.setColumnStretch(i, 2)
			for j in range(2):
				self.wbutmisc.append(QtWidgets.QPushButton("-"))
				self.hbl2.addWidget(self.wbutmisc[-1], j, i)
				self.wbutmisc[-1].setEnabled(False)
				self.wbutmisc[-1].clicked.connect(lambda x, v=i * 2 + j: self.buttonMisc(v))
		self.hbl2.setColumnStretch(6, 1)

		# Modal OK/Cancel
		if withmodal:
			self.wbutcancel = QtWidgets.QPushButton("Cancel")
			self.hbl2.addWidget(self.wbutcancel, 1, 7)
			self.wbutok = QtWidgets.QPushButton("OK")
			self.hbl2.addWidget(self.wbutok, 1, 8)
			self.wbutcancel.clicked.connect(self.buttonCancel)
			self.wbutok.clicked.connect(self.buttonOk)

		self.gbl.addLayout(self.hbl2, 4, 1)

		self.wbutback.clicked.connect(self.buttonBack)
		self.wbutfwd.clicked.connect(self.buttonFwd)
		self.wbutup.clicked.connect(self.buttonUp)
		self.wbutrefresh.clicked.connect(self.buttonRefresh)
		self.wbutinfo.clicked.connect(self.buttonInfo)
		self.selectall.clicked.connect(self.selectAll)
		self.wtree.clicked.connect(self.itemSel)
		self.wtree.activated.connect(self.itemActivate)
		self.wtree.doubleClicked.connect(self.itemDoubleClick)
		self.wtree.expanded.connect(self.itemExpand)
		self.wpath.returnPressed.connect(self.editPath)
		self.wbookmarks.actionTriggered.connect(self.bookmarkPress)
		self.wfilter.currentIndexChanged.connect(self.editFilter)

		self.setsmode = setsmode
		self.curmodel = None
		self.curpath = None
		self.curft = None
		self.curactions = []
		self.pathstack = []
		self.infowin = None

		self.view2d = []
		self.view2ds = []
		self.view3d = []
		self.viewplot2d = []
		self.viewplot3d = []
		self.viewhist = []

		self.needresize = 0
		self.expanded = set()

		# Metadata cache polling (replaces the old in-process update thread).
		# _watched: absolute folder path -> last-seen cache timestamp, for the
		# current folder and every expanded subfolder. _updaters: (folder, Popen)
		# for standalone updater processes currently running.
		self._watched = {}
		self._updaters = []
		self._cache_timer = QtCore.QTimer()
		self._cache_timer.timeout.connect(self.check_cache)

		self.setPath(startpath)
		self._cache_timer.start(1000)

		self.result = None

	def busy(self):
		QtWidgets.QApplication.instance().setOverrideCursor(Qt.BusyCursor)

	def notbusy(self):
		QtWidgets.QApplication.instance().setOverrideCursor(Qt.ArrowCursor)

	# ---- 2-D plot windows ------------------------------------------------------

	def _drop_view(self, lst, w):
		try:
			lst.remove(w)
		except ValueError:
			pass

	def _track_plot2d(self, target):
		self.viewplot2d.append(target)
		target._plot_dead = False
		target.destroyed.connect(lambda *_: setattr(target, "_plot_dead", True))
		target.destroyed.connect(lambda *_: self._drop_view(self.viewplot2d, target))

	def _last_plot2d(self):
		"""The last opened 2-D plot window that is still alive, or None."""
		while self.viewplot2d:
			w = self.viewplot2d[-1]
			if getattr(w, "_plot_dead", False):
				self.viewplot2d.pop()
				continue
			return w
		return None

	def _show_plot2d(self, data, path, new=False, column_labels=None):
		"""Show 'data' (a list of column value lists) in a 2-D plot window keyed
		by the file name. new=False reuses the last opened plot window (creating
		one if none); new=True always opens a new window."""
		from EMAN3.gui.emplot2d import EMPlot2DWidget
		key = os.path.basename(path)
		if new:
			target = EMPlot2DWidget()
			self._track_plot2d(target)
		else:
			target = self._last_plot2d()
			if target is None:
				target = EMPlot2DWidget()
				self._track_plot2d(target)
		target.set_data(data, key=key, column_labels=column_labels)
		target.setWindowTitle(key)
		target.show()
		target.raise_()

	def _track_plot3d(self, target):
		self.viewplot3d.append(target)
		target._plot_dead = False
		target.destroyed.connect(lambda *_: setattr(target, "_plot_dead", True))
		target.destroyed.connect(lambda *_: self._drop_view(self.viewplot3d, target))

	def _last_plot3d(self):
		"""The last opened 3-D plot window that is still alive, or None."""
		while self.viewplot3d:
			w = self.viewplot3d[-1]
			if getattr(w, "_plot_dead", False):
				self.viewplot3d.pop()
				continue
			return w
		return None

	def _show_plot3d(self, data, path, new=False, column_labels=None):
		"""Show 'data' (a list of column value lists) in a 3-D plot window keyed
		by the file name. new=False reuses the last opened plot window (creating
		one if none); new=True always opens a new window."""
		from EMAN3.gui.emplot3d import EMPlot3DWidget
		key = os.path.basename(path)
		if new:
			target = EMPlot3DWidget()
			self._track_plot3d(target)
		else:
			target = self._last_plot3d()
			if target is None:
				target = EMPlot3DWidget()
				self._track_plot3d(target)
		target.set_data(data, key=key, column_labels=column_labels)
		target.setWindowTitle(key)
		target.show()
		target.raise_()

	def _track_hist(self, target):
		self.viewhist.append(target)
		target._plot_dead = False
		target.destroyed.connect(lambda *_: setattr(target, "_plot_dead", True))
		target.destroyed.connect(lambda *_: self._drop_view(self.viewhist, target))

	def _last_hist(self):
		"""The last opened histogram window that is still alive, or None."""
		while self.viewhist:
			w = self.viewhist[-1]
			if getattr(w, "_plot_dead", False):
				self.viewhist.pop()
				continue
			return w
		return None

	def _show_hist(self, data, path, new=False, column_labels=None):
		"""Show 'data' (a list of column value lists) as distributions in a
		histogram window keyed by the file name. new=False reuses the last
		opened histogram window (creating one if none); new=True always opens
		a new window."""
		from EMAN3.gui.emhist import EMHistogramWidget
		key = os.path.basename(path)
		if new:
			target = EMHistogramWidget()
			self._track_hist(target)
		else:
			target = self._last_hist()
			if target is None:
				target = EMHistogramWidget()
				self._track_hist(target)
		target.set_data(data, key=key, column_labels=column_labels)
		target.setWindowTitle(key)
		target.show()
		target.raise_()

	def _track_view3d(self, target):
		self.view3d.append(target)
		target._plot_dead = False
		target.destroyed.connect(lambda *_: setattr(target, "_plot_dead", True))
		target.destroyed.connect(lambda *_: self._drop_view(self.view3d, target))

	def _show_spheres(self, point_sets, path, new=False):
		"""Add scatter plot(s) to a 3-D scene window keyed by the file name,
		like Show 3D / Show 3D+. 'point_sets' is a list of Nx3/Nx4 arrays
		(X-Y-Z[-amplitude]); each set becomes its own scatter plot node (a
		stack of images contributes one node per image, as in EMAN2). new=False
		adds to the last opened 3-D scene window (creating one if none);
		new=True always opens a new window. Returns the window."""
		from EMAN3.gui.emscene3d import EMScene3DWidget
		key = os.path.basename(path)
		if new:
			target = EMScene3DWidget()
			self._track_view3d(target)
		else:
			target = self._last_view3d()
			if target is None:
				target = EMScene3DWidget()
				self._track_view3d(target)
		nsets = len(point_sets)
		for k, pts in enumerate(point_sets):
			pts = np.asarray(pts, dtype=np.float32)
			if pts.ndim == 1:
				pts = pts[np.newaxis, :]
			if pts.shape[1] == 3:
				pts = np.column_stack([pts, np.ones(len(pts), dtype=np.float32)])
			elif pts.shape[1] > 4:
				pts = pts[:, :4]
			name = key if nsets == 1 else "%s #%d" % (key, k)
			target.add_scatter_plot(name=name, data=pts)
		# The widget fits the camera to the first object added to the scene;
		# subsequent additions preserve the user's current view.
		target.setWindowTitle(key)
		target.show()
		target.raise_()
		return target

	def _last_view3d(self):
		"""The last opened 3-D scene window that is still alive, or None."""
		while self.view3d:
			w = self.view3d[-1]
			if getattr(w, "_plot_dead", False):
				self.view3d.pop()
				continue
			return w
		return None

	def _read_volumes(self, path, first_only=False):
		"""Read an image file as a list of (data, header) 3-D volumes. Volumes
		are stored (nz, ny, nx); 2-D members are expanded to nz=1."""
		io = ImageIO(path, "r")
		vols = []
		try:
			nimg = io.nimg or 1
			if first_only:
				nimg = 1
			for i in range(nimg):
				img, header = io.read_image(i)
				arr = np.asarray(img, dtype=np.float32)
				if arr.ndim == 2:
					arr = arr[np.newaxis, :, :]
				vols.append((arr, header))
		finally:
			io.close()
		return vols

	def _add_volume_to_scene(self, widget, name, data, header, path):
		"""Add one volume as a data node with an isosurface child (visible)
		and an X-Y slice child (created but hidden initially). Returns the
		isosurface node."""
		root = widget.get_scene_root()
		data_node = widget.add_data(name=name, data=data, header=header,
			filename=path, parent=root)
		iso = widget.add_isosurface(name="Isosurface", parent=data_node)
		slice_node = widget.add_slice(name="XY slice", parent=data_node)
		slice_node.visible = False
		widget.set_selected_node(iso)
		return iso

	def _show_scene3d(self, vols, path, new=False):
		"""Show 3-D volume(s) in a 3-D scene window keyed by the file name.
		Each volume becomes a data node with an isosurface child (visible)
		and an X-Y slice child (initially hidden). new=False adds to the last
		opened 3-D scene window (creating one if none); new=True always opens
		a new window. Returns the window."""
		from EMAN3.gui.emscene3d import EMScene3DWidget
		key = os.path.basename(path)
		if new:
			target = EMScene3DWidget()
			self._track_view3d(target)
		else:
			target = self._last_view3d()
			if target is None:
				target = EMScene3DWidget()
				self._track_view3d(target)
		for i, (data, header) in enumerate(vols):
			name = key if len(vols) == 1 else "%s #%d" % (key, i)
			self._add_volume_to_scene(target, name, data, header, path)
		# The widget fits the camera to the first object added to the scene;
		# subsequent additions preserve the user's current view.
		target.setWindowTitle(key)
		target.show()
		target.raise_()
		return target

	def _read_2d_list(self, path):
		"""Read an image file as a list of 2-D numpy arrays: a 2-D image is
		[itself], a 3-D volume is its z-slices, and a stack is its images
		(3-D members expanded to their z-slices)."""
		io = ImageIO(path, "r")
		images = []
		try:
			nimg = io.nimg or 1
			for i in range(nimg):
				img, _header = io.read_image(i)
				arr = np.asarray(img, dtype=np.float32)
				if arr.ndim == 2:
					images.append(arr)
				elif arr.ndim == 3:
					# volumes are stored (nz, ny, nx): z-slices are axis 0
					for z in range(arr.shape[0]):
						images.append(arr[z, :, :])
		finally:
			io.close()
		return images

	def _track_view2d(self, target):
		self.view2d.append(target)
		target._plot_dead = False
		target.destroyed.connect(lambda *_: setattr(target, "_plot_dead", True))
		target.destroyed.connect(lambda *_: self._drop_view(self.view2d, target))

	def _last_view2d(self):
		"""The last opened 2-D image window that is still alive, or None."""
		while self.view2d:
			w = self.view2d[-1]
			if getattr(w, "_plot_dead", False):
				self.view2d.pop()
				continue
			return w
		return None

	def _show_image2d(self, images, path, new=False):
		"""Show 'images' (list of 2-D arrays) in a 2-D image window. Unlike the
		plot widgets, the new file replaces the previous contents of a window."""
		from EMAN3.gui.emimage2d import EMImage2DWidget
		if new:
			target = EMImage2DWidget()
			self._track_view2d(target)
		else:
			target = self._last_view2d()
			if target is None:
				target = EMImage2DWidget()
				self._track_view2d(target)
		target.set_data(images[0] if len(images) == 1 else images, file_name=path)
		target.show()
		target.raise_()

	def _track_view2ds(self, target):
		self.view2ds.append(target)
		target._plot_dead = False
		target.destroyed.connect(lambda *_: setattr(target, "_plot_dead", True))
		target.destroyed.connect(lambda *_: self._drop_view(self.view2ds, target))

	def _last_view2ds(self):
		"""The last opened stack (grid) window that is still alive, or None."""
		while self.view2ds:
			w = self.view2ds[-1]
			if getattr(w, "_plot_dead", False):
				self.view2ds.pop()
				continue
			return w
		return None

	def _show_imagemx(self, images, path, new=False):
		"""Show 'images' (list of 2-D arrays) as a grid in a stack window. The
		new file replaces the previous contents of a window."""
		from EMAN3.gui.emimagemx import EMImageMXWidget
		if new:
			target = EMImageMXWidget()
			self._track_view2ds(target)
		else:
			target = self._last_view2ds()
			if target is None:
				target = EMImageMXWidget()
				self._track_view2ds(target)
		target.set_data(images, filename=path)
		target.show()
		target.raise_()

	def _show_xyz_projections(self, path, nimg):
		"""Rng XYZ / All XYZ: ask for slice parameters, compute the restricted
		orthogonal projections of the requested range of volumes, and display
		them. Single-volume (or 'stack output') results go to a 2-D image
		window; otherwise the packed images are shown in a stack grid window."""
		dlg = EMSliceParamDialog(self, nimg)
		if dlg.exec() != QtWidgets.QDialog.Accepted:
			return
		p = dlg.parameters()
		step = max(int(p["step"]), 1)
		img0 = max(int(p["first"]), 0)
		img1 = int(p["last"]) if 0 < int(p["last"]) <= nimg else nimg
		if img1 <= img0:
			display_error("empty range of images to project")
			return

		mask = None
		if p["mask"]:
			try:
				mio = ImageIO(p["mask"], "r")
				mask, _mh = mio.read_image(0)
				mio.close()
				mask = np.asarray(mask, dtype=np.float32)
			except Exception as e:
				print("warning: could not read mask volume %s: %s" % (p["mask"], e))
		ref = None
		if p["reference"]:
			try:
				rio = ImageIO(p["reference"], "r")
				ref, _rh = rio.read_image(0)
				rio.close()
				ref = np.asarray(ref, dtype=np.float32)
				if ref.ndim == 2:
					ref = ref[np.newaxis, :, :]
			except Exception as e:
				print("warning: could not read reference volume %s: %s" % (p["reference"], e))

		self.busy()
		try:
			io = ImageIO(path, "r")
			apix = 1.0
			try:
				h0 = io.read_header(0)
				apix = float(h0.get("apix_x") or h0.get("apix") or 1.0)
			except Exception:
				pass
			images = []
			for i in range(img0, img1, step):
				try:
					vol, _h = io.read_image(i)
				except Exception as e:
					print("warning: could not read image #%d of %s: %s" % (i, path, e))
					continue
				vol = np.asarray(vol, dtype=np.float32)
				if vol.ndim == 2:
					vol = vol[np.newaxis, :, :]
				if mask is not None:
					if mask.shape == vol.shape:
						vol = vol * mask
					else:
						print("warning: mask shape %s does not match volume shape %s; ignoring mask" % (mask.shape, vol.shape))
				out = make_ortho_proj(vol, p["layers"], p["center"],
					p["lowpass"], p["highpass"], apix, p["stack_out"])
				if isinstance(out, list):
					images.extend(out)
				else:
					images.append(out)
				QtWidgets.QApplication.processEvents()
			if ref is not None:
				if mask is not None and mask.shape == ref.shape:
					ref = ref * mask
				out = make_ortho_proj(ref, p["layers"], p["center"],
					p["lowpass"], p["highpass"], apix, p["stack_out"])
				if isinstance(out, list):
					images.extend(out)
				else:
					images.append(out)
			io.close()
		finally:
			self.notbusy()
		if not images:
			display_error("no images could be projected")
			return
		if nimg == 1 or p["stack_out"]:
			self._show_image2d(images, path, new=not p["same_window"])
		else:
			self._show_imagemx(images, path, new=True)

	# ---- Save As -------------------------------------------------------------

	# Output formats that can only hold a single image; saving more than one
	# image to such a format produces a numbered file set (output_0000.jpg, ...).
	SINGLE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".pgm"}

	def _save_as(self, path, nimg):
		"""Save As: copy the selected image(s) of 'path' into a new file, in the
		format of the user's choice (or determined by the output extension).
		Single-image formats (e.g. JPEG) with N > 1 images produce a numbered
		file set: output_0000.jpg, output_0001.jpg, ..."""
		dlg = EMSaveAsDialog(self, default_name=os.path.basename(path), nimg=nimg)
		if dlg.exec() != QtWidgets.QDialog.Accepted:
			return
		p = dlg.parameters()
		out = p["file"] or path
		if p["format"] is not None:
			root, _cur = os.path.splitext(out)
			out = root + p["format"]
		if not os.path.isabs(out):
			out = os.path.join(os.path.dirname(os.path.abspath(path)), out)
		if os.path.normcase(os.path.normpath(out)) == os.path.normcase(os.path.normpath(path)):
			display_error("output file is the same as the input file")
			return

		first = max(0, min(int(p["first"]), nimg - 1))
		last = max(first, min(int(p["last"]), nimg - 1))

		self.busy()
		try:
			io = ImageIO(path, "r")
			images = []
			header = None
			for i in range(first, last + 1):
				data, hdr = io.read_image(i)
				images.append(_reduce_bits(np.asarray(data), int(p.get("bits", -1))))
				if header is None:
					header = hdr
			io.close()
		except Exception as e:
			self.notbusy()
			display_error("could not read %s: %s" % (path, e))
			return

		n = len(images)
		if n == 0:
			self.notbusy()
			display_error("no images to save")
			return
		root, ext = os.path.splitext(out)
		if n > 1 and ext.lower() in self.SINGLE_IMAGE_EXTS:
			files = ["%s_%04d%s" % (root, k, ext) for k in range(n)]
		else:
			files = [out]
		existing = [f for f in files if os.path.exists(f)]
		if existing:
			self.notbusy()
			msg = ("The following file(s) already exist and will be overwritten:\n"
				+ "\n".join(existing) + "\nContinue?")
			if QtWidgets.QMessageBox.question(None, "Save As", msg) != QtWidgets.QMessageBox.Yes:
				return

		self.busy()
		try:
			if len(files) == 1 and n > 1:
				# stack-capable format: all images into one file
				wio = ImageIO(files[0], "rw")
				try:
					for k, data in enumerate(images):
						wio.write_image(k, data, header)
				finally:
					wio.close()
			else:
				for fn, data in zip(files, images):
					wio = ImageIO(fn, "rw")
					try:
						wio.write_image(0, data, header)
					finally:
						wio.close()
					QtWidgets.QApplication.processEvents()
		except Exception as e:
			self.notbusy()
			display_error("could not write %s: %s" % (files[0], e))
			return
		self.notbusy()

	def _copy_file_as(self, path):
		"""Save a copy of a non-image file (e.g. a PDB model) to a new location."""
		dirname = os.path.dirname(os.path.abspath(path))
		fn = QtWidgets.QFileDialog.getSaveFileName(self, "Save As", dirname, os.path.basename(path))[0]
		if not fn:
			return
		if os.path.normcase(os.path.normpath(fn)) == os.path.normcase(os.path.normpath(path)):
			display_error("output file is the same as the input file")
			return
		if os.path.exists(fn) and QtWidgets.QMessageBox.question(
				None, "Save As", "%s already exists. Overwrite?" % fn) != QtWidgets.QMessageBox.Yes:
			return
		try:
			shutil.copy2(path, fn)
		except Exception as e:
			display_error("could not copy %s to %s: %s" % (path, fn, e))

	# ---- metadata cache polling ------------------------------------------------

	def _current_cache_ts(self):
		ts, _e, _p, _v = browser_cache_read(self.curpath)
		return ts

	def check_cache(self):
		"""Periodically check whether the browser cache for the current folder or
		any expanded subfolder has been updated (by the standalone updater) and,
		if so, refresh the metadata columns. Keeps the browser responsive: the
		file list is never blocked."""
		if self.curmodel is None or self.curpath is None:
			return
		changed = False
		for folder, last in list(self._watched.items()):
			try:
				ts, _e, _p, _v = browser_cache_read(folder)
			except Exception:
				continue
			if ts is None or ts == last:
				continue
			self._watched[folder] = ts
			print("%s: browser cache updated for %s, refreshing metadata" % (local_datetime(), folder))
			changed = True
		if changed and self.curmodel.root.refresh_meta_tree():
			self.curmodel.layoutChanged.emit()
			self.wtree.resizeColumnToContents(2)
			self.wtree.resizeColumnToContents(4)
			self.needresize = 2

	def _initial_fill(self):
		"""Fill in metadata for the top-level entries from the cache (instant if a
		cache exists; otherwise columns fill in as soon as the updater completes)."""
		if self.curmodel is None:
			return
		self.curmodel.root.fillChildEntries()
		for c in self.curmodel.root.file_children():
			c.fillDetails()
		self.wtree.resizeColumnToContents(0)
		self.wtree.resizeColumnToContents(1)
		self.wtree.resizeColumnToContents(2)
		self.wtree.resizeColumnToContents(3)
		self.wtree.resizeColumnToContents(4)

	# ---- update (standalone updater) ------------------------------------------

	def _updater_command(self, folder):
		exe = shutil.which("e3browsercache")
		if exe:
			return [exe, folder], None
		return [sys.executable, "-m", "programs.e3browsercache", folder], _REPO_ROOT

	def _cache_is_fresh(self, folder):
		"""True if a browser cache exists for folder, is of the current format
		version, and is no older than the folder itself (i.e. it reflects the
		current contents)."""
		_ts, _e, path, _version = browser_cache_read(folder)
		if _ts is None or _version != BROWSER_CACHE_VERSION:
			return False
		try:
			return os.stat(path).st_mtime >= os.stat(folder).st_mtime - 1.0
		except Exception:
			return True

	def _launch_updater(self, folder=None):
		"""Launch the standalone metadata updater (a separate, non-blocking process)
		for the given folder. Multiple folders may be updated concurrently; a
		folder that already has an updater running is skipped. When an updater
		writes a newer cache, check_cache() refreshes the columns automatically."""
		if folder is None:
			folder = self.curpath
		folder = os.path.abspath(folder)
		self._updaters = [(f, p) for f, p in self._updaters if p.poll() is None]
		for f, _p in self._updaters:
			if f == folder:
				print("Metadata updater already running for %s; skipping." % folder)
				return
		cmd, cwd = self._updater_command(folder)
		try:
			proc = subprocess.Popen(cmd, cwd=cwd,
				stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		except Exception as e:
			display_error("Failed to launch metadata updater: %s" % e)
			return
		self._updaters.append((folder, proc))
		print("%s: launching metadata updater for %s" % (local_datetime(), folder))

	def buttonRefresh(self, tog=None):
		"""Update: refresh the file list and (re)run the standalone metadata
		updater for the current folder."""
		self.setPath(self.curpath)
		self._launch_updater()

	# ---- selection / actions --------------------------------------------------

	def editFilter(self, newfilt):
		self.setPath(str(self.wpath.text()))

	def editPath(self):
		self.setPath(str(self.wpath.text()))

	def itemSel(self, qmi):
		qism = self.wtree.selectionModel().selectedRows()
		if len(qism) > 1:
			self.wpath.setText("<multiple select>")
			if self.infowin != None and not self.infowin.isHidden():
				self.infowin.set_target(None, None)
		elif len(qism) == 1:
			obj = qism[0].internalPointer()
			self.wpath.setText(obj.path())
			self.curmodel.details(qism[0])
			self.wtree.resizeColumnToContents(2)
			self.wtree.resizeColumnToContents(3)

			ftc = obj.fileTypeClass()
			if ftc != None:
				if os.path.exists(obj.truepath()):
					self.curft = ftc(obj.path())
				else:
					print("Error: file %s does not exist..." % obj.path())
					return
				self.curactions = self.curft.actions()
				for i, b in enumerate(self.wbutmisc):
					try:
						b.setText(self.curactions[i][0])
						b.setToolTip(self.curactions[i][1])
						b.setEnabled(True)
					except Exception:
						b.setText("-")
						b.setToolTip("")
						b.setEnabled(False)
			else:
				self.curft = None
				self.curactions = []
				for b in self.wbutmisc:
					b.setText("-")
					b.setToolTip("")
					b.setEnabled(False)

			if self.infowin != None and not self.infowin.isHidden():
				self.infowin.set_target(obj, ftc)

	def itemActivate(self, qmi):
		pass

	def itemDoubleClick(self, qmi):
		itm = qmi.internalPointer()
		if itm.nChildren() > 0:
			self.setPath(itm.path())
		else:
			if self.withmodal and not self.multiselect:
				self.buttonOk(True)
				return
			try:
				self.curactions[0][2](self)
			except Exception:
				pass

	def itemExpand(self, qmi):
		if qmi.internalPointer().path() in self.expanded:
			return
		self.expanded.add(qmi.internalPointer().path())
		if qmi.internalPointer().filetype != "Folder":
			return
		folder = os.path.abspath(qmi.internalPointer().path())
		# Watch this subfolder's cache, and kick off its standalone metadata
		# updater if there is no fresh cache yet (check_cache() refreshes the
		# columns when the updater finishes).
		self._watched.setdefault(folder, None)
		if not self._cache_is_fresh(folder):
			self._launch_updater(folder)
		# fill metadata for newly expanded children from the cache
		for i in range(self.curmodel.rowCount(qmi) - 1, -1, -1):
			idx = self.curmodel.index(i, 0, qmi)
			self.curmodel.details(idx)

	def buttonMisc(self, num):
		try:
			self.curactions[num][2](self)
		except Exception:
			traceback.print_exc()

	def buttonOk(self, tog=None):
		qism = self.wtree.selectionModel().selectedRows()
		self.result = [i.internalPointer().path().replace(os.getcwd(), ".") for i in qism]
		self.ok.emit()

	def buttonCancel(self, tog=None):
		self.result = []
		self.cancel.emit()
		self.close()

	def selectAll(self):
		self.wtree.selectAll()

	def buttonBack(self, tog=None):
		l = self.pathstack.index(self.curpath)
		self.setPath(self.pathstack[(l - 1)], True)
		if l == 1:
			self.wbutback.setEnabled(False)
			self.wbutfwd.setEnabled(True)

	def buttonFwd(self, tog=None):
		l = self.pathstack.index(self.curpath)
		self.setPath(self.pathstack[(l + 1)], True)
		if l == len(self.pathstack) - 2:
			self.wbutback.setEnabled(True)
			self.wbutfwd.setEnabled(False)

	def buttonUp(self, tog=None):
		if "/" in self.curpath:
			newpath = self.curpath.rsplit("/", 1)[0]
		else:
			newpath = os.path.realpath(self.curpath).rsplit("/", 1)[0]
		if len(newpath) > 0:
			self.setPath(newpath)

	def infowinClosed(self):
		self.wbutinfo.setChecked(False)

	def buttonInfo(self, tog=None):
		if tog:
			if self.infowin == None:
				self.infowin = EMInfoWin()
				self.infowin.resize(500, 600)
			self.infowin.show()
			self.infowin.raise_()
			qism = self.wtree.selectionModel().selectedRows()
			if len(qism) == 1:
				self.infowin.set_target(qism[0].internalPointer(), self.curft)
			else:
				self.infowin.set_target(None, None)
			try:
				self.infowin.winclosed.connect(self.infowinClosed)
			except Exception:
				pass
		else:
			if self.infowin != None:
				self.infowin.hide()

	def getResult(self):
		if self.result == None:
			return None
		self.close()
		for i in range(len(self.result)):
			if self.result[i][:2] == "./":
				self.result[i] = self.result[i][2:]
		return self.result

	def getCWD(self):
		if self.result and os.path.isdir(self.result[0]):
			self.close()
			return self.result[0]
		if self.curpath == None:
			return None
		self.close()
		return self.curpath

	def addBookmark(self, label, path):
		act = self.wbookmarks.addAction(label)
		act.setData(path)

	def setPath(self, path, silent=False, inimodel=EMFileItemModel):
		if path != "" and path[0] == ":":
			os.system(path[1:])
			return

		path = os.path.expandvars(path)
		if path == "":
			path = "."
		path = path.replace("\\", "/")
		if path[:2] == "./":
			path = path[2:]

		filt = str(self.wfilter.currentText()).strip()
		if filt != "" and not os.path.isdir(path):
			path = os.path.dirname(path)
			if path == "":
				path = "."

		self.curpath = str(path)
		self.wpath.setText(path)

		if filt == "" or filt == "?" or filt.lower() == "help":
			filt = None
		else:
			try:
				filt = re.compile(filt)
			except Exception:
				filt = filt

		if filt != None and filt != "":
			try:
				self.curmodel = inimodel(path, dirregex=filt)
			except Exception:
				self.curmodel = inimodel(path)
				filt = None
		else:
			self.curmodel = inimodel(path)

		self.wtree.setSortingEnabled(False)
		self.wtree.setModel(self.curmodel)
		self.wtree.setSortingEnabled(True)
		self.wtree.resizeColumnToContents(0)
		self.wtree.resizeColumnToContents(1)

		self.expanded = set()
		self._initial_fill()

		# If there is no fresh cache for this folder, kick off the standalone
		# updater in the background so the metadata columns populate themselves
		# (check_cache() refreshes the display when it finishes). No manual press
		# needed; the 'Update' button remains for forcing a refresh.
		self._watched = {os.path.abspath(self.curpath): self._current_cache_ts()}
		if not self._cache_is_fresh(self.curpath):
			self._launch_updater(self.curpath)

		if not silent:
			try:
				self.pathstack.remove(self.curpath)
			except Exception:
				pass
			self.pathstack.append(self.curpath)
			if len(self.pathstack) > 1:
				self.wbutback.setEnabled(True)
			else:
				self.wbutback.setEnabled(False)
			self.wbutfwd.setEnabled(False)

	def bookmarkPress(self, action):
		self.setPath(action.data())

	def closeEvent(self, event):
		for w in self.view2d + self.view2ds + self.view3d + self.viewplot2d + self.viewplot3d + self.viewhist:
			w.close()
		if self.infowin != None:
			self.infowin.close()
		self._cache_timer.stop()
		event.accept()
		self.module_closed.emit()


def main():
	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
	window = EMBrowserWidget(withmodal=True, multiselect=True, startpath=".")
	window.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
