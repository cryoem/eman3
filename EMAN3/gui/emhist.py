"""
EMAN3 histogram window: distribution of keyed data sets as vertical bars.

A port of the EMAN2 histogram (EMAN3/qtgui/emhist.py) to the new PySide6
system. The EMAN2 version rendered via matplotlib-in-OpenGL and depended on
the EMAN2 infrastructure; this version draws the bars directly with the Qt
painter but keeps the same model and controls:

- keyed data sets (set_data(columns, key=...)), one plotted column per set
  (selectable per set in the inspector)
- per-set color / alpha / bar width
- global nbins, bar alignment, norm, cumulative, log-Y, stacked
- x/y limits with auto-rescale
- middle-click (or 'C') opens the inspector, as with the other EMAN3 widgets
"""
import os
import weakref

import numpy as np

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

# EMAN2 color order/names (first color assigned is index 0)
COLORNAMES = ["black", "blue", "red", "green", "yellow", "cyan", "magenta", "grey"]
COLORS = [(0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0),
          (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 128)]
# black is unreadable on the inspector's dark list background
LIST_COLORS = [(70, 70, 70)] + COLORS[1:]


class EMHistogramWidget(QtWidgets.QWidget):
	"""A window drawing the distribution of keyed data sets as vertical bars."""

	def __init__(self, parent=None):
		super().__init__(parent)
		self.data = {}            # key -> list of 1-D float arrays (one per column)
		self.visibility = {}      # key -> bool
		self.axes = {}            # key -> (col,) the plotted column
		self.pparm = {}           # key -> (color_index, alpha, rwidth)
		self.column_labels = {}  # key -> list of labels (or None)

		self.nbins = 10           # start with 10. user can modify via inspector.
		self.xlimits = None
		self.ylimits = None
		self.histtype = "bar"
		self.orientation = "vertical"
		self.alignment = "edge"
		self.normed = False
		self.cumulative = False
		self.logy = False
		self.stacked = False

		self.inspector = None
		self.resize(640, 480)
		self.setAutoFillBackground(True)
		pal = self.palette()
		pal.setColor(self.backgroundRole(), QColor(240, 240, 240))
		self.setPalette(pal)
		self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)

	# ----- data model -------------------------------------------------------

	def set_data(self, input_data, key="data", replace=False, quiet=False,
				 color=-1, alpha=0.8, rwidth=0.8, column_labels=None):
		"""Set a keyed data set. input_data is a list of columns (each a
		sequence of values) or a single sequence of values. A value of None
		clears the data set."""
		if replace:
			self.data = {}
			self.axes = {}
			self.visibility = {}
			self.pparm = {}
			self.column_labels = {}
		if input_data is None:
			self.data.pop(key, None)
			self.visibility.pop(key, None)
			self.axes.pop(key, None)
			self.pparm.pop(key, None)
			self.column_labels.pop(key, None)
			if self.inspector:
				self.inspector.datachange()
			if not quiet:
				self.update()
			return
		if not isinstance(input_data[0], (list, tuple, np.ndarray)):
			# single sequence of values -> plot vs index
			input_data = [np.arange(len(input_data)), np.asarray(input_data, dtype=np.float64)]
		cols = []
		for c in input_data:
			arr = np.asarray(c, dtype=np.float64).ravel()
			arr = arr[np.isfinite(arr)]
			cols.append(arr)
		self.data[key] = cols
		self.visibility.setdefault(key, True)
		if column_labels is not None:
			self.column_labels[key] = list(column_labels)

		# default plotted column
		if len(cols) == 1:
			self.axes[key] = (-1,)
		elif len(cols) > 2 and len(cols[0]) > 1 and np.all(np.diff(cols[0]) == 1):
			self.axes[key] = (1,)     # first axis looks like a boring count
		else:
			self.axes[key] = (0,)

		# per-set parameters
		if key in self.pparm:
			pp = self.pparm[key]
			if color < 0 or color > len(COLORS):
				color = pp[0]
			if alpha > 1.0 or alpha < 0.0:
				alpha = pp[1]
			if rwidth > 1.0 or rwidth < 0.0:
				rwidth = pp[2]
		else:
			if color < 0:
				color = len(self.data) % len(COLORS)
			if color > len(COLORS):
				color = 0
			if alpha > 1.0:
				alpha = 1.0
			if alpha < 0.0:
				alpha = 0.8
			if rwidth < 0.0:
				rwidth = 0.8
			if rwidth > 1.0:
				rwidth = 1.0
		self.pparm[key] = (color, alpha, rwidth)

		self.autoscale()
		if self.inspector:
			self.inspector.datachange()
		if not quiet:
			self.update()

	def plotted_column(self, key):
		col = self.axes.get(key, (0,))[0]
		if col < 0:
			col = 0
		if col >= len(self.data[key]):
			col = len(self.data[key]) - 1
		return col

	def _counts(self, key):
		"""Binned counts for a key, with norm/cumulative applied (EMAN2 order)."""
		arr = self.data[key][self.plotted_column(key)]
		counts, _edges = np.histogram(arr, self.nbins, range=self.xlimits)
		if self.cumulative:
			counts = np.cumsum(counts)
		if self.normed:
			s = float(np.sum(counts))
			if s > 0:
				counts = counts / s / max(1, len(self.axes))
		return counts

	# ----- limits ------------------------------------------------------------

	def autoscale(self, force=False):
		xmin, xmax = np.inf, -np.inf
		for k in self.data:
			if not self.visibility.get(k, True):
				continue
			arr = self.data[k][self.plotted_column(k)]
			if len(arr) == 0:
				continue
			xmin = min(xmin, float(arr.min()))
			xmax = max(xmax, float(arr.max()))
		if np.isfinite(xmin) and np.isfinite(xmax):
			if self.xlimits is None or force:
				if xmin == xmax:
					xmin -= 0.5
					xmax += 0.5
				self.xlimits = (float(xmin), float(xmax))
		if self.xlimits is not None:
			ymax = 0.0
			for k in self.data:
				if not self.visibility.get(k, True):
					continue
				ymax = max(ymax, float(np.max(self._counts(k))))
			if self.ylimits is None or force:
				self.ylimits = (0.0, ymax if ymax > 0 else 1.0)
		self.update()
		if self.inspector:
			self.inspector.update()

	def rescale(self, xmin, xmax, ymin, ymax, force=True):
		"""Set the x/y limits (None values are left to autoscale)."""
		try:
			if xmin is not None and xmax is not None and xmax > xmin:
				self.xlimits = (float(xmin), float(xmax))
			if ymin is not None and ymax is not None and ymax > ymin:
				self.ylimits = (float(ymin), float(ymax))
		except Exception:
			self.xlimits = None
			self.ylimits = None
		if force:
			if self.xlimits is None or self.ylimits is None:
				self.autoscale(True)
			self.update()
			if self.inspector:
				self.inspector.update()

	def setNBins(self, n):
		self.nbins = max(1, int(n))
		self.autoscale(True)
		self.update()

	def setAxes(self, key, col, quiet=False):
		self.axes[key] = (int(col),)
		self.autoscale(True)
		if self.inspector:
			self.inspector.update()
		if not quiet:
			self.update()

	def setPlotParms(self, key, color, alpha=0.8, rwidth=0.8, quiet=False):
		"""Per-set color / alpha / bar width (indices into the EMAN2 color list)."""
		if color < 0:
			color = self.pparm.get(key, (0, 0.8, 0.8))[0]
		if alpha > 1.0 or alpha < 0.0:
			alpha = 0.8
		if rwidth > 1.0 or rwidth < 0.0:
			rwidth = 0.8
		self.pparm[key] = (int(color), float(alpha), float(rwidth))
		if self.inspector:
			self.inspector.datachange()
		if not quiet:
			self.update()

	def setPlotRepr(self, repr):
		"""Set the plot representation (histtype/orientation/alignment/
		normed/cumulative/logy/stacked) from a dict."""
		for attr in ("histtype", "orientation", "alignment"):
			if attr in repr:
				setattr(self, attr, repr[attr])
		for attr in ("normed", "cumulative", "logy", "stacked"):
			if attr in repr:
				setattr(self, attr, bool(repr[attr]))
		self.autoscale(True)
		self.update()

	# ----- inspector ----------------------------------------------------------

	def get_inspector(self):
		if self.inspector is None:
			self.inspector = EMHistogramInspector(self)
		return self.inspector

	def show_inspector(self, show=True):
		if show:
			insp = self.get_inspector()
			insp.show()
			insp.raise_()
			insp.activateWindow()
		else:
			if self.inspector:
				self.inspector.close()

	# ----- events --------------------------------------------------------------

	def mousePressEvent(self, event):
		# Middle click (or alt+left) -> show inspector
		if event.button() == Qt.MouseButton.MiddleButton or \
		   (event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.AltModifier):
			self.show_inspector(True)
			return
		super().mousePressEvent(event)

	def keyPressEvent(self, event):
		if event.key() == Qt.Key_C:
			self.show_inspector(True)
		else:
			super().keyPressEvent(event)

	def closeEvent(self, event):
		if self.inspector:
			self.inspector.close()
		super().closeEvent(event)

	# ----- rendering -----------------------------------------------------------

	def paintEvent(self, event):
		p = QPainter(self)
		p.setRenderHint(QPainter.Antialiasing, False)
		w, h = self.width(), self.height()
		p.fillRect(0, 0, w, h, QColor(240, 240, 240))
		ml, mr, mt, mb = 60, 12, 12, 28
		pw, ph = w - ml - mr, h - mt - mb
		if pw < 40 or ph < 40 or self.xlimits is None or self.ylimits is None or not self.data:
			p.end()
			return

		x0, x1 = self.xlimits
		y0, y1 = self.ylimits
		fg = QColor(0, 0, 0)

		# white plot area (like the EMAN2 matplotlib figure)
		p.fillRect(ml, mt, pw, ph, QColor(255, 255, 255))

		# ticks and labels
		p.setFont(QFont("Sans", 9))
		p.setPen(fg)
		for i in range(5):
			xv = x0 + (x1 - x0) * i / 4
			x = ml + pw * i / 4
			p.drawLine(int(x), mt + ph, int(x), mt + ph + 4)
			p.drawText(int(x) - 20, mt + ph + 20, "{:.3g}".format(xv))
			yv = y0 + (y1 - y0) * i / 4
			y = mt + ph - ph * i / 4
			p.drawLine(ml - 4, int(y), ml, int(y))
			p.drawText(4, int(y) + 4, "{:.3g}".format(yv))
		p.drawRect(ml, mt, pw, ph)

		def val2py(v):
			"""Map a value to a y pixel, honoring log-Y."""
			if self.logy:
				v = max(v, 1e-12)
				v = np.log10(v)
				span = np.log10(max(y1, 1.0001)) - 0.0
				if span <= 0:
					frac = 0.0
				else:
					frac = v / span
			else:
				frac = (v - y0) / (y1 - y0) if y1 > y0 else 0.0
			return mt + ph - frac * ph

		binw = pw / self.nbins
		usedcounts = []
		for k in self.data:
			if not self.visibility.get(k, True):
				continue
			coloridx, alpha, rwidth = self.pparm.get(k, (0, 0.8, 0.8))
			counts = self._counts(k)
			bottom = np.sum([c for c in usedcounts], axis=0) if (self.stacked and usedcounts) else np.zeros(self.nbins)
			usedcounts.append(counts)
			c = QColor(*COLORS[coloridx % len(COLORS)])
			bw = binw * max(0.01, min(1.0, rwidth))
			for i in range(self.nbins):
				top = bottom[i] + counts[i]
				if top <= y0:
					continue
				if self.alignment == "center":
					x = ml + i * binw + (binw - bw) / 2
				else:
					x = ml + i * binw
				bot_y = val2py(max(bottom[i], y0))
				top_y = val2py(top)
				if self.histtype == "bar":
					p.setPen(Qt.PenStyle.NoPen)
					f = QColor(c)
					f.setAlpha(int(alpha * 255))
					p.setBrush(f)
					p.drawRect(int(x), int(top_y), max(1, int(bw)), int(bot_y - top_y))
				elif self.histtype == "step":
					p.setPen(QPen(c, 2))
					p.setBrush(Qt.BrushStyle.NoBrush)
					p.drawRect(int(x), int(top_y), max(1, int(bw)), int(bot_y - top_y))
				elif self.histtype == "stepfilled":
					p.setPen(QPen(c, 1))
					f = QColor(c)
					f.setAlpha(int(alpha * 255))
					p.setBrush(f)
					p.drawRect(int(x), int(top_y), max(1, int(bw)), int(bot_y - top_y))

		# legend: set names in their colors (top-right of the white area)
		p.setFont(QFont("Sans", 9))
		y = mt + 6
		for k in self.data:
			if not self.visibility.get(k, True):
				continue
			c = COLORS[self.pparm.get(k, (0, 0.8, 0.8))[0] % len(COLORS)]
			p.setPen(QColor(*c))
			p.drawLine(ml + pw - 70, y, ml + pw - 60, y)
			p.drawText(ml + pw - 55, y + 4, str(k)[:12])
			y += 14
		p.end()


class EMHistogramInspector(QtWidgets.QWidget):
	"""Inspector for EMHistogramWidget, with the EMAN2 parameter set:
	data set visibility (list/All/None/Sel1 range), per-set color/alpha/width,
	column selector, nbins, histtype/alignment, norm/cumulative/log-Y/stacked,
	x/y limits, rescale, and data saving."""

	def __init__(self, target):
		QtWidgets.QWidget.__init__(self, None)
		self.target = weakref.ref(target)
		self.setWindowTitle("Histogram Controls")
		self.resize(420, 560)
		self.quiet = 0
		self.busy = 0

		vbl0 = QtWidgets.QVBoxLayout(self)
		hbl = QtWidgets.QHBoxLayout()

		# ---- data set list ----------------------------------------------------
		gbx = QtWidgets.QGroupBox("Data sets")
		vbl3 = QtWidgets.QVBoxLayout(gbx)
		self.setlist = QtWidgets.QListWidget()
		self.setlist.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
		vbl3.addWidget(self.setlist)

		hbl6 = QtWidgets.QHBoxLayout()
		vbl3.addLayout(hbl6)
		self.nonebut = QtWidgets.QPushButton("None")
		hbl6.addWidget(self.nonebut)
		self.allbut = QtWidgets.QPushButton("All")
		hbl6.addWidget(self.allbut)

		# Sel1 visibility range slider
		hbls = QtWidgets.QHBoxLayout()
		self.showslide = QtWidgets.QSlider(Qt.Orientation.Horizontal)
		self.showslide.setRange(0, 0)
		hbls.addWidget(QtWidgets.QLabel("Sel1:"))
		hbls.addWidget(self.showslide)
		vbl3.addLayout(hbls)

		hbl7 = QtWidgets.QHBoxLayout()
		vbl3.addLayout(hbl7)
		self.nbox = QtWidgets.QSpinBox()
		self.nbox.setRange(1, 100000)
		self.nbox.setValue(1)
		hbl7.addWidget(QtWidgets.QLabel("ns:"))
		hbl7.addWidget(self.nbox)
		self.stepbox = QtWidgets.QSpinBox()
		self.stepbox.setRange(1, 100000)
		self.stepbox.setValue(1)
		hbl7.addWidget(QtWidgets.QLabel("stp:"))
		hbl7.addWidget(self.stepbox)

		hbl.addWidget(gbx)

		# ---- right column: plot parameters ------------------------------------
		vbl = QtWidgets.QVBoxLayout()
		hbl.addLayout(vbl)

		hbl0 = QtWidgets.QHBoxLayout()
		self.saveb = QtWidgets.QPushButton("Save")
		hbl0.addWidget(self.saveb)
		self.concatb = QtWidgets.QPushButton("Concat")
		hbl0.addWidget(self.concatb)
		self.pdfb = QtWidgets.QPushButton("PDF")
		hbl0.addWidget(self.pdfb)
		vbl.addLayout(hbl0)

		hbl01 = QtWidgets.QHBoxLayout()
		self.histtype = QtWidgets.QComboBox()
		self.histtype.addItems(["bar", "step", "stepfilled"])
		hbl01.addWidget(self.histtype)
		self.align = QtWidgets.QComboBox()
		self.align.addItems(["center", "edge"])
		hbl01.addWidget(self.align)
		vbl.addLayout(hbl01)

		hbl1 = QtWidgets.QHBoxLayout()
		self.color = QtWidgets.QComboBox()
		self.color.addItems(COLORNAMES)
		hbl1.addWidget(self.color)
		vbl.addLayout(hbl1)

		hbl000 = QtWidgets.QHBoxLayout()
		self.alpha = QtWidgets.QSlider(Qt.Orientation.Horizontal)
		self.alpha.setRange(0, 100)
		self.alpha.setValue(80)
		hbl000.addWidget(QtWidgets.QLabel("Alpha:"))
		hbl000.addWidget(self.alpha)
		vbl.addLayout(hbl000)

		hbl001 = QtWidgets.QHBoxLayout()
		self.rwidth = QtWidgets.QSlider(Qt.Orientation.Horizontal)
		self.rwidth.setRange(1, 100)
		self.rwidth.setValue(80)
		hbl001.addWidget(QtWidgets.QLabel("Width:"))
		hbl001.addWidget(self.rwidth)
		vbl.addLayout(hbl001)

		gl = QtWidgets.QGridLayout()
		gl.addWidget(QtWidgets.QLabel("Column:"), 0, 0, Qt.AlignmentFlag.AlignRight)
		self.slidecol = QtWidgets.QSpinBox()
		self.slidecol.setRange(0, 1)
		self.slidecol.setValue(0)
		gl.addWidget(self.slidecol, 0, 1, Qt.AlignmentFlag.AlignLeft)
		gl.addWidget(QtWidgets.QLabel("N Bins:"), 0, 2, Qt.AlignmentFlag.AlignRight)
		self.slidenbs = QtWidgets.QSpinBox()
		self.slidenbs.setRange(1, 10000)
		self.slidenbs.setValue(10)
		gl.addWidget(self.slidenbs, 0, 3, Qt.AlignmentFlag.AlignLeft)
		vbl.addLayout(gl)

		hbl02 = QtWidgets.QHBoxLayout()
		self.normed = QtWidgets.QCheckBox("Norm")
		hbl02.addWidget(self.normed)
		self.cumulative = QtWidgets.QCheckBox("Cumulative")
		hbl02.addWidget(self.cumulative)
		vbl.addLayout(hbl02)

		hbl03 = QtWidgets.QHBoxLayout()
		self.logtogy = QtWidgets.QCheckBox("Log Y")
		hbl03.addWidget(self.logtogy)
		self.stacked = QtWidgets.QCheckBox("Stacked")
		hbl03.addWidget(self.stacked)
		vbl.addLayout(hbl03)

		self.wrescale = QtWidgets.QPushButton("Rescale")
		vbl.addWidget(self.wrescale)

		vbl0.addLayout(hbl)

		# ---- limits -------------------------------------------------------------
		hbl2a = QtWidgets.QHBoxLayout()
		self.wxmin = self._limbox()
		self.wxmax = self._limbox()
		hbl2a.addWidget(QtWidgets.QLabel("X Min:"))
		hbl2a.addWidget(self.wxmin)
		hbl2a.addWidget(QtWidgets.QLabel("X Max:"))
		hbl2a.addWidget(self.wxmax)
		vbl0.addLayout(hbl2a)

		hbl2b = QtWidgets.QHBoxLayout()
		self.wymin = self._limbox()
		self.wymax = self._limbox()
		hbl2b.addWidget(QtWidgets.QLabel("Y Min:"))
		hbl2b.addWidget(self.wymin)
		hbl2b.addWidget(QtWidgets.QLabel("Y Max:"))
		hbl2b.addWidget(self.wymax)
		vbl0.addLayout(hbl2b)

		# ---- signals -------------------------------------------------------------
		self.showslide.valueChanged.connect(self.selSlide)
		self.allbut.clicked.connect(self.selAll)
		self.nonebut.clicked.connect(self.selNone)
		self.setlist.currentRowChanged.connect(self.newSet)
		self.setlist.itemChanged.connect(self.list_item_changed)
		self.saveb.clicked.connect(self.savePlot)
		self.concatb.clicked.connect(self.saveConcatPlot)
		self.pdfb.clicked.connect(self.savePdf)
		self.normed.stateChanged.connect(self.updPlotRepr)
		self.logtogy.stateChanged.connect(self.updPlotRepr)
		self.cumulative.stateChanged.connect(self.updPlotRepr)
		self.stacked.stateChanged.connect(self.updPlotRepr)
		self.histtype.currentIndexChanged.connect(self.updPlotRepr)
		self.align.currentIndexChanged.connect(self.updPlotRepr)
		self.slidecol.valueChanged.connect(self.newCols)
		self.slidenbs.valueChanged.connect(self.newNBins)
		self.rwidth.valueChanged.connect(self.updPlot)
		self.alpha.valueChanged.connect(self.updPlot)
		self.color.currentIndexChanged.connect(self.updPlot)
		self.wxmin.valueChanged.connect(self.newLimits)
		self.wxmax.valueChanged.connect(self.newLimits)
		self.wymin.valueChanged.connect(self.newLimits)
		self.wymax.valueChanged.connect(self.newLimits)
		self.wrescale.clicked.connect(self.autoScale)

		self.newSet(0)
		self.datachange()
		# sync the limit boxes with the widget's current limits right away
		self.update()

	def _limbox(self):
		box = QtWidgets.QDoubleSpinBox()
		box.setRange(-1e12, 1e12)
		box.setDecimals(8)
		box.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
		box.setSpecialValueText("auto")
		box.setValue(1e12)
		return box

	def _get_tgt(self):
		return self.target() if self.target() else None

	def _box_val(self, box):
		"""Value of a limit box; 1e12 (special 'auto') means no limit."""
		v = box.value()
		if v >= 1e12:
			return None
		return v

	# ---- data set visibility ---------------------------------------------------

	def selSlide(self, val):
		tgt = self._get_tgt()
		if not tgt:
			return
		rngn0 = int(val)
		rngn1 = int(self.nbox.value())
		rngstp = int(self.stepbox.value())
		rng = list(range(rngn0, rngn0 + rngstp * rngn1, rngstp))
		for i, k in enumerate(sorted(tgt.visibility.keys())):
			tgt.visibility[k] = i in rng
		tgt.autoscale()
		tgt.update()
		self.datachange()

	def selAll(self):
		tgt = self._get_tgt()
		if not tgt:
			return
		for k in list(tgt.visibility.keys()):
			tgt.visibility[k] = True
		tgt.autoscale()
		tgt.update()
		self.datachange()

	def selNone(self):
		tgt = self._get_tgt()
		if not tgt:
			return
		for k in list(tgt.visibility.keys()):
			tgt.visibility[k] = False
		tgt.autoscale()
		tgt.update()
		self.datachange()

	def list_item_changed(self, item):
		tgt = self._get_tgt()
		if not tgt:
			return
		name = str(item.text())
		checked = item.checkState() == Qt.CheckState.Checked
		if tgt.visibility.get(name) != checked:
			tgt.visibility[name] = checked
			tgt.autoscale()
			tgt.update()

	# ---- per-set parameters ------------------------------------------------------

	def newSet(self, row):
		tgt = self._get_tgt()
		if not tgt:
			return
		self.quiet = 1
		try:
			i = str(self.setlist.item(row).text())
		except Exception:
			self.quiet = 0
			return
		self.slidecol.setRange(0, max(0, len(tgt.data[i]) - 1))
		self.slidecol.setValue(tgt.plotted_column(i))
		pp = tgt.pparm.get(i, (0, 0.8, 0.8))
		self.color.setCurrentIndex(pp[0])
		self.alpha.setValue(int(pp[1] * 100))
		self.rwidth.setValue(int(pp[2] * 100))
		self.quiet = 0

	def newCols(self, val):
		tgt = self._get_tgt()
		if not tgt or self.quiet:
			return
		for name in self._selected_names():
			tgt.setAxes(name, val, quiet=True)
		tgt.update()

	def updPlot(self, s=None):
		tgt = self._get_tgt()
		if not tgt or self.quiet:
			return
		for name in self._selected_names():
			tgt.setPlotParms(name, self.color.currentIndex(),
							alpha=self.alpha.value() / 100.0,
							rwidth=self.rwidth.value() / 100.0, quiet=True)
		tgt.update()

	# ---- global parameters --------------------------------------------------------

	def newNBins(self):
		tgt = self._get_tgt()
		if not tgt or self.quiet:
			return
		tgt.setNBins(self.slidenbs.value())

	def updPlotRepr(self):
		tgt = self._get_tgt()
		if not tgt or self.quiet:
			return
		tgt.setPlotRepr({
			"histtype": self.histtype.currentText(),
			"orientation": "vertical",
			"alignment": self.align.currentText(),
			"normed": self.normed.isChecked(),
			"cumulative": self.cumulative.isChecked(),
			"logy": self.logtogy.isChecked(),
			"stacked": self.stacked.isChecked(),
		})

	# ---- limits --------------------------------------------------------------------

	def newLimits(self, val=None):
		tgt = self._get_tgt()
		if not tgt or self.busy:
			return
		tgt.rescale(self._box_val(self.wxmin), self._box_val(self.wxmax),
					self._box_val(self.wymin), self._box_val(self.wymax))

	def update(self):
		"""Sync the limit boxes from the widget's current limits."""
		tgt = self._get_tgt()
		if not tgt:
			return
		self.busy = 1
		try:
			self._set_box(self.wxmin, None if tgt.xlimits is None else tgt.xlimits[0])
			self._set_box(self.wxmax, None if tgt.xlimits is None else tgt.xlimits[1])
			self._set_box(self.wymin, None if tgt.ylimits is None else tgt.ylimits[0])
			self._set_box(self.wymax, None if tgt.ylimits is None else tgt.ylimits[1])
		finally:
			self.busy = 0

	def _set_box(self, box, val):
		box.setValue(1e12 if val is None else val)

	def autoScale(self):
		tgt = self._get_tgt()
		if not tgt:
			return
		tgt.autoscale(True)
		tgt.update()

	# ---- saving ----------------------------------------------------------------------

	def _selected_names(self):
		return [str(i.text()) for i in self.setlist.selectedItems()]

	def savePlot(self):
		"""Save each selected data set to a text file (first two columns)."""
		tgt = self._get_tgt()
		if not tgt:
			return
		for name in self._selected_names():
			data = tgt.data[name]
			sname = name.replace(" ", "_").replace("(", "_").replace(")", "").replace("/", "_")
			name2 = "plt_%s.txt" % sname
			i = 0
			while os.path.exists(name2):
				name2 = "plt_%s_%02d.txt" % (sname, i)
				i += 1
			with open(name2, "w") as out:
				n = len(data[0])
				for j in range(n):
					if len(data) > 1:
						out.write("%g\t%g\n" % (data[0][j], data[1][j]))
					else:
						out.write("%g\t%g\n" % (j, data[0][j]))
			print("Wrote", name2)

	def saveConcatPlot(self):
		"""Concatenate all selected data sets into a single text file."""
		tgt = self._get_tgt()
		if not tgt:
			return
		names = self._selected_names()
		if len(names) == 0:
			return
		name2 = "plt_concat.txt"
		i = 0
		while os.path.exists(name2):
			name2 = "plt_concat_%02d.txt" % i
			i += 1
		with open(name2, "w") as out:
			for name in names:
				data = tgt.data[name]
				n = len(data[0])
				for j in range(n):
					if len(data) > 1:
						out.write("%g\t%g\n" % (data[0][j], data[1][j]))
					else:
						out.write("%g\t%g\n" % (j, data[0][j]))
		print("Wrote", name2)

	def savePdf(self):
		"""Save a snapshot of the plot (PDF export is not yet available;
		a PNG snapshot is written instead)."""
		tgt = self._get_tgt()
		if not tgt:
			return
		name2 = "plot.png"
		i = 0
		while os.path.exists(name2):
			name2 = "plot_%02d.png" % i
			i += 1
		tgt.grab().save(name2)
		print("Wrote", name2)

	# ---- list maintenance -------------------------------------------------------------

	def datachange(self):
		tgt = self._get_tgt()
		if not tgt:
			return
		self.setlist.clear()
		flags = (Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled |
				 Qt.ItemFlag.ItemIsUserCheckable)
		keys = sorted(tgt.data.keys())
		parms = tgt.pparm
		for j in keys:
			a = QtWidgets.QListWidgetItem(j)
			a.setFlags(flags)
			try:
				a.setForeground(QColor(*LIST_COLORS[parms[j][0] % len(LIST_COLORS)]))
			except Exception:
				pass
			a.setCheckState(Qt.CheckState.Checked if tgt.visibility.get(j, True) else Qt.CheckState.Unchecked)
			self.setlist.addItem(a)
		if len(keys) > 0:
			self.setlist.setCurrentRow(0)
		self.showslide.setRange(0, len(keys))

	def closeEvent(self, event):
		tgt = self._get_tgt()
		if tgt:
			tgt.inspector = None
		super().closeEvent(event)


def main():
	import sys
	app = QtWidgets.QApplication(sys.argv)
	win = EMHistogramWidget()
	win.setWindowTitle("emhist test")
	win.set_data([[1.0, 2, 3, 4, 5], [2.0, 4, 8, 16, 32]], key="demo")
	win.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
