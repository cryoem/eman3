from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QColor


class HistogramWidget(QtWidgets.QWidget):
	"""256-bin histogram of rendered image intensities (0-255).

	Draws a 1-pixel-wide vertical line per bin using Qt painter.
	No axes or labels -- just the intensity distribution shape.
	Fixed width of 256 pixels.
	"""

	def __init__(self, parent=None):
		super().__init__(parent)
		self._hist = None
		self.setFixedSize(256, 128)
		self.setAutoFillBackground(True)
		pal = self.palette()
		pal.setColor(self.backgroundRole(), QColor(20, 20, 20))
		self.setPalette(pal)

	def set_histogram(self, hist_array):
		"""Set histogram data. Expects a sequence of 256 non-negative values."""
		import numpy as np
		if hist_array is not None and len(hist_array) == 256:
			self._hist = np.asarray(hist_array).copy()
		else:
			self._hist = None
		self.update()

	def paintEvent(self, event):
		from PySide6.QtGui import QPainter
		painter = QPainter(self)
		painter.setRenderHint(QPainter.Antialiasing, False)

		w, h = 256, 128

		if self._hist is not None:
			import numpy as np
			hist = self._hist.copy()
			sorted_vals = np.sort(hist)[::-1]
			max_val = sorted_vals[0]
			second_max = sorted_vals[1] if len(sorted_vals) > 1 else max_val
			effective_max = min(max_val, second_max * 1.05)
			if effective_max > 1e-12:
				scaled = hist / effective_max
				painter.setPen(QColor(153, 217, 102))
				for i in range(256):
					bar_h = int(scaled[i] * (h - 4))
					if bar_h > 0:
						# Use drawLine instead of drawRect to avoid edge clipping
						painter.drawLine(i, h - bar_h - 1, i, h - 2)

		painter.end()
