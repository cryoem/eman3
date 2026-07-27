#!/usr/bin/env python
#
# Author: Steven Ludtke (original EMAN2), ported for EMAN3 by PySide6
# Copyright (c) 2000-2006 Baylor College of Medicine
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
# Foundation, Inc., 59 Temple Place, Suite 330, Boston MA 02111-1307 USA
#

"""Common value-entry widgets for EMAN3 GUIs.

Provides ValSlider (slider + text box), ValBox (text box only), StringBox
(arbitrary text), and CheckBox with consistent APIs: setValue, getValue,
setLabel/getLabel, setEnabled/getEnabled, setRange/getRange, and signal
emission on change.
"""

import math
from PySide6 import QtCore, QtGui, QtWidgets


def _clamp(lo, val, hi):
	return max(lo, min(val, hi))


def _make_label(text, parent=None, minimum_width=30):
	label = QtWidgets.QLabel(text, parent)
	label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
	label.setMinimumSize(minimum_width, 20)
	return label


# ------------------------------------------------------------------ #

class ValSlider(QtWidgets.QWidget):
	"""Connected QLineEdit + QSlider for floating point values.

	showenable: -1 (no checkbox), 0 (checkbox off), 1 (checkbox on)
	Typing '<value' or '>value' in the text box adjusts max/min range.

	Signals
	-------
	enableChanged : int
	valueChanged  : float
	textChanged   : float
	sliderReleased: float
	sliderPressed : float
	"""

	enableChanged = QtCore.Signal(int)
	valueChanged = QtCore.Signal(float)
	textChanged = QtCore.Signal(float)
	sliderReleased = QtCore.Signal(float)
	sliderPressed = QtCore.Signal(float)

	def __init__(self, parent=None, rng=None, label=None, value=0,
				 labelwidth=30, showenable=-1, rounding=3):
		super().__init__(parent)

		self.rng = list(rng) if rng else [0.0, 1.0]
		self.value = value if value is not None else 0.0
		self._oldvalue = value - 1.0
		self.ignore = False
		self.intonly = False
		self.rounding = rounding

		layout = QtWidgets.QHBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(6)

		self._has_enablebox = showenable >= 0
		if self._has_enablebox:
			self.enablebox = QtWidgets.QCheckBox(self)
			self.enablebox.setChecked(bool(showenable))
			layout.addWidget(self.enablebox)
			self.enablebox.toggled.connect(self.setEnabled)

		self._has_label = bool(label)
		if self._has_label:
			self.label = _make_label(label, self, labelwidth)
			layout.addWidget(self.label)

		self.text = QtWidgets.QLineEdit(self)
		self.text.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
								QtWidgets.QSizePolicy.Preferred)
		self.text.setMinimumSize(80, 0)
		layout.addWidget(self.text)

		self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
		self.slider.setMaximum(4095)
		self.slider.setSingleStep(16)
		self.slider.setPageStep(256)
		self.slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
								 QtWidgets.QSizePolicy.Preferred)
		self.slider.setMinimumSize(100, 0)
		layout.addWidget(self.slider)

		self.text.editingFinished.connect(self._on_text_change)
		self.slider.valueChanged.connect(self._on_slider_change)
		self.slider.sliderReleased.connect(lambda: self.sliderReleased.emit(self.value))
		self.slider.sliderPressed.connect(lambda: self.sliderPressed.emit(self.value))

		self._updateboth()
		if self._has_enablebox:
			self.setEnabled(bool(showenable))

	# -- public API ---------------------------------------------------

	def setEnabled(self, ena):
		self.slider.setEnabled(ena)
		self.text.setEnabled(ena)
		self.enableChanged.emit(int(ena))

	def getEnabled(self):
		if self._has_enablebox:
			return self.enablebox.isChecked()
		return True

	def setRange(self, minv, maxv):
		if maxv <= minv:
			maxv = minv + 0.001
		self.rng = [float(minv), float(maxv)]
		self._updates()

	def getRange(self):
		return list(self.rng)

	def setValue(self, val, quiet=0):
		if val is None:
			return
		if val <= self.rng[0]:
			self.rng[0] = val
		if val >= self.rng[1]:
			self.rng[1] = val

		if self.intonly:
			new = int(round(val))
		else:
			new = float(val)

		if new == self.value:
			return
		self.value = new
		self._updateboth()
		if not quiet and self.value != self._oldvalue:
			self.valueChanged.emit(self.value)
		self._oldvalue = self.value

	def getValue(self):
		return self.value

	def setIntonly(self, flag):
		self.intonly = bool(flag)
		self.rounding = 0 if flag else 3
		self.text.setMinimumWidth(50 if flag else 80)
		self._updateboth()

	def setLabel(self, label):
		self.label.setText(label)

	def getLabel(self):
		if self._has_label:
			return self.label.text()
		return ""

	def updates(self):
		self._updates()

	def updatet(self):
		self._updatet()

	def updateboth(self):
		self._updateboth()

	# -- internal -----------------------------------------------------

	def _on_text_change(self):
		if self.ignore:
			return
		x = self.text.text()
		if not x:
			return
		if x.startswith("<"):
			try:
				self.rng[1] = float(x[1:])
			except ValueError:
				pass
			self._updateboth()
		elif x.startswith(">"):
			try:
				self.rng[0] = float(x[1:])
			except ValueError:
				pass
			self._updateboth()
		else:
			try:
				if self.intonly:
					self.value = int(round(float(x)))
				else:
					self.value = float(x)

				if self.value > self.rng[1]:
					self.rng[1] = self.value
				if self.value < self.rng[0]:
					self.rng[0] = self.value

				self._updates()
				if self.value != self._oldvalue:
					self.valueChanged.emit(self.value)
					self.textChanged.emit(self.value)
				self._oldvalue = self.value
			except ValueError:
				self._updateboth()

	def _on_slider_change(self, x):
		if self.ignore:
			return
		old = self.value
		self.value = (x / 4095.0) * (self.rng[1] - self.rng[0]) + self.rng[0]
		if self.intonly:
			self.value = int(round(self.value))
			if self.value == old:
				return
		self._updatet()
		if self.value != self._oldvalue:
			self.valueChanged.emit(self.value)
		self._oldvalue = self.value

	def _updates(self):
		self.ignore = True
		self.slider.setValue(_clamp(0,
								   (self.value - self.rng[0]) / (self.rng[1] - self.rng[0]) * 4095.0,
								   4095.0))
		self.ignore = False

	def _updatet(self):
		self.ignore = True
		if self.intonly:
			self.text.setText(str(int(self.value)))
		else:
			self.text.setText(f"{self.value:.{self.rounding}g}")
		self.ignore = False

	def _updateboth(self):
		self._updates()
		self._updatet()


# ------------------------------------------------------------------ #

class ValBox(QtWidgets.QWidget):
	"""Floating-point text entry without a slider.

	Drop-in replacement for ValSlider in most cases; the only difference is
	no graphical slider control exists, so range-adjustment via typing
	'<value' / '>value' is the primary way to change bounds.
	"""

	enableChanged = QtCore.Signal(int)
	valueChanged = QtCore.Signal(float)
	textChanged = QtCore.Signal(float)

	def __init__(self, parent=None, rng=None, label=None, value=0,
				 labelwidth=30, showenable=-1):
		super().__init__(parent)

		self.rng = list(rng) if rng else [0.0, 1.0]
		self.value = value if value is not None else 0.0
		self.ignore = False
		self.intonly = False
		self.digits = 5

		layout = QtWidgets.QHBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(6)

		self._has_enablebox = showenable >= 0
		if self._has_enablebox:
			self.enablebox = QtWidgets.QCheckBox(self)
			self.enablebox.setChecked(bool(showenable))
			layout.addWidget(self.enablebox)
			self.enablebox.toggled.connect(self.setEnabled)

		self._has_label = bool(label)
		if self._has_label:
			self.label = _make_label(label, self, labelwidth)
			layout.addWidget(self.label)

		self.text = QtWidgets.QLineEdit(self)
		self.text.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
								QtWidgets.QSizePolicy.Preferred)
		self.text.setMinimumSize(60, 0)
		layout.addWidget(self.text)

		self.text.editingFinished.connect(self._on_text_change)

		if self._has_enablebox:
			self.setEnabled(bool(showenable))
		self._updateboth()

	def setEnabled(self, ena):
		self.text.setEnabled(ena)
		self.enableChanged.emit(int(ena))

	def getEnabled(self):
		if self._has_enablebox:
			return self.enablebox.isChecked()
		return True

	def setRange(self, minv, maxv):
		if maxv <= minv:
			maxv = minv + 0.001
		self.rng = [float(minv), float(maxv)]

	def getRange(self):
		return list(self.rng)

	def setDigits(self, digits):
		self.digits = int(digits)

	def setValue(self, val, quiet=0):
		if val is None:
			self.text.setText("")
			return
		if val <= self.rng[0]:
			self.rng[0] = val
		if val >= self.rng[1]:
			self.rng[1] = val

		if self.intonly:
			new = int(round(val))
		else:
			new = float(val)

		if new == self.value:
			return
		self.value = new
		self._updateboth()
		if not quiet:
			self.valueChanged.emit(self.value)

	def getValue(self):
		if self.intonly:
			return int(self.value)
		return self.value

	def setIntonly(self, flag):
		self.intonly = bool(flag)
		self._updateboth()

	def setLabel(self, label):
		self.label.setText(label)

	def getLabel(self):
		if self._has_label:
			return self.label.text()
		return ""

	def updatet(self):
		self._updatet()

	def updateboth(self):
		self._updateboth()

	def _on_text_change(self):
		if self.ignore:
			return
		x = self.text.text()
		if not x:
			return
		if x.startswith("<"):
			try:
				self.rng[1] = float(x[1:])
			except ValueError:
				pass
			self._updateboth()
		elif x.startswith(">"):
			try:
				self.rng[0] = float(x[1:])
			except ValueError:
				pass
			self._updateboth()
		else:
			try:
				if self.intonly:
					self.value = int(round(float(x)))
				else:
					self.value = float(x)
				self.valueChanged.emit(self.value)
				self.textChanged.emit(self.value)
			except ValueError:
				self._updateboth()

	def _updatet(self):
		self.ignore = True
		self.text.setText(f"{self.value:.{self.digits}g}")
		self.ignore = False

	def _updateboth(self):
		self._updatet()


# ------------------------------------------------------------------ #

class StringBox(QtWidgets.QWidget):
	"""Arbitrary text entry with optional label and enable checkbox."""

	enableChanged = QtCore.Signal(int)
	valueChanged = QtCore.Signal(str)
	textChanged = QtCore.Signal(str)

	def __init__(self, parent=None, label=None, value="",
				 labelwidth=30, showenable=-1):
		super().__init__(parent)

		self.ignore = False

		layout = QtWidgets.QHBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(6)

		self._has_enablebox = showenable >= 0
		if self._has_enablebox:
			self.enablebox = QtWidgets.QCheckBox(self)
			self.enablebox.setChecked(bool(showenable))
			layout.addWidget(self.enablebox)
			self.enablebox.toggled.connect(self.setEnabled)

		self._has_label = bool(label)
		if self._has_label:
			self.label = _make_label(label, self, labelwidth)
			layout.addWidget(self.label)

		self.text = QtWidgets.QLineEdit(self)
		self.text.setText(value if value else "")
		self.text.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
								QtWidgets.QSizePolicy.Preferred)
		self.text.setMinimumSize(80, 0)
		layout.addWidget(self.text)

		self.text.editingFinished.connect(self._on_text_change)

		if self._has_enablebox:
			self.setEnabled(bool(showenable))

	def setEnabled(self, ena):
		self.text.setEnabled(ena)
		self.enableChanged.emit(int(ena))

	def getEnabled(self):
		if self._has_enablebox:
			return self.enablebox.isChecked()
		return True

	def setValue(self, val, quiet=0):
		if self.getValue() == str(val):
			return
		self.text.setText(str(val))
		if not quiet:
			self.valueChanged.emit(str(val))

	def getValue(self):
		return self.text.text()

	def setLabel(self, label):
		self.label.setText(label)

	def getLabel(self):
		if self._has_label:
			return self.label.text()
		return ""

	def _on_text_change(self):
		if self.ignore:
			return
		val = self.getValue()
		self.valueChanged.emit(val)
		self.textChanged.emit(val)


# ------------------------------------------------------------------ #

class CheckBox(QtWidgets.QWidget):
	"""QCheckBox wrapper with optional label and enable checkbox."""

	enableChanged = QtCore.Signal(int)
	valueChanged = QtCore.Signal(bool)

	def __init__(self, parent=None, label=None, value=False,
				 labelwidth=30, showenable=-1):
		super().__init__(parent)

		self.ignore = False

		layout = QtWidgets.QHBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(6)

		self._has_enablebox = showenable >= 0
		if self._has_enablebox:
			self.enablebox = QtWidgets.QCheckBox(self)
			self.enablebox.setChecked(bool(showenable))
			layout.addWidget(self.enablebox)
			self.enablebox.toggled.connect(self.setEnabled)

		self._has_label = bool(label)
		if self._has_label:
			self.label = _make_label(label, self, labelwidth)
			layout.addWidget(self.label)

		self.check = QtWidgets.QCheckBox(self)
		self.check.setChecked(bool(value))
		layout.addWidget(self.check)

		self.check.stateChanged.connect(self._on_state_changed)

		if self._has_enablebox:
			self.setEnabled(bool(showenable))

	def setEnabled(self, ena):
		self.check.setEnabled(ena)
		self.enableChanged.emit(int(ena))

	def getEnabled(self):
		if self._has_enablebox:
			return self.enablebox.isChecked()
		return True

	def setValue(self, val, quiet=0):
		try:
			val = bool(int(val))
		except (ValueError, TypeError):
			if isinstance(val, str):
				val = val.lower() in ("true", "yes", "t", "y")

		if self.getValue() == val:
			return
		self.check.setChecked(val)
		if not quiet:
			self.valueChanged.emit(val)

	def getValue(self):
		return self.check.isChecked()

	def setLabel(self, label):
		self.label.setText(label)

	def getLabel(self):
		if self._has_label:
			return self.label.text()
		return ""

	def _on_state_changed(self, newv):
		if self.ignore:
			return
		self.valueChanged.emit(bool(newv))
