"""Shared presentation of the CoilGen squared wordmark."""

from importlib.resources import files

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QBitmap, QPixmap, QRegion
from PySide6.QtWidgets import QLabel, QSizePolicy


class LogoLabel(QLabel):
    """Keep the shared wordmark sharp and proportional at each display size."""

    def __init__(self, parent=None, *, width=420):
        super().__init__(parent)
        self._logo = QPixmap()
        try:
            self._logo.loadFromData(files("coilgen.gui").joinpath("assets/coilgen-logo.png").read_bytes())
        except OSError:
            pass
        if not self._logo.isNull():
            mask = QBitmap.fromImage(self._logo.toImage().createAlphaMask(Qt.ImageConversionFlag.ThresholdDither))
            bounds = QRegion(mask).boundingRect()
            if not bounds.isEmpty():
                horizontal, vertical = round(bounds.width() * 0.02), round(bounds.height() * 0.02)
                bounds = bounds.adjusted(-horizontal, -vertical, horizontal, vertical).intersected(self._logo.rect())
                self._logo = self._logo.copy(bounds)
        self._preferred_width = width
        self.setObjectName("coilgen_logo")
        self.setAccessibleName("CoilGen squared (CoilGen2) logo with cylindrical coil winding")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMaximumWidth(width)
        self.setFixedHeight(round(width * self._logo.height() / self._logo.width()) if not self._logo.isNull() else 32)
        self._update_pixmap()

    def sizeHint(self):
        return QSize(self._preferred_width, self.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self):
        if self._logo.isNull():
            self.setText("CoilGen\N{SUPERSCRIPT TWO}")
            return
        ratio = self.devicePixelRatioF()
        size = QSize(round(self.width() * ratio), round(self.height() * ratio))
        pixmap = self._logo.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        pixmap.setDevicePixelRatio(ratio)
        self.setPixmap(pixmap)
