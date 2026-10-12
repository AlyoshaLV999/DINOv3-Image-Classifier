"""独立图片预览窗口。"""
from __future__ import annotations

import os
from typing import Any

from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QSizePolicy



def pil_to_qimage(image: Image.Image) -> QImage:
    """把 Pillow 图像转换为拥有独立像素数据的 QImage。

    ``QImage`` 使用外部缓冲区构造时**不会接管缓冲区所有权**，必须显式
    调用 :meth:`QImage.copy` 生成独立副本，否则底层 ``bytes`` 被回收后
    会出现悬垂指针（表现为花屏或访问违例）。同时显式传入
    ``bytesPerLine``，避免依赖平台/版本对步长的隐式推断。

    :param image: 已解码的 Pillow 图像，内部会统一转换为 RGBA 模式。
    :return: 拥有独立像素副本的 QImage，可安全跨线程传递。
    """
    rgba = image.convert("RGBA")
    width, height = rgba.size
    buffer = rgba.tobytes("raw", "RGBA")
    return QImage(
        buffer, width, height, width * 4, QImage.Format.Format_RGBA8888
    ).copy()


class _PreviewDialog(QDialog):
    """提供图片前后浏览和选中切换的独立预览窗口。

    预览图片始终按容器尺寸等比缩放并完整显示，避免拉伸变形。
    """
    def __init__(self, parent: "ImageClassifierUI", paths: list[str], index: int) -> None:
        super().__init__(parent)
        self.owner = parent
        self.paths = paths
        self.index = index
        self._current_qimage: QImage | None = None
        self.setWindowTitle("图片预览")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.info = QLabel()
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.select_button = QPushButton()
        previous = QPushButton("‹ 上一张")
        following = QPushButton("下一张 ›")
        previous.clicked.connect(lambda: self.navigate(-1))
        following.clicked.connect(lambda: self.navigate(1))
        self.select_button.clicked.connect(self.toggle_selected)
        controls = QHBoxLayout()
        controls.addWidget(previous)
        controls.addWidget(self.info, 1)
        controls.addWidget(self.select_button)
        controls.addWidget(following)
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label, 1)
        layout.addLayout(controls)
        self.resize(900, 700)
        self.show_image()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_pixmap)

    def _update_pixmap(self) -> None:
        """按当前标签尺寸等比缩放并居中显示当前图片。"""
        if self._current_qimage is None or self._current_qimage.isNull():
            return
        label_size = self.image_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return
        pixmap = QPixmap.fromImage(self._current_qimage)
        scaled = pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def navigate(self, direction: int) -> None:
        if self.paths:
            self.index = (self.index + direction) % len(self.paths)
            self.show_image()

    def toggle_selected(self) -> None:
        path = self.paths[self.index]
        if path in self.owner.selected_images:
            self.owner.selected_images.remove(path)
        else:
            self.owner.selected_images.add(path)
        self.owner.refresh_selection()
        self.show_image()

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_Left:
            self.navigate(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.navigate(1)
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_selected()
        else:
            super().keyPressEvent(event)

    def show_image(self) -> None:
        path = self.paths[self.index]
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGBA")
                image.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
                qimage = pil_to_qimage(image)
            self._current_qimage = qimage
            self._update_pixmap()
            self.setWindowTitle(f"图片预览 · {os.path.basename(path)}")
        except (OSError, ValueError) as error:
            self._current_qimage = None
            self.image_label.clear()
            self.image_label.setText(f"无法打开图片：{error}")
            self.owner.set_status(f"无法打开 {os.path.basename(path)}：{error}")
        self.info.setText(f"{self.index + 1} / {len(self.paths)}")
        self.select_button.setText("取消选择" if path in self.owner.selected_images else "选择图片")
