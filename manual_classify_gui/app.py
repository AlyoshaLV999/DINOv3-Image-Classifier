"""图像分类整理桌面应用入口。"""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication, QMainWindow

from .actions import ActionsMixin
from .archive import ArchiveMixin
from .filesystem import FileSystemMixin
from .image_browser import ImageBrowserMixin
from .navigation import NavigationMixin
from .ui_core import UICoreMixin

LOGGER = logging.getLogger("manual_classify_gui")


class ImageClassifierUI(
    UICoreMixin, NavigationMixin, ImageBrowserMixin, ActionsMixin,
    ArchiveMixin, FileSystemMixin, QMainWindow,
):
    """复核 output 推理图片，并管理归类、移除和归档操作。"""

    IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    THUMBNAIL_BASE_SIZE = 164
    THUMBNAIL_CACHE_SIZE = 384
    THUMBNAIL_PREFETCH_ROWS = 1
    DIRECTORY_CACHE_SIZE = 64
    IMAGE_LIST_CACHE_SIZE = 64
    SHUTDOWN_WAIT_MS = 3000
    TARGET_DIR_LOW_COUNT_THRESHOLD = 30


def main() -> int:
    """创建并运行桌面应用。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("图像分类整理")
    window = ImageClassifierUI()
    window.show()
    return app.exec()
