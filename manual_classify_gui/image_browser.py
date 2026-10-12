"""图片列表、缩略图和预览相关行为。"""
from __future__ import annotations

import logging
import os
from typing import Any

from PIL import Image, ImageOps
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from .preview import _PreviewDialog, pil_to_qimage
from .widgets import _FunctionWorker

LOGGER = logging.getLogger("manual_classify_gui")

try:
    from send2trash import send2trash
except ImportError:  # 可选依赖。
    send2trash = None


class ImageBrowserMixin:
    """图片列表、缩略图、选择状态、预览和缩放。"""

    def load_images(self, directory_name: str, clear_selection: bool = True) -> None:
        if not self.current_dataset:
            self.set_status("请先选择数据集。")
            return
        new_dir = os.path.join("output", self.current_dataset, directory_name)
        if clear_selection and new_dir != self.current_output_dir:
            self.selected_images.clear()
        self.current_output_dir = new_dir
        try:
            self.current_output_index = self.get_label_dirs().index(directory_name)
        except ValueError:
            self.current_output_index = -1
        self.current_image_paths = self.get_image_files(new_dir)
        self.selected_images.intersection_update(self.current_image_paths)
        self.image_list.blockSignals(True)
        self.image_list.clear()
        self.image_list.clear_anchor()
        size = max(16, int(self.THUMBNAIL_BASE_SIZE * self.zoom_scale / 100))
        self.image_list.setIconSize(QSize(size, size))
        self.image_list.setGridSize(QSize(max(110, size + 26), size + 60))
        for path in self.current_image_paths:
            item = QListWidgetItem("\n".join([os.path.basename(path)[:32]]))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            item.setSizeHint(QSize(max(110, size + 26), size + 60))
            self.image_list.addItem(item)
        self.image_list.blockSignals(False)
        self.refresh_output_dirs()
        self.refresh_selection()
        if not self.current_image_paths:
            self.set_status(f"{directory_name}：0 张图片")
            return
        self.set_status(f"{os.path.basename(new_dir)}：{len(self.current_image_paths)} 张图片")
        QTimer.singleShot(0, self._schedule_thumbnail_load)

    def _schedule_thumbnail_load(self, *_args: Any) -> None:
        if self._closing or not self.current_image_paths or self.image_list.count() == 0:
            return
        rect = self.image_list.viewport().rect()
        visible = [self.image_list.item(i) for i in range(self.image_list.count())
                   if self.image_list.visualItemRect(self.image_list.item(i)).intersects(rect)]
        if not visible:
            return
        first = max(0, self.image_list.row(visible[0]) - self.THUMBNAIL_PREFETCH_ROWS * max(1, self._grid_columns()))
        last = min(self.image_list.count(), self.image_list.row(visible[-1]) + 1 + self.THUMBNAIL_PREFETCH_ROWS * max(1, self._grid_columns()))
        for index in range(first, last):
            self._request_thumbnail(index)

    def _grid_columns(self) -> int:
        item_width = max(1, self.image_list.gridSize().width())
        return max(1, self.image_list.viewport().width() // item_width)

    def _request_thumbnail(self, index: int) -> None:
        if not (0 <= index < len(self.current_image_paths)):
            return
        path = self.current_image_paths[index]
        try:
            stat = os.stat(path)
        except OSError:
            item = self.image_list.item(index)
            if item:
                item.setText("無法读取\n" + os.path.basename(path)[:24])
            return
        size = self.image_list.iconSize().width()
        key = (os.path.abspath(path), stat.st_mtime_ns, size)
        item = self.image_list.item(index)
        if item is None:
            return
        image = self._thumbnail_cache.get(key)
        if image is not None:
            self._thumbnail_cache.move_to_end(key)
            item.setIcon(QPixmap.fromImage(image))
            return
        if key in self._thumbnail_pending:
            return
        self._thumbnail_pending.add(key)
        worker = _FunctionWorker(self._decode_thumbnail, path, size)
        self._start_worker(
            worker,
            lambda result, cache_key=key, file_path=path: self._thumbnail_ready(cache_key, file_path, result),
            lambda _error, cache_key=key: self._thumbnail_failed(cache_key),
        )

    @staticmethod
    def _decode_thumbnail(path: str, size: int) -> QImage:
        """在工作线程中解码并缩放缩略图，返回拥有独立像素数据的 QImage。

        使用 :meth:`PIL.Image.Image.draft` 让 JPEG 解码器直接输出低分辨率
        结果，可显著降低大图解码时的内存峰值与耗时。

        :param path: 图片文件路径。
        :param size: 目标缩略图边长（像素）。
        :return: 已缩放的 QImage。
        """
        with Image.open(path) as source:
            source.draft("RGB", (size, size))
            image = ImageOps.exif_transpose(source).convert("RGBA")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            return pil_to_qimage(image)

    def _thumbnail_ready(self, key: tuple[str, int, int], path: str, image: QImage) -> None:
        self._thumbnail_pending.discard(key)
        self._thumbnail_cache[key] = image
        self._thumbnail_cache.move_to_end(key)
        while len(self._thumbnail_cache) > self.THUMBNAIL_CACHE_SIZE:
            self._thumbnail_cache.popitem(last=False)
        try:
            index = self.current_image_paths.index(path)
        except ValueError:
            return
        item = self.image_list.item(index)
        # 目录切换后列表项会被重建，必须校验该项仍然对应当前路径，
        # 否则会把缩略图写到错误（甚至已销毁）的条目上。
        if item is None or item.data(Qt.ItemDataRole.UserRole) != path:
            return
        if key[2] != self.image_list.iconSize().width():
            return
        item.setIcon(QPixmap.fromImage(image))
        self._schedule_thumbnail_load()

    def _thumbnail_failed(self, key: tuple[str, int, int]) -> None:
        self._thumbnail_pending.discard(key)
        LOGGER.warning("缩略图解码失败：%s", key[0])

    def _selection_changed(self) -> None:
        selected = {item.data(Qt.ItemDataRole.UserRole) for item in self.image_list.selectedItems()}
        self.selected_images = {path for path in selected if path}
        self.move_btn.setEnabled(bool(self.selected_images and self.target_dir))

    def _set_recycle_bin_enabled(self, enabled: bool) -> None:
        self.recycle_bin_enabled = enabled and send2trash is not None

    def refresh_selection(self) -> None:
        self.image_list.blockSignals(True)
        try:
            for index in range(self.image_list.count()):
                item = self.image_list.item(index)
                item.setSelected(item.data(Qt.ItemDataRole.UserRole) in self.selected_images)
        finally:
            self.image_list.blockSignals(False)
        self.move_btn.setEnabled(bool(self.selected_images and self.target_dir))

    def _open_preview_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        try:
            index = self.current_image_paths.index(path)
        except ValueError:
            return
        dialog = _PreviewDialog(self, self.current_image_paths[:], index)
        dialog.exec()

    def invert_selection(self) -> None:
        current = set(self.current_image_paths)
        self.selected_images = current - self.selected_images
        self.refresh_selection()

    def decrease_zoom(self) -> None:
        self._set_zoom(max(10, self._read_zoom() - 10))

    def increase_zoom(self) -> None:
        self._set_zoom(min(200, self._read_zoom() + 10))

    def _read_zoom(self) -> int:
        try:
            return int(self.zoom_entry.text())
        except ValueError:
            return self.zoom_scale

    def on_zoom_entry_change(self) -> None:
        try:
            self._set_zoom(max(10, min(200, int(self.zoom_entry.text()))))
        except ValueError:
            self._set_zoom(self.zoom_scale)

    def _set_zoom(self, scale: int) -> None:
        self.zoom_scale = scale
        self.zoom_entry.setText(str(scale))
        self._thumbnail_cache.clear()
        self._thumbnail_pending.clear()
        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)
