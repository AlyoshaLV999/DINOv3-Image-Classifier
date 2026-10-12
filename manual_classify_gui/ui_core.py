"""主窗口的基础 UI 组装与生命周期。"""
from __future__ import annotations

import logging
import locale
import os
from collections import OrderedDict
from typing import Any

from PySide6.QtCore import Qt, QThreadPool, QTimer, QSize
from PySide6.QtGui import QAction, QImage, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QGraphicsOpacityEffect, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QStatusBar, QVBoxLayout, QWidget, QToolButton,
)

try:
    from send2trash import send2trash
except ImportError:  # 可选依赖。
    send2trash = None

from .widgets import _DoubleClickButton, _FlowWidget, _FunctionWorker, _ImageListWidget

LOGGER = logging.getLogger("manual_classify_gui")


class UICoreMixin:
    """窗口初始化、控件构建、任务调度与通用 UI 状态。"""

    def __init__(self) -> None:
        super().__init__()
        try:
            locale.setlocale(locale.LC_ALL, "")
        except locale.Error:
            LOGGER.warning("无法启用系统区域设置排序")
        self.setWindowTitle("图像分类整理")
        self.resize(1280, 850)
        self.setMinimumSize(960, 620)
        self.current_dataset = ""
        self.current_output_dir = ""
        self.current_output_index = -1
        self.target_dataset = ""
        self.target_dir = ""
        self.zoom_scale = 100
        self.current_image_paths: list[str] = []
        self.selected_images: set[str] = set()
        self.recycle_bin_enabled = send2trash is not None
        self._directory_cache: OrderedDict[str, tuple[tuple[int, int], list[str]]] = OrderedDict()
        self._image_list_cache: OrderedDict[str, tuple[tuple[int, int], list[str]]] = OrderedDict()
        self._thumbnail_cache: OrderedDict[tuple[str, int, int], QImage] = OrderedDict()
        self._thumbnail_pending: set[tuple[str, int, int]] = set()
        self._active_workers: set[_FunctionWorker] = set()
        self._closing = False
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(max(2, min(6, os.cpu_count() or 2)))
        self._archive_running = False
        self._letter_widgets: dict[str, list[QWidget]] = {}
        self._build_ui()
        self._bind_shortcuts()
        self.refresh_dataset_nav()
        self.refresh_target_dataset_selector()
        self.refresh_output_dirs()
        self.load_target_dirs()

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f2f2f7; color: #1c1c1e; font: 10pt Arial; }
            QGroupBox { background: white; border: 1px solid #d1d1d6; border-radius: 9px; margin-top: 10px; padding: 10px; font-weight: bold; }
            QPushButton, QToolButton { background: white; border: 1px solid #d1d1d6; border-radius: 6px; padding: 7px 10px; }
            QPushButton:hover, QToolButton:hover { background: #e9e9ee; }
            QPushButton:checked, QToolButton:checked { background: #007aff; color: white; border-color: #007aff; }
            QListWidget { background: white; border: 0; outline: 0; }
            QListWidget::item { border: 1px solid #e5e5ea; border-radius: 7px; padding: 5px; }
            QListWidget::item:selected { background: #eaf4ff; border: 2px solid #007aff; color: #1c1c1e; }
            QLineEdit { background: white; border: 1px solid #d1d1d6; border-radius: 5px; padding: 5px; }
        """)
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 8)
        title = QLabel("图像分类整理")
        title.setStyleSheet("font-size: 20pt; font-weight: bold;")
        subtitle = QLabel("复核 output 推理结果，选择独立目标数据集与目录后归类，并可安全归档。")
        subtitle.setStyleSheet("color: #6e6e73;")
        root.addWidget(title)
        root.addWidget(subtitle)

        root.addWidget(self._create_labeled_section("数据集", "dataset_nav", empty_text="暂无目录"))
        root.addWidget(self._create_labeled_section("标签", "output_nav", empty_text="暂无目录"))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        image_group = QWidget()
        image_layout = QVBoxLayout(image_group)
        image_layout.setContentsMargins(8, 8, 8, 8)
        image_layout.addWidget(QLabel("待复核图片 · output"))
        self.image_list = _ImageListWidget()
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.setMovement(QListWidget.Movement.Static)
        self.image_list.setWrapping(True)
        self.image_list.setWordWrap(True)
        self.image_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list.setUniformItemSizes(True)
        self.image_list.setIconSize(QSize(self.THUMBNAIL_BASE_SIZE, self.THUMBNAIL_BASE_SIZE))
        self.image_list.setGridSize(QSize(190, 220))
        self.image_list.verticalScrollBar().valueChanged.connect(self._schedule_thumbnail_load)
        self.image_list.itemDoubleClicked.connect(self._open_preview_item)
        self.image_list.itemSelectionChanged.connect(self._selection_changed)
        image_layout.addWidget(self.image_list, 1)
        image_group.setStyleSheet("background:white; border:1px solid #d1d1d6; border-radius:8px;")

        target = QWidget()
        target_layout = QVBoxLayout(target)
        target_layout.setContentsMargins(8, 8, 8, 8)
        target_layout.addWidget(QLabel("目标位置 · output/<目标数据集>/<子目录>"))
        target_dataset_section = self._create_labeled_section(
            "目标数据集", "target_nav", empty_text="无可选数据集", scale=1.2
        )
        target_layout.addWidget(target_dataset_section)
        self.target_hint = QLabel("请先选择目标数据集")
        self.target_hint.setStyleSheet("color:#6e6e73;")
        target_layout.addWidget(self.target_hint)

        subdir_label = QLabel("目标目录")
        subdir_label.setStyleSheet(f"""
            background-color: #E67E22;
            color: white;
            border-radius: {3 * 1.2:.1f}px;
            padding: {4 * 1.2:.1f}px {12 * 1.2:.1f}px;
            font-weight: bold;
            font-size: {10 * 1.2:.1f}pt;
        """)
        subdir_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        target_layout.addWidget(subdir_label)

        subdir_container = QWidget()
        subdir_layout = QHBoxLayout(subdir_container)
        subdir_layout.setContentsMargins(0, 0, 0, 0)
        self.letter_nav = QWidget()
        self.letter_layout = QVBoxLayout(self.letter_nav)
        self.letter_layout.setContentsMargins(2, 2, 2, 2)
        self.letter_layout.setSpacing(3)
        subdir_layout.addWidget(self.letter_nav)
        self.subdir_scroll = QScrollArea()
        self.subdir_scroll.setWidgetResizable(True)
        self.subdir_content = QWidget()
        self.subdir_grid = QGridLayout(self.subdir_content)
        self.subdir_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.subdir_grid.setContentsMargins(0, 0, 0, 0)
        self.subdir_grid.setHorizontalSpacing(4)
        self.subdir_grid.setVerticalSpacing(4)
        self.subdir_scroll.setWidget(self.subdir_content)
        subdir_layout.addWidget(self.subdir_scroll, 1)
        target_layout.addWidget(subdir_container, 1)

        actions = QHBoxLayout()
        create_folder = QPushButton("新建目标文件夹")
        create_folder.clicked.connect(self.create_new_folder)
        self.move_btn = QPushButton("移动图片")
        self.move_btn.clicked.connect(self.move_images)
        actions.addWidget(create_folder)
        actions.addWidget(self.move_btn)
        target_layout.addLayout(actions)
        splitter.addWidget(image_group)
        splitter.addWidget(target)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        controls = QHBoxLayout()
        self._add_action_button(controls, "反选", self.invert_selection)
        self._add_action_button(controls, "移除选中图片（仅 output）", self.remove_selected_images)
        delete_button = self._add_action_button(controls, "删除选中图片", self.delete_selected_images)
        delete_button.setStyleSheet("background:#ff3b30;color:white;")
        archive_button = self._add_action_button(controls, "归档全部 output", self.archive_output)
        archive_button.setStyleSheet("background:#007aff;color:white;")
        controls.addSpacing(8)
        controls.addWidget(QLabel("缩略图"))
        minus = QPushButton("−")
        minus.clicked.connect(self.decrease_zoom)
        controls.addWidget(minus)
        self.zoom_entry = QLineEdit("100")
        self.zoom_entry.setFixedWidth(54)
        self.zoom_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_entry.editingFinished.connect(self.on_zoom_entry_change)
        controls.addWidget(self.zoom_entry)
        controls.addWidget(QLabel("%"))
        plus = QPushButton("+")
        plus.clicked.connect(self.increase_zoom)
        controls.addWidget(plus)
        controls.addStretch(1)
        self.recycle_checkbox = QCheckBox("删除到回收站" if send2trash else "回收站不可用（将永久删除）")
        self.recycle_checkbox.setChecked(send2trash is not None)
        self.recycle_checkbox.setEnabled(send2trash is not None)
        self.recycle_checkbox.toggled.connect(self._set_recycle_bin_enabled)
        controls.addWidget(self.recycle_checkbox)
        root.addLayout(controls)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.set_status("请选择 output 中的数据集开始整理")

    def _create_labeled_section(
            self,
            title: str,
            attribute_name: str,
            empty_text: str,
            scale: float = 1.0,
    ) -> QWidget:
        """创建一个带有前置标签的流式布局区域，用于数据集和标签的紧凑展示。

        :param title: 前置标签的文本内容。
        :param attribute_name: 内部流式容器在实例上绑定的属性名称。
        :param empty_text: 流式容器为空时展示的占位文案。
        :param scale: 视觉缩放比例，用于调整字体、内边距和圆角大小，默认为 1.0。
        :return: 包含前置标签和流式布局容器的 QWidget。
        """
        container = QWidget()
        # 显式声明水平扩展：否则在紧凑面板中容器可能只按 sizeHint 占位，
        # 导致内部的 _FlowWidget 拿不到完整可用宽度、退化成单列。
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QGridLayout(container)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)  # 仅控制橙色标签与右侧流式区之间的距离

        label = QLabel(title)
        label.setStyleSheet(f"""
            background-color: #E67E22;
            color: white;
            border-radius: {3 * scale:.1f}px;
            padding: {4 * scale:.1f}px {12 * scale:.1f}px;
            font-weight: bold;
            font-size: {10 * scale:.1f}pt;
        """)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(label, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        flow_widget = _FlowWidget(empty_text=empty_text)
        setattr(self, attribute_name, flow_widget)
        layout.addWidget(flow_widget, 0, 1)
        layout.setColumnStretch(1, 1)

        # 限定样式作用域，仅影响该区域内的 QToolButton。
        container.setStyleSheet(f"""
            QToolButton {{
                background-color: #4CAF50;
                color: white;
                border-radius: {3 * scale:.1f}px;
                padding: {2 * scale:.1f}px {4 * scale:.1f}px;
                margin: 0px;
                font-size: {9.5 * scale:.1f}pt;
                border: none;
            }}
            QToolButton:hover {{
                background-color: #45a049;
            }}
            QToolButton:checked {{
                background-color: #2E7D32;
                border: 1px solid #1B5E20;
            }}
        """)
        return container

    @staticmethod
    def _add_action_button(layout: QHBoxLayout, text: str, callback: Any) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _bind_shortcuts(self) -> None:
        next_action = QAction(self)
        next_action.setShortcut(QKeySequence("Ctrl+Tab"))
        next_action.triggered.connect(self.switch_to_next_output_dir)
        self.addAction(next_action)
        previous_action = QAction(self)
        previous_action.setShortcut(QKeySequence("Ctrl+Shift+Tab"))
        previous_action.triggered.connect(self.switch_to_prev_output_dir)
        self.addAction(previous_action)

    def _start_worker(
        self,
        worker: _FunctionWorker,
        on_finished: Callable[[Any], None],
        on_failed: Callable[[str], None] | None = None,
    ) -> None:
        """把后台任务提交到线程池，并保证回调在 GUI 主线程执行。

        在 PySide6 中，信号连接到普通 Python 可调用对象（lambda、函数）时
        没有接收者上下文，可能以 **直接连接** 的方式在**工作线程**中执行。
        一旦回调触碰 ``QPixmap`` / ``QListWidgetItem`` 等只允许在 GUI 线程
        使用的对象，就会产生访问违例（0xC0000005）或控件状态随机丢失。
        这里统一使用 ``Qt.ConnectionType.QueuedConnection`` 强制回调回到
        主线程，并在回调执行完毕前保持对 worker 的强引用，避免
        ``QRunnable`` 自动删除后其 ``_WorkerSignals`` 被提前回收。

        :param worker: 待执行的 :class:`_FunctionWorker`。
        :param on_finished: 成功时在主线程调用的回调，接收返回值。
        :param on_failed: 失败时在主线程调用的回调，接收错误文本；可为 ``None``。
        """
        if self._closing:
            return
        self._active_workers.add(worker)

        def _release() -> None:
            self._active_workers.discard(worker)

        def _handle_finished(result: Any) -> None:
            try:
                on_finished(result)
            finally:
                _release()

        def _handle_failed(message: str) -> None:
            try:
                if on_failed is not None:
                    on_failed(message)
            finally:
                _release()

        worker.signals.finished.connect(_handle_finished, Qt.ConnectionType.QueuedConnection)
        worker.signals.failed.connect(_handle_failed, Qt.ConnectionType.QueuedConnection)
        self._pool.start(worker)

    def closeEvent(self, event: Any) -> None:
        """关闭窗口前等待后台任务收敛，避免回调访问已销毁的控件。"""
        self._closing = True
        if not self._pool.waitForDone(self.SHUTDOWN_WAIT_MS):
            LOGGER.warning("关闭时仍有后台任务未结束，等待超时")
        super().closeEvent(event)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._schedule_thumbnail_load)
        if hasattr(self, "dataset_nav"):
            self.refresh_dataset_nav()
            self.refresh_output_dirs()
            self.refresh_target_dataset_selector()

    def _fill_flow(self, widget: _FlowWidget, names: list[str], selected: str, callback: Any) -> None:
        """把条目写入 :class:`_FlowWidget`。

        列数由容器在尺寸稳定后自行计算并重排，调用方不再依赖瞬时的
        ``width()``，从而避免“选中后被挤成一列”的问题。

        :param widget: 目标流式容器。
        :param names: 待展示的条目名称。
        :param selected: 当前应被勾选的条目名；不在 ``names`` 中则无勾选。
        :param callback: 条目被点击时调用的回调，签名 ``callback(name: str) -> None``。
        """
        widget.set_entries(names, selected, callback)

    def _set_controls_enabled(self, enabled: bool) -> None:
        if self.centralWidget():
            self.centralWidget().setEnabled(enabled)

    def _clear_fs_caches(self) -> None:
        self._directory_cache.clear()
        self._image_list_cache.clear()
        self._thumbnail_cache.clear()
        self._thumbnail_pending.clear()

    @staticmethod
    def _format_errors(errors: list[str], heading: str) -> str:
        visible = errors[:5]
        suffix = "\n…" if len(errors) > len(visible) else ""
        return f"\n\n{heading}（{len(errors)} 个）：\n" + "\n".join(visible) + suffix

    def set_status(self, message: object) -> None:
        self.statusBar().showMessage(" ".join(str(message).split()))
