"""通用 GUI 工作线程和小型自定义控件。"""
from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QGridLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QToolButton, QWidget,
)

LOGGER = logging.getLogger("manual_classify_gui")


class _WorkerSignals(QObject):
    """后台任务向 GUI 线程安全发送结果。"""
    finished = Signal(object)
    failed = Signal(str)

class _FunctionWorker(QRunnable):
    """在线程池中运行纯计算或文件 I/O 函数。

    ``run`` 只负责产生结果并发射信号；任何 UI 更新都必须由调用方通过
    :meth:`ImageClassifierUI._start_worker` 以队列连接的方式在主线程完成。
    """
    def __init__(self, function: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception as error:  # 交由 GUI 线程报告，避免线程异常静默丢失
            LOGGER.exception("后台任务失败")
            self.signals.failed.emit(str(error))
            return
        self.signals.finished.emit(result)

class _FlowWidget(QWidget):
    """按内容尺寸靠左排列的可勾选按钮流式容器。

    容器自行保存条目、选中项与回调，并在尺寸变化时按当前宽度重新计算
    列数，保证按钮先横向铺满可用宽度，放不下时才换到下一行。

    重新布局被推迟到事件循环空闲时执行：在 ``__init__`` 或父级
    ``resizeEvent`` 触发的瞬间，``width()`` 往往返回尚未稳定的临时值
    （可能是默认 640，也可能是刚被压缩的极小值），若立刻据此计算列数，
    会把按钮一次性放入单列且不再重排——这正是“刚打开就是一列”的根因。
    """

    # 单个按钮的最小占位宽度（像素）；宽度不足以容纳该值时退化为单列。
    # 该值仅用于估算列数，网格单元格仍会按内容自适应实际宽度。
    BUTTON_MIN_WIDTH = 60

    def __init__(self, empty_text: str, parent: QWidget | None = None) -> None:
        """
        :param empty_text: 条目为空时展示的占位文案。
        :param parent: 可选的父控件。
        """
        super().__init__(parent)
        self._empty_text = empty_text
        self._names: list[str] = []
        self._selected: str = ""
        self._callback: Callable[[str], None] | None = None
        self._columns: int = 0
        self._rebuild_pending: bool = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

    def set_entries(
        self,
        names: list[str],
        selected: str,
        callback: Callable[[str], None],
    ) -> None:
        """用最新的条目、选中项和回调重建按钮。

        :param names: 待展示的条目名称列表，顺序即显示顺序。
        :param selected: 当前被勾选的条目名；不在 ``names`` 中时表示无勾选。
        :param callback: 条目被点击时的回调，签名为 ``callback(name: str) -> None``。
        """
        self._names = list(names)
        self._selected = selected
        self._callback = callback
        self._columns = 0  # 强制下一次重建时重新计算列数
        self._schedule_rebuild()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._schedule_rebuild()

    def _schedule_rebuild(self) -> None:
        """把重建推迟到事件循环空闲时执行；多次请求会被合并为一次。"""
        if self._rebuild_pending:
            return
        self._rebuild_pending = True
        QTimer.singleShot(0, self._do_rebuild)

    def _do_rebuild(self) -> None:
        self._rebuild_pending = False
        self._rebuild()

    def _rebuild(self) -> None:
        columns = max(1, self.width() // self.BUTTON_MIN_WIDTH)
        if columns == self._columns and self.layout().count() > 0:
            return
        self._columns = columns
        layout = self.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not self._names:
            layout.addWidget(QLabel(self._empty_text), 0, 0)
            return
        for index, name in enumerate(self._names):
            button = QToolButton()
            button.setText(name)
            button.setCheckable(True)
            button.setChecked(name == self._selected)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.clicked.connect(
                lambda _checked=False, value=name: self._invoke(value)
            )
            layout.addWidget(button, index // columns, index % columns)
        # 列拉伸保持为 0；容器和按钮均按内容尺寸靠左，不吸收剩余宽度。
        for column in range(layout.columnCount()):
            layout.setColumnStretch(column, 0)

    def _invoke(self, name: str) -> None:
        if self._callback is not None:
            self._callback(name)

class _ImageListWidget(QListWidget):
    """让普通单击独立切换图片选择状态，保持旧版交互习惯。

    同时禁用框选，并支持 Shift 连选：先单击一个图片作为锚点，
    再按住 Shift 单击另一个图片，自动选中两者之间的所有图片。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._anchor_index: int | None = None

    def clear_anchor(self) -> None:
        """重置 Shift 连选锚点，通常在图片列表重建后调用。"""
        self._anchor_index = None

    def mousePressEvent(self, event: Any) -> None:
        item = self.itemAt(event.position().toPoint())
        if item is not None and event.button() == Qt.MouseButton.LeftButton:
            modifiers = event.modifiers()
            current_row = self.row(item)
            if modifiers & Qt.KeyboardModifier.ShiftModifier and self._anchor_index is not None:
                # Shift 连选：选中锚点与当前项之间的所有图片。
                start = min(self._anchor_index, current_row)
                end = max(self._anchor_index, current_row)
                for index in range(start, end + 1):
                    target = self.item(index)
                    if target is not None:
                        target.setSelected(True)
                event.accept()
                return
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                item.setSelected(not item.isSelected())
                self._anchor_index = current_row
                event.accept()
                return
            # 普通单击：切换当前项选中状态，并记录为新的锚点。
            item.setSelected(not item.isSelected())
            self._anchor_index = current_row
            event.accept()
            return
        self._anchor_index = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        # 禁用框选，避免误触；列表滚动仍可通过滚轮或滚动条进行。
        event.ignore()

    def mouseDoubleClickEvent(self, event: Any) -> None:
        item = self.itemAt(event.position().toPoint())
        if item is not None and event.button() == Qt.MouseButton.LeftButton:
            self.itemDoubleClicked.emit(item)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: Any) -> None:
        """把图片列表的滚轮滚动量调整为默认值的 1/3。"""
        scrollbar = self.verticalScrollBar()
        before = scrollbar.value()
        # 先让 QListWidget 按默认行为滚动一次
        super().wheelEvent(event)
        after = scrollbar.value()
        delta = after - before
        if delta:
            # 默认步长通常是 3 个条目，这里改成 1/3
            new_delta = int(delta / 3)
            if new_delta == 0:
                # 避免小步长被截断为 0，至少滚动 1 个单位
                new_delta = 1 if delta > 0 else -1
            scrollbar.setValue(before + new_delta)

class _DoubleClickButton(QPushButton):
    """提供可选目标目录双击后立即移动的交互。"""
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)
