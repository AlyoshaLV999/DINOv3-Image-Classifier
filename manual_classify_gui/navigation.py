"""数据集和目标目录导航逻辑。"""
from __future__ import annotations

import os
import locale
from collections import defaultdict
from PySide6.QtCore import Qt, QTimer, QRect, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget, QGraphicsOpacityEffect

from .widgets import _DoubleClickButton


class NavigationMixin:
    """数据集、输出目录和目标目录导航。"""

    def refresh_dataset_nav(self) -> None:
        names = self.get_output_datasets()
        if self.current_dataset not in names:
            self.current_dataset = ""
        self._fill_flow(self.dataset_nav, names, self.current_dataset, self.select_dataset)

    def refresh_output_dirs(self) -> None:
        names = self.get_label_dirs()
        selected = os.path.basename(self.current_output_dir) if self.current_output_dir else ""
        self._fill_flow(self.output_nav, names, selected, self.load_images)

    def select_dataset(self, dataset_name: str) -> None:
        self.current_dataset = dataset_name
        self.current_output_dir = ""
        self.current_output_index = -1
        self.selected_images.clear()
        self.current_image_paths = []
        self.image_list.clear()
        self.refresh_dataset_nav()
        self.refresh_output_dirs()
        self.set_status(f"已选择数据集：{dataset_name}。请选择 output 标签目录。")

    def get_output_datasets(self) -> list[str]:
        return self._get_child_dirs("output")

    def get_label_dirs(self) -> list[str]:
        return self._get_child_dirs(os.path.join("output", self.current_dataset)) if self.current_dataset else []

    def get_target_datasets(self) -> list[str]:
        names = set(self.get_output_datasets()) | set(self._get_child_dirs("cache"))
        return sorted(names, key=locale.strxfrm)

    def refresh_target_dataset_selector(self) -> None:
        names = self.get_target_datasets()
        if self.target_dataset not in names:
            self.target_dataset = ""
            self.target_dir = ""
        self._fill_flow(self.target_nav, names, self.target_dataset, self.select_target_dataset)
        self.move_btn.setEnabled(bool(self.target_dataset and self.target_dir and self.selected_images))

    def select_target_dataset(self, name: str) -> None:
        if name == self.target_dataset:
            return
        self.target_dataset = name
        self.target_dir = ""
        self.refresh_target_dataset_selector()
        self.load_target_dirs()
        self.set_status(f"已选择目标数据集：{name}。请选择目标子目录。")

    def get_target_root(self) -> str:
        return os.path.join("cache", self.target_dataset, "images") if self.target_dataset else ""

    def get_target_dirs(self) -> list[str]:
        if not self.target_dataset:
            return []
        names = set(self._get_child_dirs(self.get_target_root()))
        names.update(self._get_child_dirs(os.path.join("output", self.target_dataset)))
        return sorted(names, key=locale.strxfrm)

    def _count_cache_tag_files(self, tag: str) -> int:
        """统计 cache 中当前目标数据集下指定 tag 目录内的文件数量。

        数量用于在目标目录按钮上显示，并据此切换按钮配色。只统计直接
        子文件，不递归子目录；目录不存在或不可读时返回 0。

        :param tag: 目标子目录名称。
        :return: 文件数量。
        """
        if not self.target_dataset:
            return 0
        path = os.path.join("cache", self.target_dataset, "images", tag)
        try:
            with os.scandir(path) as entries:
                return sum(1 for entry in entries if entry.is_file())
        except OSError:
            return 0

    def load_target_dirs(self) -> None:
        self._clear_layout(self.subdir_grid)
        self._clear_layout(self.letter_layout)
        self._letter_widgets = defaultdict(list)
        if not self.target_dataset:
            self.target_hint.setText("请先选择目标数据集")
            self.refresh_target_dataset_selector()
            return
        self.target_hint.setText(f"目录来自 {self.get_target_root()} 和 output/{self.target_dataset}")
        names = self.get_target_dirs()
        if self.target_dir not in names:
            self.target_dir = ""
        grouped: dict[str, list[str]] = defaultdict(list)
        for name in names:
            grouped[self.get_first_letter(name)].append(name)
        row = 0
        for letter, group in sorted(grouped.items()):
            label = QLabel(letter)
            label.setStyleSheet("font-weight:bold;color:#6e6e73;")
            label.setObjectName(f"letter_{letter}")
            self.subdir_grid.addWidget(label, row, 0, 1, 3)
            self._letter_widgets[letter].append(label)
            row += 1
            for index, name in enumerate(group):
                count = self._count_cache_tag_files(name)
                display_text = f"{name}({count})"
                button = _DoubleClickButton(display_text)
                button.setProperty("target_dir_name", name)
                button.setCheckable(True)
                button.setChecked(name == self.target_dir)
                if count < self.TARGET_DIR_LOW_COUNT_THRESHOLD:
                    # 数量不足时使用淡红色，提示该目录样本较少。
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: #FFCDD2;
                            color: #B71C1C;
                            border-radius: {3 * 1.2:.1f}px;
                            padding: {2 * 1.2:.1f}px {4 * 1.2:.1f}px;
                            margin: 0px;
                            font-size: {9.5 * 1.2:.1f}pt;
                            border: none;
                        }}
                        QPushButton:hover {{
                            background-color: #EF9A9A;
                        }}
                        QPushButton:checked {{
                            background-color: #E57373;
                            border: 1px solid #B71C1C;
                            color: white;
                        }}
                    """)
                else:
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: #4CAF50;
                            color: white;
                            border-radius: {3 * 1.2:.1f}px;
                            padding: {2 * 1.2:.1f}px {4 * 1.2:.1f}px;
                            margin: 0px;
                            font-size: {9.5 * 1.2:.1f}pt;
                            border: none;
                        }}
                        QPushButton:hover {{
                            background-color: #45a049;
                        }}
                        QPushButton:checked {{
                            background-color: #2E7D32;
                            border: 1px solid #1B5E20;
                        }}
                    """)
                button.clicked.connect(lambda _checked=False, value=name: self.select_target_dir(value))
                button.doubleClicked.connect(self.move_images)
                self.subdir_grid.addWidget(button, row + index // 3, index % 3)
                self._letter_widgets[letter].append(button)
            row += (len(group) + 2) // 3
            jump = QToolButton()
            jump.setText(letter)
            jump.setFixedSize(22, 22)
            jump.setStyleSheet("""
                QToolButton {
                    background: white;
                    color: #007aff;
                    border: 1px solid #007aff;
                    border-radius: 11px;
                    font-size: 8pt;
                    padding: 0px;
                }
                QToolButton:hover {
                    background: #eaf4ff;
                }
            """)
            jump.clicked.connect(lambda _checked=False, value=letter: self._jump_to_subdir_letter(value))
            self.letter_layout.addWidget(jump)
        self.letter_layout.addStretch(1)
        self.move_btn.setEnabled(bool(self.target_dir and self.selected_images))

    @staticmethod
    def _clear_layout(layout: Any) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                NavigationMixin._clear_layout(item.layout())

    def _jump_to_subdir_letter(self, letter: str) -> None:
        """将对应字母分组的标签滚动到子目录滚动区顶部，并高亮该分组。

        索引点击后强制把该字母所在行对齐到滚动区顶部，方便用户快速定位，
        同时短暂高亮该字母分组区域，帮助视觉确认。

        :param letter: 要跳转并高亮的字母分组。
        """
        label = self.subdir_content.findChild(QLabel, f"letter_{letter}")
        if label:
            self.subdir_scroll.verticalScrollBar().setValue(label.y())
            QTimer.singleShot(0, lambda: self._highlight_letter_group(letter))

    def _highlight_letter_group(self, letter: str) -> None:
        """高亮指定字母分组的整个矩形区域，并在 0.5 秒内淡出。

        覆盖层作为 ``subdir_content`` 的子控件，不拦截鼠标事件，
        仅用于视觉反馈。动画结束后自动删除。

        :param letter: 要高亮显示的字母分组。
        """
        widgets = self._letter_widgets.get(letter)
        if not widgets:
            return
        rect = QRect()
        for widget in widgets:
            rect = rect.united(widget.geometry())
        if rect.isNull() or rect.isEmpty():
            return
        padding = 4
        rect.adjust(-padding, -padding, padding, padding)

        overlay = QWidget(self.subdir_content)
        overlay.setGeometry(rect)
        overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        overlay.setStyleSheet("background-color: rgba(255, 235, 59, 180); border-radius: 6px;")
        overlay.show()

        effect = QGraphicsOpacityEffect(overlay)
        overlay.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", overlay)
        animation.setDuration(3000)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(overlay.deleteLater)
        animation.start()

    def select_target_dir(self, directory_name: str) -> None:
        self.target_dir = directory_name
        self.refresh_target_selector_buttons()
        self.move_btn.setEnabled(bool(self.selected_images))
        self.set_status(f"目标目录：output/{self.target_dataset}/{directory_name}")

    def refresh_target_selector_buttons(self) -> None:
        """根据 ``self.target_dir`` 同步目标目录按钮的选中状态。

        按钮文本包含数量后缀（如 ``tag(54)``），因此不能直接比较文本，
        而是通过创建时保存的 ``target_dir_name`` 属性判断。
        """
        for button in self.subdir_content.findChildren(QPushButton):
            button.setChecked(button.property("target_dir_name") == self.target_dir)
