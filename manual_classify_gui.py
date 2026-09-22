"""高响应图像分类整理界面。

该工具将 ``output/<dataset>/<label>`` 中的推理结果作为待整理图片，
目标数据集可独立选择；目标标签由 ``cache/<dataset>/images`` 与
``output/<dataset>`` 的一级子目录共同提供。归档会把 ``output`` 中的所有
数据集递归合并到 ``datasets``；忽略用于保留目录的 ``.gitkeep``，并在每个
文件均成功移动后清理 ``input`` 中相应数据集内同名的源文件。
"""

from __future__ import annotations

import locale
import os
import platform
import shutil
import tkinter as tk
from collections import OrderedDict, defaultdict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from tkinter import simpledialog, ttk

from PIL import Image, ImageOps, ImageTk

try:
    from send2trash import send2trash
except ImportError:  # 可选依赖；未安装时仍可选择永久删除。
    send2trash = None


class ImageClassifierUI:
    """用于复核和整理推理分类结果的 Tkinter 应用。"""

    IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    THUMBNAIL_BASE_SIZE = 164
    THUMBNAIL_CACHE_SIZE = 384
    THUMBNAIL_PREFETCH_ROWS = 1
    THUMBNAIL_MAX_PENDING = 32
    THUMBNAIL_BATCH_RESULT_LIMIT = 12
    DIRECTORY_CACHE_SIZE = 64
    IMAGE_LIST_CACHE_SIZE = 64
    SUBDIR_INDEX_BAR_WIDTH = 26
    SUBDIR_INDEX_FONT_SIZE = 9
    SUBDIR_INDEX_FOREGROUND = "#007AFF"
    SUBDIR_INDEX_BACKGROUND = "#FFFFFF"
    SUBDIR_INDEX_HOVER_BACKGROUND = "#EAF4FF"
    SUBDIR_INDEX_HOVER_FOREGROUND = "#0055CC"

    def __init__(self, root: tk.Tk) -> None:
        """
        初始化图像分类整理界面及其有界缓存、异步缩略图解码器。
        :param root: 作为应用主窗口的 Tk 根对象
        :return: 无返回值
        """
        self.root = root
        self.root.title("图像分类整理")
        self.root.configure(background="#F2F2F7")

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = min(max(1080, int(screen_width * 0.82)), screen_width)
        window_height = min(max(720, int(screen_height * 0.82)), screen_height)
        x = max(0, (screen_width - window_width) // 2)
        y = max(0, (screen_height - window_height) // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(960, 620)

        try:
            locale.setlocale(locale.LC_ALL, "")
        except locale.Error:
            pass

        self.selected_images = set()
        self.current_output_dir = ""
        self.current_dataset = ""
        self.target_dataset = ""
        self.current_output_index = -1
        self.target_dir = ""
        self.zoom_scale = 100
        self.current_image_paths = []

        self.click_timer = None
        self._resize_timer = None
        self._scroll_update_job = None
        self._render_job = None
        self._render_generation = 0
        self._rendered_cards = {}
        self._grid_columns = 1
        self._grid_cell_width = 1
        self._grid_cell_height = 1
        self._grid_thumb_size = self.THUMBNAIL_BASE_SIZE
        self._thumbnail_cache = OrderedDict()
        self._thumbnail_futures = {}
        self._thumbnail_results = Queue()
        self._thumbnail_poll_job = None
        worker_count = max(2, min(6, (os.cpu_count() or 2)))
        self._thumbnail_executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="thumbnail")
        self._directory_cache = OrderedDict()
        self._image_list_cache = OrderedDict()
        self._is_closing = False
        self._thumbnail_error_count = 0
        self._failed_thumbnail_keys = set()
        self._last_layout_signature = None
        self._last_status_message = ""
        self._subdir_letter_labels: dict[str, ttk.Label] = {}
        self._subdir_letter_offsets: dict[str, int] = {}
        self._subdir_index_job = None

        self.recycle_bin_enabled = tk.BooleanVar(value=send2trash is not None)
        self.status_var = tk.StringVar(value="请选择 output 中的数据集开始整理")
        self.target_dataset_var = tk.StringVar()
        self.target_hint_var = tk.StringVar(value="请先选择目标数据集")

        self._setup_styles()
        self.create_widgets()
        self.setup_scroll_management()
        self.bind_events()

        self.refresh_dataset_nav()
        self.refresh_target_dataset_selector()
        self.root.bind("<Control-Tab>", self.switch_to_next_output_dir)
        self.root.bind("<Control-Shift-Tab>", self.switch_to_prev_output_dir)
        self.root.bind("<Configure>", self.on_window_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------- 界面和样式 ----------------- #

    def _setup_styles(self) -> None:
        """
        配置接近 iOS 的浅色卡片、圆润留白与高对比操作色。
        :return: 无返回值
        """
        style = ttk.Style(self.root)
        style.theme_use("clam")

        background = "#F2F2F7"
        card = "#FFFFFF"
        text = "#1C1C1E"
        secondary_text = "#6E6E73"
        separator = "#D1D1D6"
        blue = "#007AFF"
        blue_active = "#0066D6"
        red = "#FF3B30"
        red_active = "#D92D25"

        style.configure(".", background=background, foreground=text, font=("Arial", 10))
        style.configure("TFrame", background=background)
        style.configure("Header.TFrame", background=background)
        style.configure("Card.TFrame", background=card, relief="flat")
        style.configure("TLabel", background=background, foreground=text)
        style.configure("Title.TLabel", background=background, foreground=text, font=("Arial", 20, "bold"))
        style.configure("Subtitle.TLabel", background=background, foreground=secondary_text, font=("Arial", 10))
        style.configure("Status.TLabel", background=background, foreground=secondary_text, font=("Arial", 9))
        style.configure("Hint.TLabel", background=card, foreground=secondary_text, font=("Arial", 9))
        style.configure(
            "TLabelframe",
            background=card,
            borderwidth=1,
            relief="solid",
            bordercolor=separator,
            padding=10,
        )
        style.configure("TLabelframe.Label", background=card, foreground=text, font=("Arial", 10, "bold"))
        style.configure("TButton", background=card, foreground=text, bordercolor=separator, padding=(10, 7), relief="flat")
        style.map(
            "TButton",
            background=[("active", "#E9E9EE"), ("disabled", "#E5E5EA")],
            foreground=[("disabled", "#8E8E93")],
        )
        style.configure("Nav.TButton", padding=(10, 6), font=("Arial", 9))
        style.configure("SubDir.TButton", padding=(9, 6), font=("Arial", 9))
        style.configure("Selected.TButton", background=blue, foreground="white", bordercolor=blue, padding=(10, 6))
        style.map(
            "Selected.TButton",
            background=[("active", blue_active), ("!active", blue)],
            foreground=[("!disabled", "white")],
        )
        style.configure("Accent.TButton", background=blue, foreground="white", bordercolor=blue, padding=(12, 8))
        style.map(
            "Accent.TButton",
            background=[("active", blue_active), ("disabled", "#A9D5FF")],
            foreground=[("disabled", "#F7FBFF")],
        )
        style.configure("Danger.TButton", background=red, foreground="white", bordercolor=red, padding=(12, 8))
        style.map(
            "Danger.TButton",
            background=[("active", red_active), ("disabled", "#FFB3AE")],
            foreground=[("disabled", "#FFF7F6")],
        )
        style.configure("Thumbnail.TFrame", background=card, borderwidth=1, relief="solid", bordercolor="#E5E5EA")
        style.configure("Selected.TFrame", background="#EAF4FF", borderwidth=2, relief="solid", bordercolor=blue)
        style.configure("TCheckbutton", background=background, foreground=text, padding=(4, 2))
        style.configure("TEntry", fieldbackground=card, bordercolor=separator, padding=5)

    def create_widgets(self) -> None:
        """
        创建主窗口、导航、图片网格以及操作面板。
        :return: 无返回值
        """
        header = ttk.Frame(self.root, style="Header.TFrame")
        header.pack(fill=tk.X, padx=18, pady=(16, 8))
        ttk.Label(header, text="图像分类整理", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="复核 output 推理结果，选择独立目标数据集与目录后归类，并可安全归档。",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(3, 0))

        dataset_bar = ttk.LabelFrame(self.root, text="数据集 · output")
        dataset_bar.pack(fill=tk.X, padx=18, pady=(0, 8))
        self.dataset_nav_frame = ttk.Frame(dataset_bar, style="Card.TFrame")
        self.dataset_nav_frame.pack(fill=tk.X)

        label_bar = ttk.LabelFrame(self.root, text="当前数据集的标签目录 · output")
        label_bar.pack(fill=tk.X, padx=18, pady=(0, 8))
        self.output_nav_frame = ttk.Frame(label_bar, style="Card.TFrame")
        self.output_nav_frame.pack(fill=tk.X)

        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 8))
        self.main_paned.add(self.setup_image_panel(), weight=7)
        self.main_paned.add(self.setup_control_panel(), weight=3)

        self.setup_bottom_controls()

    def setup_image_panel(self) -> ttk.LabelFrame:
        """
        创建带滚动条的虚拟化图片展示面板。
        :return: 已配置完成的图片面板
        """
        panel = ttk.LabelFrame(self.main_paned, text="待复核图片 · output")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(panel, background="#FFFFFF", highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas, style="Card.TFrame")
        self.scroll_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor=tk.NW)
        self.canvas.configure(yscrollcommand=self._on_canvas_yview)

        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.scrollbar.grid(row=0, column=1, sticky=tk.NS)
        return panel

    def setup_control_panel(self) -> ttk.Frame:
        """
        创建目标数据集、目标目录和移动操作面板。

        目标子目录区域从左到右依次为首字母索引条、子目录画布与滚动条，
        索引条用于在大批标签中快速定位到指定首字母分组。
        :return: 已配置完成的控制面板
        """
        right_frame = ttk.Frame(self.main_paned)

        target_container = ttk.LabelFrame(right_frame, text="目标位置 · output/<目标数据集>/<子目录>")
        target_container.pack(fill=tk.BOTH, expand=True)
        dataset_selector = ttk.Frame(target_container, style="Card.TFrame")
        dataset_selector.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(dataset_selector, text="目标数据集：", style="Hint.TLabel").pack(side=tk.LEFT)
        self.target_dataset_combobox = ttk.Combobox(
            dataset_selector,
            textvariable=self.target_dataset_var,
            state="readonly",
        )
        self.target_dataset_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.target_dataset_combobox.bind("<<ComboboxSelected>>", self.on_target_dataset_selected)
        ttk.Label(target_container, textvariable=self.target_hint_var, style="Hint.TLabel").pack(anchor=tk.W, pady=(0, 8))

        self.subdir_area = ttk.Frame(target_container, style="Card.TFrame")
        self.subdir_area.pack(fill=tk.BOTH, expand=True)

        self.subdir_index_bar = ttk.Frame(
            self.subdir_area,
            style="Card.TFrame",
            width=self.SUBDIR_INDEX_BAR_WIDTH,
        )
        self.subdir_index_bar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        self.subdir_index_bar.pack_propagate(False)

        self.subdir_canvas = tk.Canvas(self.subdir_area, background="#FFFFFF", highlightthickness=0, borderwidth=0)
        self.subdir_scrollbar = ttk.Scrollbar(self.subdir_area, orient=tk.VERTICAL, command=self.subdir_canvas.yview)
        self.subdir_frame = ttk.Frame(self.subdir_canvas, style="Card.TFrame")
        self.subdir_canvas_window = self.subdir_canvas.create_window((0, 0), window=self.subdir_frame, anchor=tk.NW)
        self.subdir_canvas.configure(yscrollcommand=self.subdir_scrollbar.set)
        self.subdir_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.subdir_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.subdir_frame.bind("<Configure>", self._on_subdir_frame_configure)
        self.subdir_canvas.bind("<Configure>", self._on_subdir_canvas_configure)
        self.subdir_canvas.bind("<Enter>", self._bind_subdir_mousewheel)
        self.subdir_canvas.bind("<Leave>", self._unbind_subdir_mousewheel)

        actions = ttk.Frame(right_frame)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="新建目标文件夹", command=self.create_new_folder).pack(side=tk.LEFT, padx=(0, 6))
        self.move_btn = ttk.Button(actions, text="移动图片", style="Accent.TButton", command=self.move_images, state=tk.DISABLED)
        self.move_btn.pack(side=tk.LEFT)
        return right_frame

    def setup_bottom_controls(self) -> None:
        """
        创建选择、删除、归档、缩放和状态栏控制项。
        :return: 无返回值
        """
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=18, pady=(0, 7))

        left_buttons = ttk.Frame(control_frame)
        left_buttons.pack(side=tk.LEFT)
        ttk.Button(left_buttons, text="反选", command=self.invert_selection).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(left_buttons, text="移除选中图片（仅 output）", command=self.remove_selected_images).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(left_buttons, text="删除选中图片", style="Danger.TButton", command=self.delete_selected_images).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(left_buttons, text="归档全部 output", style="Accent.TButton", command=self.archive_output).pack(side=tk.LEFT)

        zoom_frame = ttk.LabelFrame(left_buttons, text="缩略图")
        zoom_frame.pack(side=tk.LEFT, padx=12)
        ttk.Button(zoom_frame, text="−", width=3, command=self.decrease_zoom).pack(side=tk.LEFT, padx=(0, 3))
        self.zoom_entry = ttk.Entry(zoom_frame, width=5, justify=tk.CENTER)
        self.zoom_entry.insert(0, "100")
        self.zoom_entry.pack(side=tk.LEFT)
        self.zoom_entry.bind("<Return>", self.on_zoom_entry_change)
        self.zoom_entry.bind("<FocusOut>", self.on_zoom_entry_change)
        ttk.Label(zoom_frame, text="%").pack(side=tk.LEFT, padx=(3, 0))
        ttk.Button(zoom_frame, text="+", width=3, command=self.increase_zoom).pack(side=tk.LEFT, padx=(3, 0))

        recycle_text = "删除到回收站" if send2trash else "回收站不可用（将永久删除）"
        self.recycle_checkbox = ttk.Checkbutton(control_frame, text=recycle_text, variable=self.recycle_bin_enabled)
        self.recycle_checkbox.pack(side=tk.RIGHT)
        if send2trash is None:
            self.recycle_checkbox.state(["disabled"])

        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side=tk.RIGHT, padx=(0, 16))

    def setup_scroll_management(self) -> None:
        """
        绑定图片画布和内容区域的尺寸变化处理。
        :return: 无返回值
        """
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.scroll_frame.bind("<Configure>", self.on_frame_configure)

    def bind_events(self) -> None:
        """
        绑定图片区域的鼠标滚轮事件。
        :return: 无返回值
        """
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.scroll_frame.bind("<MouseWheel>", self.on_mousewheel)
        self.scroll_frame.bind("<Button-4>", self.on_mousewheel)
        self.scroll_frame.bind("<Button-5>", self.on_mousewheel)

    def _bind_mousewheel_to_widget(self, widget: tk.Misc) -> None:
        """
        为指定控件绑定跨平台的鼠标滚轮事件，使鼠标位于图片上时仍可滚动主画布。
        :param widget: 需要绑定滚轮事件的 Tk 控件
        :return: 无返回值
        """
        widget.bind("<MouseWheel>", self.on_mousewheel)
        widget.bind("<Button-4>", self.on_mousewheel)
        widget.bind("<Button-5>", self.on_mousewheel)

    def on_close(self) -> None:
        """
        取消待处理任务、释放异步解码器并关闭主窗口。
        :return: 无返回值
        """
        self._is_closing = True
        self._cancel_thumbnail_render()
        if self._thumbnail_poll_job is not None:
            self.root.after_cancel(self._thumbnail_poll_job)
            self._thumbnail_poll_job = None
        if self._subdir_index_job is not None:
            self.root.after_cancel(self._subdir_index_job)
            self._subdir_index_job = None
        self._thumbnail_executor.shutdown(wait=False, cancel_futures=True)
        self._thumbnail_cache.clear()
        self._thumbnail_futures.clear()
        self._failed_thumbnail_keys.clear()
        self.root.destroy()

    # ----------------- 导航与图片加载 ----------------- #

    def on_window_resize(self, event: tk.Event) -> None:
        """
        延迟处理主窗口尺寸变化，避免连续 Configure 事件重复重建导航。
        :param event: Tkinter 窗口尺寸变化事件
        :return: 无返回值
        """
        if event.widget != self.root:
            return
        if self._resize_timer is not None:
            self.root.after_cancel(self._resize_timer)
        self._resize_timer = self.root.after(180, self._refresh_responsive_layout)

    def _refresh_responsive_layout(self) -> None:
        """
        在窗口尺寸稳定后刷新导航布局和可视图片网格。
        :return: 无返回值
        """
        self._resize_timer = None
        self.adjust_dataset_nav_layout()
        self.adjust_nav_layout()
        self._schedule_virtual_render()

    def adjust_dataset_nav_layout(self) -> None:
        """
        按窗口宽度重建 output 数据集导航。
        :return: 无返回值
        """
        max_cols = max(3, (self.root.winfo_width() - 70) // 130)
        self._clear_children(self.dataset_nav_frame)
        for index, dataset_name in enumerate(self.get_output_datasets()):
            button = ttk.Button(
                self.dataset_nav_frame,
                text=dataset_name,
                style="Selected.TButton" if dataset_name == self.current_dataset else "Nav.TButton",
                command=lambda name=dataset_name: self.select_dataset(name),
            )
            button.grid(row=index // max_cols, column=index % max_cols, padx=3, pady=3, sticky=tk.EW)
        for column in range(max_cols):
            self.dataset_nav_frame.columnconfigure(column, weight=1)

    def adjust_nav_layout(self) -> None:
        """
        按窗口宽度重建当前数据集的 output 标签导航。
        :return: 无返回值
        """
        max_cols = max(3, (self.root.winfo_width() - 70) // 120)
        self._clear_children(self.output_nav_frame)
        selected_name = os.path.basename(self.current_output_dir) if self.current_output_dir else ""
        for index, directory_name in enumerate(self.get_label_dirs()):
            button = ttk.Button(
                self.output_nav_frame,
                text=directory_name,
                style="Selected.TButton" if directory_name == selected_name else "Nav.TButton",
                command=lambda name=directory_name: self.load_images(name),
            )
            button.grid(row=index // max_cols, column=index % max_cols, padx=3, pady=3, sticky=tk.EW)
        for column in range(max_cols):
            self.output_nav_frame.columnconfigure(column, weight=1)

    def refresh_dataset_nav(self) -> None:
        """
        刷新 output 数据集导航。
        :return: 无返回值
        """
        self.adjust_dataset_nav_layout()

    def refresh_output_dirs(self) -> None:
        """
        刷新当前数据集的 output 标签导航。
        :return: 无返回值
        """
        self.adjust_nav_layout()

    def select_dataset(self, dataset_name: str) -> None:
        """
        选择待复核的 output 数据集，不影响独立选择的目标数据集。
        :param dataset_name: output 下已存在的数据集目录名
        :return: 无返回值
        """
        self.current_dataset = dataset_name
        self.current_output_dir = ""
        self.current_output_index = -1
        self.selected_images.clear()
        self._cancel_thumbnail_render()
        self._clear_children(self.scroll_frame)
        self._rendered_cards.clear()
        self._last_layout_signature = None
        self.current_image_paths = []
        self.scroll_frame.configure(width=max(1, self.canvas.winfo_width()), height=1)
        self.refresh_dataset_nav()
        self.refresh_output_dirs()
        self._set_status(f"已选择数据集：{dataset_name}。请选择 output 标签目录。")
        self._schedule_scrollbar_update()

    def get_output_datasets(self) -> list[str]:
        """
        获取 output 下的一级数据集目录名。
        :return: 按本地化排序的数据集目录名列表
        """
        return self._get_child_dirs("output")

    def get_label_dirs(self) -> list[str]:
        """
        获取当前数据集下的一级标签目录名。
        :return: 按本地化排序的标签目录名列表
        """
        if not self.current_dataset:
            return []
        return self._get_child_dirs(os.path.join("output", self.current_dataset))

    def get_target_datasets(self) -> list[str]:
        """
        返回可作为移动目标的数据集，兼容仅存在于 cache 或 output 的数据集。
        :return: 按本地化排序的目标数据集目录名列表
        """
        dataset_names = set(self.get_output_datasets())
        dataset_names.update(self._get_child_dirs("cache"))
        return sorted(dataset_names, key=locale.strxfrm)

    def refresh_target_dataset_selector(self) -> None:
        """
        刷新独立目标数据集选择器，并保留仍然有效的选择。
        :return: 无返回值
        """
        target_datasets = self.get_target_datasets()
        self.target_dataset_combobox.configure(values=target_datasets)
        if self.target_dataset not in target_datasets:
            self.target_dataset = ""
            self.target_dataset_var.set("")
            self.target_dir = ""
            self.move_btn.config(state=tk.DISABLED)

    def on_target_dataset_selected(self, event: tk.Event | None = None) -> None:
        """
        切换移动目标数据集，并刷新其可选子目录。
        :param event: Tkinter 下拉框选择事件，可为空
        :return: 无返回值
        """
        self.target_dataset = self.target_dataset_var.get().strip()
        self.target_dir = ""
        self.move_btn.config(state=tk.DISABLED)
        self.load_target_dirs()
        self._set_status(f"已选择目标数据集：{self.target_dataset}。请选择目标子目录。")

    def get_target_root(self) -> str:
        """
        返回当前目标数据集对应的 cache 图像根目录。
        :return: cache 图像根目录路径；未选择时为空字符串
        """
        if not self.target_dataset:
            return ""
        return os.path.join("cache", self.target_dataset, "images")

    def get_target_dirs(self) -> list[str]:
        """
        合并 cache 标签目录与已有 output 目录，供图片移动时选择。
        :return: 按本地化排序的目标子目录名列表
        """
        if not self.target_dataset:
            return []
        target_root = self.get_target_root()
        directory_names = set(self._get_child_dirs(target_root))
        directory_names.update(self._get_child_dirs(os.path.join("output", self.target_dataset)))
        return sorted(directory_names, key=locale.strxfrm)

    def load_target_dirs(self) -> None:
        """
        加载目标数据集可用的 cache 标签目录与 output 目录，并重建首字母跳转索引。
        :return: 无返回值
        """
        self._clear_children(self.subdir_frame)
        self._reset_subdir_index()

        target_root = self.get_target_root()
        if not target_root:
            self.target_hint_var.set("请先选择目标数据集")
            return

        output_root = os.path.join("output", self.target_dataset)
        self.target_hint_var.set(f"目录来自 {target_root} 和 {output_root}")
        target_dirs = self.get_target_dirs()
        if self.target_dir not in target_dirs:
            self.target_dir = ""
            self.move_btn.config(state=tk.DISABLED)
        if not target_dirs:
            ttk.Label(self.subdir_frame, text="未找到目标子目录", style="Hint.TLabel").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
            return

        grouped = defaultdict(list)
        for directory_name in target_dirs:
            grouped[self.get_first_letter(directory_name)].append(directory_name)

        current_row = 0
        for letter, names in sorted(grouped.items()):
            header = ttk.Label(self.subdir_frame, text=letter, style="Hint.TLabel", font=("Arial", 11, "bold"))
            header.grid(row=current_row, column=0, columnspan=3, sticky=tk.W, padx=4, pady=(8, 2))
            self._subdir_letter_labels[letter] = header
            current_row += 1
            for index, directory_name in enumerate(names):
                column = index % 3
                if column == 0 and index:
                    current_row += 1
                button = ttk.Button(
                    self.subdir_frame,
                    text=directory_name,
                    style="Selected.TButton" if directory_name == self.target_dir else "SubDir.TButton",
                )
                button.configure(command=lambda name=directory_name, btn=button: self.select_target_dir(name, btn))
                button.bind("<Double-1>", lambda event, name=directory_name: self.move_images())
                button.grid(row=current_row, column=column, padx=3, pady=3, sticky=tk.EW)
            current_row += 1

        for column in range(3):
            self.subdir_frame.columnconfigure(column, weight=1)

        self._build_subdir_index(list(self._subdir_letter_labels))
        self._schedule_subdir_index_measurement()

    def _reset_subdir_index(self) -> None:
        """
        清空首字母索引栏、待测量的布局任务以及已记录的分组位置。
        :return: 无返回值
        """
        if self._subdir_index_job is not None:
            self.root.after_cancel(self._subdir_index_job)
            self._subdir_index_job = None
        self._clear_children(self.subdir_index_bar)
        self._subdir_letter_labels.clear()
        self._subdir_letter_offsets.clear()

    def _build_subdir_index(self, letters: list[str]) -> None:
        """
        在目标子目录列表左侧生成可点击的首字母跳转条。

        只为当前实际存在的分组生成索引项，避免出现点击后毫无反馈的死按钮。
        鼠标进入索引项时切换为高亮配色，离开时恢复常态，便于用户辨认当前光标所在索引。
        :param letters: 需要生成索引项的分组首字母列表，顺序与内容区分组顺序一致
        :return: 无返回值
        """
        for letter in letters:
            item = tk.Label(
                self.subdir_index_bar,
                text=letter,
                font=("Arial", self.SUBDIR_INDEX_FONT_SIZE, "bold"),
                foreground=self.SUBDIR_INDEX_FOREGROUND,
                background=self.SUBDIR_INDEX_BACKGROUND,
                cursor="hand2",
            )
            item.pack(side=tk.TOP, fill=tk.X)
            item.bind("<Button-1>", lambda event, name=letter: self._jump_to_subdir_letter(name))
            item.bind("<Enter>", lambda event, widget=item: self._set_subdir_index_hover(widget, True))
            item.bind("<Leave>", lambda event, widget=item: self._set_subdir_index_hover(widget, False))

    def _set_subdir_index_hover(self, widget: tk.Label, hovered: bool) -> None:
        """
        切换首字母索引项在鼠标悬停时的视觉状态。

        仅改变背景与前景色，不改变字体与布局，避免触发索引条重新测量或画布滚动。
        :param widget: 触发悬停状态变化的索引标签控件
        :param hovered: True 表示鼠标进入，False 表示鼠标离开
        :return: 无返回值
        """
        if hovered:
            widget.configure(
                background=self.SUBDIR_INDEX_HOVER_BACKGROUND,
                foreground=self.SUBDIR_INDEX_HOVER_FOREGROUND,
            )
        else:
            widget.configure(
                background=self.SUBDIR_INDEX_BACKGROUND,
                foreground=self.SUBDIR_INDEX_FOREGROUND,
            )

    def _schedule_subdir_index_measurement(self) -> None:
        """
        安排一次空闲期测量，记录各首字母分组在子目录画布中的像素位置。
        :return: 无返回值
        """
        if self._subdir_index_job is not None:
            self.root.after_cancel(self._subdir_index_job)
        self._subdir_index_job = self.root.after_idle(self._measure_subdir_index)

    def _measure_subdir_index(self) -> None:
        """
        测量各分组标题的相对纵坐标，作为索引跳转的目标位置。
        :return: 无返回值
        """
        self._subdir_index_job = None
        self._subdir_letter_offsets.clear()
        if not self._subdir_letter_labels:
            return
        self.subdir_frame.update_idletasks()
        for letter, label in self._subdir_letter_labels.items():
            self._subdir_letter_offsets[letter] = label.winfo_y()
        self.subdir_canvas.configure(scrollregion=self.subdir_canvas.bbox("all"))

    def _jump_to_subdir_letter(self, letter: str) -> None:
        """
        将目标子目录列表滚动到指定首字母分组的位置。
        :param letter: 目标分组首字母，例如 "A"、"Z" 或 "#"
        :return: 无返回值
        """
        offset = self._subdir_letter_offsets.get(letter)
        if offset is None:
            return
        bbox = self.subdir_canvas.bbox("all")
        if bbox is None:
            return
        total_height = max(1, bbox[3] - bbox[1])
        self.subdir_canvas.yview_moveto(max(0.0, min(1.0, offset / total_height)))

    def select_target_dir(self, directory_name: str, button: ttk.Button | None = None) -> None:
        """
        选择一个目标子目录并更新按钮状态。
        :param directory_name: 目标子目录名
        :param button: 触发选择的按钮，可为空
        :return: 无返回值
        """
        self.target_dir = directory_name
        self.move_btn.config(state=tk.NORMAL)
        if button is not None:
            for child in self.subdir_frame.winfo_children():
                if isinstance(child, ttk.Button):
                    child.configure(style="SubDir.TButton")
            button.configure(style="Selected.TButton")
        self._set_status(f"目标目录：output/{self.target_dataset}/{directory_name}")

    def load_images(self, directory_name: str, clear_selection: bool = True) -> None:
        """
        加载当前 output 标签目录，并只渲染可视区域附近的图片卡片。
        :param directory_name: 当前数据集下的标签目录名
        :param clear_selection: 切换目录时是否清理旧目录的选中状态
        :return: 无返回值
        """
        if not self.current_dataset:
            self._set_status("请先选择数据集。")
            return

        new_output_dir = os.path.join("output", self.current_dataset, directory_name)
        if clear_selection and new_output_dir != self.current_output_dir:
            self.selected_images.clear()
        self.current_output_dir = new_output_dir
        try:
            self.current_output_index = self.get_label_dirs().index(directory_name)
        except ValueError:
            self.current_output_index = -1

        self._cancel_thumbnail_render()
        self._clear_children(self.scroll_frame)
        self._rendered_cards.clear()
        self._last_layout_signature = None
        self.current_image_paths = self.get_image_files(new_output_dir)
        self.scroll_frame.configure(width=max(1, self.canvas.winfo_width()), height=1)
        self.canvas.yview_moveto(0)
        self.selected_images.intersection_update(self.current_image_paths)
        self.refresh_output_dirs()

        if not self.current_image_paths:
            ttk.Label(self.scroll_frame, text="此目录暂无可显示的图片", style="Hint.TLabel").grid(padx=16, pady=16, sticky=tk.W)
            self._set_status(f"{directory_name}：0 张图片")
            self._schedule_scrollbar_update()
            return

        self._thumbnail_error_count = 0
        self._set_status(f"正在加载 {len(self.current_image_paths)} 张图片…")
        self._schedule_virtual_render()

    def _render_thumbnail_batch(self) -> None:
        """
        根据当前滚动位置虚拟化渲染可视区域及一行预取区域。
        :return: 无返回值
        """
        self._render_job = None
        if self._is_closing or not self.current_output_dir or not self.current_image_paths:
            return

        self._update_virtual_layout()
        y_top = max(0.0, self.canvas.canvasy(0))
        y_bottom = y_top + max(1, self.canvas.winfo_height())
        first_row = max(0, int(y_top // self._grid_cell_height) - self.THUMBNAIL_PREFETCH_ROWS)
        last_row = min(
            (len(self.current_image_paths) - 1) // self._grid_columns,
            int(y_bottom // self._grid_cell_height) + self.THUMBNAIL_PREFETCH_ROWS,
        )
        first_index = first_row * self._grid_columns
        last_index = min(len(self.current_image_paths), (last_row + 1) * self._grid_columns)
        visible_indices = set(range(first_index, last_index))

        for task_key, future in list(self._thumbnail_futures.items()):
            generation, index = task_key
            if generation == self._render_generation and index not in visible_indices:
                if future.cancel():
                    self._thumbnail_futures.pop(task_key, None)

        for index, frame in list(self._rendered_cards.items()):
            if index not in visible_indices:
                frame.destroy()
                del self._rendered_cards[index]

        for index in range(first_index, last_index):
            if index not in self._rendered_cards:
                self._create_thumbnail_card(
                    self.current_image_paths[index], index, self._grid_thumb_size, self._grid_columns
                )
            else:
                frame = self._rendered_cards[index]
                self._position_thumbnail_card(frame, index, self._grid_columns)
                if getattr(frame, "image_photo", None) is None:
                    self._ensure_thumbnail_task(frame, self.current_image_paths[index], index, self._grid_thumb_size)
        self._schedule_scrollbar_update()
        self._set_status(f"{os.path.basename(self.current_output_dir)}：{len(self.current_image_paths)} 张图片")

    def _position_thumbnail_card(self, frame: ttk.Frame, index: int, columns: int) -> None:
        """
        按当前虚拟网格参数定位一个已存在的缩略图卡片。
        :param frame: 要定位的缩略图卡片控件
        :param index: 图片在当前目录列表中的索引
        :param columns: 当前网格列数
        :return: 无返回值
        """
        frame.place(
            x=index % columns * self._grid_cell_width + 5,
            y=index // columns * self._grid_cell_height + 5,
            width=max(32, self._grid_cell_width - 10),
            height=max(32, self._grid_cell_height - 10),
        )

    def _create_thumbnail_card(self, image_path: str, index: int, thumb_size: int, columns: int) -> None:
        """
        创建一个轻量占位卡片，并为未命中缓存的缩略图提交后台解码任务。
        :param image_path: 图片文件路径
        :param index: 图片在当前目录列表中的索引
        :param thumb_size: 缩略图边长上限
        :param columns: 当前网格列数
        :return: 无返回值
        """
        frame = ttk.Frame(self.scroll_frame, style="Thumbnail.TFrame", padding=4)
        label = ttk.Label(frame, text="加载中…", style="Hint.TLabel", anchor=tk.CENTER, justify=tk.CENTER)
        label.img_path = image_path
        label.bind("<Button-1>", lambda event, path=image_path: self.on_image_single_click(event, path))
        label.bind("<Double-1>", lambda event, path=image_path: self.on_image_double_click(event, path))
        label.pack(fill=tk.BOTH, expand=True)

        indicator = tk.Label(frame, text="✓", font=("Arial", max(16, thumb_size // 7), "bold"), foreground="#FFFFFF", background="#007AFF")
        indicator.place(relx=0.92, rely=0.08, anchor=tk.CENTER)
        indicator.place_forget()
        frame.selection_indicator = indicator
        frame.img_path = image_path
        frame.image_label = label
        frame.image_photo = None
        frame.index = index
        self._position_thumbnail_card(frame, index, columns)
        self._rendered_cards[index] = frame
        self.update_selection_ui(frame, image_path)
        self._ensure_thumbnail_task(frame, image_path, index, thumb_size)

        # 确保鼠标位于图片卡片任意位置时滚轮都能滚动主画布
        self._bind_mousewheel_to_widget(frame)
        self._bind_mousewheel_to_widget(label)
        self._bind_mousewheel_to_widget(indicator)

    def _ensure_thumbnail_task(self, frame: ttk.Frame, image_path: str, index: int, thumb_size: int) -> None:
        """
        确保指定卡片已显示缓存缩略图或已提交后台解码任务。
        :param frame: 缩略图卡片控件
        :param image_path: 图片文件路径
        :param index: 图片在当前目录列表中的索引
        :param thumb_size: 缩略图边长上限
        :return: 无返回值
        """
        cache_key = self._thumbnail_cache_key(image_path, thumb_size)
        if cache_key is None:
            self._set_thumbnail_error(frame, image_path, "文件不可访问")
            return
        if cache_key in self._failed_thumbnail_keys:
            self._set_thumbnail_error(frame, image_path, "无法加载")
            return
        photo = self._get_cached_thumbnail(cache_key)
        if photo is not None:
            self._set_thumbnail_photo(frame, photo)
            return
        task_key = (self._render_generation, index)
        if task_key in self._thumbnail_futures:
            return
        pending_current = sum(1 for generation, _ in self._thumbnail_futures if generation == self._render_generation)
        if pending_current >= self.THUMBNAIL_MAX_PENDING:
            frame.image_label.configure(text="排队中…", image="")
            frame.image_label.image = None
            frame.image_photo = None
            return
        try:
            future = self._thumbnail_executor.submit(self._decode_thumbnail, image_path, thumb_size)
        except RuntimeError:
            self._set_thumbnail_error(frame, image_path, "缩略图解码器已关闭")
            return
        self._thumbnail_futures[task_key] = future
        future.add_done_callback(
            lambda completed, generation=self._render_generation, item_index=index, path=image_path, key=cache_key:
            self._thumbnail_results.put((generation, item_index, path, key, completed))
        )
        self._schedule_thumbnail_result_poll()

    @staticmethod
    def _decode_thumbnail(image_path: str, size: int) -> Image.Image:
        """
        在工作线程中读取图片并缩放为内存中的 PIL 图像。
        :param image_path: 待读取的图片文件路径
        :param size: 缩略图边长上限
        :return: 已完成 EXIF 方向修正和缩放的 PIL 图像
        """
        with Image.open(image_path) as source:
            if source.format == "JPEG":
                try:
                    source.draft("RGB", (size, size))
                except Exception:
                    pass
            image = ImageOps.exif_transpose(source)
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            return image.copy()

    def _thumbnail_cache_key(self, image_path: str, size: int) -> tuple[str, int, int] | None:
        """
        根据文件绝对路径、修改时间和尺寸生成稳定的缩略图缓存键。
        :param image_path: 图片文件路径
        :param size: 缩略图边长上限
        :return: 缓存键；文件不可读取时返回 None
        """
        try:
            stat = os.stat(image_path)
        except OSError:
            return None
        return os.path.abspath(image_path), stat.st_mtime_ns, size

    def _get_cached_thumbnail(self, key: tuple[str, int, int]) -> ImageTk.PhotoImage | None:
        """
        从 LRU 缩略图缓存中读取并提升命中项的新鲜度。
        :param key: 缩略图缓存键
        :return: Tk 缩略图对象；未命中时返回 None
        """
        photo = self._thumbnail_cache.pop(key, None)
        if photo is not None:
            self._thumbnail_cache[key] = photo
        return photo

    def _store_thumbnail(self, key: tuple[str, int, int], image: Image.Image) -> ImageTk.PhotoImage:
        """
        将 PIL 图像转换为 Tk 图像并放入有界 LRU 缓存。
        :param key: 缩略图缓存键
        :param image: 已解码的 PIL 图像
        :return: 可供 Tk 控件显示的图像对象
        """
        photo = ImageTk.PhotoImage(image)
        self._thumbnail_cache[key] = photo
        self._thumbnail_cache.move_to_end(key)
        while len(self._thumbnail_cache) > self.THUMBNAIL_CACHE_SIZE:
            self._thumbnail_cache.popitem(last=False)
        return photo

    def _set_thumbnail_photo(self, frame: ttk.Frame, photo: ImageTk.PhotoImage) -> None:
        """
        把缓存中的 Tk 缩略图绑定到指定可视卡片。
        :param frame: 缩略图卡片控件
        :param photo: 要显示的 Tk 图像对象
        :return: 无返回值
        """
        frame.image_label.configure(image=photo, text="")
        frame.image_label.image = photo
        frame.image_photo = photo

    def _set_thumbnail_error(self, frame: ttk.Frame, image_path: str, message: str) -> None:
        """
        在缩略图卡片中显示非阻塞的加载错误。
        :param frame: 缩略图卡片控件
        :param image_path: 发生错误的图片路径
        :param message: 面向用户的错误简述
        :return: 无返回值
        """
        frame.image_label.configure(text=f"{message}\n{os.path.basename(image_path)}", image="")
        frame.image_label.image = None
        frame.image_photo = None

    def _drain_thumbnail_results(self) -> None:
        """
        在 Tk 主线程中消费后台解码结果并更新仍然可见的卡片。
        :return: 无返回值
        """
        self._thumbnail_poll_job = None
        processed = 0
        while processed < self.THUMBNAIL_BATCH_RESULT_LIMIT:
            try:
                generation, index, image_path, key, future = self._thumbnail_results.get_nowait()
            except Empty:
                break
            processed += 1
            self._thumbnail_futures.pop((generation, index), None)
            try:
                image = future.result()
            except CancelledError:
                continue
            except (OSError, ValueError, RuntimeError):
                if generation == self._render_generation and index in self._rendered_cards:
                    self._failed_thumbnail_keys.add(key)
                    self._set_thumbnail_error(self._rendered_cards[index], image_path, "无法加载")
                    self._thumbnail_error_count += 1
                continue
            except Exception:
                if generation == self._render_generation and index in self._rendered_cards:
                    self._failed_thumbnail_keys.add(key)
                    self._set_thumbnail_error(self._rendered_cards[index], image_path, "无法加载")
                    self._thumbnail_error_count += 1
                continue

            if self._is_closing or generation != self._render_generation:
                image.close()
                continue
            frame = self._rendered_cards.get(index)
            if frame is None or frame.img_path != image_path:
                image.close()
                continue
            try:
                photo = self._store_thumbnail(key, image)
                image.close()
            except Exception:
                image.close()
                self._failed_thumbnail_keys.add(key)
                self._set_thumbnail_error(frame, image_path, "无法显示")
                continue
            self._set_thumbnail_photo(frame, photo)

        if self._thumbnail_futures or not self._thumbnail_results.empty():
            self._schedule_thumbnail_result_poll()
        elif self.current_output_dir and self.current_image_paths:
            self._schedule_virtual_render()

    def _schedule_thumbnail_result_poll(self) -> None:
        """
        安排一次受控的后台缩略图结果轮询，避免每个文件创建独立 Tk 定时器。
        :return: 无返回值
        """
        if self._is_closing or self._thumbnail_poll_job is not None:
            return
        self._thumbnail_poll_job = self.root.after(20, self._drain_thumbnail_results)

    def _update_virtual_layout(self) -> None:
        """
        根据画布宽度和缩放比例计算虚拟网格尺寸及滚动区域高度。
        :return: 无返回值
        """
        thumb_size = max(32, int(self.THUMBNAIL_BASE_SIZE * self.zoom_scale / 100))
        available_width = max(160, self.canvas.winfo_width() - 8)
        minimum_cell_width = thumb_size + 24
        columns = max(1, available_width // minimum_cell_width)
        cell_width = max(minimum_cell_width, available_width // columns)
        cell_height = thumb_size + 42
        total_rows = (len(self.current_image_paths) + columns - 1) // columns
        self._grid_columns = columns
        self._grid_cell_width = cell_width
        self._grid_cell_height = cell_height
        self._grid_thumb_size = thumb_size
        signature = (available_width, total_rows, cell_height, columns, thumb_size)
        if signature != self._last_layout_signature:
            self._last_layout_signature = signature
            self.scroll_frame.configure(width=available_width, height=max(1, total_rows * cell_height + 10))

    def _schedule_virtual_render(self) -> None:
        """
        合并连续滚动、缩放和尺寸变化事件，安排一次虚拟卡片刷新。
        :return: 无返回值
        """
        if self._is_closing or self._render_job is not None:
            return
        self._render_job = self.root.after(16, self._render_thumbnail_batch)

    def _cancel_thumbnail_render(self) -> None:
        """
        取消当前渲染批次和不可见缩略图解码任务，并使旧结果失效。
        :return: 无返回值
        """
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
            self._render_job = None
        self._render_generation += 1
        for future in list(self._thumbnail_futures.values()):
            future.cancel()
        if self._thumbnail_futures:
            self._schedule_thumbnail_result_poll()

    def decrease_zoom(self) -> None:
        """
        将缩略图缩放比例降低 10 个百分点。
        :return: 无返回值
        """
        self._set_zoom(max(10, self._read_zoom() - 10))

    def increase_zoom(self) -> None:
        """
        将缩略图缩放比例提高 10 个百分点。
        :return: 无返回值
        """
        self._set_zoom(min(200, self._read_zoom() + 10))

    def on_zoom_entry_change(self, event: tk.Event | None = None) -> None:
        """
        读取输入框中的缩放比例并应用合法范围内的值。
        :param event: Tkinter 输入框事件，可为空
        :return: 无返回值
        """
        try:
            self._set_zoom(max(10, min(200, int(self.zoom_entry.get()))))
        except ValueError:
            self._set_zoom(self.zoom_scale)

    def _read_zoom(self) -> int:
        """
        读取当前缩放比例，输入非法时回退到已保存的比例。
        :return: 10 到 200 之间的缩放百分比整数
        """
        try:
            return int(self.zoom_entry.get())
        except ValueError:
            return self.zoom_scale

    def _set_zoom(self, scale: int) -> None:
        """
        保存缩放比例并重新计算当前虚拟图片网格。
        :param scale: 期望的缩放百分比
        :return: 无返回值
        """
        self.zoom_scale = scale
        self.zoom_entry.delete(0, tk.END)
        self.zoom_entry.insert(0, str(scale))
        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)

    def switch_to_next_output_dir(self, event: tk.Event | None = None) -> str:
        """
        通过快捷键切换到下一个 output 标签目录。
        :param event: Tkinter 快捷键事件，可为空
        :return: 用于停止事件继续传播的 break 字符串
        """
        directories = self.get_label_dirs()
        if directories:
            self.load_images(directories[(self.current_output_index + 1) % len(directories)])
        return "break"

    def switch_to_prev_output_dir(self, event: tk.Event | None = None) -> str:
        """
        通过快捷键切换到上一个 output 标签目录。
        :param event: Tkinter 快捷键事件，可为空
        :return: 用于停止事件继续传播的 break 字符串
        """
        directories = self.get_label_dirs()
        if directories:
            self.load_images(directories[(self.current_output_index - 1) % len(directories)])
        return "break"

    # ----------------- 目标目录和图片移动 ----------------- #

    def create_new_folder(self) -> None:
        """
        仅在目标数据集的 output 目录创建文件夹，并立即作为移动目标显示。
        :return: 无返回值
        """
        if not self.target_dataset:
            self._set_status("请先选择目标数据集。")
            return

        folder_name = simpledialog.askstring(
            "新建目标文件夹",
            f"在 output/{self.target_dataset} 下新建文件夹：",
            parent=self.root,
        )
        if folder_name is None:
            return
        folder_name = folder_name.strip()
        validation_error = self._validate_directory_name(folder_name)
        if validation_error:
            self._set_status(f"无法创建目标文件夹：{validation_error}")
            return

        output_path = os.path.join("output", self.target_dataset, folder_name)
        if os.path.exists(output_path):
            self._set_status(f"目标文件夹已存在：{output_path}")
            return

        try:
            os.makedirs(output_path, exist_ok=False)
        except OSError as error:
            self._set_status(f"无法创建输出目录：{error}")
            return

        self._clear_fs_caches()
        self.target_dir = folder_name
        self.move_btn.config(state=tk.NORMAL)
        self.refresh_dataset_nav()
        self.refresh_target_dataset_selector()
        self.refresh_output_dirs()
        self.load_target_dirs()
        self._set_status(f"已创建 output/{self.target_dataset}/{folder_name}，可直接作为移动目标。")

    def move_images(self) -> None:
        """
        将选中图片移动到独立选择的目标数据集及其目标子目录。
        :return: 无返回值
        """
        if not self.selected_images:
            self._set_status("请先选择要移动的图片。")
            return
        if not self.target_dataset or not self.target_dir:
            self._set_status("请先选择目标数据集和目标子目录。")
            return

        target_dir = os.path.join("output", self.target_dataset, self.target_dir)
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as error:
            self._set_status(f"移动失败：无法创建目标目录：{error}")
            return

        moved = 0
        failures = []
        moved_paths = set()
        for image_path in sorted(self.selected_images):
            if not os.path.isfile(image_path):
                failures.append(f"{os.path.basename(image_path)}：源文件不存在")
                continue
            destination = os.path.join(target_dir, os.path.basename(image_path))
            if self._same_path(image_path, destination):
                failures.append(f"{os.path.basename(image_path)}：已在目标目录中")
                continue
            if os.path.exists(destination):
                failures.append(f"{os.path.basename(image_path)}：目标位置存在同名文件")
                continue
            try:
                shutil.move(image_path, destination)
                self._discard_thumbnail(image_path)
                moved_paths.add(image_path)
                moved += 1
            except OSError as error:
                failures.append(f"{os.path.basename(image_path)}：{error}")

        self.selected_images.difference_update(moved_paths)
        if moved_paths:
            self._clear_fs_caches()
        self.refresh_dataset_nav()
        self.refresh_target_dataset_selector()
        self.load_target_dirs()
        self.refresh_output_dirs()
        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)

        if moved:
            message = f"已移动 {moved} 张图片到 output/{self.target_dataset}/{self.target_dir}。"
            if failures:
                message += self._format_errors(failures, "未移动")
            self._set_status(message)
        elif failures:
            self._set_status(self._format_errors(failures, "没有图片被移动").strip())

    # ----------------- 删除与归档 ----------------- #

    def remove_selected_images(self) -> None:
        """
        仅从 output 删除选中图片，不处理 input 中的同名源文件。
        :return: 无返回值
        """
        output_images = sorted(
            path for path in self.selected_images if os.path.isfile(path) and self._is_within(path, "output")
        )
        if not output_images:
            self._set_status("请先选择要从 output 移除的图片。")
            return
        if self.recycle_bin_enabled.get() and send2trash is None:
            self._set_status("回收站不可用：请取消“删除到回收站”后重试。")
            return

        removed_output, errors = self._delete_paths(output_images)
        for image_path in removed_output:
            self._discard_thumbnail(image_path)
        self.selected_images.difference_update(removed_output)
        if removed_output:
            self._clear_fs_caches()

        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)

        message = f"已从 output 移除 {len(removed_output)} 张图片；input 未作修改。"
        if errors:
            message += self._format_errors(errors, "以下文件未能移除")
        self._set_status(message)

    def delete_selected_images(self) -> None:
        """
        删除选中 output 图片及 input 中对应数据集内的同名文件。
        :return: 无返回值
        """
        output_images = sorted(
            path for path in self.selected_images if os.path.isfile(path) and self._is_within(path, "output")
        )
        if not output_images:
            self._set_status("请先选择要删除的 output 图片。")
            return
        if self.recycle_bin_enabled.get() and send2trash is None:
            self._set_status("回收站不可用：请取消“删除到回收站”后重试。")
            return

        input_matches = set()
        for output_path in output_images:
            input_matches.update(self.find_input_matches(output_path))

        deleted_output, output_errors = self._delete_paths(output_images)
        deleted_input, input_errors = self._delete_paths(sorted(input_matches))
        for image_path in deleted_output:
            self._discard_thumbnail(image_path)
        self.selected_images.difference_update(deleted_output)
        if deleted_output or deleted_input:
            self._clear_fs_caches()

        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)

        message = f"已删除 output 图片 {len(deleted_output)} 张，input 同名文件 {len(deleted_input)} 个。"
        errors = output_errors + input_errors
        if errors:
            message += self._format_errors(errors, "以下文件未能删除")
        self._set_status(message)

    def archive_output(self) -> None:
        """
        递归合并 output 到 datasets，成功后清理 input 中的同名源文件。
        :return: 无返回值
        """
        dataset_names = self.get_output_datasets()
        if not dataset_names:
            self._set_status("无需归档：output 中没有可归档的数据集目录。")
            return
        if self.recycle_bin_enabled.get() and send2trash is None:
            self._set_status("回收站不可用：请取消“删除到回收站”后重试。")
            return

        snapshots = {}
        source_file_count = 0
        for dataset_name in dataset_names:
            source_dir = os.path.join("output", dataset_name)
            files = self._iter_archivable_files(source_dir)
            directories = self._iter_dirs(source_dir)
            snapshots[dataset_name] = {"source_dir": source_dir, "files": files, "directories": directories}
            source_file_count += len(files)

        if source_file_count == 0:
            self._set_status("无需归档：output 中没有可归档文件（已忽略 .gitkeep）。")
            return

        destination_root = "datasets"
        try:
            os.makedirs(destination_root, exist_ok=True)
        except OSError as error:
            self._set_status(f"归档失败：无法创建 datasets 目录：{error}")
            return

        moved_count = 0
        archive_errors = []
        output_names_by_dataset = {}

        for dataset_name, snapshot in snapshots.items():
            moved, errors = self._merge_dataset_into_archive(
                snapshot["source_dir"],
                os.path.join(destination_root, dataset_name),
                snapshot["directories"],
                snapshot["files"],
            )
            moved_count += moved
            archive_errors.extend(errors)
            output_names_by_dataset[dataset_name] = {os.path.basename(path) for path in snapshot["files"]}

        if archive_errors or moved_count != source_file_count:
            message = f"归档未完成：已移动 {moved_count} / {source_file_count} 个文件；input 未清理。"
            if archive_errors:
                message += self._format_errors(archive_errors, "归档错误")
            self._clear_fs_caches()
            self._set_status(message)
            self.refresh_dataset_nav()
            self.refresh_output_dirs()
            return

        deleted_input, input_errors = self._delete_archived_input_files(output_names_by_dataset)
        self.selected_images.clear()
        self.current_dataset = ""
        self.current_output_dir = ""
        self.current_output_index = -1
        self._cancel_thumbnail_render()
        self._clear_children(self.scroll_frame)
        self._rendered_cards.clear()
        self._last_layout_signature = None
        self.current_image_paths = []
        self.scroll_frame.configure(width=max(1, self.canvas.winfo_width()), height=1)
        self._clear_fs_caches()
        self.refresh_target_dataset_selector()
        self.load_target_dirs()
        self.refresh_dataset_nav()
        self.refresh_output_dirs()
        self._schedule_scrollbar_update()

        message = (
            f"归档完成：已将 {source_file_count} 个文件合并到 datasets，"
            f"已清理 input 中 {len(deleted_input)} 个同名源文件。"
        )
        if input_errors:
            message += self._format_errors(input_errors, "input 清理失败")
        self._set_status(message)

    def _merge_dataset_into_archive(
        self, source_dir: str, destination_dir: str, source_dirs: list[str], source_files: list[str]
    ) -> tuple[int, list[str]]:
        """
        把一个 dataset 递归合并到 datasets，冲突文件始终保留并改名。
        :param source_dir: output 中 dataset 的源目录
        :param destination_dir: datasets 中的目标目录
        :param source_dirs: 需要创建的源目录快照
        :param source_files: 需要移动的源文件快照
        :return: 已移动数量及错误信息列表
        """
        moved = 0
        errors = []
        try:
            for directory in source_dirs:
                relative = os.path.relpath(directory, source_dir)
                target_directory = destination_dir if relative == "." else os.path.join(destination_dir, relative)
                os.makedirs(target_directory, exist_ok=True)
        except OSError as error:
            return moved, [f"{source_dir}：无法创建归档目录：{error}"]

        for source_path in source_files:
            if not os.path.isfile(source_path):
                errors.append(f"{source_path}：源文件不存在")
                continue
            try:
                relative = os.path.relpath(source_path, source_dir)
                target_directory = os.path.join(destination_dir, os.path.dirname(relative))
                os.makedirs(target_directory, exist_ok=True)
                destination = self._unique_destination(target_directory, os.path.basename(source_path))
                shutil.move(source_path, destination)
                self._discard_thumbnail(source_path)
                moved += 1
            except OSError as error:
                errors.append(f"{source_path}：{error}")

        self._remove_empty_directories(source_dir)
        return moved, errors

    def _delete_archived_input_files(self, output_names_by_dataset: dict[str, set[str]]) -> tuple[list[str], list[str]]:
        """
        在每个 input/<dataset> 中删除和原 output 同名的文件。
        :param output_names_by_dataset: 按数据集分组的 output 文件名集合
        :return: 已删除路径列表及错误信息列表
        """
        targets = set()
        input_root = "input"
        if not os.path.isdir(input_root):
            return [], []

        for dataset_name, output_names in output_names_by_dataset.items():
            if not output_names:
                continue
            dataset_input_root = os.path.join("input", dataset_name)
            search_root = dataset_input_root if os.path.isdir(dataset_input_root) else input_root
            for file_path in self._iter_files(search_root):
                if os.path.basename(file_path) in output_names:
                    targets.add(file_path)
        return self._delete_paths(sorted(targets))

    def _delete_paths(self, paths: list[str]) -> tuple[list[str], list[str]]:
        """
        按当前回收站设置删除一组文件路径。
        :param paths: 待删除文件路径列表
        :return: 已删除路径列表及错误信息列表
        """
        deleted = []
        errors = []
        for path in paths:
            try:
                if not os.path.isfile(path):
                    continue
                if self.recycle_bin_enabled.get():
                    if send2trash is None:
                        raise RuntimeError("send2trash 不可用")
                    send2trash(path)
                else:
                    os.remove(path)
                deleted.append(path)
            except OSError as error:
                errors.append(f"{os.path.basename(path)}：{error}")
            except RuntimeError as error:
                errors.append(f"{os.path.basename(path)}：{error}")
        return deleted, errors

    def find_input_matches(self, output_path: str) -> list[str]:
        """
        查找 input 中同一 dataset 内与 output 图片同名的源文件。
        :param output_path: output 中的图片路径
        :return: 匹配到的 input 文件路径列表
        """
        if not self._is_within(output_path, "output"):
            return []
        try:
            relative = os.path.relpath(output_path, "output")
            dataset_name = Path(relative).parts[0]
        except (IndexError, ValueError):
            return []

        input_dataset_dir = os.path.join("input", dataset_name)
        search_root = input_dataset_dir if os.path.isdir(input_dataset_dir) else "input"
        target_name = os.path.basename(output_path)
        return [path for path in self._iter_files(search_root) if os.path.basename(path) == target_name]

    # ----------------- 选择、预览和滚动 ----------------- #

    def update_selection_ui(self, frame: ttk.Frame, image_path: str) -> None:
        """
        根据选中集合更新缩略图卡片的视觉状态。
        :param frame: 缩略图卡片控件
        :param image_path: 卡片对应的图片路径
        :return: 无返回值
        """
        selected = image_path in self.selected_images
        frame.configure(style="Selected.TFrame" if selected else "Thumbnail.TFrame")
        if selected:
            frame.selection_indicator.place(relx=0.92, rely=0.08, anchor=tk.CENTER)
        else:
            frame.selection_indicator.place_forget()

    def on_image_single_click(self, event: tk.Event, image_path: str) -> None:
        """
        延迟处理单击，以便与双击预览操作区分。
        :param event: Tkinter 鼠标事件
        :param image_path: 被点击的图片路径
        :return: 无返回值
        """
        if self.click_timer is not None:
            self.root.after_cancel(self.click_timer)
        self.click_timer = self.root.after(220, lambda widget=event.widget, path=image_path: self.execute_single_click(widget, path))

    def on_image_double_click(self, event: tk.Event, image_path: str) -> None:
        """
        取消单击延迟并打开图片预览窗口。
        :param event: Tkinter 鼠标事件
        :param image_path: 被双击的图片路径
        :return: 无返回值
        """
        if self.click_timer is not None:
            self.root.after_cancel(self.click_timer)
            self.click_timer = None
        self.show_fullsize_image(image_path)

    def execute_single_click(self, widget: tk.Widget, image_path: str) -> None:
        """
        切换一张图片的选中状态并刷新对应卡片。
        :param widget: 触发事件的标签控件
        :param image_path: 图片路径
        :return: 无返回值
        """
        frame = widget.master
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
        else:
            self.selected_images.add(image_path)
        self.update_selection_ui(frame, image_path)
        self.click_timer = None

    def invert_selection(self) -> None:
        """
        反转当前目录中所有图片的选中状态。
        :return: 无返回值
        """
        current_images = set(self.current_image_paths)
        self.selected_images = current_images - self.selected_images
        self.refresh_selection_ui()

    def refresh_selection_ui(self) -> None:
        """
        刷新当前可视缩略图卡片的选中状态。
        :return: 无返回值
        """
        for frame in self._rendered_cards.values():
            self.update_selection_ui(frame, frame.img_path)

    def show_fullsize_image(self, image_path: str) -> None:
        """
        打开单张图片预览窗口，并以当前目录列表提供前后导航。
        :param image_path: 要预览的图片路径
        :return: 无返回值
        """
        all_images = self.current_image_paths[:]
        if image_path not in all_images:
            return
        top = tk.Toplevel(self.root)
        top.title("图片预览")
        top.transient(self.root)
        top.current_index = all_images.index(image_path)
        top.all_images = all_images

        container = ttk.Frame(top, padding=18)
        container.pack(fill=tk.BOTH, expand=True)
        image_label = ttk.Label(container)
        image_label.pack()
        indicator = tk.Label(container, text="✓", font=("Arial", 42, "bold"), foreground="#FFFFFF", background="#007AFF")
        indicator.place_forget()
        top.selection_indicator = indicator

        navigation = ttk.Frame(top, padding=(18, 0, 18, 14))
        navigation.pack(fill=tk.X)
        ttk.Button(navigation, text="‹ 上一张", command=lambda: self.navigate_image(top, image_label, -1)).pack(side=tk.LEFT)
        top.info_label = ttk.Label(navigation, text="")
        top.info_label.pack(side=tk.LEFT, expand=True)
        ttk.Button(navigation, text="下一张 ›", command=lambda: self.navigate_image(top, image_label, 1)).pack(side=tk.RIGHT)

        top.bind("<Left>", lambda event: self.navigate_image(top, image_label, -1))
        top.bind("<Right>", lambda event: self.navigate_image(top, image_label, 1))
        top.bind("<space>", lambda event: self.toggle_image_selection(top))
        top.focus_set()
        self.load_preview_image(top, image_label, top.current_index)

    def navigate_image(self, window: tk.Toplevel, label: ttk.Label, direction: int) -> None:
        """
        在预览窗口中按方向切换图片。
        :param window: 当前图片预览窗口
        :param label: 显示预览图像的标签控件
        :param direction: -1 表示上一张，1 表示下一张
        :return: 无返回值
        """
        if not window.all_images:
            return
        window.current_index = (window.current_index + direction) % len(window.all_images)
        self.load_preview_image(window, label, window.current_index)

    def toggle_image_selection(self, window: tk.Toplevel) -> None:
        """
        切换预览窗口当前图片的选中状态并同步可视缩略图。
        :param window: 当前图片预览窗口
        :return: 无返回值
        """
        image_path = window.all_images[window.current_index]
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
        else:
            self.selected_images.add(image_path)
        self.update_preview_selection_indicator(window, image_path)
        self.sync_thumbnail_selection(image_path)

    def load_preview_image(self, window: tk.Toplevel, label: ttk.Label, index: int) -> None:
        """
        读取并按屏幕尺寸缩放预览窗口中的图片。
        :param window: 当前图片预览窗口
        :param label: 显示预览图像的标签控件
        :param index: 当前图片在预览列表中的索引
        :return: 无返回值
        """
        image_path = window.all_images[index]
        try:
            with Image.open(image_path) as source:
                max_width = max(320, self.root.winfo_screenwidth() - 220)
                max_height = max(240, self.root.winfo_screenheight() - 300)
                if source.format == "JPEG":
                    try:
                        source.draft("RGB", (max_width, max_height))
                    except Exception:
                        pass
                image = ImageOps.exif_transpose(source)
                scale = min(max_width / image.width, max_height / image.height, 1.0)
                rendered = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            photo = ImageTk.PhotoImage(rendered)
            label.configure(image=photo)
            label.image = photo
            window.title(f"图片预览 · {os.path.basename(image_path)}")
            window.info_label.configure(text=f"{index + 1} / {len(window.all_images)}")
            self.update_preview_selection_indicator(window, image_path)
            if not getattr(window, "geometry_initialized", False):
                window.update_idletasks()
                width = min(rendered.width + 36, self.root.winfo_screenwidth() - 80)
                height = min(rendered.height + 100, self.root.winfo_screenheight() - 80)
                window.geometry(f"{width}x{height}")
                window.geometry_initialized = True
        except (OSError, ValueError) as error:
            self._set_status(f"无法打开 {os.path.basename(image_path)}：{error}")

    def update_preview_selection_indicator(self, window: tk.Toplevel, image_path: str) -> None:
        """
        根据图片选中状态显示或隐藏预览窗口中的标记。
        :param window: 当前图片预览窗口
        :param image_path: 图片路径
        :return: 无返回值
        """
        if image_path in self.selected_images:
            window.selection_indicator.place(relx=0.95, rely=0.05, anchor=tk.NE)
        else:
            window.selection_indicator.place_forget()

    def sync_thumbnail_selection(self, image_path: str) -> None:
        """
        只更新当前已虚拟化的对应缩略图卡片。
        :param image_path: 需要同步选中状态的图片路径
        :return: 无返回值
        """
        for frame in self._rendered_cards.values():
            if getattr(frame, "img_path", None) == image_path:
                self.update_selection_ui(frame, image_path)
                return

    def on_canvas_configure(self, event: tk.Event) -> None:
        """
        响应图片画布尺寸变化并触发虚拟网格重新布局。
        :param event: Tkinter 画布尺寸变化事件
        :return: 无返回值
        """
        self.canvas.itemconfigure(self.scroll_window, width=event.width)
        self._schedule_scrollbar_update()
        self._schedule_virtual_render()

    def on_frame_configure(self, event: tk.Event) -> None:
        """
        响应虚拟内容框尺寸变化并更新滚动区域。
        :param event: Tkinter 内容框尺寸变化事件
        :return: 无返回值
        """
        self._schedule_scrollbar_update()

    def _on_canvas_yview(self, first: str, last: str) -> None:
        """
        接收画布滚动回调，同时安排可视卡片的增删。
        :param first: 滚动区域起点比例
        :param last: 滚动区域终点比例
        :return: 无返回值
        """
        self.scrollbar.set(first, last)
        self._schedule_virtual_render()

    def _schedule_scrollbar_update(self) -> None:
        """
        合并连续尺寸事件，安排一次滚动区域更新。
        :return: 无返回值
        """
        if self._scroll_update_job is None:
            self._scroll_update_job = self.root.after_idle(self.update_scrollbar_state)

    def update_scrollbar_state(self) -> None:
        """
        更新画布滚动范围及滚动条的显示状态。
        :return: 无返回值
        """
        self._scroll_update_job = None
        bbox = self.canvas.bbox("all")
        if bbox is None:
            return
        self.canvas.configure(scrollregion=bbox)
        needs_scrollbar = (bbox[3] - bbox[1]) > self.canvas.winfo_height() + 8
        if needs_scrollbar:
            self.scrollbar.grid()
        else:
            self.scrollbar.grid_remove()
            self.canvas.yview_moveto(0)

    def on_mousewheel(self, event: tk.Event) -> str:
        """
        将平台相关的鼠标滚轮事件转换为画布滚动。
        :param event: Tkinter 鼠标滚轮事件
        :return: 用于停止事件继续传播的 break 字符串
        """
        if not self.scrollbar.winfo_ismapped():
            return "break"
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        elif platform.system() == "Windows":
            delta = -int(event.delta / 120)
        else:
            delta = -int(event.delta)
        if delta:
            self.canvas.yview_scroll(delta, "units")
        return "break"

    def _on_subdir_frame_configure(self, event: tk.Event) -> None:
        """
        更新目标子目录列表画布的滚动范围。
        :param event: Tkinter 子目录框尺寸变化事件
        :return: 无返回值
        """
        self.subdir_canvas.configure(scrollregion=self.subdir_canvas.bbox("all"))

    def _on_subdir_canvas_configure(self, event: tk.Event) -> None:
        """
        让目标子目录内容框随画布宽度变化。
        :param event: Tkinter 子目录画布尺寸变化事件
        :return: 无返回值
        """
        self.subdir_canvas.itemconfigure(self.subdir_canvas_window, width=event.width)

    def _bind_subdir_mousewheel(self, event: tk.Event) -> None:
        """
        鼠标进入目标子目录区域时接管滚轮事件。
        :param event: Tkinter 鼠标进入事件
        :return: 无返回值
        """
        self.subdir_canvas.bind_all("<MouseWheel>", self._on_subdir_mousewheel)
        self.subdir_canvas.bind_all("<Button-4>", self._on_subdir_mousewheel)
        self.subdir_canvas.bind_all("<Button-5>", self._on_subdir_mousewheel)

    def _unbind_subdir_mousewheel(self, event: tk.Event) -> None:
        """
        鼠标离开目标子目录区域时恢复主图片区域滚轮绑定。
        :param event: Tkinter 鼠标离开事件
        :return: 无返回值
        """
        self.subdir_canvas.unbind_all("<MouseWheel>")
        self.subdir_canvas.unbind_all("<Button-4>")
        self.subdir_canvas.unbind_all("<Button-5>")
        self.bind_events()

    def _on_subdir_mousewheel(self, event: tk.Event) -> str:
        """
        将平台相关滚轮事件转换为目标子目录画布滚动。
        :param event: Tkinter 鼠标滚轮事件
        :return: 用于停止事件继续传播的 break 字符串
        """
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        elif platform.system() == "Windows":
            delta = -int(event.delta / 120)
        else:
            delta = -int(event.delta)
        if delta:
            self.subdir_canvas.yview_scroll(delta, "units")
        return "break"

    # ----------------- 文件系统辅助方法 ----------------- #

    @staticmethod
    def _clear_children(widget: tk.Misc) -> None:
        """
        销毁指定 Tk 容器中的全部直接子控件。
        :param widget: 要清空的 Tk 容器
        :return: 无返回值
        """
        for child in widget.winfo_children():
            child.destroy()

    def _clear_fs_caches(self) -> None:
        """
        清理目录与图片列表缓存，使文件移动、删除和新建后的视图立即一致。
        :return: 无返回值
        """
        self._directory_cache.clear()
        self._image_list_cache.clear()

    def _get_child_dirs(self, path: str) -> list[str]:
        """
        获取目录的一级子目录，并按目录状态缓存扫描结果。
        :param path: 要扫描的父目录路径
        :return: 按本地化排序的一级子目录名列表
        """
        normalized_path = os.path.abspath(path)
        try:
            stat = os.stat(normalized_path)
            signature = (stat.st_mtime_ns, stat.st_ctime_ns)
        except OSError:
            self._directory_cache.pop(normalized_path, None)
            return []
        cached = self._directory_cache.pop(normalized_path, None)
        if cached is not None and cached[0] == signature:
            self._directory_cache[normalized_path] = cached
            return list(cached[1])
        try:
            with os.scandir(path) as entries:
                result = sorted(
                    (entry.name for entry in entries if entry.is_dir()),
                    key=locale.strxfrm,
                )
        except OSError:
            return []
        self._directory_cache[normalized_path] = (signature, result)
        while len(self._directory_cache) > self.DIRECTORY_CACHE_SIZE:
            self._directory_cache.popitem(last=False)
        return list(result)

    def get_image_files(self, folder: str) -> list[str]:
        """
        获取目录中的图片文件列表，并按目录时间戳缓存扫描结果。
        :param folder: 要扫描的图片目录路径
        :return: 按文件名本地化排序的图片路径列表
        """
        normalized_folder = os.path.abspath(folder)
        try:
            stat = os.stat(normalized_folder)
            signature = (stat.st_mtime_ns, stat.st_ctime_ns)
        except OSError:
            self._image_list_cache.pop(normalized_folder, None)
            return []
        cached = self._image_list_cache.pop(normalized_folder, None)
        if cached is not None and cached[0] == signature:
            self._image_list_cache[normalized_folder] = cached
            return list(cached[1])
        try:
            with os.scandir(folder) as entries:
                paths = [
                    entry.path
                    for entry in entries
                    if entry.is_file() and Path(entry.name).suffix.lower() in self.IMAGE_EXTENSIONS
                ]
        except OSError:
            return []
        result = sorted(paths, key=lambda path: locale.strxfrm(os.path.basename(path)))
        self._image_list_cache[normalized_folder] = (signature, result)
        while len(self._image_list_cache) > self.IMAGE_LIST_CACHE_SIZE:
            self._image_list_cache.popitem(last=False)
        return list(result)

    @staticmethod
    def _iter_files(root: str) -> list[str]:
        """
        递归收集目录下的普通文件。
        :param root: 递归扫描根目录
        :return: 普通文件路径列表
        """
        if not os.path.isdir(root):
            return []
        files = []
        for directory, _, filenames in os.walk(root):
            files.extend(os.path.join(directory, name) for name in filenames if os.path.isfile(os.path.join(directory, name)))
        return files

    @staticmethod
    def _iter_archivable_files(root: str) -> list[str]:
        """
        返回归档文件，显式忽略用于保留目录的 .gitkeep。
        :param root: 归档扫描根目录
        :return: 可归档文件路径列表
        """
        return [path for path in ImageClassifierUI._iter_files(root) if os.path.basename(path) != ".gitkeep"]

    @staticmethod
    def _iter_dirs(root: str) -> list[str]:
        """
        递归收集目录自身及其所有子目录。
        :param root: 递归扫描根目录
        :return: 按深度和路径排序的目录路径列表
        """
        if not os.path.isdir(root):
            return []
        directories = []
        for directory, subdirs, _ in os.walk(root):
            directories.append(directory)
            directories.extend(os.path.join(directory, name) for name in subdirs)
        return sorted(set(directories), key=lambda path: (path.count(os.sep), path))

    @staticmethod
    def _remove_empty_directories(root: str) -> None:
        """
        自底向上删除空目录，忽略无法删除的非空目录。
        :param root: 待清理的根目录
        :return: 无返回值
        """
        if not os.path.isdir(root):
            return
        for directory, _, _ in os.walk(root, topdown=False):
            try:
                os.rmdir(directory)
            except OSError:
                pass

    @staticmethod
    def _unique_destination(directory: str, filename: str) -> str:
        """
        为冲突文件名生成不覆盖既有文件的目标路径。
        :param directory: 目标目录路径
        :param filename: 原始文件名
        :return: 可安全使用的目标文件路径
        """
        candidate = os.path.join(directory, filename)
        if not os.path.exists(candidate):
            return candidate
        stem, extension = os.path.splitext(filename)
        index = 1
        while True:
            candidate = os.path.join(directory, f"{stem} ({index}){extension}")
            if not os.path.exists(candidate):
                return candidate
            index += 1

    @staticmethod
    def _same_path(left: str, right: str) -> bool:
        """
        判断两个路径是否指向同一规范化位置。
        :param left: 第一个路径
        :param right: 第二个路径
        :return: 两个路径相同时为 True，否则为 False
        """
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))

    @staticmethod
    def _is_within(path: str, root: str) -> bool:
        """
        判断路径是否位于指定根目录内。
        :param path: 待判断路径
        :param root: 根目录路径
        :return: 路径位于根目录内时为 True，否则为 False
        """
        try:
            return os.path.commonpath((os.path.abspath(path), os.path.abspath(root))) == os.path.abspath(root)
        except ValueError:
            return False

    @staticmethod
    def _validate_directory_name(name: str) -> str:
        """
        校验用户输入的单级目录名，防止路径穿越和平台非法字符。
        :param name: 待校验目录名
        :return: 空字符串表示合法，否则返回面向用户的错误信息
        """
        if not name:
            return "文件夹名称不能为空。"
        if name in {".", ".."} or os.path.basename(name) != name:
            return "文件夹名称不能包含路径层级。"
        invalid_chars = set('/\\:*?"<>|')
        if any(character in invalid_chars for character in name):
            return "名称包含非法字符：\\/:*?\"<>|"
        if name.endswith((".", " ")):
            return "名称不能以句点或空格结尾。"
        return ""

    def _discard_thumbnail(self, image_path: str) -> None:
        """
        删除指定路径对应的全部缩略图缓存项。
        :param image_path: 已移动、删除或失效的图片路径
        :return: 无返回值
        """
        normalized_path = os.path.abspath(image_path)
        stale_keys = [key for key in self._thumbnail_cache if key[0] == normalized_path]
        for key in stale_keys:
            self._thumbnail_cache.pop(key, None)
        stale_failure_keys = [key for key in self._failed_thumbnail_keys if key[0] == normalized_path]
        for key in stale_failure_keys:
            self._failed_thumbnail_keys.discard(key)

    def get_first_letter(self, text: str) -> str:
        """
        获取标签名称用于分组显示的首字母。
        :param text: 标签名称文本
        :return: 拉丁字母首字母或 # 分组标记
        """
        if not text:
            return "#"
        first_character = text[0].upper()
        if "A" <= first_character <= "Z":
            return first_character
        if first_character.isdigit():
            return "#"
        try:
            import pypinyin

            pinyin_list = pypinyin.pinyin(first_character, style=pypinyin.NORMAL)
            if pinyin_list and pinyin_list[0]:
                first_letter = pinyin_list[0][0][0].upper()
                if "A" <= first_letter <= "Z":
                    return first_letter
        except (ImportError, IndexError, TypeError):
            pass
        return "#"

    def _set_status(self, message: object) -> None:
        """
        在右下角显示简洁的非阻塞操作结果。
        :param message: 要显示的状态信息
        :return: 无返回值
        """
        normalized = " ".join(str(message).split())
        if normalized == self._last_status_message:
            return
        self._last_status_message = normalized
        self.status_var.set(normalized)

    @staticmethod
    def _format_errors(errors: list[str], heading: str) -> str:
        """
        将错误列表压缩为适合状态栏显示的有限长度文本。
        :param errors: 错误信息列表
        :param heading: 错误分组标题
        :return: 格式化后的错误摘要
        """
        visible = errors[:5]
        suffix = "\n…" if len(errors) > len(visible) else ""
        return f"\n\n{heading}（{len(errors)} 个）：\n" + "\n".join(visible) + suffix


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierUI(root)
    root.mainloop()