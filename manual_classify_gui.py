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
from pathlib import Path
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
    THUMBNAIL_BATCH_SIZE = 20
    THUMBNAIL_CACHE_SIZE = 180

    def __init__(self, root):
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
        self._render_queue = []
        self._render_index = 0
        self._thumbnail_cache = OrderedDict()

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

    def _setup_styles(self):
        """配置接近 iOS 的浅色卡片、圆润留白与高对比操作色。"""
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

    def create_widgets(self):
        """创建主窗口、导航、图片网格以及操作面板。"""
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

    def setup_image_panel(self):
        panel = ttk.LabelFrame(self.main_paned, text="待复核图片 · output")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(panel, background="#FFFFFF", highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas, style="Card.TFrame")
        self.scroll_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor=tk.NW)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.scrollbar.grid(row=0, column=1, sticky=tk.NS)
        return panel

    def setup_control_panel(self):
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

        self.subdir_canvas = tk.Canvas(target_container, background="#FFFFFF", highlightthickness=0, borderwidth=0)
        self.subdir_scrollbar = ttk.Scrollbar(target_container, orient=tk.VERTICAL, command=self.subdir_canvas.yview)
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

    def setup_bottom_controls(self):
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

    def setup_scroll_management(self):
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.scroll_frame.bind("<Configure>", self.on_frame_configure)

    def bind_events(self):
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.scroll_frame.bind("<MouseWheel>", self.on_mousewheel)
        self.scroll_frame.bind("<Button-4>", self.on_mousewheel)
        self.scroll_frame.bind("<Button-5>", self.on_mousewheel)

    def on_close(self):
        self._cancel_thumbnail_render()
        self._thumbnail_cache.clear()
        self.root.destroy()

    # ----------------- 导航与图片加载 ----------------- #

    def on_window_resize(self, event):
        if event.widget != self.root:
            return
        if self._resize_timer is not None:
            self.root.after_cancel(self._resize_timer)
        self._resize_timer = self.root.after(180, self._refresh_responsive_layout)

    def _refresh_responsive_layout(self):
        self._resize_timer = None
        self.adjust_dataset_nav_layout()
        self.adjust_nav_layout()

    def adjust_dataset_nav_layout(self):
        """按窗口宽度重建 output 数据集导航。"""
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

    def adjust_nav_layout(self):
        """按窗口宽度重建当前数据集的 output 标签导航。"""
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

    def refresh_dataset_nav(self):
        self.adjust_dataset_nav_layout()

    def refresh_output_dirs(self):
        self.adjust_nav_layout()

    def select_dataset(self, dataset_name):
        """选择待复核的 output 数据集，不影响独立选择的目标数据集。"""
        self.current_dataset = dataset_name
        self.current_output_dir = ""
        self.current_output_index = -1
        self.selected_images.clear()
        self._cancel_thumbnail_render()
        self._clear_children(self.scroll_frame)
        self.current_image_paths = []
        self.refresh_dataset_nav()
        self.refresh_output_dirs()
        self._set_status(f"已选择数据集：{dataset_name}。请选择 output 标签目录。")
        self._schedule_scrollbar_update()

    def get_output_datasets(self):
        return self._get_child_dirs("output")

    def get_label_dirs(self):
        if not self.current_dataset:
            return []
        return self._get_child_dirs(os.path.join("output", self.current_dataset))

    def get_target_datasets(self):
        """返回可作为移动目标的数据集，兼容仅存在于 cache 或 output 的数据集。"""
        dataset_names = set(self.get_output_datasets())
        dataset_names.update(self._get_child_dirs("cache"))
        return sorted(dataset_names, key=locale.strxfrm)

    def refresh_target_dataset_selector(self):
        """刷新独立目标数据集选择器，并保留仍然有效的选择。"""
        target_datasets = self.get_target_datasets()
        self.target_dataset_combobox.configure(values=target_datasets)
        if self.target_dataset not in target_datasets:
            self.target_dataset = ""
            self.target_dataset_var.set("")
            self.target_dir = ""
            self.move_btn.config(state=tk.DISABLED)

    def on_target_dataset_selected(self, event=None):
        """切换移动目标数据集，并刷新其可选子目录。"""
        self.target_dataset = self.target_dataset_var.get().strip()
        self.target_dir = ""
        self.move_btn.config(state=tk.DISABLED)
        self.load_target_dirs()
        self._set_status(f"已选择目标数据集：{self.target_dataset}。请选择目标子目录。")

    def get_target_root(self):
        if not self.target_dataset:
            return ""
        return os.path.join("cache", self.target_dataset, "images")

    def get_target_dirs(self):
        """合并 cache 标签目录与已有 output 目录，供图片移动时选择。"""
        if not self.target_dataset:
            return []
        target_root = self.get_target_root()
        directory_names = set(self._get_child_dirs(target_root))
        directory_names.update(self._get_child_dirs(os.path.join("output", self.target_dataset)))
        return sorted(directory_names, key=locale.strxfrm)

    def load_target_dirs(self):
        """加载目标数据集可用的 cache 标签目录与 output 目录。"""
        self._clear_children(self.subdir_frame)
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
            ttk.Label(self.subdir_frame, text=letter, style="Hint.TLabel", font=("Arial", 11, "bold")).grid(
                row=current_row, column=0, columnspan=3, sticky=tk.W, padx=4, pady=(8, 2)
            )
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

    def select_target_dir(self, directory_name, button=None):
        self.target_dir = directory_name
        self.move_btn.config(state=tk.NORMAL)
        if button is not None:
            for child in self.subdir_frame.winfo_children():
                if isinstance(child, ttk.Button):
                    child.configure(style="SubDir.TButton")
            button.configure(style="Selected.TButton")
        self._set_status(f"目标目录：output/{self.target_dataset}/{directory_name}")

    def load_images(self, directory_name, clear_selection=True):
        """异步分批渲染当前 output 标签目录，避免大量图片阻塞界面。"""
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
        self.current_image_paths = self.get_image_files(new_output_dir)
        self.selected_images.intersection_update(self.current_image_paths)
        self.refresh_output_dirs()

        if not self.current_image_paths:
            ttk.Label(self.scroll_frame, text="此目录暂无可显示的图片", style="Hint.TLabel").grid(padx=16, pady=16, sticky=tk.W)
            self._set_status(f"{directory_name}：0 张图片")
            self._schedule_scrollbar_update()
            return

        self._render_queue = self.current_image_paths[:]
        self._render_index = 0
        self._set_status(f"正在加载 {len(self._render_queue)} 张图片…")
        self._render_job = self.root.after_idle(self._render_thumbnail_batch)

    def _render_thumbnail_batch(self):
        self._render_job = None
        if not self.current_output_dir:
            return

        thumb_size = max(32, int(self.THUMBNAIL_BASE_SIZE * self.zoom_scale / 100))
        columns = max(1, int(6 * 100 / self.zoom_scale))
        for column in range(columns):
            self.scroll_frame.columnconfigure(column, weight=1)

        stop_index = min(self._render_index + self.THUMBNAIL_BATCH_SIZE, len(self._render_queue))
        for index in range(self._render_index, stop_index):
            image_path = self._render_queue[index]
            self._create_thumbnail_card(image_path, index, thumb_size, columns)
        self._render_index = stop_index
        self._schedule_scrollbar_update()

        if self._render_index < len(self._render_queue):
            self._set_status(f"正在加载图片：{self._render_index} / {len(self._render_queue)}")
            self._render_job = self.root.after(1, self._render_thumbnail_batch)
        else:
            self._set_status(f"{os.path.basename(self.current_output_dir)}：{len(self._render_queue)} 张图片")

    def _create_thumbnail_card(self, image_path, index, thumb_size, columns):
        frame = ttk.Frame(self.scroll_frame, style="Thumbnail.TFrame", padding=4)
        try:
            photo = self._get_thumbnail(image_path, thumb_size)
            label = ttk.Label(frame, image=photo, style="Thumbnail.TLabel")
            label.image = photo
            label.img_path = image_path
            label.bind("<Button-1>", lambda event, path=image_path: self.on_image_single_click(event, path))
            label.bind("<Double-1>", lambda event, path=image_path: self.on_image_double_click(event, path))
            label.pack()
        except Exception as error:
            label = ttk.Label(frame, text=f"无法加载\n{os.path.basename(image_path)}", style="Hint.TLabel", justify=tk.CENTER, width=18)
            label.img_path = image_path
            label.pack(padx=6, pady=12)
            self._set_status(f"部分图片无法加载：{error}")

        indicator = tk.Label(frame, text="✓", font=("Arial", max(16, thumb_size // 7), "bold"), foreground="#FFFFFF", background="#007AFF")
        indicator.place(relx=0.92, rely=0.08, anchor=tk.CENTER)
        indicator.place_forget()
        frame.selection_indicator = indicator
        frame.img_path = image_path
        frame.grid(row=index // columns, column=index % columns, padx=5, pady=5, sticky=tk.NW)
        self.update_selection_ui(frame, image_path)

    def _get_thumbnail(self, image_path, size):
        stat = os.stat(image_path)
        key = (os.path.abspath(image_path), stat.st_mtime_ns, size)
        photo = self._thumbnail_cache.pop(key, None)
        if photo is not None:
            self._thumbnail_cache[key] = photo
            return photo

        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source)
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            display_image = image.copy()
        photo = ImageTk.PhotoImage(display_image)
        self._thumbnail_cache[key] = photo
        while len(self._thumbnail_cache) > self.THUMBNAIL_CACHE_SIZE:
            self._thumbnail_cache.popitem(last=False)
        return photo

    def _cancel_thumbnail_render(self):
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
            self._render_job = None
        self._render_queue = []
        self._render_index = 0

    def decrease_zoom(self):
        self._set_zoom(max(10, self._read_zoom() - 10))

    def increase_zoom(self):
        self._set_zoom(min(200, self._read_zoom() + 10))

    def on_zoom_entry_change(self, event=None):
        try:
            self._set_zoom(max(10, min(200, int(self.zoom_entry.get()))))
        except ValueError:
            self._set_zoom(self.zoom_scale)

    def _read_zoom(self):
        try:
            return int(self.zoom_entry.get())
        except ValueError:
            return self.zoom_scale

    def _set_zoom(self, scale):
        self.zoom_scale = scale
        self.zoom_entry.delete(0, tk.END)
        self.zoom_entry.insert(0, str(scale))
        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)

    def switch_to_next_output_dir(self, event=None):
        directories = self.get_label_dirs()
        if directories:
            self.load_images(directories[(self.current_output_index + 1) % len(directories)])
        return "break"

    def switch_to_prev_output_dir(self, event=None):
        directories = self.get_label_dirs()
        if directories:
            self.load_images(directories[(self.current_output_index - 1) % len(directories)])
        return "break"

    # ----------------- 目标目录和图片移动 ----------------- #

    def create_new_folder(self):
        """仅在目标数据集的 output 目录创建文件夹，并立即作为移动目标显示。"""
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

        self.target_dir = folder_name
        self.move_btn.config(state=tk.NORMAL)
        self.refresh_dataset_nav()
        self.refresh_target_dataset_selector()
        self.refresh_output_dirs()
        self.load_target_dirs()
        self._set_status(f"已创建 output/{self.target_dataset}/{folder_name}，可直接作为移动目标。")

    def move_images(self):
        """将选中图片移动到独立选择的目标数据集及其目标子目录。"""
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

    def remove_selected_images(self):
        """仅从 output 删除选中图片，不处理 input 中的同名源文件。"""
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

        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)

        message = f"已从 output 移除 {len(removed_output)} 张图片；input 未作修改。"
        if errors:
            message += self._format_errors(errors, "以下文件未能移除")
        self._set_status(message)

    def delete_selected_images(self):
        """删除选中 output 图片及 input 中对应数据集内的同名文件。"""
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

        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)

        message = f"已删除 output 图片 {len(deleted_output)} 张，input 同名文件 {len(deleted_input)} 个。"
        errors = output_errors + input_errors
        if errors:
            message += self._format_errors(errors, "以下文件未能删除")
        self._set_status(message)

    def archive_output(self):
        """递归合并 output 到 datasets，成功后清理 input 中的同名源文件。"""
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
        self.current_image_paths = []
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

    def _merge_dataset_into_archive(self, source_dir, destination_dir, source_dirs, source_files):
        """把一个 dataset 递归合并到 datasets，冲突文件始终保留并改名。"""
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

    def _delete_archived_input_files(self, output_names_by_dataset):
        """在每个 input/<dataset> 中删除和原 output 同名的文件。"""
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

    def _delete_paths(self, paths):
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

    def find_input_matches(self, output_path):
        """查找 input 中同一 dataset 内与 output 图片同名的源文件。"""
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

    def update_selection_ui(self, frame, image_path):
        selected = image_path in self.selected_images
        frame.configure(style="Selected.TFrame" if selected else "Thumbnail.TFrame")
        if selected:
            frame.selection_indicator.place(relx=0.92, rely=0.08, anchor=tk.CENTER)
        else:
            frame.selection_indicator.place_forget()

    def on_image_single_click(self, event, image_path):
        if self.click_timer is not None:
            self.root.after_cancel(self.click_timer)
        self.click_timer = self.root.after(220, lambda widget=event.widget, path=image_path: self.execute_single_click(widget, path))

    def on_image_double_click(self, event, image_path):
        if self.click_timer is not None:
            self.root.after_cancel(self.click_timer)
            self.click_timer = None
        self.show_fullsize_image(image_path)

    def execute_single_click(self, widget, image_path):
        frame = widget.master
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
        else:
            self.selected_images.add(image_path)
        self.update_selection_ui(frame, image_path)
        self.click_timer = None

    def invert_selection(self):
        current_images = set(self.current_image_paths)
        self.selected_images = current_images - self.selected_images
        self.refresh_selection_ui()

    def refresh_selection_ui(self):
        for frame in self.scroll_frame.winfo_children():
            if hasattr(frame, "img_path"):
                self.update_selection_ui(frame, frame.img_path)

    def show_fullsize_image(self, image_path):
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

    def navigate_image(self, window, label, direction):
        if not window.all_images:
            return
        window.current_index = (window.current_index + direction) % len(window.all_images)
        self.load_preview_image(window, label, window.current_index)

    def toggle_image_selection(self, window):
        image_path = window.all_images[window.current_index]
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
        else:
            self.selected_images.add(image_path)
        self.update_preview_selection_indicator(window, image_path)
        self.sync_thumbnail_selection(image_path)

    def load_preview_image(self, window, label, index):
        image_path = window.all_images[index]
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source)
                max_width = max(320, self.root.winfo_screenwidth() - 220)
                max_height = max(240, self.root.winfo_screenheight() - 300)
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

    def update_preview_selection_indicator(self, window, image_path):
        if image_path in self.selected_images:
            window.selection_indicator.place(relx=0.95, rely=0.05, anchor=tk.NE)
        else:
            window.selection_indicator.place_forget()

    def sync_thumbnail_selection(self, image_path):
        for frame in self.scroll_frame.winfo_children():
            if getattr(frame, "img_path", None) == image_path:
                self.update_selection_ui(frame, image_path)
                return

    def on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.scroll_window, width=event.width)
        self._schedule_scrollbar_update()

    def on_frame_configure(self, event):
        self._schedule_scrollbar_update()

    def _schedule_scrollbar_update(self):
        if self._scroll_update_job is None:
            self._scroll_update_job = self.root.after_idle(self.update_scrollbar_state)

    def update_scrollbar_state(self):
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

    def on_mousewheel(self, event):
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

    def _on_subdir_frame_configure(self, event):
        self.subdir_canvas.configure(scrollregion=self.subdir_canvas.bbox("all"))

    def _on_subdir_canvas_configure(self, event):
        self.subdir_canvas.itemconfigure(self.subdir_canvas_window, width=event.width)

    def _bind_subdir_mousewheel(self, event):
        self.subdir_canvas.bind_all("<MouseWheel>", self._on_subdir_mousewheel)
        self.subdir_canvas.bind_all("<Button-4>", self._on_subdir_mousewheel)
        self.subdir_canvas.bind_all("<Button-5>", self._on_subdir_mousewheel)

    def _unbind_subdir_mousewheel(self, event):
        self.subdir_canvas.unbind_all("<MouseWheel>")
        self.subdir_canvas.unbind_all("<Button-4>")
        self.subdir_canvas.unbind_all("<Button-5>")
        self.bind_events()

    def _on_subdir_mousewheel(self, event):
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
    def _clear_children(widget):
        for child in widget.winfo_children():
            child.destroy()

    @staticmethod
    def _get_child_dirs(path):
        try:
            with os.scandir(path) as entries:
                return sorted(
                    (entry.name for entry in entries if entry.is_dir()),
                    key=locale.strxfrm,
                )
        except OSError:
            return []

    def get_image_files(self, folder):
        try:
            with os.scandir(folder) as entries:
                paths = [
                    entry.path
                    for entry in entries
                    if entry.is_file() and Path(entry.name).suffix.lower() in self.IMAGE_EXTENSIONS
                ]
        except OSError:
            return []
        return sorted(paths, key=lambda path: locale.strxfrm(os.path.basename(path)))

    @staticmethod
    def _iter_files(root):
        if not os.path.isdir(root):
            return []
        files = []
        for directory, _, filenames in os.walk(root):
            files.extend(os.path.join(directory, name) for name in filenames if os.path.isfile(os.path.join(directory, name)))
        return files

    @staticmethod
    def _iter_archivable_files(root):
        """返回归档文件，显式忽略用于保留目录的 .gitkeep。"""
        return [path for path in ImageClassifierUI._iter_files(root) if os.path.basename(path) != ".gitkeep"]

    @staticmethod
    def _iter_dirs(root):
        if not os.path.isdir(root):
            return []
        directories = []
        for directory, subdirs, _ in os.walk(root):
            directories.append(directory)
            directories.extend(os.path.join(directory, name) for name in subdirs)
        return sorted(set(directories), key=lambda path: (path.count(os.sep), path))

    @staticmethod
    def _remove_empty_directories(root):
        if not os.path.isdir(root):
            return
        for directory, _, _ in os.walk(root, topdown=False):
            try:
                os.rmdir(directory)
            except OSError:
                pass

    @staticmethod
    def _unique_destination(directory, filename):
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
    def _same_path(left, right):
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))

    @staticmethod
    def _is_within(path, root):
        try:
            return os.path.commonpath((os.path.abspath(path), os.path.abspath(root))) == os.path.abspath(root)
        except ValueError:
            return False

    @staticmethod
    def _validate_directory_name(name):
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

    def _discard_thumbnail(self, image_path):
        normalized_path = os.path.abspath(image_path)
        stale_keys = [key for key in self._thumbnail_cache if key[0] == normalized_path]
        for key in stale_keys:
            self._thumbnail_cache.pop(key, None)

    def get_first_letter(self, text):
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

    def _set_status(self, message):
        """在右下角显示简洁的非阻塞操作结果。"""
        self.status_var.set(" ".join(str(message).split()))

    @staticmethod
    def _format_errors(errors, heading):
        visible = errors[:5]
        suffix = "\n…" if len(errors) > len(visible) else ""
        return f"\n\n{heading}（{len(errors)} 个）：\n" + "\n".join(visible) + suffix


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierUI(root)
    root.mainloop()
