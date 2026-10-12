"""用户触发的图片移动和删除操作。"""
from __future__ import annotations

import os
import shutil
from PySide6.QtWidgets import QInputDialog

try:
    from send2trash import send2trash
except ImportError:  # 可选依赖。
    send2trash = None


class ActionsMixin:
    """新建目标目录、移动及删除图片等用户操作。"""

    def switch_to_next_output_dir(self) -> None:
        names = self.get_label_dirs()
        if names:
            self.load_images(names[(self.current_output_index + 1) % len(names)])

    def switch_to_prev_output_dir(self) -> None:
        names = self.get_label_dirs()
        if names:
            self.load_images(names[(self.current_output_index - 1) % len(names)])

    def create_new_folder(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        if not self.target_dataset:
            self.set_status("请先选择目标数据集。")
            return
        folder_name, accepted = QInputDialog.getText(self, "新建目标文件夹", f"在 output/{self.target_dataset} 下新建文件夹：")
        if not accepted:
            return
        folder_name = folder_name.strip()
        error_message = self._validate_directory_name(folder_name)
        if error_message:
            self.set_status(f"无法创建目标文件夹：{error_message}")
            return
        path = os.path.join("output", self.target_dataset, folder_name)
        if os.path.exists(path):
            self.set_status(f"目标文件夹已存在：{path}")
            return
        try:
            os.makedirs(path, exist_ok=False)
        except OSError as error:
            self.set_status(f"无法创建输出目录：{error}")
            return
        self._clear_fs_caches()
        self.target_dir = folder_name
        self.refresh_dataset_nav()
        self.refresh_target_dataset_selector()
        self.refresh_output_dirs()
        self.load_target_dirs()
        self.set_status(f"已创建 output/{self.target_dataset}/{folder_name}，可直接作为移动目标。")

    def move_images(self) -> None:
        if not self.selected_images:
            self.set_status("请先选择要移动的图片。")
            return
        if not self.target_dataset or not self.target_dir:
            self.set_status("请先选择目标数据集和目标子目录。")
            return
        target_dir = os.path.join("output", self.target_dataset, self.target_dir)
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as error:
            self.set_status(f"移动失败：无法创建目标目录：{error}")
            return
        moved_paths: set[str] = set()
        failures: list[str] = []
        for source in sorted(self.selected_images):
            if not os.path.isfile(source):
                failures.append(f"{os.path.basename(source)}：源文件不存在")
                continue
            destination = os.path.join(target_dir, os.path.basename(source))
            if self._same_path(source, destination):
                failures.append(f"{os.path.basename(source)}：已在目标目录中")
            elif os.path.exists(destination):
                failures.append(f"{os.path.basename(source)}：目标位置存在同名文件")
            else:
                try:
                    shutil.move(source, destination)
                    moved_paths.add(source)
                except OSError as error:
                    failures.append(f"{os.path.basename(source)}：{error}")
        self.selected_images.difference_update(moved_paths)
        self._clear_fs_caches()
        self.refresh_dataset_nav()
        self.refresh_target_dataset_selector()
        self.load_target_dirs()
        self.refresh_output_dirs()
        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)
        if moved_paths:
            message = f"已移动 {len(moved_paths)} 张图片到 output/{self.target_dataset}/{self.target_dir}。"
            if failures:
                message += self._format_errors(failures, "未移动")
            self.set_status(message)
        elif failures:
            self.set_status(self._format_errors(failures, "没有图片被移动").strip())

    def remove_selected_images(self) -> None:
        output_images = sorted(path for path in self.selected_images if os.path.isfile(path) and self._is_within(path, "output"))
        if not output_images:
            self.set_status("请先选择要从 output 移除的图片。")
            return
        if self.recycle_bin_enabled and send2trash is None:
            self.set_status("回收站不可用：请取消“删除到回收站”后重试。")
            return
        removed, errors = self._delete_paths(output_images, self.recycle_bin_enabled)
        self.selected_images.difference_update(removed)
        self._clear_fs_caches()
        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)
        message = f"已从 output 移除 {len(removed)} 张图片；input 未作修改。"
        if errors:
            message += self._format_errors(errors, "以下文件未能移除")
        self.set_status(message)

    def delete_selected_images(self) -> None:
        output_images = sorted(path for path in self.selected_images if os.path.isfile(path) and self._is_within(path, "output"))
        if not output_images:
            self.set_status("请先选择要删除的 output 图片。")
            return
        if self.recycle_bin_enabled and send2trash is None:
            self.set_status("回收站不可用：请取消“删除到回收站”后重试。")
            return
        input_matches = {path for output in output_images for path in self.find_input_matches(output)}
        deleted_output, output_errors = self._delete_paths(output_images, self.recycle_bin_enabled)
        deleted_input, input_errors = self._delete_paths(sorted(input_matches), self.recycle_bin_enabled)
        self.selected_images.difference_update(deleted_output)
        self._clear_fs_caches()
        if self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)
        message = f"已删除 output 图片 {len(deleted_output)} 张，input 同名文件 {len(deleted_input)} 个。"
        errors = output_errors + input_errors
        if errors:
            message += self._format_errors(errors, "以下文件未能删除")
        self.set_status(message)
