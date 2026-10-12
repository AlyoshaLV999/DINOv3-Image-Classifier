"""output 归档流程。"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from . import filesystem
from .widgets import _FunctionWorker

try:
    from send2trash import send2trash
except ImportError:  # 可选依赖；未安装时仍可永久删除。
    send2trash = None

LOGGER = logging.getLogger("manual_classify_gui")


class ArchiveMixin:
    """后台归档流程以及其文件合并和删除辅助逻辑。"""

    # output 中不参与归档、而是被直接删除的特殊目录名。
    # 定义在此处作为单一事实来源，供归档流程统一引用。
    OUTOFTAG_DIR_NAME = "OUTOFTAG"

    def archive_output(self) -> None:
        if self._archive_running:
            self.set_status("归档正在进行，请稍候。")
            return
        if self.recycle_bin_enabled and send2trash is None:
            self.set_status("回收站不可用：请取消“删除到回收站”后重试。")
            return
        datasets = self.get_output_datasets()
        if not datasets:
            self.set_status("无需归档：output 中没有可归档的数据集目录。")
            return
        use_trash = self.recycle_bin_enabled
        self._archive_running = True
        self._set_controls_enabled(False)
        self.set_status("正在归档 output 文件…")
        worker = _FunctionWorker(self._archive_output_job, datasets, use_trash)
        self._start_worker(worker, self._archive_finished, self._archive_failed)

    @staticmethod
    def _archive_output_job(dataset_names: list[str], use_trash: bool) -> dict[str, Any]:
        """在后台线程中执行 output 归档。

        ``output/OUTOFTAG`` 属于“超出标签”的丢弃目录，不参与归档合并，
        无论其中是否包含文件，都会先被整体删除（按 ``use_trash`` 决定
        进回收站还是永久删除）。其余数据集文件合并进
        ``datasets/<name>``，成功后清理 input 中的同名源文件。

        :param dataset_names: ``output`` 下的数据集目录名列表。
        :param use_trash: 为 ``True`` 时删除走回收站，否则永久删除。
        :return: 描述归档结果的字典，包含键 ``empty``、``partial``、
            ``count``、``moved``、``errors``、``deleted_input``、
            ``input_errors``、``out_of_tag_deleted``、``out_of_tag_error``。
            未触发对应分支时相关键可能缺失。
        """
        # 第一步：清理 OUTOFTAG（不归档）。
        out_of_tag_path = os.path.join("output", ArchiveMixin.OUTOFTAG_DIR_NAME)
        out_of_tag_deleted = False
        out_of_tag_error = ""
        if os.path.isdir(out_of_tag_path):
            out_of_tag_deleted, out_of_tag_error = ArchiveMixin._delete_directory_tree(
                out_of_tag_path, use_trash
            )

        # 第二步：收集其余数据集的可归档内容。
        archive_names = [name for name in dataset_names if name != ArchiveMixin.OUTOFTAG_DIR_NAME]
        snapshots: dict[str, dict[str, Any]] = {}
        count = 0
        for name in archive_names:
            source = os.path.join("output", name)
            files = filesystem.iter_archivable_files(source)
            snapshots[name] = {"source": source, "files": files, "dirs": filesystem.iter_dirs(source)}
            count += len(files)

        result: dict[str, Any] = {
            "out_of_tag_deleted": out_of_tag_deleted,
            "out_of_tag_error": out_of_tag_error,
        }
        if count == 0:
            result["empty"] = True
            return result

        os.makedirs("datasets", exist_ok=True)
        moved = 0
        errors: list[str] = []
        names_by_dataset: dict[str, set[str]] = {}
        for name, snapshot in snapshots.items():
            completed, current_errors = ArchiveMixin._merge_dataset_into_archive(
                snapshot["source"], os.path.join("datasets", name), snapshot["dirs"], snapshot["files"]
            )
            moved += completed
            errors.extend(current_errors)
            names_by_dataset[name] = {os.path.basename(path) for path in snapshot["files"]}
        if errors or moved != count:
            result.update({"partial": True, "count": count, "moved": moved, "errors": errors})
            return result
        deleted_input, input_errors = ArchiveMixin._delete_archived_input_files(names_by_dataset, use_trash)
        result.update({"count": count, "deleted_input": len(deleted_input), "input_errors": input_errors})
        return result

    def _archive_finished(self, result: dict[str, Any]) -> None:
        self._archive_running = False
        self._set_controls_enabled(True)
        self._clear_fs_caches()

        out_of_tag_suffix = ""
        if result.get("out_of_tag_deleted"):
            out_of_tag_suffix = f"；已删除 output/{ArchiveMixin.OUTOFTAG_DIR_NAME} 目录。"
        elif result.get("out_of_tag_error"):
            out_of_tag_suffix = (
                f"；删除 output/{ArchiveMixin.OUTOFTAG_DIR_NAME} 失败：{result['out_of_tag_error']}"
            )

        # 当前查看的数据集若恰好是被删除的 OUTOFTAG，需同步清空列表，
        # 避免残留指向已删除文件的列表项与缩略图。
        if result.get("out_of_tag_deleted") and self.current_dataset == ArchiveMixin.OUTOFTAG_DIR_NAME:
            self.current_dataset = ""
            self.current_output_dir = ""
            self.current_output_index = -1
            self.current_image_paths = []
            self.selected_images.clear()
            self.image_list.clear()

        if result.get("empty"):
            self.set_status(
                f"无需归档：output 中没有可归档文件（已忽略 .gitkeep）{out_of_tag_suffix}"
            )
        elif result.get("partial"):
            message = (
                f"归档未完成：已移动 {result['moved']} / {result['count']} 个文件；"
                f"input 未清理{out_of_tag_suffix}"
            )
            if result["errors"]:
                message += self._format_errors(result["errors"], "归档错误")
            self.set_status(message)
        else:
            self.selected_images.clear()
            self.current_dataset = ""
            self.current_output_dir = ""
            self.current_output_index = -1
            self.current_image_paths = []
            self.image_list.clear()
            message = (
                f"归档完成：已将 {result['count']} 个文件合并到 datasets，"
                f"已清理 input 中 {result['deleted_input']} 个同名源文件{out_of_tag_suffix}"
            )
            if result["input_errors"]:
                message += self._format_errors(result["input_errors"], "input 清理失败")
            self.set_status(message)

        self.refresh_dataset_nav()
        self.refresh_output_dirs()
        self.refresh_target_dataset_selector()
        self.load_target_dirs()
        if result.get("partial") and self.current_output_dir:
            self.load_images(os.path.basename(self.current_output_dir), clear_selection=False)

    def _archive_failed(self, message: str) -> None:
        self._archive_running = False
        self._set_controls_enabled(True)
        LOGGER.error("归档失败：%s", message)
        self.set_status(f"归档失败：{message}")

    @staticmethod
    def _merge_dataset_into_archive(source_dir: str, destination_dir: str,
                                    source_dirs: list[str], source_files: list[str]) -> tuple[int, list[str]]:
        moved, errors = 0, []
        try:
            for directory in source_dirs:
                relative = os.path.relpath(directory, source_dir)
                target = destination_dir if relative == "." else os.path.join(destination_dir, relative)
                os.makedirs(target, exist_ok=True)
        except OSError as error:
            return moved, [f"{source_dir}：无法创建归档目录：{error}"]
        for source in source_files:
            if not os.path.isfile(source):
                errors.append(f"{source}：源文件不存在")
                continue
            try:
                relative = os.path.relpath(source, source_dir)
                target_dir = os.path.join(destination_dir, os.path.dirname(relative))
                os.makedirs(target_dir, exist_ok=True)
                shutil.move(source, filesystem.unique_destination(target_dir, os.path.basename(source)))
                moved += 1
            except OSError as error:
                errors.append(f"{source}：{error}")
        filesystem.remove_empty_directories(source_dir)
        return moved, errors

    @staticmethod
    def _delete_archived_input_files(names_by_dataset: dict[str, set[str]], use_trash: bool) -> tuple[list[str], list[str]]:
        targets: set[str] = set()
        if not os.path.isdir("input"):
            return [], []
        for dataset, names in names_by_dataset.items():
            search_root = os.path.join("input", dataset)
            if not os.path.isdir(search_root):
                search_root = "input"
            for path in filesystem.iter_files(search_root):
                if os.path.basename(path) in names:
                    targets.add(path)
        return ArchiveMixin._delete_paths(sorted(targets), use_trash)

    @staticmethod
    def _delete_paths(paths: list[str], use_trash: bool = False) -> tuple[list[str], list[str]]:
        deleted, errors = [], []
        for path in paths:
            try:
                if not os.path.isfile(path):
                    continue
                if use_trash:
                    if send2trash is None:
                        raise RuntimeError("send2trash 不可用")
                    send2trash(path)
                else:
                    os.remove(path)
                deleted.append(path)
            except (OSError, RuntimeError) as error:
                errors.append(f"{os.path.basename(path)}：{error}")
        return deleted, errors

    @staticmethod
    def _delete_directory_tree(path: str, use_trash: bool) -> tuple[bool, str]:
        """删除整个目录树，用于清除 OUTOFTAG 等不参与归档的目录。

        与 :meth:`_delete_paths` 只处理文件不同，本方法处理的是目录本身
        及其全部内容：``use_trash`` 为真时整体送入回收站，否则用
        :func:`shutil.rmtree` 永久删除。

        :param path: 待删除的目录路径。
        :param use_trash: 为 ``True`` 时走 ``send2trash``，否则永久删除。
        :return: ``(是否成功, 错误消息)``；成功时错误消息为空字符串。
        """
        if not os.path.isdir(path):
            return False, "目录不存在"
        try:
            if use_trash:
                if send2trash is None:
                    raise RuntimeError("send2trash 不可用")
                send2trash(path)
            else:
                shutil.rmtree(path)
        except (OSError, RuntimeError) as error:
            return False, str(error)
        return True, ""

    def find_input_matches(self, output_path: str) -> list[str]:
        if not self._is_within(output_path, "output"):
            return []
        try:
            relative = os.path.relpath(output_path, "output")
            dataset = Path(relative).parts[0]
        except (IndexError, ValueError):
            return []
        root = os.path.join("input", dataset)
        if not os.path.isdir(root):
            root = "input"
        name = os.path.basename(output_path)
        return [path for path in self._iter_files(root) if os.path.basename(path) == name]