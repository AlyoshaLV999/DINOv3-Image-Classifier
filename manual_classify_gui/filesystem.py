"""文件系统浏览、缓存以及归档路径辅助函数。"""
from __future__ import annotations

import locale
import os
from pathlib import Path


class FileSystemMixin:
    """带缓存的目录浏览与常用路径/归档文件操作。"""

    def _get_child_dirs(self, path: str) -> list[str]:
        normalized = os.path.abspath(path)
        try:
            stat = os.stat(normalized)
            signature = (stat.st_mtime_ns, stat.st_ctime_ns)
        except OSError:
            self._directory_cache.pop(normalized, None)
            return []
        cached = self._directory_cache.pop(normalized, None)
        if cached and cached[0] == signature:
            self._directory_cache[normalized] = cached
            return list(cached[1])
        try:
            with os.scandir(path) as entries:
                names = sorted((entry.name for entry in entries if entry.is_dir()), key=locale.strxfrm)
        except OSError:
            return []
        self._directory_cache[normalized] = (signature, names)
        while len(self._directory_cache) > self.DIRECTORY_CACHE_SIZE:
            self._directory_cache.popitem(last=False)
        return list(names)

    def get_image_files(self, folder: str) -> list[str]:
        normalized = os.path.abspath(folder)
        try:
            stat = os.stat(normalized)
            signature = (stat.st_mtime_ns, stat.st_ctime_ns)
        except OSError:
            self._image_list_cache.pop(normalized, None)
            return []
        cached = self._image_list_cache.pop(normalized, None)
        if cached and cached[0] == signature:
            self._image_list_cache[normalized] = cached
            return list(cached[1])
        try:
            with os.scandir(folder) as entries:
                paths = [entry.path for entry in entries if entry.is_file() and Path(entry.name).suffix.lower() in self.IMAGE_EXTENSIONS]
        except OSError:
            return []
        paths.sort(key=lambda path: locale.strxfrm(os.path.basename(path)))
        self._image_list_cache[normalized] = (signature, paths)
        while len(self._image_list_cache) > self.IMAGE_LIST_CACHE_SIZE:
            self._image_list_cache.popitem(last=False)
        return list(paths)

    @staticmethod
    def _iter_files(root: str) -> list[str]:
        if not os.path.isdir(root):
            return []
        files: list[str] = []
        for directory, _, filenames in os.walk(root):
            files.extend(os.path.join(directory, name) for name in filenames
                         if os.path.isfile(os.path.join(directory, name)))
        return files

    @staticmethod
    def _iter_archivable_files(root: str) -> list[str]:
        return [path for path in FileSystemMixin._iter_files(root) if os.path.basename(path) != ".gitkeep"]

    @staticmethod
    def _iter_dirs(root: str) -> list[str]:
        if not os.path.isdir(root):
            return []
        directories: list[str] = []
        for directory, subdirs, _ in os.walk(root):
            directories.append(directory)
            directories.extend(os.path.join(directory, name) for name in subdirs)
        return sorted(set(directories), key=lambda path: (path.count(os.sep), path))

    @staticmethod
    def _remove_empty_directories(root: str) -> None:
        if os.path.isdir(root):
            for directory, _, _ in os.walk(root, topdown=False):
                try:
                    os.rmdir(directory)
                except OSError:
                    pass

    @staticmethod
    def _unique_destination(directory: str, filename: str) -> str:
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
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))

    @staticmethod
    def _is_within(path: str, root: str) -> bool:
        try:
            return os.path.commonpath((os.path.abspath(path), os.path.abspath(root))) == os.path.abspath(root)
        except ValueError:
            return False

    @staticmethod
    def _validate_directory_name(name: str) -> str:
        if not name:
            return "文件夹名称不能为空。"
        if name in {".", ".."} or os.path.basename(name) != name:
            return "文件夹名称不能包含路径层级。"
        invalid_chars = set('/\\:*?"<>|')
        if any(char in invalid_chars for char in name):
            return '名称包含非法字符：\\/:*?"<>|'
        if name.endswith((".", " ")):
            return "名称不能以句点或空格结尾。"
        return ""

    @staticmethod
    def get_first_letter(text: str) -> str:
        if not text:
            return "#"
        first = text[0].upper()
        if "A" <= first <= "Z":
            return first
        if first.isdigit():
            return "#"
        try:
            import pypinyin
            result = pypinyin.pinyin(first, style=pypinyin.NORMAL)
            if result and result[0]:
                letter = result[0][0][0].upper()
                if "A" <= letter <= "Z":
                    return letter
        except (ImportError, IndexError, TypeError):
            pass
        return "#"


def iter_files(root: str) -> list[str]:
    """递归列出目录中的普通文件。"""
    return FileSystemMixin._iter_files(root)


def iter_archivable_files(root: str) -> list[str]:
    """递归列出排除占位文件后的归档内容。"""
    return FileSystemMixin._iter_archivable_files(root)


def iter_dirs(root: str) -> list[str]:
    """按深度列出目录树中的目录。"""
    return FileSystemMixin._iter_dirs(root)


def remove_empty_directories(root: str) -> None:
    """移除目录树中空目录。"""
    FileSystemMixin._remove_empty_directories(root)


def unique_destination(directory: str, filename: str) -> str:
    """为归档文件生成不覆盖现有内容的目标路径。"""
    return FileSystemMixin._unique_destination(directory, filename)
