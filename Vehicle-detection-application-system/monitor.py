from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "缺少依赖 watchdog。请先在你的 conda 环境中安装：\n"
        "- `pip install watchdog` 或\n"
        "- `conda install -c conda-forge watchdog`\n"
        f"原始错误: {e!r}"
    ) from e


IMAGE_EXT_DEFAULT = (".png", ".jpg", ".jpeg", ".bmp")


def _is_image(path: str, exts: Tuple[str, ...]) -> bool:
    p = path.lower()
    return any(p.endswith(ext) for ext in exts)


@dataclass
class CommonArgs:
    mode: str
    path: Optional[str]
    recursive: bool
    debounce_ms: int
    image_ext: Tuple[str, ...]
    max_workers: int


class Debouncer:
    def __init__(self, debounce_ms: int):
        self.debounce_ms = max(0, debounce_ms)
        self._last_ts = 0.0

    def allow(self) -> bool:
        now = time.time()
        if self.debounce_ms == 0:
            self._last_ts = now
            return True
        if (now - self._last_ts) * 1000.0 >= self.debounce_ms:
            self._last_ts = now
            return True
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        required=True,
        choices=["django-tree", "folder-modified", "new-images"],
        help="监听模式：django-tree / folder-modified / new-images",
    )
    p.add_argument("--path", default=None, help="监听路径（folder-modified/new-images 必填；django-tree 可选）")
    p.add_argument("--recursive", action="store_true", help="是否递归监听子目录")
    p.add_argument("--debounce-ms", type=int, default=800, help="防抖毫秒（folder-modified 推荐）")
    p.add_argument(
        "--image-ext",
        default=",".join(IMAGE_EXT_DEFAULT),
        help="图片扩展名，逗号分隔，例如 .jpg,.png",
    )
    p.add_argument("--max-workers", type=int, default=4, help="线程池大小（django-tree 使用）")
    return p


def parse_args(argv: Optional[Iterable[str]] = None) -> CommonArgs:
    opt = build_parser().parse_args(list(argv) if argv is not None else None)
    exts = tuple(e.strip().lower() for e in opt.image_ext.split(",") if e.strip())
    if not exts:
        exts = IMAGE_EXT_DEFAULT
    return CommonArgs(
        mode=opt.mode,
        path=opt.path,
        recursive=bool(opt.recursive),
        debounce_ms=int(opt.debounce_ms),
        image_ext=exts,
        max_workers=max(1, int(opt.max_workers)),
    )


# ------------------------
# mode: folder-modified
# ------------------------


class FolderModifiedHandler(FileSystemEventHandler):
    def __init__(self, debouncer: Debouncer):
        self.debouncer = debouncer

    def on_modified(self, event):
        if not self.debouncer.allow():
            return
        try:
            import connecthanshu

            print(f"文件夹发生变化: {event.src_path} - {event.event_type}")
            connecthanshu.connectfunchanshu()
        except Exception as e:
            print(f"处理失败: {e!r}")


# ------------------------
# mode: new-images
# ------------------------


class NewImagesHandler(FileSystemEventHandler):
    def __init__(self, image_ext: Tuple[str, ...]):
        self.image_ext = image_ext
        self.processed: Set[str] = set()

    def on_created(self, event):
        if event.is_directory:
            return
        if not _is_image(event.src_path, self.image_ext):
            return

        filename = os.path.basename(event.src_path)
        if filename in self.processed:
            return

        self.processed.add(filename)
        print(f"新添加的图片: {filename}")
        try:
            import connecthanshu

            connecthanshu.connectfunchanshu()
        except Exception as e:
            print(f"处理失败: {e!r}")


# ------------------------
# mode: django-tree
# ------------------------


def _resolve_django_images_root(path_override: Optional[str]) -> str:
    if path_override:
        return path_override

    try:
        from django.conf import settings

        return os.path.join(settings.MEDIA_ROOT, "data", "images_o")
    except Exception as e:
        raise RuntimeError(
            "django-tree 模式需要 Django settings 可用，或者显式传入 --path。\n"
            f"当前无法导入 django.conf.settings: {e!r}"
        )


class DjangoRootFolderHandler(FileSystemEventHandler):
    def __init__(self, watcher: "DjangoTreeWatcher"):
        self.watcher = watcher

    def on_created(self, event):
        if event.is_directory:
            folder_name = os.path.basename(event.src_path)
            print(f"have new position: {folder_name}")
            self.watcher.add_location_watcher(event.src_path)


class DjangoTimeFolderHandler(FileSystemEventHandler):
    def __init__(self, location_path: str, watcher: "DjangoTreeWatcher"):
        self.location_path = location_path
        self.watcher = watcher

    def on_created(self, event):
        if not event.is_directory:
            return
        folder_name = os.path.basename(event.src_path)
        print(f"have new time: {folder_name} in {os.path.basename(self.location_path)}")

        try:
            # 保持与原 monitorF.py 一致：延迟导入 detect
            from app.meth.meth_1 import detect  # type: ignore
        except Exception as e:
            print(f"无法导入 detect: {e!r}")
            return

        position = os.path.basename(self.location_path)
        time_str = folder_name

        try:
            from django.conf import settings

            images_path = os.path.join(settings.MEDIA_ROOT, "data", "images_o")
            face_db = os.path.join(settings.MEDIA_ROOT, "data", "face_db")
        except Exception:
            # 如果无法导入 settings，就退化到监听根目录同级 data/ 结构
            images_path = self.watcher.images_root
            face_db = os.path.join(os.path.dirname(self.watcher.images_root), "face_db")

        fut = self.watcher.executor.submit(detect, position, time_str, images_path, face_db)
        if fut:
            print("detect going")


class DjangoTreeWatcher:
    def __init__(self, images_root: str, max_workers: int):
        self.images_root = images_root
        self.observer = Observer()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.monitored_locations: Set[str] = set()

    def add_location_watcher(self, location_path: str) -> None:
        if location_path in self.monitored_locations:
            return
        self.monitored_locations.add(location_path)
        print(f"began to listen: {os.path.basename(location_path)}")
        handler = DjangoTimeFolderHandler(location_path, self)
        self.observer.schedule(handler, location_path, recursive=False)

    def run(self) -> None:
        os.makedirs(self.images_root, exist_ok=True)
        root_handler = DjangoRootFolderHandler(self)
        self.observer.schedule(root_handler, self.images_root, recursive=False)
        self.observer.start()

        # 监听已有的地点目录
        for name in os.listdir(self.images_root):
            p = os.path.join(self.images_root, name)
            if os.path.isdir(p):
                self.add_location_watcher(p)

        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()


def run_observer(path: str, handler: FileSystemEventHandler, recursive: bool) -> None:
    if not path:
        raise ValueError("path is required")
    os.makedirs(path, exist_ok=True)
    observer = Observer()
    observer.schedule(handler, path=path, recursive=recursive)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    if args.mode == "folder-modified":
        if not args.path:
            print("folder-modified 模式必须传入 --path", file=sys.stderr)
            raise SystemExit(2)
        run_observer(args.path, FolderModifiedHandler(Debouncer(args.debounce_ms)), args.recursive)
        return

    if args.mode == "new-images":
        if not args.path:
            print("new-images 模式必须传入 --path", file=sys.stderr)
            raise SystemExit(2)
        run_observer(args.path, NewImagesHandler(args.image_ext), args.recursive)
        return

    if args.mode == "django-tree":
        images_root = _resolve_django_images_root(args.path)
        DjangoTreeWatcher(images_root=images_root, max_workers=args.max_workers).run()
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()

