import os
import shutil
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".mpg", ".mpeg"}
USE_FILTER = False   # set to True to limit to VIDEO_EXTS
SKIP_NAMES = {"Thumbs.db", "desktop.ini"}

def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    i = 1
    while True:
        candidate = p.with_name(f"{stem} ({i}){suffix}")
        if not candidate.exists():
            return candidate
        i += 1

def get_executable_dir() -> Path:
    if getattr(sys, 'frozen', False):  # running as exe
        return Path(sys.executable).parent
    else:  # running as script
        return Path(__file__).parent

def main():
    root = tk.Tk()
    root.withdraw()
    root.update()

    default_dir = get_executable_dir()

    folder = filedialog.askdirectory(
        title="Select the folder to process",
        initialdir=str(default_dir)
    )
    if not folder:
        return

    folder_path = Path(folder)
    moved = 0
    skipped = 0
    errors = 0

    try:
        for entry in os.scandir(folder_path):
            if not entry.is_file():
                continue

            name = entry.name
            if name in SKIP_NAMES:
                skipped += 1
                continue

            src = Path(entry.path)
            ext = src.suffix.lower()
            if USE_FILTER and ext not in VIDEO_EXTS:
                skipped += 1
                continue

            base = src.stem or src.name
            dest_dir = folder_path / base

            try:
                dest_dir.mkdir(exist_ok=True)
                dest_file = dest_dir / src.name
                dest_file = unique_path(dest_file)
                shutil.move(str(src), str(dest_file))
                moved += 1
            except Exception:
                errors += 1

    except Exception:
        messagebox.showerror("Error", "Failed to read the folder. Do you have permission?")
        return

    messagebox.showinfo(
        "Done",
        f"Processed: {folder}\n\nMoved: {moved}\nSkipped: {skipped}\nErrors: {errors}"
    )

if __name__ == "__main__":
    main()
