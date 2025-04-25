import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Tuple, List, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, DirModifiedEvent, FileModifiedEvent, DirCreatedEvent, \
    FileCreatedEvent


class ChmodHandler(FileSystemEventHandler):
    mode = 0o2777

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        try:
            os.chmod(event.src_path, self.mode)
        except Exception as e:
            pass

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        try:
            os.chmod(event.src_path, self.mode)
        except Exception as e:
            pass


def mkdtemp_watchdog(path: str) -> Tuple[str, Observer]:
    temp_path: str = mkdtemp(path)
    observer: Observer = Observer()
    observer.schedule(event_handler=ChmodHandler(),
                      path=temp_path,
                      recursive=True,
                      event_filter=[DirCreatedEvent, FileCreatedEvent, DirModifiedEvent, FileModifiedEvent])
    observer.start()
    return temp_path, observer


def free_watchdog_dir(path: str, observer: Observer) -> None:
    observer.stop()
    observer.join()
    shutil.rmtree(path)


def mkdtemp(path: str) -> str:
    name: str = str(uuid.uuid4())
    temp_path: str = f'{path}/{name}'
    subprocess.run(f'mkdir -p {temp_path}', shell=True)
    os.chmod(temp_path, 0o2777)
    return temp_path


def is_absolute_path(path: str) -> bool:
    return path.startswith('/')


def get_files_with_suffix(dir_path: str, suffix: str) -> List[str]:
    folder_path = Path(dir_path)
    return list(folder_path.glob(f'*{suffix}'))


def get_dir_from_exception(ex: Exception) -> Optional[str]:
    if isinstance(ex, FileNotFoundError):
        ex_not_found: FileNotFoundError = ex
        dir_path: str = os.path.dirname(ex_not_found.filename)
        return dir_path
    return None
