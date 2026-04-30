from pathlib import Path
import logging
logger = logging.getLogger(__name__)


def assert_file_exists(filename_path: Path):
    if filename_path.is_file():
        return True
    else:
        logger.error(f"{str(filename_path)} does not exist!\n")
