import logging
import pathlib
from typing import Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger("main")


@ dataclass
class Dataset(ABC):
    path: pathlib.Path
    data: Any  # To be defined later

    @abstractmethod
    def clean(self):
        ...


if __name__ == "__main__":
    # Do nothing
    # This is just a package for definitions
    pass
