"""
This module contains classes and definitions for test categories.

The Test class is a dataclass that represents a test in the system.
Each test should be unique, and should have a name and a description.
"""
import logging
from dataclasses import dataclass


# use main logger, shared by the whole module
logger = logging.getLogger("main")

@dataclass
class Test:
    """
    This class represents a test.

    Attributes:
        name (str): The name of the test.
        description (str): The description of the test.
    """
    name: str = "Unknown"
    description: str = "No description provided"


if __name__ == "__main__":
    # Do nothing
    # This is just a package for definitions
    pass