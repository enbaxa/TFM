import logging
from dataclasses import dataclass


# use main logger, shared by the whole module
logger = logging.getLogger("main")

# Define the "Test" class. This represents 1 test in the system.
# Each test should be unique.
@dataclass
class Test:
    name: str = "Unknown"
    description: str = "No description provided"

if __name__ == "__main__":
    # Do nothing
    # This is just a package for definitions
    pass