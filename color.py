#!/usr/bin/env python3
# Author: Enric Basso

"""
This module contains a class that defines color codes for the terminal
and a method to color a string with a given color
"""
from dataclasses import dataclass

#  """
#                 BASH                   C                  Python3
#  literal	      \e, \E	           \e (non-standard)	Python 3.3+
#  octal	      \033                 \33, \033            \33, \033
#  hexadecimal	  \x1b                 \x1b                 \x1b
#  Unicode	      \u1b,\U1b              â                  \u001b, \U0000001b
#
#  This are escape characters to tell whoever is parsing above
#  what to do. \033 is a good choice because works for all.
#
#  What follows in the bracket [
#      The bracket signals the start of the byte sequence that gives
#      the instructions
#
#      instructions can be separated with ;
#
#      instructions are closed by m character, known as "last byte"
#
#      0 -> reset
#      1 -> bold
#      2 -> faint
#      4 -> underlined
#      5 -> blinking
#
#      30-37 -> color
#      40-47 -> background color
#
#      90-97 -> bright foreground color
#      10-107-> bright background color
#  """


@dataclass(init=False, frozen=True)
class Color:
    """
    This class contains color codes for the terminal.
    """
    red: str = "\033[31m"
    green: str = "\033[32m"
    yellow: str = "\033[33m"
    blue: str = "\033[34m"
    magenta: str = "\033[35m"
    cyan: str = "\033[36m"
    white: str = "\033[37m"
    bold: str = "\033[1m"
    faint: str = "\033[2m"
    underline: str = "\033[4m"
    blink: str = "\033[5m"
    reset: str = "\033[0m"


def color_message(msg: str, color: str) -> str:
    """
    Adds color codes at the beginning and end of a string.

    Args:
        msg (str): The string to which to add color codes.
        color (str): The color name. Must be defined inside the Color object.

    Returns:
        str: The same string as `msg` with the color code prepended and a reset code appended.

    Raises:
        AttributeError: If no color named `color` is found.
    """
    try:
        return "".join((getattr(Color, color), msg, getattr(Color, "reset")))
    except AttributeError as e:
        print(e)
        print(f"No color named \"{color}\" was found. Ignoring coloring of message.")
        return msg
