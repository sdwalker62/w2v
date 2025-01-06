"""A handful of utility functions that don't fit any major themes."""

import shutil

terminal_width = shutil.get_terminal_size().columns

def horizontal_rule():
    print(terminal_width * "=")    
