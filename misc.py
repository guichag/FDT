"""Miscellaneous"""

import re


### FUNC ###

def atoi(text):
    """www.tutorialspoint.com/"""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


