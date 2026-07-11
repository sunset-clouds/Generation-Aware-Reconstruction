"""
Miscellaneous utility functions.
"""

import argparse


def str2bool(v):
    """
    Convert string to boolean for argparse.
    
    Supports: yes/no, true/false, 1/0, y/n, t/f
    
    Usage:
        parser.add_argument('--flag', type=str2bool, default=True)
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}')
