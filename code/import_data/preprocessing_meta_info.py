"""
Importing metadata for preprocessing
ultraleap data
"""

# Import public packages and fucntions
import os
import numpy as np
import pandas as pd

# impoprt own functions
from import_data.find_paths import find_onedrive_path


def load_block_timestamps(
    sub: str, task: str, side: str
):
    """"
    .....
    """
    # prevent incorrent side variable
    if side == 'lh': side = 'left'
    elif side == 'rh': side = 'right'

    if 'ul' in sub:
        # prevent incorrect task variable
        if sub[:2].lower() == 'ul': sub = sub[2:]

        blocktimes = pd.read_excel(
            os.path.join(
                find_onedrive_path('patientdata'),
                f'ul{sub}',
                f'ul{sub}_block_timestamps.xlsx'),
            sheet_name=f'{task}_{side}',
        )
        blocktimes.set_index('cond_cam', inplace = True)
    
    elif 'control' in sub:
        blocktimes = pd.read_excel(
            os.path.join(
                find_onedrive_path('control'),
                f'{sub}',
                f'{sub}_block_timestamps.xlsx'),
            sheet_name=f'{task}_{side}',
        )
        blocktimes.set_index('cam', inplace = True)
    return blocktimes
