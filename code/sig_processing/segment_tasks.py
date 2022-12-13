"""
Importing Handtrack-Files from Ultraleap
version data output 01.09.2022

Most recent version prior to repo-deletion
"""

# Import public packages and fucntions
import numpy as np
import pandas as pd
import datetime

import import_data.preprocessing_meta_info as meta_info


def task_block_extraction(
    df, sub, task, side, cond, cam
):
    """
    Divides the cleaned data between two time points (can 
    be used in cases where all task are in the same file).

    CHANGE FUNCTIONALITY FOR ONLY 1 BLOCK PRESENT
        
        Input:
            - cleaned df (DataFrame),
            time1: global_time of the start of a task, 
            time2: global_time of the end of a task.
        
        Output:
            - new dataframe for specific task/block.
    """
    # get block timestamps corresponding to data
    blocktimes = meta_info.load_block_timestamps(
        sub=sub, task=task, side=side,)
    # # deal with Error:
    # try:
    #     # normal code
    # except ValueError:
    #     # code will only run when a valueError occurs
    
    if cam == 'desktop': cam = 'dt'
    
    run_row = blocktimes.loc[f'{cond.lower()}_{cam.lower()}']

    block_times = {}
    for block in ['b1', 'b2']:
        for time in ['start', 'end']:
            t = run_row[f'{block}_{time}_ul']
            # correct if timestamps is datetime
            if type(t) == datetime.time: t = t.strftime("%H:%M:%S")
            
            block_times[f'{block}_{time}'] = t

    block1_df = df[
        np.logical_and(
            df['global_time'] >= block_times['b1_start'],
            df['global_time'] <= block_times['b1_end']
        )
    ].reset_index(drop=True)
    block2_df = df[
        np.logical_and(
            df['global_time'] >= block_times['b2_start'],
            df['global_time'] <= block_times['b2_end']
        )
    ].reset_index(drop=True)

    return block1_df, block2_df