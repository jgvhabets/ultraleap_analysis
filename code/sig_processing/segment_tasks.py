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


def block_cut(df, block_times, block):
    block_df = df[np.logical_and(
                df['global_time'] >= block_times[f'{block}_start'],
                df['global_time'] <= block_times[f'{block}_end']
            )
            ].reset_index(drop = True)
    return block_df


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

    # the excel table has two columns 'Unnamed' if the task has 3 blocks
    if set(['Unnamed: 8', 'Unnamed: 14']).issubset(run_row.keys()):
        ls_blocks = ['b1', 'b2', 'b3']
    elif set(['Unnamed: 8']).issubset(run_row.keys()):
        ls_blocks = ['b1', 'b2']
    else: 
        ls_blocks = ['b1']

    block_times = {}
    block_df = {}
    new_blocks = []

    for block in ls_blocks:
        for time in ['start', 'end']:
            t = run_row[f'{block}_{time}_ul']
            if type(t) == datetime.time: 
                t = t.strftime("%H:%M:%S")
                block_times[f'{block}_{time}'] = t
                new_blocks.append(block)
                new_blocks = list(set(new_blocks))
            else:
                continue    
    
    for b_idx, b in enumerate(new_blocks):
        b_idx +=1
      
        block_df[f'b{b_idx}_df'] = df[
            np.logical_and(
                df['global_time'] >= block_times[f'{b}_start'],
                df['global_time'] <= block_times[f'{b}_end']
                )
                ].reset_index(drop=True)
        locals().update(block_df)
    return block_df