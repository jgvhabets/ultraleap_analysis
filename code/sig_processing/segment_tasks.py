# Import public packages and fucntions
import traces
import numpy as np
import pandas as pd
import datetime 
from itertools import product 
import os 
import json 
import import_data.preprocessing_meta_info as meta_info
from datetime import datetime
from datetime import timedelta

from import_data import import_and_convert_data as import_dat


def block_extraction(
    df, sub, task, side, cond, cam, to_save = False
):
    """
    Divides the cleaned data between two time points (can 
    be used in cases where all task are in the same file).

        
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

    if cam == 'desktop': cam = 'dt'

    if 'control' in str(sub):
        run_row = blocktimes.loc[f'{cam.lower()}']

    elif 'ul' in str(sub):
        run_row = blocktimes.loc[f'{cond.lower()}_{cam.lower()}']

    block_times = {}
    blocks_dict = {}
    new_blocks = []

    for block, time in product(['b1', 'b2', 'b3'],['start','end']):
        try:
            t = run_row[f'{block}_{time}_ul']  

        except KeyError:
            continue

        if type(t) != float: 
            t = t.strftime("%H:%M:%S")
            block_times[f'{block}_{time}'] = t
            new_blocks.append(block)

    new_blocks = list(set(new_blocks))

    for b_idx, b in enumerate(new_blocks):
        b_idx +=1

        old_df = df[
            np.logical_and(
                df['global_time'] >= block_times[f'{b}_start'],
                df['global_time'] <= block_times[f'{b}_end']
                )
                ].reset_index(drop=True)

        blocks_dict[f'b{b_idx}'] = regularize_block(old_df, 10)

    return blocks_dict


def regularize_block(
    or_block, 
    new_timedelta_ms = int
):

    '''

    This is used to regularize unevenly-spaced dataframes.
    If sampling rate differs between blocks/ dataframes, this function
    helps to normalize them.

    Input: 
        - or_block (pandas Dataframe): csv file of time series data 
        - new_timedelta_ms (int): the new sampling rate 

    Returns: 
        - clean_new_df (pandas Dataframe): dataframe with regular intervals 
                                           between samples 

    '''

    glob_time_to_date_time = [datetime.strptime(or_block['global_time'].iloc[row], '%H:%M:%S:%f') for row in np.arange(0, or_block.shape[0])]

    if 'date_time' not in or_block.columns:
        or_block.insert(loc = 0, column = 'date_time', value = glob_time_to_date_time)

    new_df = pd.DataFrame()

    for col in or_block.columns:
        if col.startswith('pinch'): continue
        if col[-1] not in ['x', 'y', 'z']: continue

        ls_tuple = [
            (or_block['date_time'][row], or_block[col][row])
            for row in np.arange(0, or_block.shape[0])
        ]
        ts = traces.TimeSeries(ls_tuple)

        new_col = ts.sample(
            sampling_period = timedelta(milliseconds = new_timedelta_ms),
            start = or_block['date_time'][0].round('1s'),
            end = or_block['date_time'][or_block.shape[0]-1],
            interpolate = 'linear',
        )

        new_times = []
        new_values = []

        for t in new_col:
            new_times.append(t[0])
            new_values.append(t[1])

        if 'date_time' not in new_df.columns:
            new_df.insert(loc = 0, column = 'date_time', value = new_times)

        new_df.insert(loc = new_df.shape[1], column = col, value = new_values)

    clean_new_df = import_dat.cleaning_data(new_df)

    clean_new_df.insert(loc = 1, column = 'program_time', value = np.linspace(0, new_timedelta_ms, clean_new_df.shape[0]))

    return clean_new_df