"""
Importing Handtrack-Files from Ultraleap
version data output 01.09.2022

Most recent version prior to repo-deletion
"""

# Import public packages and fucntions
import numpy as np
import pandas as pd

def task_segments(df,cutoff):

    """
    Function to convert UltraLeap-data
    with incorrect commas and strings
    in to DataFrame

    Input:
        - clean dataframe, cut-off of time 
        differences
    
    Returns:
        - store_dfs: list w/ segmented tasks
    """

    i_start = 0
    store_dfs = []

    for idx,dif in enumerate(np.diff(df['program_time'])):
        if dif < cutoff:
            continue
        else:
            i_end = idx
            new_df = df.iloc[i_start:i_end].reset_index(drop=True)
            store_dfs.append(new_df)

            i_start = i_end+1

    if np.diff(df['program_time'])[-1] < cutoff:
        i_end = df.shape[0]
        new_df = df.iloc[i_start:i_end].reset_index(drop=True)
        store_dfs.append(new_df)

    return store_dfs
