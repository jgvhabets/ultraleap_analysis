"""
Importing Handtrack-Files from Ultraleap
version data output 01.09.2022

Most recent version prior to repo-deletion
"""

# Import public packages and fucntions
import os
import numpy as np
import pandas as pd

# import own functions
from import_data import find_paths


def import_string_data(
    file_path: str,
    removeNaNs: bool=True,
):
    """
    Function to convert UltraLeap-data
    with incorrect commas and strings
    in to DataFrame

    Input:
        - file_path (str): directory and
            name of data file
        - removeNaNs: defines if double- and
            nan-rows have to be deleted,
            defaults to True
    
    Returns:
        - df: pd DataFrame with correct data
    """

    try:

        # read in original data
        dat = np.loadtxt(file_path, dtype=str)

        # split keys to list
        keys = dat[0]
        keys = keys.split(',')

        # remove key row from data
        dat = dat[1:]

        list_of_values = []

        for row in np.arange(len(dat)):

            # create value list per row

            # split big string in pieces
            datsplit = dat[row].split(',')

            # take out single values global time and is pinching
            glob_time = datsplit[0]
            
            # take pinching boolean value
            try:
                is_pinch = int(datsplit[-5])

            # if is_pinching is missing (nan) bcs
            # hand was not recorded
            except ValueError:
                
                if datsplit[-5] == 'nan':

                    is_pinch = np.nan

            # remove boolean values from rest
            datsplit.pop(0)
            datsplit.pop(-5)

            # fill new list with floats
            values = []

            for i in np.arange(0, len(datsplit) - 1, 2):

                # create float from two string parts
                try:
                    values.append(
                        float(f'{datsplit[i]}.{datsplit[i + 1]}')
                    )
                
                # add nan if no values are recorded
                except ValueError:
                    
                    if np.logical_or(
                        datsplit[i] == 'nan',
                        datsplit[i + 1] == 'nan'
                    ):
                        values.append(np.nan)

            # insert single values in correct order to keys
            values.insert(0, glob_time)
            values.insert(-4, is_pinch)

            list_of_values.append(values)

        # convert list of lists to DataFrame
        df = pd.DataFrame(data=list_of_values, columns=keys)    

    except (ValueError, AssertionError):
        df = pd.read_csv(file_path)

    if removeNaNs:
            df = remove_double_and_onlyNan_rows(df)
    

    return df



def remove_double_and_onlyNan_rows(
    df
):
    """
    Removes every row which contains only
    NaN values, or which is identical to the
    previous row.
    
    Input:
        - df (DataFrame): dataframe which needs
        to be cleaned.
    
    Output:
        - cleaned_df (DataFrame): df without rows
        which are only-nan, or double
    """
    values = df.values  # use np-array for computational-speed
    # create list to store selection
    to_keep = [False]  # start with 1 bool because of range - 1 in for-loop
    for i in np.arange(1, df.shape[0]):
        # loop over every row
        if np.isnan(list(values[i, 3:])).all():
            # if all values are nan
            to_keep.append(False)
            continue
        if (values[i, 3:] == values[i - 1, 3:]).all():
            # if all values are identical
            to_keep.append(False)

        elif sum(values[i, 3:] == values[i - 1, 3:]) / len(values[i, 3:]) > .8:
            # if xx% of values are identical
            to_keep.append(False)

        
        else:
            # keep row if not all-nan, or all-identical
            to_keep.append(True)
    # create new dataframe with rows-to-keep
    clean_df = df[to_keep].reset_index(drop=True)
    
    return clean_df


def get_data(folder: str,
    sub: str, task, condition, side,
    cam_pos,
):
    """
    explanation of funcction

    Arguments:
        - sub: subject code as string
    
    Returns:
        - df_out
    
    Raises:
        - ValueErrors if campos or side are incorrect

    """
    if side.lower() not in ['left', 'right']:
        raise ValueError('incorrect side variable')
    
    if cam_pos.lower() not in ['vr', 'dt', 'st']:
        raise ValueError('incorrect camera variable')
    
    # find path of defined data
    pathfile = find_paths.find_raw_data_filepath(folder=folder,
        sub=sub, cam_pos=cam_pos, task=task,
        condition=condition, side=side,
    )


    if len(pathfile) == 0: return
    
    assert os.path.exists(pathfile), (
        f'selected path does not exist {pathfile}')
    

    # load selected file
    data = import_string_data(pathfile)

    # clean data
    data = remove_double_and_onlyNan_rows(data)    

    return data



