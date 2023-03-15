# Import public packages and fucntions
import os
import numpy as np
import pandas as pd

# import own functions
from import_data import find_paths

def cleaning_data(
    df
):
    """

    Removes every row which contains only
    NaN values, which is identical to the
    previous row, and where more than 80% of 
    the values are the same.

    This is necessary due to sampling 
    infrequencies or occlusion of hands
    
    Input:
        - df (DataFrame): dataframe which needs
        to be cleaned.
    
    Output:
        - cleaned pandas Dataframe 
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
            
    return df[to_keep].reset_index(drop=True)


def import_string_data(
    file_path: str,
    removeNaNs: bool=True,
):
    """
    Function to convert UltraLeap-data into a pandas Dataframe.
    Can handle input with incorrect commas and strings.

    Input:
        - file_path (str): directory and
            name of data file
        - removeNaNs: defines if double- and
            nan-rows have to be deleted,
            defaults to True
    
    Returns:
        - df: pd DataFrame 
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
            df = cleaning_data(df)

    return df


def get_data(folder: str,
    sub: str, task, condition, side,
    cam_pos,
):
    """
    Imports the data for a given subject, 
    task, condition, camera position and hand-side.

    Arguments:
        - sub (string): subject code as string
        - folder (string): motherfolder where the data
                           can be found (e.g., Patientdata or control)
        - task (string): task name 
        - condition (string): names of the different conditions a 
                              subject performed tasks in 
        - side (string): sie of the hand with which the tasks where perfomed
        - cam_pos (string): cname of the camera position which was used for 
                            recording 
        
        - Note: Each variable can also be a lists of strings to loop over 
    
    Returns:
        - cleaned pandas Dataframe for a specific combination of inputs
    
    Raises:
        - ValueErrors if cam_pos or side are incorrect

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
    data = cleaning_data(data)    

    return data



