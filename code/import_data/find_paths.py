import os
import numpy as np
import import_data.import_and_convert_data as import_dat


def find_onedrive_path(
    subfolder = str
):

    '''

    Locates the motherfolder where the data is stored 
    (in the ondrive).

    Input: 
        - subfolder (string): name of the motherfolder where 
                              the data is stored

    Returns:
        - path to folder depending on subfolder input
    '''
        
    path = os.getcwd()
    # Use the individual Users/username for path
    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path)
    
    onedrive_f = [
        f for f in os.listdir(path) if np.logical_and(
            'onedrive' in f.lower(),
            'charit' in f.lower()
        ) 
    ]  
    onedrivepath = os.path.join(path, onedrive_f[0])
    
    if subfolder.lower() == 'data':
        onedrivepath = os.path.join(
            onedrivepath,
            'Ultraleap-hand-tracking',  
            'data', 
        )
    
    elif subfolder.lower() == 'patientdata':
        onedrivepath = os.path.join(
            onedrivepath,
            'Ultraleap-hand-tracking',  
            'data',
            'patientdata'
        )
    
    elif subfolder.lower() == 'control':
        onedrivepath = os.path.join(
            onedrivepath,
            'Ultraleap-hand-tracking',  
            'data',
            'control'
        )
    
    return onedrivepath


def find_available_subs(
    folder = str
    ):

    '''

    Can be used to find the names of the subjects for 
    which there is data recorded and stored.

    Input:
        - folder (string): name of the motherfolder where the 
                           data is stored (e.g. patientdata or control)
    
    Returns:
        - List of the names/ keys of the subjects 

    '''

    subs = os.listdir(find_onedrive_path(folder))

    if folder == 'patientdata':
        subs = [s for s in subs if s[:2].lower() == 'ul']
    elif folder == 'control':
        subs = [s for s in subs if s[:7].lower() == 'control']

    return subs


def find_raw_data_filepath(
    folder: str,
    sub: str, cam_pos: str, task: str,
    condition: str, side: str
):
    """
    Function to find specific path with defined
    files.

    Input:
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

    Reutrns:
        - Filepath for specific datafiles based in input
    """
    assert side in {
        'left',
        'right',
    }, f'given side ({side}) should be "lh" or "rh"'

    if folder == 'control':
        if len(sub) == 3: sub = f'control{sub}'
        subpath = os.path.join(find_onedrive_path(folder), sub)

        cam_folder = os.path.join(subpath, cam_pos.lower())

        # only take folder with defined task
        files = os.listdir(cam_folder)
        sel_files = [f for f in files if (
            task.lower() in f)]


    elif folder == 'patientdata':
        # find folder with defined data
        if len(sub) == 3: sub = f'ul{sub}'
        subpath = os.path.join(find_onedrive_path(folder), sub)

        cam_folder = os.path.join(subpath, cam_pos.lower())

        # only take folder with defined task
        files = os.listdir(cam_folder)
        sel_files = [f for f in files if (
            task.lower() in f and condition.lower() in f)]

    if not sel_files:
        return ''

    sel_folder = os.path.join(cam_folder, sel_files[0])
    data_files = os.listdir(sel_folder)

    # select one side
    if side.lower() == 'left':
        data_files = [f for f in data_files if 'lh' in f]

    elif side.lower() == 'right':
        data_files = [f for f in data_files if 'rh' in f]

    return os.path.join(sel_folder, data_files[0])
