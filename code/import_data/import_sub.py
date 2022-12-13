"""
Loop over Ultraleap Patient files
"""

import os
import import_data.import_and_convert_data as import_dat

def find_available_subs():

    subs = os.listdir(import_dat.find_onedrive_path()[0])
    subs = [s for s in subs if s[:2].lower() == 'ul']

    return subs


def find_file_path(
    sub, cam_pos, task, condition, side
):

    # find folder with defined data
    subpath = os.path.join(import_dat.find_onedrive_path()[0], f'ul{sub}')
    cam_folder = os.path.join(subpath, cam_pos.lower())
    # only take folder with defined task
    files = os.listdir(cam_folder)
    sel_files = [f for f in files if (
        task.lower() in f and condition.lower() in f)]
    
    sel_folder = os.path.join(cam_folder, sel_files[0])
    data_files = os.listdir(sel_folder)
    # select on side
    if side.lower() == 'left':
        data_files = [f for f in data_files if 'lh' in f]
    elif side.lower() == 'right':
        data_files = [f for f in data_files if 'rh' in f]

    pathfile = (os.path.join(sel_folder, data_files[0]))

    return pathfile

for sub in subs:
    dir = show_sub_files(
        sub[2:],
        'desktop',
        None,
        'm1',
        'oc',
        'rh'
    )

find_file_path(
    sub = '001', cam_pos='vR', task='oc', condition='M1', side='right')
