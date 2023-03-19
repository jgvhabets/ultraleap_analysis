import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import operator
import json

def saveasjson(path, 
    file, 
    data
):
    filename = file.replace('dist', 'feat')
    file_path_name = path + '/' + filename + '.json'
    with open(file_path_name, 'w') as fp:
        json.dump(data, fp)
    return

def calc_fps(
    df 
):
    """
    Calculates the sampling rate of a given data frame 

    Input: 
        - df: pandas Dataframe

    Returns:
        - sampling frequency (float)
    """
    try: 
        time = list(df['program_time']) 

    except KeyError:
        time = list(df['time'])

    dur = time[-1] - time[0]

    return len(df) / dur


def calc_distances(xyz_data, point1, point2):
    """
    Calculates the euclidean distance between two points.
    This function returns a Dataframe with all the distances between point1 and point2.
    
    Input:
        - xyz_data: cleaned df (DataFrame) which contains 3D coordinates of the two points.
        - point1: a single point to calculate the distances from.
        - point2: the second point for calculating the distances.
    
    Output:
        - Dataframe with the distances.
    """

    distances = []
    key = f"{point1}_{point2}"

    for i in np.arange(0, xyz_data.shape[0]):
        dist = dist_2_points(xyz_data, point1, point2, i)
        distances.append(dist)

    return pd.DataFrame({key: distances})


def dist_2_points(xyz_data,
 point1, 
 point2, 
 i
):

    x1 = xyz_data.iloc[i][f'{point1}_x']
    y1 = xyz_data.iloc[i][f'{point1}_y']
    z1 = xyz_data.iloc[i][f'{point1}_z']

    x2 = xyz_data.iloc[i][f'{point2}_x']
    y2 = xyz_data.iloc[i][f'{point2}_y']
    z2 = xyz_data.iloc[i][f'{point2}_z']

    pos1 = (x1, y1, z1)
    pos2 = (x2, y2, z2)

    return distance.euclidean(pos1, pos2)


def calc_prosup_angle(df, 
    thumb = str, 
    middle = str, 
    palm = str
):
    """
    This can be sued to calculate the roll angle of the palm from a pronation/supination movement.

    NEEDS REVISION!

    Parameters:
    df (pandas DataFrame): DataFrame containing the coordinates of (at least) the thumb, middle finger, and palm
    thumb (str): Name of the thumb column in the df DataFrame (if possible, use the coordinates of the the fingertip)
    middle (str): Name of the middlefinger column in the df DataFrame (if possible, use the coordinates of the the fingertip)
    palm (str): Name of the palm column in the df DataFrame

    Returns:
    pandas DataFrame: Roll angles of the palm in degrees over time
    """

    pro_sup_angle = []

    for i in range(df.shape[0]):
        # Thumb coordinates
        xt = df.iloc[i][f'{thumb}_x']
        yt = df.iloc[i][f'{thumb}_y']
        zt = df.iloc[i][f'{thumb}_z']

        # Mid-finger coordinates
        xm = df.iloc[i][f'{middle}_x']
        ym = df.iloc[i][f'{middle}_y']
        zm = df.iloc[i][f'{middle}_z']

        # Palm coordinates
        xp = df.iloc[i][f'{palm}_x']
        yp = df.iloc[i][f'{palm}_y']
        zp = df.iloc[i][f'{palm}_z']

        t = (xt, yt, zt)
        m = (xm, ym, zm)
        p = (xp, yp, zp)

        # Calculate vectors
        vector_t = np.array(t) - np.array(p)
        vector_m = np.array(m) - np.array(p)

        # Calculate cross product and angle between vectors
        cross = np.cross(vector_t, vector_m)
        angle_rad = np.arctan2(np.linalg.norm(cross), np.dot(vector_t, vector_m))

        # Convert angle to degrees
        angle_deg = np.rad2deg(angle_rad)

        # Determine the sign of the angle based on the direction of rotation
        if vector_t[1] > 0:
            angle_deg *= -1

        pro_sup_angle.append(angle_deg)

    # Create dataframe with time and pronation/supination angles
    dist = pd.DataFrame()
    dist['roll_angle'] = pro_sup_angle

    return dist


def find_min_max(
    dist_dataframe, task
):
    '''
    Function to calculate the indexes of the maximum values 
    in a movement trace. Then the index of a local minimum between 
    two maxima is calculated 

    NEEDS REVISION! Check for different moevemt profiles!

    Input:
        - dist_dataframe: dataframe with values for distance betwenn two points over time

    Output:
        - Arrays for the maximum indexes and the minimum indexes seperately 

    '''


    fps = calc_fps(dist_dataframe)

    dist_array = np.array(dist_dataframe.iloc[:,0])

    if task in ['oc', 'ft']:

        peaks_idx_min, _ = find_peaks(
        -dist_array, 
        height= np.mean(-dist_array) - np.std(-dist_array),
        prominence= 0.01,  # prominence of 1 cm
        distance= fps/4 # there cannot more than 4 peaks (minima in this case) per second
        )

        peaks_idx_max = np.array([np.argmax(dist_array[peaks_idx_min[i]:peaks_idx_min[i+1]]) + peaks_idx_min[i] for i in range(len(peaks_idx_min)-1)])
    
    elif task == 'ps':
                
        peaks_idx_max, _ = find_peaks(
        dist_array, 
        height= 0,      
        distance= fps/ 3    # there cannot more than 3 pronation events (maxima in this case) per second
        )

        peaks_idx_min = np.array([np.argmin(dist_array[peaks_idx_max[i]:peaks_idx_max[i+1]]) + peaks_idx_max[i] for i in range(len(peaks_idx_max)-1)])
    
    else:
        print('The task (or task name) you specified does not exist')
        
    return peaks_idx_min, peaks_idx_max