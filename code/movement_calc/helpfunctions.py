import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import operator

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
    time = list(df['program_time'])
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
    dist_dataframe
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

    dist_array = np.array(dist_dataframe.iloc[:,0])
    peaks_idx_max, _ = find_peaks(
        dist_array, 
        height = np.mean(dist_array)-np.std(dist_array),
        prominence = 0.02,
    )

    peaks_idx_min = np.array([np.where(dist_dataframe.iloc[:,0] == np.array(dist_dataframe.iloc[peaks_idx_max[i]:peaks_idx_max[i+1]]).min())[0][0]
        for i in np.arange(0, len(peaks_idx_max[:-1]))
    ])
        
    return peaks_idx_max, peaks_idx_min


'''
BELOW NEEDS REVISION AND IS INCLUDED IN ANOTHER PY.FILE 
'''
# def find_zeroPasses(
#     signal
# ):
#     """
#     Finding slope changes / zeros (?)
#     """

#     zeropasses = []
#     for i in np.arange(len(signal) - 1):

#         prod = signal[i] * signal[i + 1]
        
#         if prod <= 0:

#             zeropasses.append(i)
    
#     return zeropasses





# def extract_tap_features(
#     distance_over_time, min_idx,
# ):
#     """
#     Calculates features -> explain more! 
#     """
    
#     # block_feature_dict= {}
#     block_feature_list= [] # list to collect features
#     # tap_feature_list =[]

#     # lists to store features per tap
#     max_vel_pertap = []
#     mean_vel_pertap = []
#     sd_vel_pertap = []
#     max_amp_pertap = []
#     sd_amp_pertap = []
#     rms_pertap = []


#     for i in np.arange(0, len(min_idx[:-1])):

#         #calculating featrues per tap
#         # min_time1 = distance_over_time.iloc[min_idx[i]]['program_time']
#         # min_time2 = distance_over_time.iloc[min_idx[i+1]]['program_time']

#         # tap_duration = min_time2 - min_time1

#         tap_distances = np.array(distance_over_time.iloc[min_idx[i]:min_idx[i+1]]['distance'])
#         tap_durations = np.array(distance_over_time.iloc[min_idx[i]:min_idx[i+1]]['program_time'])
        
#         df_dist = np.diff(tap_distances)
#         df_time = np.diff(tap_durations)
        
#         speed_during_tap = abs(df_dist) / df_time

#         if any([v == np.inf for v in speed_during_tap]) or np.isnan(speed_during_tap).any():
                
#             bad_sel = np.array([v == np.inf for v in speed_during_tap]) + np.isnan(speed_during_tap)

#             speed_during_tap = speed_during_tap[~bad_sel]


#         max_vel_pertap.append(np.nanmax((speed_during_tap)))
#         mean_vel_pertap.append(np.nanmean((speed_during_tap)))
#         sd_vel_pertap.append(np.nanstd((speed_during_tap)))
        
#         max_amp_pertap.append(np.nanmax(tap_distances))
#         sd_amp_pertap.append(np.nanstd(tap_distances))

#         rms_pertap.append(np.sqrt(np.nanmean(tap_distances**2)))
#         rms_normed_pertap = (np.sqrt(np.nanmean(tap_distances**2))) / tap_durations

#         # tap_fefature_list.append([mean_vel_pertap, max_vel_pertap, rms_pertap, sd_vel_pertap, max_amp_pertap, sd_amp_pertap, rms_pertap, rms_normed_pertap])

#     #calculating features per block using the input from each tap
#     max_vel_blk = np.nanmean(max_vel_pertap)
#     mean_vel_blk = np.nanmean(mean_vel_pertap)
#     sd_vl_blk = np.nanstd(max_vel_pertap)
#     mean_maxamp_blk = np.nanmean(max_amp_pertap)
#     sd_maxamp_blk = np.nanstd(max_amp_pertap)
#     mean_rms_blk = np.nanmean(rms_pertap)
#     sd_rms_blk = np.nanstd(rms_pertap)
#     nrms_blk = np.nanmean(rms_normed_pertap)

#     # block_feature_dict = {'max_velocity_block': max_vel_blk,
#     #                     'mean_velocity_block': mean_vel_blk, 
#     #                     'sd_velocity_block' :sd_vl_blk, 
#     #                     'mean_maxamplitude_block': mean_maxamp_blk, 
#     #                     'mean_maxamplitude_block': sd_maxamp_blk, 
#     #                     'mean_rms_block': mean_rms_blk}

#     block_feature_list.append([max_vel_blk, mean_vel_blk, sd_vl_blk, mean_maxamp_blk, sd_maxamp_blk, mean_rms_blk, sd_rms_blk, nrms_blk])

#     return block_feature_list