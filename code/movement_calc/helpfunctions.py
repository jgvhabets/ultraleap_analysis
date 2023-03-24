import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import operator


"""
Fingertapping Function
"""

def calc_distances(
    xyz_data, point1, point2,
):

    """
    Calculates the euclidean distance between
    two fingers.
    EXPLAIN LIST OPTION

    Input:
        - cleaned df (DataFrame), 3D coordinates
        of the two fingers.

    Output:
        - dataframe with the distance between
        fingers and time.
    """
    distances = []

    for i in np.arange(0, xyz_data.shape[0]):

        if type(point1) == list:

            i_dist = []
            for p1 in point1:

                d = dist_2_points(xyz_data, p1, point2, i)
                i_dist.append(d)

            dist = np.mean(i_dist)

        else:

            dist = dist_2_points(xyz_data, point1, point2, i)

        distances.append(dist)


    return distances


def dist_2_points(xyz_data, point1, point2, i):

    x1 = xyz_data.iloc[i][f'{point1}_x']
    y1 = xyz_data.iloc[i][f'{point1}_y']
    z1 = xyz_data.iloc[i][f'{point1}_z']

    x2 = xyz_data.iloc[i][f'{point2}_x']
    y2 = xyz_data.iloc[i][f'{point2}_y']
    z2 = xyz_data.iloc[i][f'{point2}_z']

    pos1 = (x1, y1, z1)
    pos2 = (x2, y2, z2)

    dist = distance.euclidean(pos1, pos2)

    return dist




"""
Pronation-Supination Function
"""

def PS_ang(df, thumb, middle, palm):

    """
    Calculates the angle between the normal
    vector (normal vector between mid-finger
    and thump) of the palm and the vertical
    axis of the ultraleap.

    Input:
        - df (cleaned DataFrame), thumb, middle
        finger and palm.

    Output:
        - pro_sup_angle: list with PS angles
        in degrees.
   """

    df_time_ang = df[['program_time']].copy()
    pro_sup_angle = []

    for i in np.arange(0, df.shape[0]):
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

        vector_t = tuple(map(operator.sub,t,p))
        vector_m = tuple(map(operator.sub,m,p))

        cross_vector = np.cross(vector_t, vector_m)
        vert_vector = (0, 1, 0)
        #div = vector_vert_ax/cross

        # Normalization of cross_vector and vert_vector
        unit_vect1 = cross_vector / np.linalg.norm(cross_vector)
        unit_vect2 = vert_vector / np.linalg.norm(vert_vector)

        dotprod = np.dot(unit_vect1, unit_vect2)

        pro_sup_angle.append(np.arccos(dotprod)*(180/np.pi))
        # print(pro_sup_angle)
    df_time_ang.insert(1, 'angle', pro_sup_angle)
    df_time_ang = df_time_ang.reset_index(drop = True)
    # print(df_time_ang)
    return df_time_ang



"""
Finding minima and maxima
"""

def find_min_max(distance_array, cam):
    """"
    ...
    """


    # if cam == 'vr':

    peaks_idx_max, _ = find_peaks(
        distance_array,
        height=.03,  # not adding something for now
        prominence = 0.01,  # prominence of 1 cm
        wlen=30,  # prominence versus a window of 20 samples around the peak
        distance=90 / 6,  # peaks are at least 1 / 6 seconds from each other
        )

    peaks_idx_min, _ = find_peaks(
        -distance_array,
        height=np.mean(-distance_array) + 0.5*np.std(-distance_array),
        distance=90 / 5,
        prominence=.001,
        )

    # # #     # in case none or only a few maxima and minima are detected
    # # #     if (len(peaks_idx_max) <= 5) or (len(peaks_idx_min) <= 5):
    # # #         peaks_idx_max, _ = find_peaks(
    # # #             distance_array,
    # # #             prominence=.0005,
    # # #             wlen=90,
    # # #             distance=90 / 5
    # # #             )

    # # #         peaks_idx_min, _ = find_peaks(
    # # #             -distance_array,
    # # #             prominence=.0005,
    # # #             wlen=90,
    # # #             distance=90 / 5
    # # #             )

    # # # elif cam == 'desktop' or cam == 'dt':

    # # #     peaks_idx_max, _ = find_peaks(
    # # #         distance_array,
    # # #         height = np.mean(distance_array)-np.std(distance_array),
    # # #         prominence = 0.02
    # # #         )

    # # #     peaks_idx_min, _ = find_peaks(
    # # #         -distance_array,
    # # #         height = np.mean(-distance_array)-np.std(-distance_array),
    # # #         # (np.mean(-np.array(df2_dist_lh['distance']))+4*np.std(-np.array(df2_dist_lh['distance'])))),
    # # #         prominence = 0.02
    # # #         )

    # # #     # in case none or only a few maxima and minima are detected
    # # #     if (len(peaks_idx_max) <= 5) or (len(peaks_idx_min) <= 5):
    # # #         print('check')
    # # #         peaks_idx_max, _ = find_peaks(
    # # #             distance_array,
    # # #             prominence=.0005,
    # # #             wlen=90,
    # # #             distance=90 / 5
    # # #             )

    # # #         peaks_idx_min, _ = find_peaks(
    # # #             -distance_array,
    # # #             prominence=.0005,
    # # #             wlen=90,
    # # #             distance=90 / 5
    # # #             )

    # # # elif cam == 'st':

    # # #     peaks_idx_max, _ = find_peaks(
    # # #         distance_array,
    # # #         height=.03,  # not adding something for now
    # # #         prominence = 0.01,  # prominence of 1 cm
    # # #         wlen=30,  # prominence versus a window of 20 samples around the peak
    # # #         distance=90 / 6,  # peaks are at least 1 / 6 seconds from each other
    # # #         )

    # # #     peaks_idx_min, _ = find_peaks(
    # # #         -distance_array,
    # # #         height=np.mean(-distance_array) - 2*np.std(-distance_array),
    # # #         distance=90 / 5,
    # # #         prominence=.001, wlen=30,
    # # #         )
    # # #     # in case none or only a few maxima and minima are detected

    # # #     if (len(peaks_idx_max) <= 5) or (len(peaks_idx_min) <= 5):
    # # #         print('check')
    # # #         peaks_idx_max, _ = find_peaks(
    # # #             distance_array,
    # # #             prominence=.0005,
    # # #             wlen=90,
    # # #             distance=90 / 5
    # # #             )

    # # #         peaks_idx_min, _ = find_peaks(
    # # #             -distance_array,
    # # #             prominence=.0005,
    # # #             wlen=90,
    # # #             distance=90 / 5
    # # #             )

    return peaks_idx_max, peaks_idx_min



#### POTENTIALLY MAKE FUNCTION TO EXTRACT MINIMA IN BETWEEN MAXIMA
# GET DISTANCE[max1 : max2]
# GET INDEX OF LOWEST POINT (minimum) np.argmin(distance between maxima)




"""
Finding slope changes / zeros
"""

def find_zeroPasses(signal):

    zeropasses = []
    for i in np.arange(len(signal) - 1):

        prod = signal[i] * signal[i + 1]

        if prod <= 0:

            zeropasses.append(i)

    return zeropasses



"""
Speed Function w/ time slicing for the whole movement
"""

def speed_over_time(df_dist_time, k):

    """
        Calculates the speed of movements.

        Input:
            - dataframe with time and distance values
             (from calc_amp_OC() function).

        Output:
            - speed (list).
    """

    speed = []

    for i in np.arange(0, df_dist_time.shape[0] - k, k):
        dist1 = df_dist_time.iloc[i]['distance']
        dist2 = df_dist_time.iloc[i + k]['distance']

        time1 = df_dist_time.iloc[i]['program_time']
        time2 = df_dist_time.iloc[i + k]['program_time']

        delta_dist = dist2-dist1
        delta_time = time2-time1

        vel = delta_dist/delta_time

        speed.append(abs(vel))

    # dict_speed = {'speed': speed}
    # df_speed = pd.DataFrame(dict_speed)

    return speed


"""
Speed of opening and closing
"""

def speed_OC_time_series(df_time_amp, max_idx, min_idx):

    """
        Calculates the speed of opening and closing over
        time series.

        Input:
            - dataframe with time and dist values
             (from OC_amp() function).

        Output:
            - dict_speedOC: dictionary with speedO and speedC.
    """

    speedO = []
    speedC = []

    for i, (max,min) in enumerate(zip(max_idx[:-1], min_idx)):

        max_time = df_time_amp.iloc[max]['program_time']
        max_amp = df_time_amp.iloc[max]['distance']
        min_time = df_time_amp.iloc[min]['distance']
        min_amp = df_time_amp.iloc[min]['inv_distance']
        max_amp2 = df_time_amp.iloc[max_idx[i+1]]['distance']
        max_time2 = df_time_amp.iloc[max_idx[i+1]]['program_time']

        speed_O_amp = max_amp2-min_amp
        speed_O_time = max_time2-min_time

        vel_O = speed_O_amp/speed_O_time
        speedO.append(vel_O)

        speed_C_amp = min_amp-max_amp
        speed_C_time = min_time-max_time

        vel_C = speed_C_amp/speed_C_time
        speedC.append(vel_C)

    dict_speedOC = {'opening speed': speedO,'closing speed': speedC}

    return  dict_speedOC


"""
Speed per tap
"""


def speed_tap(df_time_amp, min_idx):

    # Speed per tap = Speed per closing
    speed_per_tap = []
    counter = 0
    counter_ls = []


    for i in np.arange(0,len(min_idx[:-1])):

        min_time1 = df_time_amp.iloc[min_idx[i]]['program_time']
        min_amp1 = df_time_amp.iloc[min_idx[i]]['inv_distance']
        min_time2 = df_time_amp.iloc[min_idx[i+1]]['program_time']
        min_amp2 = df_time_amp.iloc[min_idx[i+1]]['inv_distance']

        speed_C_amp = min_amp2-min_amp1
        speed_C_time = min_time2-min_time1

        vel_C = speed_C_amp/speed_C_time
        speed_per_tap.append(vel_C)

        counter += 1
        counter_ls.append(counter)


    plt.scatter(counter_ls, speed_per_tap)
    plt.xlabel('tap')
    plt.ylabel('speed_C')
    plt.title('Speed per Tap')

    print(f'# tap: {len(counter_ls)}')

    return speed_per_tap