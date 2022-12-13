import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import operator

"""
Block Extraction
"""

def block_extraction(df, cutoff):

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



"""
Fingertapping Function
"""

def FT_amp(
    df, point1, point2
):

    """
        Calculates the euclideian distance between 
        two fingers.
        
        Input:
            - cleaned df (DataFrame), 3D coordinates 
            of the two fingers.
        
        Output:
            - dataframe with the distance between 
            fingers and time.
    """

    df_time = df[['program_time']].copy()

    distances = []

    for i in np.arange(0, df.shape[0]):

        x1 = df.iloc[i][f'{point1}_x']
        y1 = df.iloc[i][f'{point1}_y']
        z1 = df.iloc[i][f'{point1}_z']
        
        x2 = df.iloc[i][f'{point2}_x']
        y2 = df.iloc[i][f'{point2}_y']
        z2 = df.iloc[i][f'{point2}_z']

        pos1 = (x1, y1, z1)
        pos2 = (x2, y2, z2)

        distances.append(distance.euclidean(pos1, pos2))

    df_time.insert(1, 'distance', distances)
    df_time = df_time.reset_index(drop = True)

    return df_time



"""
Opening-closing distance for each invividual finger
"""

def OC_amp(df, finger_ls):

        """
        Calculates the distance between each finger 
        individually and palm.
        
        Input:
            - cleaned df (DataFrame), finger list.
        
        Output:
            - dataframe with finger-palm distances 
            (columns have the finger's name)
            and 'program_time'.
        """
        

        distances = []
        for finger in finger_ls: 
            dist = []

            for idx in np.arange(0, df.shape[0]):

                x = df.iloc[idx][f'{finger}_x']
                y = df.iloc[idx][f'{finger}_y']
                z = df.iloc[idx][f'{finger}_z']

                x_pal = df.iloc[idx]['palm_x']
                y_pal = df.iloc[idx]['palm_y']
                z_pal = df.iloc[idx]['palm_z']

                fing = (x, y, z)
                palm = (x_pal,y_pal,z_pal)

                fing_pal = distance.euclidean(fing, palm)

                dist.append(fing_pal)
                
            distances.append(dist)
            
        
        key_ls = [f'{f}_dist' for f in finger_ls]
    
        # key_ls = ['program_time'] + key_ls

        # df_amp_fing = pd.DataFrame(columns = key_ls)
        # df_amp_fing.rename(columns = {finger_ls: key_ls}, inplace = True)
        df_amp_fing = pd.DataFrame(
            data = np.array(distances).T,
            columns = key_ls,
            )
        
        # if df_amp_fing.shape[1] >= 2: 
        # print(df_amp_fing.index)

        df_amp_fing.insert(0, 'program_time', df['program_time'].values)


        # if len(df_amp_fing.keys()) == 2:
        #     df_amp_fing.rename(columns = {finger_ls[0]:'distance'}, inplace = True)
        #     # df_amp_fing.insert(0, 'program_time', df['program_time'])

        return df_amp_fing



"""
Opening-Closing distance considering only 1 finger
"""
def OC_amp_fing(
    df,
    finger
):
    """
        Calculates the mean distance between 1 finger 
        and palm.
        
        Input:
            - cleaned df (DataFrame), finger name 
            (str - ex: 'middle_tip').
        
        Output:
            - dataframe with the finger-palm 
            distance and 'program_time'.
    """

    df_time = df[['program_time']].copy()

    distances = []

    for i in np.arange(0, df.shape[0]):
            
        # finger coordinates
        xf = df.iloc[i][f'{finger}_x']
        yf = df.iloc[i][f'{finger}_y']
        zf = df.iloc[i][f'{finger}_z']
        

        # Palm coordinates
        xp = df.iloc[i]['palm_x']
        yp = df.iloc[i]['palm_y']
        zp = df.iloc[i]['palm_z']

        fing = (xf, yf, zf)
        pal = (xp, yp, zp)
        
        fing_pal = distance.euclidean(fing, pal)

        distances.append(fing_pal)

    df_time.insert(1, 'distance', distances)
    df_time = df_time.reset_index(drop = True)

    return df_time



"""
Opening-Closing Amplitude - Mean Position of the 4 fingers
"""
def OC_amp_mean(
    df,
    index,
    middle,
    ring,
    pinky
):
    """
        Calculates the mean distance between pinky, 
        ring, mid, index finger and palm.
        
        Input:
            - cleaned df (DataFrame), finger names 
            (str - ex: 'middle_tip').
        
        Output:
            - dataframe with the mean fingers-palm 
            distance and 'program_time'.
    """

    df_time = df[['program_time']].copy()

    distances = []

    for i in np.arange(0, df.shape[0]):
            
        # Index coordinates
        x1 = df.iloc[i][f'{index}_x']
        y1 = df.iloc[i][f'{index}_y']
        z1 = df.iloc[i][f'{index}_z']
        
        # Middle coordinates
        x2 = df.iloc[i][f'{middle}_x']
        y2 = df.iloc[i][f'{middle}_y']
        z2 = df.iloc[i][f'{middle}_z']

        # Ring coordinates
        x3 = df.iloc[i][f'{ring}_x']
        y3 = df.iloc[i][f'{ring}_y']
        z3 = df.iloc[i][f'{ring}_z']

        # Pinky coordinates
        x4 = df.iloc[i][f'{pinky}_x']
        y4 = df.iloc[i][f'{pinky}_y']
        z4 = df.iloc[i][f'{pinky}_z']

        # Palm coordinates
        x5 = df.iloc[i][f'palm_x']
        y5 = df.iloc[i][f'palm_y']
        z5 = df.iloc[i][f'palm_z']

        ind = (x1, y1, z1)
        mid = (x2, y2, z2)
        rin = (x3, y3, z3)
        pin = (x4, y4, z4)
        pal = (x5, y5, z5)
        
        ind_pal = distance.euclidean(ind, pal)
        mid_pal = distance.euclidean(mid, pal)
        rin_pal = distance.euclidean(rin, pal)
        pin_pal = distance.euclidean(pin, pal)

        distances.append((ind_pal+mid_pal+rin_pal+pin_pal)/4)

    df_time.insert(1, 'distance', distances)
    df_time = df_time.reset_index(drop = True)

    return df_time


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
        
        #vert_axis_xyz = (xp, 1, zp)
        #vector_vert_ax = tuple(map(operator.sub, vert_axis_xyz, p))

        cross_vector = np.cross(vector_t, vector_m)
        vert_vector = (0, 1, 0)
        #div = vector_vert_ax/cross
       
        # Normalization of cross_vector and vert_vector
        unit_vect1 = cross_vector / np.linalg.norm(cross_vector)
        unit_vect2 = vert_vector / np.linalg.norm(vert_vector)

        dotprod = np.dot(unit_vect1, unit_vect2)
        # div = unit_vect1/unit_vect2
        # div = vert_vector/cross_vector
    
        pro_sup_angle.append(np.arccos(dotprod)*(180/np.pi))
        # pro_sup_angle.append(np.arccos(div)*(180/np.pi))
        # print(pro_sup_angle)
        #print(pro_sup_angle)
    df_time_ang.insert(1, 'angle', pro_sup_angle)
    df_time_ang = df_time_ang.reset_index(drop = True)
    # print(df_time_ang)
    return df_time_ang



"""
Finding minima and maxima
"""

def find_min_max(
    df_time_amp,
    # distmax,
    # distmin,
    # hgt_min,
    # hgt_max,
    hgt,
    # dist,
    col_name: str,
    prom,
    # prom_min,
    # prom_max,
    # wid,
    ):

    """
        Calculates the minima and maxima of a dataframe.
        
        Input:
            - dataframe with time and distance values
             (from a calculating distance/angle function),
             height_max, height_min, str ('distance'/'angle')
             of the dataframe's column name, prominence.
        
        Output:
            - dictionary with max/min_idx, max/min_values.
    """

    inv_df_time_amp = -np.array(df_time_amp[col_name])
    print(np.min(inv_df_time_amp))

    # Adding a column with inverted distances to df_time_dist dataframe
    if df_time_amp.shape[1] == 2:
        df_time_amp.insert(2,'inv_distance', inv_df_time_amp)

    # Maxima
    max_peaks = find_peaks(
        df_time_amp[col_name],
        # height = np.mean(df_time_amp[col_name]),
        # height = hgt_max,
        height = hgt,
        # distance = dist,
        # height = hgt,
        # distance = distmax,
        prominence = prom,
        # prominence = prom_max,
        # width = wid,
        ) # distance = 15 -> assuming the patient needs 15 s to perform a tap
    max_idx = max_peaks[0]

    max_values = max_peaks[1]['prominences']

    # Minima
    min_peaks = find_peaks(
        inv_df_time_amp,
        # height = np.mean(inv_df_time_amp),
        # height = hgt_min,
        height = -hgt,
        # distance = dist,
        # distance = distmin, 
        prominence = prom,
        # prominence = prom_min 
        # width = wid
        ) # distance = 15 -> assuming the patient needs 15 s to perform a tap
    min_idx = min_peaks[0]
    min_values = min_peaks[1]['prominences']

    dict_ind_values = {'max_idx': max_idx, 'max_values': max_values, 'min_idx': min_idx, 'min_values': min_values}

    return dict_ind_values




"""
Finding slope changes / zeros (?)
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