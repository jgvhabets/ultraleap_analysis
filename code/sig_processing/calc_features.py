
import numpy as np
import movement_calc.helpfunctions as hp
import matplotlib.pyplot as plt
import os
import json

def get_repo_path_in_notebook():
    """
    Finds path of repo from Notebook.
    Start running this once to correctly find
    other modules/functions
    """
    path = os.getcwd()
    repo_name = 'ultraleap_analysis'

    while path[-len(repo_name):] != 'ultraleap_analysis':

        path = os.path.dirname(path)

    return path

repo_path = get_repo_path_in_notebook()


def calculate_feat(dist, block, sub, cond, cam, task, side, to_plot = False, to_save_plot = False, to_save = False):
    
    """
    
    """

    if task == 'ft':
        dist_col = 'dist'
    elif task == 'oc':
        dist_col = 'mean'

    # get minima and maxima
    idx_max, idx_min = hp.find_min_max(np.array(dist[dist_col]), cam=cam)

    # get time & dist lists per tap and tap duration
    ls_time, ls_dist, tap_duration = tap_times(dist, dist_col, idx_min)
    spe_over_taps = speed_over_time_tap(ls_time, ls_dist)

    # extract features per tap
    ft_dict_tap, ft_name = get_feat_tap(ls_dist, spe_over_taps, tap_duration)
    ft_dict_block = get_feat_block(ft_dict_tap)

    if to_save:
        # save features per tap in json files
        ft_tap_path = os.path.join(repo_path, 'features', sub, task, cond, 'features_tap')
        if not os.path.exists(ft_tap_path):
            os.makedirs(ft_tap_path)
        
        ft_dict_open = open(os.path.join(ft_tap_path, f'{block}_{sub}_{cond}_{cam}_{task}_{side}.json'), 'w')
        json.dump(ft_dict_tap, ft_dict_open)
        ft_dict_open.close()

        # save features per block in json files
        ft_block_path = os.path.join(repo_path, 'features', sub, task, cond,'features_block')
        if not os.path.exists(ft_block_path):
            os.makedirs(ft_block_path)

        ft_dict_block_open = open(os.path.join(ft_block_path, f'{block}_{sub}_{cond}_{cam}_{task}_{side}.json'), 'w')
        json.dump(ft_dict_block, ft_dict_block_open)
        ft_dict_open.close()

    # plot max_min
    if to_plot:
        plot_max_min(dist, dist_col, block, sub, cond, cam, task, side, idx_max, idx_min, to_save = to_save_plot)

    return

def plot_max_min(dist, dist_col, block, sub, cond, cam, task, side, idx_max, idx_min, to_save = False):

    """
    Function thats plot the block with the
    minima & maxima calculated with the 
    find_min_max function.

    Input:
        - dist (Dataframe) with the euclidean distances,
        maxima indexes, minima indexes, camera position,
        distance column (e.g. for  oc: 'mean', for ft: 'dist') 
        
    Output: 
        - figures with minima and maxima.
    """

    # if task == 'ft':
    #     dist_col = 'dist'
    # elif task == 'oc':
    #     dist_col = 'mean'

    x = np.linspace(0,1,len(dist))

    fig = plt.figure(figsize=(8,6))
    plt.plot(x, dist[dist_col], color='grey')
    plt.plot(x[idx_max],
                np.array(dist[dist_col])[idx_max], 
                "o", label="max", color='blue')
    plt.plot(x[idx_min],
                np.array(dist[dist_col])[idx_min], 
                "o", label="max", color='red')
   
    if to_save:
        fig_path = os.path.join(repo_path,
                                'figures',
                                 'distances_min_max',
                                 f'{cam}_max_min'
                                )
        if not os.path.exists(fig_path): 
            os.makedirs(fig_path)

        fig.savefig(os.path.join(fig_path, 
                                    f'{block}_{sub}_{cond}_{cam}_{task}_{side}'),
                                    dpi = 300, facecolor = 'w',
                                    )
    plt.close()
    return


def tap_times(block_df, dist_col, min_idx):
    
    """ Function that gives all time and
        dist values in between two minima
        -> values per tap

        Input:
            - block dataframe, column name of the dist_time df,
            min indices - numpy.ndarray

        Output:
            - list of lists for each tap containing tap_start_time, 
            tap_end_time, tap_duration
    """
    
    ls_time = []
    ls_dist = []
    ls_tap_duration = []
    # times = []

    for idx, i_min in enumerate(min_idx[:-1]):

        t_start = block_df.iloc[i_min]['time']
        t_end = block_df.iloc[min_idx[idx+1]]['time']
        tap_duration = t_end-t_start

        df = block_df[np.logical_and(block_df['time']>=t_start, block_df['time']<=t_end)]
        # df = block_df[block_df.time.between(t_start, t_end)]
      
        ls_time.append(df['time'].tolist())
        ls_dist.append(df[dist_col].tolist())
        # times.append([t_start, t_end])

        ls_tap_duration.append(tap_duration)
    
    return ls_time, ls_dist, ls_tap_duration


def speed_over_time_tap(ls_time, ls_dist):

    speed = []
    
    for j in range(0, len(ls_dist)-1):
        dif_dist = abs(np.diff(ls_dist[j]))
        dif_time = np.diff(ls_time[j])
        speed_single = dif_dist/dif_time
        # nans = np.nonzero(np.isnan(speed_single))[0]
        # infs = np.nonzero(np.isinf(speed_single))[0]
        # if nans.size:
        #     print(f"{dif_time[nans] = }")
        #     speed_single = speed_single[~nans]
        # if infs.size:
        #     print(f"{dif_time[infs] = }")
        #     speed_single = speed_single[~infs]
        speed.append(speed_single)   

        # if any([v == np.inf for v in speed]) or np.isnan(speed).any():
        #     bad_sel = np.array([v == np.inf for v in speed]) + np.isnan(speed)
        #     speed = speed[~bad_sel]
     
    return speed


def calc_feat(ls_feat, task, feat):

    if feat == 'rms':
        feat = np.sqrt(np.mean(ls_feat)**2)
    elif feat == 'sd':
        feat = np.nanstd(ls_feat)
    elif feat == 'slope':
        # feat, intercept = np.polyfit(
        #     np.arange(len(ls_feat)), ls_feat, 1)
        feat = np.polyfit(
            np.arange(len(ls_feat)), ls_feat, 1)
    elif feat == 'max':
        feat = np.nanmax(ls_feat)
    elif feat == 'mean':
        feat = np.nanmean(ls_feat)
    elif feat == 'coef_var':
        feat = np.nanstd(ls_feat)/np.nanmean(ls_feat)

    return feat


def get_feat_tap(ls_dist, ls_vel, ls_tap_duration):
    
    total_ft_dict = {}
    ft_names = [
        'num_events',
        'max_dist', 
        'max_vel', 'mean_vel',
        'deltat',
        'rms'
        ]

    for ft in ft_names:
        total_ft_dict[ft] = []

    # num_events
    total_ft_dict['num_events'].append(len(ls_dist))

    for d, s, t in zip(ls_dist, ls_vel, ls_tap_duration):

        # distance
        total_ft_dict['max_dist'].append(np.nanmax(d))
        
        # speed
        total_ft_dict['max_vel'].append(np.nanmax(s))
        total_ft_dict['mean_vel'].append(np.nanmean(s))

        # tap_duration
        total_ft_dict['deltat'].append(t)

        # root mean square
        total_ft_dict['rms'].append(np.sqrt(np.mean(d)**2))

    return total_ft_dict, ft_names


def get_feat_block(feat_dict):

    feat_names = ['num_events', 
    'mean_max_dist', 'sd_max_dist', 'coef_var_max_dist', 
    # 'slope_max_dist',
    'mean_max_vel', 'sd_max_vel', 'coef_var_max_vel', 
    # 'slope_max_vel', 
    'mean_mean_vel', 'sd_mean_vel', 'coef_var_mean_vel', 
    # 'slope_mean_vel', 
    'mean_deltat', 'sd_deltat', 'coef_var_deltat', 
    # 'slope_deltat',
    'mean_rms', 'sd_rms', 
    # 'slope_rms', 
    'sum_rms']

    dict_feat_block = {}

    for feat in feat_names:
        dict_feat_block[feat] = []

    dict_feat_block['num_events'] = feat_dict['num_events']

    # distance
    dict_feat_block['mean_max_dist'].append(np.nanmean(feat_dict['max_dist']))
    dict_feat_block['sd_max_dist'].append(np.nanstd(feat_dict['max_dist']))
    dict_feat_block['coef_var_max_dist'].append(np.nanstd(feat_dict['max_dist'])/np.nanmean(feat_dict['max_dist']))
    # print(len(np.arange(len(feat_dict['max_dist']))), len(feat_dict['max_dist']))
    # print(np.polyfit(np.arange(len(feat_dict['max_dist'])), feat_dict['max_dist'], 1)[0])
    # dict_feat_block['slope_max_dist'].append(np.polyfit(np.arange(len(feat_dict['max_dist'])), feat_dict['max_dist'], 1)[0])
    
    # speed
    dict_feat_block['mean_max_vel'].append(np.nanmean(feat_dict['max_vel']))
    dict_feat_block['sd_max_vel'].append(np.nanstd(feat_dict['max_vel']))
    dict_feat_block['coef_var_max_vel'].append(np.nanstd(feat_dict['max_vel'])/np.nanmean(feat_dict['max_vel']))
    # dict_feat_block['slope_max_vel'].append(np.polyfit(np.arange(len(feat_dict['max_vel'])), feat_dict['max_vel'], 1)[0])
    dict_feat_block['mean_mean_vel'].append(np.nanmean(feat_dict['mean_vel']))
    dict_feat_block['sd_mean_vel'].append(np.nanstd(feat_dict['mean_vel']))
    dict_feat_block['coef_var_mean_vel'].append(np.nanstd(feat_dict['mean_vel'])/np.nanmean(feat_dict['mean_vel']))
    # dict_feat_block['slope_mean_vel'].append(np.polyfit(np.arange(len(feat_dict['mean_vel'])), feat_dict['mean_vel'], 1)[0])

    # tap_duration
    dict_feat_block['mean_deltat'].append(np.nanmean(feat_dict['deltat']))
    dict_feat_block['sd_deltat'].append(np.nanstd(feat_dict['deltat']))
    dict_feat_block['coef_var_deltat'].append(np.nanstd(feat_dict['deltat'])/np.nanmean(feat_dict['deltat']))
    # dict_feat_block['slope_deltat'].append(np.polyfit(np.arange(len(feat_dict['deltat'])), feat_dict['deltat'], 1)[0])

    # root mean square
    dict_feat_block['mean_rms'].append(np.nanmean(feat_dict['rms']))
    dict_feat_block['sd_rms'].append(np.nanstd(feat_dict['rms']))
    # dict_feat_block['slope_rms'].append(np.polyfit(np.arange(len(feat_dict['rms'])), feat_dict['rms'], 1)[0])
    dict_feat_block['sum_rms'].append(np.sum(feat_dict['rms']))

    return dict_feat_block