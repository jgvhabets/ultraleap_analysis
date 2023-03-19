import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from movement_calc import helpfunctions as hp

# create one dict for block features 
def features_across_block(block, task):

    if task in ['ft', 'oc']:
        tap_features = features_within_block(block, task)

        features = ['num_events',
        'mean_max_amp', 'sd_max_amp', 'coef_var_max_amp', 'slope_max_amp',
        'mean_max_vel', 'sd_max_vel', 'coef_var_max_vel', 'slope_max_vel', 'mean_mean_vel', 'sd_mean_vel', 'coef_var_mean_vel', 'slope_mean_vel',
        'mean_tap_dur', 'sd_tap_dur', 'coef_var_tap_dur', 'slope_tap_dur',
        'mean_rms', 'sd_rms', 'slope_rms', 'sum_rms']

        block_features = {feat: [] for feat in features}

        block_features['num_events'] = tap_features[0]

        # distance
        block_features['mean_max_amp'].append(np.nanmean(tap_features['max_amp']))
        block_features['sd_max_amp'].append(np.nanstd(tap_features['max_amp']))
        block_features['coef_var_max_amp'].append(np.nanstd(tap_features['max_amp'])/np.nanmean(tap_features['max_amp']))
        # block_features['slope_max_amp'].append(np.polyfit(np.arange(len(tap_features['max_amp'])), x['max_amp'], 1)[0])
        
        # speed
        block_features['mean_max_vel'].append(np.nanmean(tap_features['max_vel']))
        block_features['sd_max_vel'].append(np.nanstd(tap_features['max_vel']))
        block_features['coef_var_max_vel'].append(np.nanstd(tap_features['max_vel'])/np.nanmean(tap_features['max_vel']))
        # block_features['slope_max_vel'].append(np.polyfit(np.arange(len(tap_features['max_vel'])), x['max_vel'], 1)[0])
        block_features['mean_mean_vel'].append(np.nanmean(tap_features['mean_vel']))
        block_features['sd_mean_vel'].append(np.nanstd(tap_features['mean_vel']))
        block_features['coef_var_mean_vel'].append(np.nanstd(tap_features['mean_vel'])/np.nanmean(tap_features['mean_vel']))
        # block_features['slope_mean_vel'].append(np.polyfit(np.arange(len(tap_features['mean_vel'])), x['mean_vel'], 1)[0])

        # tap_duration
        block_features['mean_tap_dur'].append(np.nanmean(tap_features['tap_dur']))
        block_features['sd_tap_dur'].append(np.nanstd(tap_features['tap_dur']))
        block_features['coef_var_tap_dur'].append(np.nanstd(tap_features['tap_dur'])/np.nanmean(tap_features['tap_dur']))
        # block_features['slope_tap_dur'].append(np.polyfit(np.arange(len(tap_features['tap_dur'])), x['tap_dur'], 1)[0])

        # root mean square
        block_features['mean_rms'].append(np.nanmean(tap_features['rms']))
        block_features['sd_rms'].append(np.nanstd(tap_features['rms']))
        # block_features['slope_rms'].append(np.polyfit(np.arange(len(tap_features['rms'])), x['rms'], 1)[0])
        block_features['sum_rms'].append(np.sum(tap_features['rms']))

        # normalized root mean square
        block_features['mean_nrms'].append(np.nanmean(tap_features['nrms']))
        block_features['sd_nrms'].append(np.nanstd(tap_features['nrms']))
        # block_features['slope_nrms'].append(np.polyfit(np.arange(len(tap_features['nrms'])), x['nrms'], 1)[0])
        block_features['sum_nrms'].append(np.sum(tap_features['nrms']))


    return block_features

# get features for each 'tap -> needede for above function 
def features_within_block(block, task):

    if task in ['ft', 'oc']:

        features = ['num_events', 'max_amp', 'max_vel', 'mean_vel', 'tap_dur', 'rms', 'nrms']

        tap_features = {feat: [] for feat in features}

        min_idx, _ = hp.find_min_max(block, task)

        tap_features['num_events'] = (len(min_idx))

        for i in np.arange(0, len(min_idx[:-1])):

            try:
                distances = np.array(block.iloc[min_idx[i]:min_idx[i+1]]['index_tip_thumb_tip'])

            except KeyError:
                distances = np.array(block.iloc[min_idx[i]:min_idx[i+1]]['middle_tip_palm'])

            durations = np.array(block.iloc[min_idx[i]:min_idx[i+1]]['time'])

            tap_dur = block.iloc[min_idx[i+1]]['time'] - block.iloc[min_idx[i]]['time']


            df_dist = np.diff(distances)
            df_time = np.diff(durations)
            
            vel = abs(df_dist) / df_time

            tap_features['max_amp'].append(np.nanmax(distances))
            tap_features['max_vel'].append(np.nanmax(vel))
            tap_features['mean_vel'].append(np.nanmean(vel))
            tap_features['tap_dur'].append(tap_dur)
            tap_features['rms'].append(np.sqrt(np.nanmean(distances**2)))
            tap_features['nrms'].append((np.sqrt(np.nanmean(distances**2))) / tap_dur)


    # if task == 'ps':
    #     sup_idx, pro_idx = hp.find_min_max(block, task) # i.e.: sup = min, pro = max
    return tap_features