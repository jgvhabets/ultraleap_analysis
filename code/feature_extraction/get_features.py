import numpy as np
import pandas as pd


from movement_calc import helpfunctions as hp

# create one dict for block features 
def features_across_block(block, task):

    block_features = {}

    within_features = features_within_block(block, task)

    try:
        #number of events
        block_features['num_events'] = within_features['num_events']

        # distance
        block_features['mean_max_amp'] = (np.nanmean(within_features['max_amp']))
        block_features['sd_max_amp'] = (np.nanstd(within_features['max_amp']))
        block_features['coef_var_max_amp'] = (np.nanstd(within_features['max_amp'])/np.nanmean(within_features['max_amp']))
        block_features['slope_max_amp'] = (np.polyfit(np.arange(len(within_features['max_amp'])), within_features['max_amp'], 1)[0])
        
        # speed
        block_features['mean_max_vel'] = (np.nanmean(within_features['max_vel']))
        block_features['sd_max_vel'] = (np.nanstd(within_features['max_vel']))
        block_features['coef_var_max_vel'] = (np.nanstd(within_features['max_vel'])/np.nanmean(within_features['max_vel']))
        block_features['slope_max_vel'] = (np.polyfit(np.arange(len(within_features['max_vel'])), within_features['max_vel'], 1)[0])
        block_features['mean_mean_vel'] = (np.nanmean(within_features['mean_vel']))
        block_features['sd_mean_vel'] = (np.nanstd(within_features['mean_vel']))
        block_features['coef_var_mean_vel'] = (np.nanstd(within_features['mean_vel'])/np.nanmean(within_features['mean_vel']))
        block_features['slope_mean_vel'] = (np.polyfit(np.arange(len(within_features['mean_vel'])), within_features['mean_vel'], 1)[0])

        # tap_duration
        block_features['mean_tap_dur'] = (np.nanmean(within_features['tap_dur']))
        block_features['sd_tap_dur'] = (np.nanstd(within_features['tap_dur']))
        block_features['coef_var_tap_dur'] = (np.nanstd(within_features['tap_dur'])/np.nanmean(within_features['tap_dur']))
        block_features['slope_tap_dur'] = (np.polyfit(np.arange(len(within_features['tap_dur'])), within_features['tap_dur'], 1)[0])

        # root mean square
        block_features['mean_rms'] = (np.nanmean(within_features['rms']))
        block_features['sd_rms'] = (np.nanstd(within_features['rms']))
        block_features['sum_rms'] = (np.sum(within_features['rms']))

        # normalized root mean square
        block_features['mean_nrms'] = (np.nanmean(within_features['nrms']))
        block_features['sd_nrms'] = (np.nanstd(within_features['nrms']))
        block_features['sum_nrms'] = (np.sum(within_features['nrms']))

    except TypeError:
        features = ['mean_max_amp', 'sd_max_amp', 'coef_var_max_amp', 'slope_max_amp',
                    'mean_max_vel', 'sd_max_vel', 'coef_var_max_vel', 'slope_max_vel', 'mean_mean_vel', 'sd_mean_vel', 'coef_var_mean_vel', 'slope_mean_vel',
                    'mean_tap_dur', 'sd_tap_dur', 'coef_var_tap_dur', 'slope_tap_dur',
                    'mean_rms', 'sd_rms', 'sum_rms', 
                    'mean_nrms', 'sd_nrms', 'sum_nrms']

        for feat in features:
            if feat == 'num_events':
                block_features[feat] = within_features['num_event']
            else:
                block_features[feat] = 'nan'


    return block_features

# get features for each tap/ movement -> needed for function above
def features_within_block(block, task):

    if task in ['ft', 'oc']:
        within_features = tap_features_within_block(block, task)

    if task == 'ps':
        within_features = pro_sup_features_within_block(block, task)

    return within_features



def pro_sup_features_within_block(block, task):
    sup_idx, pro_idx = hp.find_min_max(block, task)

    # make sure that the first index is from a pronation position. I.e. from the first to second index
    # should reflect a supinatino movement, just like is practice
    prosup_idx = ( sorted(sup_idx[1:] + pro_idx) if pro_idx[0] > sup_idx[0] else sorted(sup_idx + pro_idx) )
    
    feature_names = ['num_pro_events', 'num_sup_events', 'num_pro_sup_events',
                'max_pro_ang', 'max_sup_ang', 
                'pro_vel', 'sup_vel', 'pro_sup_vel',
                'pro_dur', 'sup_dur', 'pro_sup_dur', 
    ]
    features = {feat: [] for feat in feature_names}

    features['num_pro_events'].append(len(pro_idx))
    features['num_sup_events'].append(len(sup_idx))
    features['num_pro_sup_events'].append(len(pro_idx)-1)

    for i in np.arange(0, len(pro_idx[:-1])):
        features['max_pro_ang'].append(block.iloc[:, 0][i])

        # duration of one "pronation-supination" movement starting in the pronatino position (turning hand until palm faces down again)
        features['pro_sup_dur'].append(
            block.iloc[:, 1][i + 1] - block.iloc[:, 0][i]) 


    for i in sup_idx:
        features['max_sup_ang'].append(block.iloc[:, 0][i])


    for i in np.arange(0, len(prosup_idx[:-1])):
            # duration of movement from pronation to supination all in one list (from each idx to another). 
            # for further extraction of each movement seperately
        pro_sup_dur_all = [ block.iloc[: ,1][i + 1] - block.iloc[:, 1][i] ]

    features['sup_dur'].append(pro_sup_dur_all[::2])
    features['pro_dur'].append(pro_sup_dur_all[1::2])

    return features



def tap_features_within_block(block, task):
    min_idx, _ = hp.find_min_max(block, task)

    feature_names = ['num_events', 'max_amp', 'max_vel', 'mean_vel', 'tap_dur', 'rms', 'nrms']

    features = {feat: [] for feat in feature_names}

    if len(min_idx) <= 1:
        features = {feat: ['nan'] for feat in features}

    features['num_events'] = (len(min_idx))

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

        features['max_amp'].append(np.nanmax(distances))
        features['max_vel'].append(np.nanmax(vel))
        features['mean_vel'].append(np.nanmean(vel))
        features['tap_dur'].append(tap_dur)
        features['rms'].append(np.sqrt(np.nanmean(distances**2)))
        features['nrms'].append((np.sqrt(np.nanmean(distances**2))) / tap_dur)

    return features
