import numpy as np
import pandas as pd


from movement_calc import helpfunctions as hp

# create one dict for block features 
def features_across_block(block, task): 

    block_features = {}

    within_features = features_within_block(block, task)

    if task in ['ft', 'oc']:

        try:
            #number of events
            block_features['num_events'] = within_features['num_events']

            # distance
            block_features['mean_max_amp'] = (np.nanmean(within_features['max_amp']))
            block_features['sd_max_amp'] = (np.nanstd(within_features['max_amp']))
            block_features['coef_var_max_amp'] = (np.nanstd(within_features['max_amp'])/np.nanmean(within_features['max_amp']))
            block_features['dec_max_amp'] = abs((np.nanmean(within_features['max_amp'][:2])) - (np.nanmean(within_features['max_amp'][-2:])))
            block_features['perc_dec_max_amp'] = abs(((np.nanmean(within_features['max_amp'][:2])) - (np.nanmean(within_features['max_amp'][-2:])) / (np.nanmean(within_features['max_amp'][:2])) * 100))

            # speed
            block_features['mean_max_vel'] = (np.nanmean(within_features['max_vel']))
            block_features['sd_max_vel'] = (np.nanstd(within_features['max_vel']))
            block_features['coef_var_max_vel'] = (np.nanstd(within_features['max_vel'])/np.nanmean(within_features['max_vel']))
            block_features['dec_max_vel'] = abs((np.nanmean(within_features['max_vel'][:2])) - (np.nanmean(within_features['max_vel'][-2:])))
            block_features['perc_dec_max_vel'] = abs(((np.nanmean(within_features['max_vel'][:2])) - (np.nanmean(within_features['max_vel'][-2:])) / (np.nanmean(within_features['max_vel'][:2])) * 100))
            block_features['mean_mean_vel'] = (np.nanmean(within_features['mean_vel']))
            block_features['sd_mean_vel'] = (np.nanstd(within_features['mean_vel']))
            block_features['coef_var_mean_vel'] = (np.nanstd(within_features['mean_vel'])/np.nanmean(within_features['mean_vel']))

            # tap_duration
            block_features['mean_tap_dur'] = (np.nanmean(within_features['tap_dur']))
            block_features['sd_tap_dur'] = (np.nanstd(within_features['tap_dur']))
            block_features['coef_var_tap_dur'] = (np.nanstd(within_features['tap_dur'])/np.nanmean(within_features['tap_dur']))
            block_features['dec_tap_dur'] = abs((np.nanmean(within_features['tap_dur'][:2])) - (np.nanmean(within_features['tap_dur'][-2:])))
            block_features['perc_dec_tap_dur'] = abs(((np.nanmean(within_features['tap_dur'][:2])) - (np.nanmean(within_features['tap_dur'][-2:])) / (np.nanmean(within_features['tap_dur'][:2])) * 100))

            # root mean square
            block_features['mean_rms'] = (np.nanmean(within_features['rms']))
            block_features['sd_rms'] = (np.nanstd(within_features['rms']))
            block_features['sum_rms'] = (np.sum(within_features['rms']))

            # normalized root mean square
            block_features['mean_nrms'] = (np.nanmean(within_features['nrms']))
            block_features['sd_nrms'] = (np.nanstd(within_features['nrms']))
            block_features['sum_nrms'] = (np.sum(within_features['nrms']))

        except TypeError:
            features = ['mean_max_amp', 'sd_max_amp', 'coef_var_max_amp', 'dec_max_amp', 'perc_dec_max_amp',
                        'mean_max_vel', 'sd_max_vel', 'coef_var_max_vel', 'dec_max_vel', 'mean_mean_vel', 'sd_mean_vel', 'coef_var_mean_vel', 'perc_dec_max_vel',
                        'mean_tap_dur', 'sd_tap_dur', 'coef_var_tap_dur', 'dec_tap_dur', 'perc_dec_tap_dur',
                        'mean_rms', 'sd_rms', 'sum_rms', 
                        'mean_nrms', 'sd_nrms', 'sum_nrms']

            for feat in features:
                if feat == 'num_events':
                    block_features[feat] = within_features['num_event']
                else:
                    block_features[feat] = 'nan'

    elif task == 'ps':
        try:
            #number of events
            block_features['num_pro_position'] = within_features['num_pro_position']
            block_features['num_sup_position'] = within_features['num_sup_position']
            block_features['num_pro_sup_events'] = within_features['num_pro_sup_events']

            # angle
            block_features['mean_max_pro_ang'] = (np.nanmean(within_features['max_pro_ang']))
            block_features['sd_max_pro_ang'] = (np.nanstd(within_features['max_pro_ang']))
            block_features['coef_var_pro_max_ang'] = (np.nanstd(within_features['max_pro_ang'])/np.nanmean(within_features['max_pro_ang']))
            block_features['perc_dec_max_pro_ang'] = abs(((within_features['max_pro_ang'][0] - within_features['max_pro_ang'][-1]) / within_features['max_pro_ang'][0]) * 100)

            block_features['mean_max_sup_ang'] = (np.nanmean(within_features['max_sup_ang']))
            block_features['sd_max_sup_ang'] = (np.nanstd(within_features['max_sup_ang']))
            block_features['coef_var_sup_max_ang'] = (np.nanstd(within_features['max_sup_ang'])/np.nanmean(within_features['max_sup_ang']))
            block_features['perc_dec_max_sup_ang'] = abs(((within_features['max_sup_ang'][0] - within_features['max_sup_ang'][-1]) / within_features['max_sup_ang'][0]) * 100)

            # movement duration for pro and sup and both together
            block_features['mean_pro_dur'] = (np.nanmean(within_features['pro_dur']))
            block_features['sd_pr_dur'] = (np.nanstd(within_features['pro_dur']))
            block_features['coef_var_pro_dur'] = (np.nanstd(within_features['pro_dur'])/np.nanmean(within_features['pro_dur']))
            block_features['dec_pro_dur'] = abs(within_features['pro_dur'][0] - within_features['pro_dur'][-1])
            block_features['perc_dec_pro_dur'] = abs(((within_features['pro_dur'][0] - within_features['pro_dur'][-1]) / within_features['pro_dur'][0]) * 100)

            block_features['mean_sup_dur'] = (np.nanmean(within_features['sup_dur']))
            block_features['sd_sup_dur'] = (np.nanstd(within_features['sup_dur']))
            block_features['coef_var_sup_dur'] = (np.nanstd(within_features['sup_dur'])/np.nanmean(within_features['sup_dur']))
            block_features['dec_sup_dur'] = abs(within_features['sup_dur'][0] - within_features['sup_dur'][-1])
            block_features['perc_dec_sup_dur'] = abs(((within_features['sup_dur'][0] - within_features['sup_dur'][-1]) / within_features['sup_dur'][0]) * 100)

            block_features['mean_prosup_dur'] = (np.nanmean(within_features['pro_sup_dur']))
            block_features['sd_prosup_dur'] = (np.nanstd(within_features['pro_sup_dur']))
            block_features['coef_var_prosup_dur'] = (np.nanstd(within_features['pro_sup_dur'])/np.nanmean(within_features['pro_sup_dur']))
            block_features['dec_prosup_dur'] = abs(within_features['pro_sup_dur'][0] - within_features['pro_sup_dur'][-1])
            block_features['perc_dec_prosup_dur'] = abs(((within_features['pro_sup_dur'][0] - within_features['pro_sup_dur'][-1]) / within_features['pro_sup_dur'][0]) * 100)
        
        except (TypeError, IndexError):
            features = ['num_pro_sup_events',
                        'mean_max_pro_ang', 'sd_max_pro_ang', 'coef_var_pro_max_ang', 'perc_dec_max_pro_ang',
                        'mean_max_sup_ang', 'sd_max_sup_ang', 'coef_var_sup_max_ang', 'perc_dec_max_sup_ang',  
                        'mean_pro_dur', 'sd_sup_dur', 'coef_var_pro_dur', 'dec_pro_dur', 'perc_dec_pro_dur', 
                        'mean_sup_dur', 'sd_pr_dur', 'coef_var_sup_dur', 'dec_sup_dur', 'perc_dec_sup_dur',
                        'mean_prosup_dur', 'sd_prosup_dur', 'coef_var_prosup_dur', 'dec_prosup_dur', 'perc_dec_prosup_dur']

            for feat in features:
                if feat == 'num_pro_position':
                    block_features[feat] = within_features['num_pro_position']
                if feat == 'num_sup_position':
                    block_features[feat] = within_features['num_sup_position']
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

    feature_names = ['num_pro_position', 'num_sup_position', 'num_pro_sup_events',
                'max_pro_ang', 'max_sup_ang', 
                'pro_vel', 'sup_vel', 'pro_sup_vel',
                'pro_dur', 'sup_dur', 'pro_sup_dur', 
    ]

    features = {feat: [] for feat in feature_names}

    try:
        full_movement_features(
            sup_idx, pro_idx, features, block
        )

    except IndexError:
        features = {feat: ['nan'] for feat in features}

        features['num_pro_position'] = len(pro_idx)
        features['num_sup_position'] = len(sup_idx)

    return features


def full_movement_features(sup_idx, pro_idx, features, block):

    # make sure that the first index is from a pronation position. I.e. from the first to second index
    # should reflect a supination movement, just like is practice
    prosup_idx = ( sorted(sup_idx[1:] + pro_idx) if pro_idx[0] > sup_idx[0] else sorted(sup_idx + pro_idx) )

    features['num_pro_position'] = len(pro_idx)
    features['num_sup_position'] = len(sup_idx)
    features['num_pro_sup_events'] = len(pro_idx) - 1


    for i in np.arange(0, len(pro_idx[:-1])):
        features['max_pro_ang'].append(block.iloc[:, 0][pro_idx[i+1]])

        # duration of one "pronation-supination" movement starting in the pronation position (turning hand until palm faces down again)
        features['pro_sup_dur'].append(
            block.iloc[:, 1][pro_idx[i+1]] - block.iloc[:, 1][pro_idx[i]]) 


    for i in np.arange(0, len(sup_idx[:-1])):
        features['max_sup_ang'].append(block.iloc[:, 0][sup_idx[i+1]])

    # calculate duration for each moevemt and create a list of all values. used for later extract sup and pro dur across whole trace seperately
    pro_sup_dur_all = [ block.iloc[:, 1][prosup_idx[i + 1]] - block.iloc[:, 1][prosup_idx[i]]
    for i in np.arange(0, len(prosup_idx[:-1])) ]

    features['sup_dur'] = pro_sup_dur_all[::2]
    features['pro_dur'] = pro_sup_dur_all[1::2]


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
