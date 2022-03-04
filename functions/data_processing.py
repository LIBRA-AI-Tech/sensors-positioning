import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn.preprocessing import StandardScaler

import random
import copy

anchors = ['anchor1', 'anchor2', 'anchor3', 'anchor4']
channels = ['37','38','39']
polarities = ['V','H']

def iq_processing(data):
    
    """
    Input: Data
    Output: Processed Data

    Processing: Power Scaling, IQ shifting
    """

    cols_real = ['pdda_input_real_{}'.format(x+1) for x in range(5)]
    cols_imag = ['pdda_input_imag_{}'.format(x+1) for x in range(5)]

    iq_values = pd.DataFrame(data['pdda_input_real'].tolist(), columns=cols_real, index=data.index)
    iq_values[cols_imag] = pd.DataFrame(data['pdda_input_imag'].tolist(), columns=cols_imag, index=data.index)
    
    phase = pd.DataFrame(np.arctan2(iq_values['pdda_input_imag_1'],iq_values['pdda_input_real_1']), columns=['phase_1'])
    
    cos = np.cos(phase).values.ravel()
    sin = np.sin(phase).values.ravel()
    
    out = data.copy()
    iq_ref = np.abs(iq_values[f'pdda_input_real_1']*cos + iq_values[f'pdda_input_imag_1']*sin)
    for i in range(1,6):
        out[f'pdda_input_real_{i}'] = (iq_values[f'pdda_input_real_{i}']*cos + iq_values[f'pdda_input_imag_{i}']*sin)
        out[f'pdda_input_imag_{i}'] = (-iq_values[f'pdda_input_real_{i}']*sin + iq_values[f'pdda_input_imag_{i}']*cos)
        iq_ref +=  iq_values[f'pdda_input_real_{i}']**2 + iq_values[f'pdda_input_imag_{i}']**2

    power_norm =  StandardScaler().fit_transform((out['reference_power'] + out['relative_power']).values.reshape(-1,1))/10
    
    out.insert(22, 'power', power_norm)
    out.insert(21, 'iq_ref', iq_ref)
    out.drop(columns=['pdda_input_imag_1', 'pdda_input_real', 'pdda_input_imag'], inplace=True)

    return out.iloc[:,-10:]

def create_set(data, rooms, points):

    """
    Input: Data and points for set that we want
    Output: x and y for set that we want
    """
    x = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    y = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for room in rooms:
        for anchor in anchors:
            for channel in channels:
                util_data = {polarity: points[['point']].merge(data[room][anchor][channel][polarity], on='point') for polarity in polarities}
                h,v = util_data['H'], util_data['V']
                m = h.where(h['relative_power']+h['reference_power'] > v['reference_power']+v['relative_power'], v)
                x[room][anchor][channel] = iq_processing(m)
                y[room][anchor][channel] = util_data['H'][['true_phi', 'true_theta']]
    
    return x, y

def create_iq_images(data, augm = False, anchors = None, channels = None):
    
    '''
    Preprocess input for CNN model
    '''

    chanls = []
    powers = []

    for channel in channels:    
        iqs = []
        for anchor in anchors:
            dt = data[anchor][channel] if augm else iq_processing(data[anchor][channel])
            powers.append(dt['power'])
            iqs.append(dt)
        chanls.append(pd.concat(iqs, axis=1).values.reshape((-1, 4, 10)))

    iq_images = np.concatenate(chanls, axis=1).reshape((-1, 3, 4, 10)).transpose(0,3,2,1)
    powers = pd.concat(powers, axis=1)
        
    return iq_images, powers

def create_set_cnn(data, points_set, rooms = None, concrete_rooms = None, other_scenarios = None, anchors = None, channels = None):

    """
    Input: Data and points for set that we want
    Output: x -> (IQ Image (10x4x3 : IQs + RSSI x anchors x channels)), y
    """

    tmp = defaultdict(lambda: defaultdict(dict))
    x = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    y = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for room in rooms + concrete_rooms + other_scenarios:
        for anchor in anchors:
            for channel in channels:
                util_data = {polarity: points_set[['point']].merge(data[room][anchor][channel][polarity], on='point') for polarity in polarities}
                h,v = util_data['H'], util_data['V']
                tmp[room][anchor][channel] = h.where(h['relative_power']+h['reference_power'] > v['reference_power']+v['relative_power'], v)
            y[room][anchor] = util_data['H'][['true_phi', 'true_theta']]
        x[room]['iq_image'], x[room]['powers'] = create_iq_images(tmp[room], anchors = anchors, channels = channels)
    
    return x,y
  
def decreasing_signal(df, rooms, scale_util=5, mode='amplitude'):
    """
    Input: Training dictionary
    Output: Augmented training dictionary
    """

    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for room in rooms:
        for anchor in anchors:
            for channel in channels:
                out[room][anchor][channel] = copy.deepcopy(df[room][anchor][channel])

    for room in rooms:
        for channel in channels:
            for index in range(df[room]['anchor1'][channel].shape[0]):
                
                scale = np.random.uniform(1.1, scale_util)
                scale_phase = np.random.choice([-1,1]) * scale

                num_of_mutes = np.random.choice([1,2,3], p = [0.7,0.2,0.1])
                anchs = set(range(1,5))
                anch = []

                for i in range(num_of_mutes):
                    el = random.sample(anchs, 1)[0]
                    anch.append(el)
                    anchs.remove(el)

                anchs = [f'anchor{i}' for i in anch]

                for anchor in anchs:
                    util = df[room][anchor][channel].iloc[index]
                    polar = {}
                    
                    if mode == 'amplitude':
                        out[room][anchor][channel]['pdda_input_real_1'].iloc[index] = util['pdda_input_real_1'] / scale
                        out[room][anchor][channel]['power'].iloc[index] = util['power'] / scale

                        for i in range(2,6):
                            polar[f'amplitude_{i}'] = np.sqrt(util[f'pdda_input_real_{i}']**2+util[f'pdda_input_imag_{i}']**2) / scale
                            polar[f'phase_{i}'] = np.arctan2(util[f'pdda_input_imag_{i}'],util[f'pdda_input_real_{i}'])
                    
                    if mode == 'phase':
                        out[room][anchor][channel]['pdda_input_real_1'].iloc[index] = util['pdda_input_real_1']
                        out[room][anchor][channel]['power'].iloc[index] = util['power']

                        for i in range(2,6):
                            polar[f'amplitude_{i}'] = np.sqrt(util[f'pdda_input_real_{i}']**2+util[f'pdda_input_imag_{i}']**2)
                            polar[f'phase_{i}'] = np.arctan2(util[f'pdda_input_imag_{i}'],util[f'pdda_input_real_{i}']) + scale_phase
                    
                    if mode == 'both':
                        out[room][anchor][channel]['pdda_input_real_1'].iloc[index] = util['pdda_input_real_1'] / scale
                        out[room][anchor][channel]['power'].iloc[index] = util['power'] / scale

                        for i in range(2,6):
                            polar[f'amplitude_{i}'] = np.sqrt(util[f'pdda_input_real_{i}']**2+util[f'pdda_input_imag_{i}']**2) / scale
                            polar[f'phase_{i}'] = np.arctan2(util[f'pdda_input_imag_{i}'],util[f'pdda_input_real_{i}']) + scale_phase
                    
                    for i in range(2,6):
                        out[room][anchor][channel][f'pdda_input_real_{i}'].iloc[index] = polar[f'amplitude_{i}'] * np.cos(polar[f'phase_{i}'])
                        out[room][anchor][channel][f'pdda_input_imag_{i}'].iloc[index] = polar[f'amplitude_{i}'] * np.sin(polar[f'phase_{i}'])                
    
    return out
