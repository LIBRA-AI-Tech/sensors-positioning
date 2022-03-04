from itertools import combinations

from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import layers, Input

anchors = ['anchor1', 'anchor2', 'anchor3', 'anchor4']
channels = ['37','38','39']

class triplets_arch():

    def __init__(self, lr):
        
        combs = list(combinations(anchors,3))
        x,p,y = defaultdict(dict), defaultdict(dict), defaultdict(list)
        for channel in channels:
            for anchor in anchors:
                x[anchor][channel] = Input((10,))
                p[anchor][channel] = layers.Lambda(lambda x: x[:,1:2])(x[anchor][channel])

        w = []
        for i,comb in enumerate(combs):
            for channel in channels:
                y[i].append(self.aoa_mlp(comb)([x[anchor][channel] for anchor in comb]))
            y[i] = layers.concatenate(y[i])
            powers = layers.concatenate([p[anchor][channel] for anchor in comb for channel in channels])
            y[i] = self.smol_mlp()([y[i], powers])
            w.append(y[i])
            w.append(powers)
        w = layers.concatenate(w)
        
        w = layers.Dropout(0.1)(w)
        w = layers.Dense(32, activation='relu')(w)
        w = layers.Dropout(0.1)(w)
        w = layers.Dense(16, activation='relu')(w)
        out = layers.Dense(8)(w)

        self.model = tf.keras.Model(
            inputs=x,
            outputs=out,
        )

        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss = 'mse', metrics = 'mae')
    
    def aoa_mlp(self, anchors, arch=[[24], [0.1]]):
        '''
        Input: 3 x (8 IQ Values, 1 x RSSI, 1 x IQ Reference Values) (for three anchors)
        Output: A latence space 9 x 1
        '''

        inp = {anchor: Input((10,)) for anchor in anchors}
        x = [inp[anchor] for anchor in anchors]

        x = layers.concatenate(x)
        x = layers.BatchNormalization()(x)
        
        for i in range(len(arch[0])):
            x = layers.Dense(arch[0][i], activation='relu')(x)
            x = layers.Dropout(arch[1][i])(x)

        out = layers.Dense(9)(x)

        model = tf.keras.Model(
            inputs=inp,
            outputs=out,
        )

        return model

    def smol_mlp(self):

        '''
        Input: The outputs of the aoa_mlps concatenated
        Output: A latence space 12x1 
        '''

        inp1 = Input((27,), name='AoAs')
        inp2 = Input((9), name='Powers ')

        x = layers.concatenate([inp1, inp2])

        out = layers.Dense(27)(x)
        out = layers.Dropout(0.15)(out)
        out = layers.Dense(12)(out)
        
        model = tf.keras.Model(
            inputs=[inp1, inp2],
            outputs=out,
        )

        return model

class pairs_arch():

    def __init__(self, lr):
        '''
        Input: 3 x 4 x (8 IQ Values, 1 x RSSI, 1 x IQ Reference Values) (for the 4 anchors and the 3 channels)
        Output: The 8 predicted Azimuthian and Elevation AoAs of the four anchors.
        '''
        combs = list(combinations(anchors,2))
        
        x,p,y = defaultdict(dict), defaultdict(dict), defaultdict(list)
        for channel in channels:
            for anchor in anchors:
                x[anchor][channel] = Input((10,))
                p[anchor][channel] = layers.Lambda(lambda x: x[:,1:2])(x[anchor][channel])

        w = []
        for i,comb in enumerate(combs):
            for channel in channels:
                y[i].append(self.aoa_mlp(comb)([x[anchor][channel] for anchor in comb]))
            y[i] = layers.concatenate(y[i])
            powers = layers.concatenate([p[anchor][channel] for anchor in comb for channel in channels])
            y[i] = self.smol_mlp()([y[i], powers])
            w.append(y[i])
            w.append(powers)
        w = layers.concatenate(w)

        w = layers.Dense(64, activation='relu')(w)
        w = layers.Dropout(0.1)(w)
        w = layers.Dense(32, activation='relu')(w)
        w = layers.Dropout(0.1)(w)
        out = layers.Dense(8)(w)

        self.model = tf.keras.Model(
            inputs=x,
            outputs=out,
        )

        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss = 'mse', metrics = 'mae')

    def aoa_mlp(self, anchors, arch=[[12], [0.1]]):
        '''
        Input: 2 x (8 IQ Values, 1 x RSSI, 1 x IQ Reference Values) (for two anchors)
        Output: The 4 predicted Azimuthian and Elevation AoAs of the two anchors.
        '''

        inp = {anchor: Input((10,)) for anchor in anchors}
        x = [inp[anchor] for anchor in anchors]

        x = layers.concatenate(x)
        x = layers.BatchNormalization()(x)
        
        for i in range(len(arch[0])):
            x = layers.Dense(arch[0][i], activation='relu')(x)
            x = layers.Dropout(arch[1][i])(x)

        out = layers.Dense(7)(x)
        out = layers.Dropout(arch[1][i])(out)

        model = tf.keras.Model(
            inputs=inp,
            outputs=out,
        )

        return model

    def smol_mlp(self):
        '''
        Input: 3 x (4 Predicted AoAs of the 2 anchors) for the 3 channels, 3 x (2 x RSSI of the 2 anchors) for the 3 channels
        Output: The 4 predicted Azimuthian and Elevation AoAs of the two anchors.
        '''
        inp1 = Input((21,), name='AoAs')
        inp2 = Input((6), name='Powers ')

        x = layers.concatenate([inp1, inp2])
        x = layers.Dense(12, activation='relu')(x)

        out = layers.Dense(12)(x)
        x = layers.Dropout(0.05)(x)
        
        model = tf.keras.Model(
            inputs=[inp1, inp2],
            outputs=out,
        )

        return model