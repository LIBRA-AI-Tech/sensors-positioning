from itertools import combinations

from collections import defaultdict

import tensorflow as tf
from tensorflow import layers, Input

anchors = ['anchor1', 'anchor2', 'anchor3', 'anchor4']
channels = ['37','38','39']

class independentArch():

    def __init__(self, lr=0.005, arch=[[24,24], [0.1,0.1]], batch=False):
        '''
        Input: 4 x 3 x (8 IQ Values, 1 x RSSI, 1 x IQ Reference Values) (for each channel and for each anchor)
        Output: The predicted Azimuthian and Elevation AoAs for the 4 anchors.
        '''

        y,x = [],{}
        for anchor in anchors:
            x[anchor] = {channel: Input((10,)) for channel in channels}
            y.append(self.anchor_model(arch=arch, batch=batch)(x[anchor]))

        y = layers.concatenate(y, name='AoAs')

        self.model = tf.keras.Model(
            inputs=[x],
            outputs=y,
        )

        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss = 'mse', metrics = 'mae')

    def anchor_model(self, arch, batch):
        '''
        Input: 3 x (8 IQ Values, 1 x RSSI, 1 x IQ Reference Values) (for each channel)
        Output: The predicted Azimuthian and Elevation AoAs of the anchor.
        '''

        x,y = {},[]
        for channel in channels:
            x[channel] = Input((10,))
            y.append(self.aoa_mlp(arch=arch, batch = batch)(x[channel]))
            y.append(tf.keras.layers.Lambda(lambda x: x[:,1:2])(x[channel]))

        y = layers.concatenate(y)
        y = self.smol_mlp()(y)

        model = tf.keras.Model(
            inputs=[x.values()],
            outputs=y,
        )

        return model

    def aoa_mlp(self, arch, batch):
        '''
        Input: 8 Normalized IQ Values, RSSI value and IQ Reference value (the magnitude of the IQ value of the first antenna). 
                Those refer to a single anchor and a single BLE channel
        Output: The predicted Azimuthian and Elevation AoAs
        '''

        inp = Input((10,))

        x = inp

        if batch:
            x = layers.BatchNormalization()(x)
        for i in range(len(arch[0])):
            x = layers.Dense(arch[0][i], activation='relu')(x)
            x = layers.Dropout(arch[1][i])(x)
        
        out = layers.Dense(5)(x)
        out = layers.Dropout(0.02)(out)

        model = tf.keras.Model(
            inputs=inp,
            outputs=out,
        )

        return model

    def smol_mlp(self):
        '''
        Input: AoAs predictions from each of the 3 BLE channels along with the RSSI value for each channel.
        Output: The predicted Azimuthian and Elevation AoAs of the anchor.
        '''
        inp = Input((18,))

        x = inp
        x = layers.Dense(36, activation='relu')(x)
        x = layers.Dropout(0.05)(x)
        x = layers.Dense(24, activation='relu')(x)
        x = layers.Dropout(0.05)(x)
        out = layers.Dense(2)(x)
        
        model = tf.keras.Model(
            inputs=inp,
            outputs=out,
        )

        return model

class jointArch():
    def __init__(self, lr=0.001):
        '''
        Input: 3 x 4 x (8 IQ Values, 1 x RSSI, 1 x IQ Reference Values) (for the 4 anchors and the 3 channels)
        Output: The 8 predicted Azimuthian and Elevation AoAs of the four anchors.
        '''

        x,y = defaultdict(dict), defaultdict(dict)
        p = []
        for channel in channels:
            for anchor in anchors:
                x[anchor][channel] = Input((10,))
                y[channel][anchor] = x[anchor][channel]
                p.append(layers.Lambda(lambda x: x[:,1:2])(x[anchor][channel]))

        w = [self.aoa_mlp()(y[channel]) for channel in channels]
        w = layers.concatenate(w)
        p = layers.concatenate(p)
        out = self.smol_mlp()([w,p])
        
        self.model = tf.keras.Model(
            inputs=x,
            outputs=out,
        )

        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss = 'mse', metrics = 'mae')

    def aoa_mlp(self, arch=[[60,40], [0.1,0.1]], batch_norm=False):
        '''
        Input: 4 x (8 IQ Values, 1 x RSSI, 1 x IQ Reference Values) (for each anchor)
        Output: The 8 predicted Azimuthian and Elevation AoAs of the four anchors.
        '''


        inp = {anchor: Input((10,)) for anchor in anchors}
        x = [inp[anchor] for anchor in anchors]

        x = layers.concatenate(x)
        x = layers.BatchNormalization()(x)

        for i in range(len(arch[0])):
            x = layers.Dense(arch[0][i], activation='relu')(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Dropout(arch[1][i])(x)

        out = layers.Dense(12)(x)

        model = tf.keras.Model(
            inputs=inp,
            outputs=out,
        )

        return model   

    def smol_mlp(self):
        '''
        Input: 3 x (8 Predicted AoAs of the 4 anchors) for the 3 channels, 3 x (4 RSSI of the 4 anchors) for the 3 channels
        Output: The 8 predicted Azimuthian and Elevation AoAs of the four anchors.
        '''
        inp1 = Input((36,), name='AoAs')
        inp2 = Input((12), name='Powers ')

        x = layers.concatenate([inp1, inp2])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(8)(x)
        
        model = tf.keras.Model(
            inputs=[inp1, inp2],
            outputs=out,
        )

        return model

class tripletsArch():

    def __init__(self, lr=0.002):
        
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

class pairsArch():

    def __init__(self, lr=0.002):
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

class cnnArch():
    
    def __init__(self, lr = 0.0005, conv_arch=[(12, (4,1), (1,1)), (16, (3,2), (1,1))], dense_arch=[[64,32], [0.1,0.1]], batch_norm=False):

            iq_input = Input((10,4,3), name='iq_image')
            power_input = Input(shape=(12,), name='powers') 

            x = iq_input
            for (num_channels, kernel, strides) in conv_arch: 
                if batch_norm == True:
                    x = layers.BatchNormalization()(x)
                else:
                    x = layers.Dropout(0.05)(x)
                x = layers.Conv2D(filters=num_channels, kernel_size=kernel, strides=strides, padding='valid', activation='relu')(x)

            p = layers.Dense(16)(power_input)
            p = layers.Dropout(0.1)(p)
            x = layers.Flatten()(x)
            x = layers.concatenate([x, p])

            for i in range(len(dense_arch[0])):
                x = layers.Dense(dense_arch[0][i], activation='relu')(x)
                x = layers.Dropout(dense_arch[1][i])(x)

            out = layers.Dense(8)(x)

            self.model = tf.keras.Model(
                inputs=[iq_input, power_input],
                outputs=out,
            )

            self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr), loss = 'mse', metrics = 'mae')