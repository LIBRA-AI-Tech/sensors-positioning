![](images/python_icon.png) ![](images/jupyter_icon.png) ![](images/tf_icon.png) ![](images/pandas_icon.png)

# Bluetooth Indoor Positioning with DNNs

## Overview

🔭 This repository contains 5 different machine learning approaches that predict a tag's location based on received signals from 4 different anchor points. The models estimate the Angles of Arrival (AoA) for each anchor point and then the estimations are combined with the least squares method to produce the final position prediction.

🔗 The dataset on which the machine learning models were trained and evaluated can be found here:
https://zenodo.org/record/6303184

🔬 functions contains the 5 machine learning models and functions that are used in the notebooks

🔬 training is an example notebook for the machine learing models
🔬 training_cnn is an example notebook for the cnn model, since it requires some minor tweaks. 

## Dataset

The dataset consists of 9 different directories. Each directory contains the data of one particular scenario of the 14 x 7m room.

Room Setup                              | Description
-------------                           | -------------
testbench_01                            | No line-of-sight blocking furniture
testbench_01_furniture_low              | One line-of-sight blocking furniture
testbench_01_furniture_mid              | Three line-of-sight blocking furniture
testbench_01_furniture_high             | Six line-of-sight blocking furniture
testbench_01_furniture_low_concrete     | Same as low but with concrete furniture
testbench_01_furniture_mid_concrete     | Same as mid but with concrete furniture
testbench_01_furniture_high_concrete    | Same as high but with concrete furniture
testbench_01_rotated_anchors            | Same as testbench_01 but the anchors have been rotated clockwise 5 degrees
testbench_01_translated_anchors         | Same as testbench_01 but the anchors have been translated 10cm degrees

In each scenario there are 12 different json files - 6 for the data collected from the anchors and 6 for the anchor properties. 
The 6 different files correspond to all the channel-polarization combinations.

Anchor data jsons:
* anchor : number of anchor
* x_anchor, y_anchor, z_anchor : anchor's position
* az_anchor : anchor's horizontal rotation
* el_anchor : anchor's vertical rotation
* reference_power : 

Tag data jsons:
* anchor : number of anchor 
* point : number of point
* x_tag, y_tag, z_tag : point's position
* los : line or no line of sight (1, 0)
* relative power : 
* pdda_input_real : real parts of the IQ values
* pdda_input_image : imaginary parts of the IQ values
* pdda_phi : pdda prediction for phi angle
* pdda_theta : pdda prediction for theta angle
* pdda_out_az : 
* pdda_out_el : 
* true_phi : label for phi angle
* true_theta : label for theta angle

Further technical information about the data can be found in the dataset description at zenodo.

## Functions

### :mag: Data_processing

data_processing.py contains functions that are used during the processing phase of the data

### :mag: Models

models.py contains the 5 different model architectures:
* independentArch : Each anchor estimates its AoA independently
* jointArch : The predictions are computed by jointly exploiting all anchor inputs
* tripletsArch : The predictions are computed by jointly exploiting all possible triplet of anchors inputs
* pairsArch : The predictions are computed by jointly exploiting all possible pair of anchors inputs
* cnnArch : The predictions are computed from an image that contains all the anchor data and uses the BLE channels as image channels.

### :mag: Prediction

predictions.py contains functions that are use for making the position and angles estimations.

### :mag: Visualization

visualization.py :bar_chart: contains functions that are used for visualizing the results.

## General Comments

### Training Points

We picked 140 training points in order to imitate real conditions where the collection of data more than 200 points is not practical. One can train with different set of points:

![](images/training_points.png)

### Models' Complexity

We tuned all the models around 20k parameters so as to compare them on common grounds. Lower complexities can be tried as well:

![](images/complexities.png)