# Bluetooth Indoor Positioning with DNNs

🔭 5 different machine learning approaches were designed and implemented to perform indoor positioning using bluetooth signals collected by 4 different anchor points. A tag transmitts a signal which, upon arrival to the anchor points, is fed to the developed models in order to estimate the angles of arrival to each anchor point. Finally, the angle of arrival estimations are fed to a positioning algorithm that utilizes the least squares method to produce a position estimate.

🔗 The dataset on which the machine learning models were trained and evaluated can be found here:
https://zenodo.org/record/6303184

## Overview

🔬 functions folder contains files with definitions for the 5 machine learning model classes and various functions  used for data processing, model evaluation and visualizations.
🔬 training.ipynb is an example notebook for the machine learing models
🔬 training_cnn.ipynb is an example notebook for the cnn model, since it requires some minor tweaks. 

## Dataset

The data used was generated via ray-tracing simulations in an environment of dimensions 14m x 7m. Multiple setups were examined with varying parameters that affect indoor positioning tasks. The different room setups are presented.

Room Setup                              | Description
-------------                           | -------------
testbench_01                            | No line-of-sight blocking furniture
testbench_01_furniture_low              | One line-of-sight blocking furniture
testbench_01_furniture_mid              | Three line-of-sight blocking furniture
testbench_01_furniture_high             | Six line-of-sight blocking furniture
testbench_01_furniture_low_concrete     | Same as low but with concrete furniture
testbench_01_furniture_mid_concrete     | Same as mid but with concrete furniture
testbench_01_furniture_high_concrete    | Same as high but with concrete furniture
testbench_01_rotated_anchors            | Same as testbench_01 but the anchors have been rotated clockwise by 5 degrees
testbench_01_translated_anchors         | Same as testbench_01 but the anchors have been translated by 10cm

In each scenario there are 12 different json files - 6 for the data collected from the anchors and 6 for the anchor properties. 
The 6 different files correspond to all the channel-polarization combinations.

Anchor data jsons:
* anchor: anchor's index
* x_anchor, y_anchor, z_anchor: anchor point's coordinates
* az_anchor: anchor's horizontal rotation
* el_anchor: anchor's vertical rotation
* reference_power: received signal strength (RSS) reference value in dB

Tag data jsons:
* anchor: anchor's index
* point: point's index
* x_tag, y_tag, z_tag: tag point's coordinates
* los: point is in line of sight of anchor {0=false,1=true}
* relative power: RSS value in dB
* pdda_input_real: in-phase components of the anchor's antennas' measurements
* pdda_input_image: quadrature-phase components of the anchor's antennas' measurements
* pdda_phi: pdda prediction for azimuth angle
* pdda_theta: pdda prediction for elevation angle
* pdda_out_az: pdda's spatial power spectrum for azimuth angle
* pdda_out_el: pdda's spatial power spectrum for elevation angle
* true_phi: actual azimuth angle
* true_theta:  actual azimuth angle

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
