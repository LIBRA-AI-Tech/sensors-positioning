import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn.metrics import mean_absolute_error

anchors = ['anchor1', 'anchor2', 'anchor3', 'anchor4']
channels = ['37','38','39']
polarities = ['V','H']

def lsq(angle_preds, anch_info, iter=None, theta=False, alt=False):

    """
    Given the AoA predictions for each anchor applies least squares to estimate position.
    This function works for (phi, theta) predicions for multiple points.
    Input: Dictionary with entry for each anchor containing a Nx2 array with the phi and theta AoA predictions for N points
    Output: The estimated position for each of the N points

    Note: Theta can be skipped and the prediction will be in ty xy-plane.
    """

    As, Bs = [], []
    for anchor in angle_preds:
        a, b = lsq_aux(angle_preds[anchor], anch_info[anch_info['anchor'] == int(anchor[-1])], theta, alt)
        As.append(a)
        Bs.append(b)
        
    A = np.concatenate(As, axis=1)
    B = np.concatenate(Bs) if alt else np.concatenate(Bs, axis=1)
    preds = [np.linalg.lstsq(a,b,rcond=None)[0] for a,b in zip(A,B)]

    if iter:
        for i in range(iter):
            W = compute_weights(np.array(preds),anch_info,theta) 
            preds = []
            for a,b,w in zip(A,B,W):
                wd = np.zeros((len(a),len(a)))
                for i in range(len(a)):
                    wd[i,i] = w[i//2]
                new_a = (a.T)@wd@a
                new_b = (a.T)@wd@b
                preds.append(np.linalg.lstsq(new_a,new_b,rcond=None)[0])

    return preds

def lsq_aux(angle_preds, anchor_info, theta = False, alt=False): 

    """
    Calculates the matrices A,B which are used for least squares position calculation
    Input: Nx2 array containing phi and theta AoA predictions for a specific anchor for N points
    Output: Nx2x3 array containing the equation coefficients for the intersecting planes for each point

    Note: Theta can be skipped and the prediction will be in ty xy-plane.
    """

    phi = angle_preds[:,0] + anchor_info['az_anchor'].values
    th = anchor_info['el_anchor'].values + (angle_preds[:,1] if theta else np.zeros_like(angle_preds[:,1]))

    a = np.cos(phi*np.pi/180)*np.cos(th*np.pi/180)
    b = np.sin(phi*np.pi/180)*np.cos(th*np.pi/180)
    c = np.sin(th*np.pi/180)
    
    if alt:
        Ax,Ay,Az = a,b,c
        B = a*anchor_info['x_anchor'].values + b*anchor_info['y_anchor'].values + c*anchor_info['z_anchor'].values
        A = np.concatenate((Ax,Ay,Az)).reshape((3,len(phi))).T
        return A,B

    Ax = np.vstack((b,c))
    Ay = np.vstack((-a,np.zeros(a.shape)))
    Az = np.vstack((np.zeros(a.shape),-a))

    B1 = b*anchor_info['x_anchor'].values - a*anchor_info['y_anchor'].values
    B2 = c*anchor_info['x_anchor'].values - a*anchor_info['z_anchor'].values

    A = np.concatenate((Ax,Ay,Az)).reshape((3,2,len(phi))).T
    B = np.vstack((B1,B2)).T

    return A,B

def compute_weights(pos_preds, anch_info, theta):
    w = np.zeros((len(pos_preds),4))
    for i,_ in enumerate(anchors):
        info = anch_info[anch_info['anchor'] == i+1]
        w[:,i] = (info['x_anchor'].values - pos_preds[:,0])**2 + (info['y_anchor'].values - pos_preds[:,1])**2
        if theta:
            w[:,i] += (info['z_anchor'].values - pos_preds[:,2])**2
        w[:,i] = 1/(w[:,i])
    return w

def make_predictions(df_x, df_y, model, training_rooms, testing_rooms, test_points = None,
                    anchor_data = None, cnn = False):

    """
    Input: x, y
    Output: predictions and errors
    """

    angle_maes = defaultdict(lambda: (np.empty((2,4,10))))
    pos_maes = np.zeros((4,10))

    if cnn:
        y_test = pd.concat([df_y['testbench_01'][anchor] for anchor in anchors], axis=1)
    else:
        y_test = pd.concat([df_y['testbench_01'][anchor]['37'] for anchor in anchors], axis=1)

    angle_preds = defaultdict(dict)
    pos_angle_preds = defaultdict(dict)
    for i,training in enumerate(training_rooms):
        for j,testing in enumerate(testing_rooms):
            pos_preds = {}
            angle_preds[training][testing] = model[training].predict(df_x[testing])
            angle_preds_mae_angles = mean_absolute_error(angle_preds[training][testing], y_test,  multioutput='raw_values')

            for k,anchor in enumerate(anchors):
                angle_maes[anchor][:,i,j] = angle_preds_mae_angles[(2*k):(2*k + 2)]
                pos_preds[anchor] = angle_preds[training][testing][:,(k*2):(k*2 + 2)]
                
            pos_angle_preds[training][testing] = np.array(lsq(pos_preds, anchor_data[testing][anchors[0]]['37']['H']))
            true_pos = test_points[['x_tag', 'y_tag', 'z_tag']].values

            pos_maes[i,j] = np.mean(np.sqrt(np.sum((true_pos[:,:2] - pos_angle_preds[training][testing][:,:2])**2, axis=1)))
    
    predictions = {"angle_maes": angle_maes,
                   "pos_maes": pos_maes,
                   "angle_preds": angle_preds,
                   "pos_preds": pos_angle_preds
                   }
    
    return predictions, true_pos

def generate_pdda_preds(data, rooms, points, anchor_data):

    """
    Input: basic parameters
    Output: PDDA predictions
    """

    pdda_specs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for room in rooms:
        for anchor in anchors:
            for channel in channels:
                util_data = {polarity: points[['point']].merge(data[room][anchor][channel][polarity], on='point') for polarity in polarities}
                h = util_data['H']
                v = util_data['V']
                m = h.where(h['relative_power']+h['reference_power'] > v['reference_power']+v['relative_power'], v)
                pdda_specs[room][anchor][channel] = m.loc[:,['pdda_out_az','pdda_out_el']]

    pdda_angle_preds = defaultdict(dict)
    pdda_pos_preds = defaultdict()
    pdda_pos_maes = np.zeros((10,))
    pdda_angle_maes = defaultdict(lambda: (np.empty((2,10))))
    pdda_angle_stds = defaultdict(lambda: (np.empty((2,10))))

    for i,room in enumerate(rooms):  
        true_pos = points.iloc[:,1:4].values
        for anchor in anchors:
            util_data = {polarity: points[['point']].merge(data[room][anchor][channel][polarity], on='point') for polarity in polarities}
            true_angles = util_data['H'][['true_phi', 'true_theta']].values
            pddas_az = np.ones((len(points), 181))
            pddas_el = np.ones((len(points), 181))
            for channel in channels:
                pddas_az *= np.array(np.array(pdda_specs[room][anchor][channel]['pdda_out_az']).tolist())
                pddas_el *= np.array(np.array(pdda_specs[room][anchor][channel]['pdda_out_el']).tolist())
            pdda_angle_preds[room][anchor] = np.array(pd.concat((pd.DataFrame(np.argmax(pddas_az, axis = 1) - 90), pd.DataFrame(np.argmax(pddas_el, axis = 1) - 90)), axis = 1))
            pdda_angle_maes[anchor][:,i] = mean_absolute_error(true_angles, pdda_angle_preds[room][anchor], multioutput='raw_values')
            pdda_angle_stds[anchor][:,i] = np.std(np.abs(true_angles - pdda_angle_preds[room][anchor]),axis=0)
        pdda_pos_preds[room] = np.array(lsq(pdda_angle_preds[room], anchor_data[room][anchors[0]]['37']['H']))
        pdda_pos_maes[i] = np.mean(np.sqrt(np.sum((true_pos[:,:2] - pdda_pos_preds[room][:,:2])**2, axis=1)))

    pdda_results = {'angle_preds': pdda_angle_preds,
                    'pos_preds': pdda_pos_preds,
                    'pos_maes': pdda_pos_maes,
                    'angle_maes': pdda_angle_maes,
                    'angle_stds': pdda_angle_stds}
    
    return pdda_results