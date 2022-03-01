def iq_processing(data, power_scale=400, epsilon=10**(-8)):
    
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

    #out.insert(22, 'power', (out['reference_power'] + out['relative_power'].values)/power_scale)
    power_norm =  StandardScaler().fit_transform((out['reference_power'] + out['relative_power']).values.reshape(-1,1))/10
    out.insert(22, 'power', power_norm)
    
    out.insert(21, 'iq_ref', iq_ref)
    out.drop(columns=['pdda_input_imag_1', 'pdda_input_real', 'pdda_input_imag'], inplace=True)

    return out



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
    if alt:
        B = np.concatenate(Bs)
    else:
        B = np.concatenate(Bs, axis=1)

    preds = []
    for a,b in zip(A,B):
        preds.append(np.linalg.lstsq(a,b,rcond=None)[0])
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
    th = anchor_info['el_anchor'].values + angle_preds[:,1]
    if not theta:
        th = anchor_info['el_anchor'].values + np.zeros_like(angle_preds[:,1])

    a = np.cos(phi*np.pi/180)*np.cos(th*np.pi/180)
    b = np.sin(phi*np.pi/180)*np.cos(th*np.pi/180)
    c = np.sin(th*np.pi/180)
    
    if alt:
        Ax = a
        Ay = b
        Az = c
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



def create_set(data, points_set, rooms = rooms, concrete_rooms = concrete_rooms, other_scenarios = other_scenarios, 
               anchors = anchors, channels = channels, polarities = polarities):

    """
    Input: Data and points for set that we want
    Output: x and y for set that we want
    """

    x = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    y = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for room in rooms + concrete_rooms + other_scenarios:
        for anchor in anchors:
            for channel in channels:
                util_data = defaultdict()
                for polarity in polarities:
                    util_data[polarity] = points_set[['point']].merge(data[room][anchor][channel][polarity], on='point')

                h = util_data['H']
                v = util_data['V']
                m = h.where(h['relative_power']+h['reference_power'] > v['reference_power']+v['relative_power'], v)
                x[room][anchor][channel] = iq_processing(m).iloc[:,-10:]
                y[room][anchor][channel] = util_data[polarity][['true_phi', 'true_theta']]
    
    return x, y



def decreasing_signal(df, scale_util = 5, mode = 'amplitude', rooms = rooms, anchors = anchors, channels = channels):

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



def make_predictions(df_x, df_y, model, rooms = rooms, concrete_rooms = concrete_rooms, test_points = test_points,
                     other_scenarios = other_scenarios, anchors = anchors, anchor_data = anchor_data):

    """
    Input: x, y
    Output: predictions and errors
    """

    angle_maes = defaultdict(lambda: (np.empty((2,4,10))))
    pos_maes = np.zeros((4,10))

    y_test = pd.concat([df_y['testbench_01'][anchor]['37'] for anchor in anchors], axis=1)

    angle_preds = defaultdict(dict)
    pos_angle_preds = defaultdict(dict)
    for i,training in enumerate(rooms):
        for j,testing in enumerate(rooms + concrete_rooms + other_scenarios):
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



def default_to_regular(d):
    if isinstance(d, (defaultdict, dict)):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d