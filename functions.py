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



def create_set(data, points_set, rooms = None, concrete_rooms = None, other_scenarios = None, 
               anchors = None, channels = None, polarities = None):

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



def create_iq_images(data, augm = False, anchors = None, channels = None):
    
    '''
    Preprocess input for CNN model
    '''

    chanls = []
    powers = []

    for channel in channels:    
        iqs = []
        for anchor in anchors:
            if augm:
                dt = data[anchor][channel]
            else:
                dt = iq_processing(data[anchor][channel]).iloc[:,-10:]
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
                util_data = defaultdict()
                for polarity in polarities:
                    util_data[polarity] = points_set[['point']].merge(data[room][anchor][channel][polarity], on='point')
                h = util_data['H']
                v = util_data['V']
                tmp[room][anchor][channel] = h.where(h['relative_power']+h['reference_power'] > v['reference_power']+v['relative_power'], v)
            y[room][anchor] = util_data['H'][['true_phi', 'true_theta']]
        x[room]['iq_image'], x[room]['powers'] = create_iq_images(tmp[room], anchors = anchors, channels = channels)
    
    return x,y


    
def decreasing_signal(df, scale_util = 5, mode = 'amplitude', rooms = None, anchors = None, channels = None):

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



def make_predictions(df_x, df_y, model, rooms = None, concrete_rooms = None, test_points = None,
                     other_scenarios = None, anchors = None, anchor_data = None, cnn = False):

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



def posHeatmapXY(maes, pdda_maes = None, figsize=(15,7), scenarios = False):

    '''
    Plots a heatmap given the mean euclidean distance errors of the model.
    Input:  A (3,7) numpy array containing the mean euclidean distance errors of the model trained and tested in different room combinations
    Output: Prints the heatmap
    '''
    
    training_room_names = ['No Furniture', 'Low Furniture', 'Mid Furniture', 'High Furniture']
    testing_room_names = ['No Furniture', 'Low Furniture', 'Mid Furniture', 'High Furniture']
    testing_room_names += ['Low Furniture\nConcrete', 'Mid Furniture\nConcrete', 'High Furniture\nConcrete']

    if scenarios:
        testing_room_names += ['Rotated\nAnchors', 'Translated\nAnchors']

    fig, ax = plt.subplots(2,1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [1,4]})
    cbar_ax = fig.add_axes([.91, .12, .03, .75])

    euclid_pdda = pd.DataFrame(pdda_maes, testing_room_names)
    euclid_pdda.rename(columns = {0: ''}, inplace = True)
    euclid = pd.DataFrame(maes, training_room_names, testing_room_names)

    vmax = np.max(euclid_pdda)
    vmin = np.min(euclid.values)
    
    sn.heatmap(euclid_pdda.T, annot=True, annot_kws={"size": 20}, cmap="YlGnBu", cbar_ax = cbar_ax, norm=colors.LogNorm(vmin, vmax), ax=ax[0])
    sn.heatmap(euclid, annot=True, annot_kws={"size": 20}, cbar=False, vmax = vmax, vmin = vmin, cmap="YlGnBu", cbar_ax = None, norm=colors.LogNorm(vmin, vmax), ax=ax[1])

    ax[0].set_ylabel('PDDA', rotation=0, ha='right')
    sn.set(font_scale=1.2)
    
    fig.text(0.52, -0.1, 'Testing Room', ha='center')
    fig.text(0, 0.4, 'Training Room', va='center', rotation='vertical')

    cbar_ax.set_ylabel('MEDE (m)', fontsize=14)

    ax[1].set_yticklabels(ax[1].get_yticklabels(), rotation=0) 
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='center') 
    
    plt.show()



def addFurniture(ax, room, anchors=[1,2,3,4]):
    a = np.array([1,1,1,1,1,1])
    a_low = 0.5
    if room in ['testbench_01_furniture_mid', 'testbench_01_furniture_mid_concrete']:
        a[1] = a[3] = a_low
    if room in ['testbench_01_furniture_low', 'testbench_01_furniture_low_concrete']:
        a[2] = a[0] = a_low
    if room in ['testbench_01', 'testbench_01_scenario2', 'testbench_01_scenario3']:
        a = 6*[a_low]
    furniture = [plt.Rectangle((44.+1.9, 43.1+0.2), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[0]),
                 plt.Rectangle((44.+4.45, 43.1+1), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[1]),
                 plt.Rectangle((44.+6.4, 43.1+2.6), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[2]),
                 plt.Rectangle((44.+1.7, 43.1+4.1), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[3]),
                 plt.Rectangle((44.+4.2, 43.1+3.4), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[4]),
                 plt.Rectangle((44.+5.4, 43.1+5.15), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[5]),]
    
    anchrs = [plt.Circle((57.9, 43.3), 0.2, fc='firebrick', ec='black', lw=2),
               plt.Circle((57.9, 50.0), 0.2, fc='firebrick', ec='black', lw=2),
               plt.Circle((44.3, 50.0), 0.2, fc='firebrick', ec='black', lw=2),
               plt.Circle((44.3, 43.3), 0.2, fc='firebrick', ec='black', lw=2)]

    for anchor in anchors:
        ax.add_patch(anchrs[anchor-1])
    for item in furniture:
        ax.add_patch(item)



def spatial_plot(preds, true_pos, testing_room = None, mode='xy', vmin=None, vmax=None, cmap='PuBu'):
    errors = pd.DataFrame()
    errors['xy'] = np.sqrt((preds[:,0] - true_pos[:,0])**2 + (preds[:,1] - true_pos[:,1])**2)
    errors['xyz'] = np.sqrt((preds[:,0] - true_pos[:,0])**2 + (preds[:,1] - true_pos[:,1])**2 + (preds[:,2] - true_pos[:,2])**2)
    errors[['x_tag', 'y_tag', 'z_tag']] = true_pos
    ax = plt.gca()
    errors.plot.hexbin('x_tag', 'y_tag', mode, gridsize=(35,12), figsize = (17,7), cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
    addFurniture(ax, testing_room)
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_ylim(43,50.3)
    ax.set_xlim(43.9,58.2)
    ax.set_xticklabels(list(range(-2,15,2)))
    ax.set_yticklabels(list(range(0,8)))
    ax.text(60.3, 46, 'MEDE (m)', rotation=90)
    plt.show()



def default_to_regular(d):
    if isinstance(d, (defaultdict, dict)):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d