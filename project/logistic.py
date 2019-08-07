import numpy as np

import matplotlib.pyplot as plt
import keras, time, os
from keras.layers import (Dense, Dropout, Activation, Flatten, Conv2D,
                          AveragePooling2D, MaxPooling2D, BatchNormalization)
import sys
from IPython.display import display_svg
import pickle
#import platform # platform.system() returns 'Darwin' 'Linux' 'Windows' etc
#backend.tf.reset_default_graph()#can't run script twice in a row if this is on

def load_data(path):
    isPU_t = np.load(path + "isPU_j0_EM_train.npy")
    isPU_v = np.load(path + "isPU_j0_EM_val.npy")
    isPU_T = np.load(path + "isPU_j0_EM_test.npy")
    rwi_t  = np.load(path + "revisedWeights_j0_EM_train.npy")
    rwi_v  = np.load(path + "revisedWeights_j0_EM_val.npy")
    rwi_T  = np.load(path + "revisedWeights_j0_EM_test.npy")
    image_t = np.load(path + "pixel_image_clus_trks_j0_EM_train.npy")
    image_v = np.load(path + "pixel_image_clus_trks_j0_EM_val.npy")
    image_T = np.load(path + "pixel_image_clus_trks_j0_EM_test.npy")
    rew_t = np.load(path + "rawEventWeights_j0_EM_train.npy")
    rew_v = np.load(path + "rawEventWeights_j0_EM_val.npy")
    rew_T = np.load(path + "rawEventWeights_j0_EM_test.npy")
    Rpt_t = np.load(path + "jet_Rpt_j0_EM_train.npy")
    Rpt_v = np.load(path + "jet_Rpt_j0_EM_val.npy")
    Rpt_T = np.load(path + "jet_Rpt_j0_EM_test.npy")
    j0pt_t = np.load(path + "recopts_j0_EM_train.npy")
    j0pt_v = np.load(path + "recopts_j0_EM_val.npy")
    j0pt_T = np.load(path + "recopts_j0_EM_test.npy")

#    # debug. is there a bias between train and val?
#    nn = len(isPU_t)
#    nn_t = int(nn*.9)
#    indices = np.random.permutation(range(nn))
#
#    def debug(isPU_t, indices, nn_t):
#        isPU_v = isPU_t[indices[nn_t:]]
#        isPU_t = isPU_t[indices[:nn_t]]
#        return isPU_t, isPU_v
#
#    isPU_t , isPU_v  = debug(isPU_t , indices, nn_t)
#    rwi_t  , rwi_v   = debug(rwi_t  , indices, nn_t)
#    image_t, image_v = debug(image_t, indices, nn_t)
#    rew_t  , rew_v   = debug(rew_t  , indices, nn_t)
#    Rpt_t  , Rpt_v   = debug(Rpt_t  , indices, nn_t)
#    j0pt_t , j0pt_v  = debug(j0pt_t , indices, nn_t)
#
#    rwi_t /= sum(rwi_t)
#    rwi_v /= sum(rwi_v)

    return (isPU_t, isPU_v, isPU_T,
            image_t, image_v, image_T,
            rwi_t, rwi_v, rwi_T,
            rew_t, rew_v, rew_T,
            Rpt_t, Rpt_v, Rpt_T,
            j0pt_t, j0pt_v, j0pt_T)

def plot_model(model, path='', prefix=''):
    dot = keras.utils.vis_utils.model_to_dot(
            model, show_shapes=True, show_layer_names=True)
    svg = dot.create(prog='dot', format='svg')
    display_svg(svg, raw=True)
    if path and prefix:#37427362
        keras.utils.plot_model(model, to_file=path+prefix+'_mod.png')

def pred(model, batch_size, XX, yy, weight, p_thresh, ifprint=False):
    y_pred = np.squeeze(model.predict(XX, batch_size=batch_size, verbose=1))
    if type(yy) is np.ndarray:
        yy_array = yy
    else:  # if not ndarray, must be a dict
        yy_array = yy[list(yy.keys())[0]]
    acc = np.sum(((y_pred > p_thresh) == yy_array)*weight)
    los = model.evaluate(XX, yy, batch_size=batch_size,
                         verbose=1, sample_weight=weight)
    if ifprint:
        print('### los', los, '### weighted acc', acc, '###')
    return (los, acc)

def plot_filters(filters, path='', prefix='', nrowsmax=0):
    '''expect dimension [:, :, num_channels, num_filters]'''
    ncols = filters.shape[2]
    if nrowsmax==0:
        nrows = filters.shape[-1]
    else:
        nrows = min([nrowsmax, filters.shape[-1]])

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows),
                            sharey=True, sharex=True)
    if len(axs.shape) == 1:
        axs = np.expand_dims(axs, 0)

    for idx in range(nrows):
        for jdx in range(ncols):
            axs[idx, jdx].imshow(filters[:, :, jdx, idx], cmap='gray')
            axs[idx, jdx].get_xaxis().set_visible(False)
            axs[idx, jdx].get_yaxis().set_visible(False)

    if path and prefix:
        fig.savefig(path+prefix+'_fil.png', bbox_inches='tight')

def plot_history_loss(his, iflog=False, path='', prefix='', ifyy=False):
    '''
    his is a dict
    Say, if
      history = model.fit(...)
    use history.history for his
    '''
    if ifyy:
        fig, ax1 = plt.subplots()
        ax1.plot(his['loss'], 'b-')
        ax2 = ax1.twinx()
        ax2.plot(his['val_loss'], 'r-')

        if iflog:
            ax1.set_yscale('log')
            ax2.set_yscale('log')
        else:
            ax1.set_yscale('linear')
            ax2.set_yscale('linear')
    
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training loss', color='b')
        ax1.tick_params('y', colors='b')
        ax2.set_ylabel('Val loss', color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
    else:
        fig = plt.figure()
        plt.plot(his['loss'], 'b-', label='Train')
        plt.plot(his['val_loss'], 'r-', label='Val')
        plt.gca().set_xlabel('Epoch')
        plt.gca().set_ylabel('Loss')
        if iflog:
            plt.gca().set_yscale('log')
        plt.legend()
    if path and prefix:
        fig.savefig(path+prefix+'_los.png', bbox_inches='tight')

def roc(rawEventWeight, isPU, TorF, prob, NUM=5000, MAXPROB=1):
    '''
    rawEventWeight: Nx1 (N is number of samples)
    isPU: Nx1 bool
    TorF: True (PU) or False (non PU)
    prob: Nx1, output of classifier
    NUM: number of points for roc curve
    MAXPROB: 1 or 1000, max of probability
    '''
    rwe = np.array(rawEventWeight[isPU==TorF])  # raw event weight of one kind
    rweS = np.sum(rwe)
    pro = prob[isPU==TorF].squeeze()
    eff = []
    for thresh in np.linspace(0, MAXPROB, num=NUM):
        eff.append(rwe[pro<=thresh].sum() / rweS)
    return eff

def save_obj(obj, path, name):
    with open(path+name, 'wb') as ff:
        pickle.dump(obj, ff, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name):
    with open(path+name, 'rb') as ff:
        return pickle.load(ff)

if __name__ == "__main__":
    ttt = time.time()
    
    path_file    = 'data/'
    path_results = 'results/'
    
    (isPU_train, isPU_val, _,
     image_train, image_val, _,
     rwi_train, rwi_val, _,
     rew_train, rew_val, _,
     Rpt_train, Rpt_val, _,
     j0pt_train, j0pt_val, _) = load_data(path_file)
    
    y_train = isPU_train.astype(int)
    y_val   = isPU_val.astype(int)
    
    X_train = image_train
    X_val   = image_val
    
    seed = 7
    np.random.seed(seed)
    batch_size = 10000
    
    model = keras.models.Sequential()
    numfil = 1
    
    regds = 0#1e-8
    
    model.add(Conv2D(filters=numfil, kernel_size=10, strides=(1, 1),
                     padding='valid', data_format='channels_first',
                     input_shape=(3, 10, 10),
                     kernel_regularizer=keras.regularizers.l2(regds)))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    '''
    keras expects a probability at the end of the network
    github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
    '''
    model.compile(loss="binary_crossentropy",
                  optimizer='adam', metrics=['accuracy'])
    '''
    binary cross entropy vs softmax
    www.quora.com/For-a-classification-problem-two-classes-is-it-better-to-use-two-
    outputs-with-softmax-or-one-output-with-binary-cross-entropy
    '''
    #sanity check
    los, acc = pred(model, batch_size, X_val, y_val, rwi_val, .5, True)
    
    current_file_name = os.path.basename(sys.argv[0])
    # __file__ is different in python and ipython.. 4152963
    
    chkpnt_name = path_results + current_file_name + '_' + str(regds) + '_chk_temp'
    chkpnt_ojb = keras.callbacks.ModelCheckpoint(
                     chkpnt_name, monitor='val_loss', verbose=0,
                     save_best_only=True, save_weights_only=False,
                     mode='auto', period=1)
    early_stop = keras.callbacks.EarlyStopping(
                     monitor='val_loss', min_delta=0, patience=50, verbose=1,
                     mode='auto')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
                     monitor='val_loss', factor=0.2, patience=10, verbose=1,
                     mode='auto', epsilon=1e-9)
    
    history = model.fit(X_train, y_train, epochs=1000, batch_size=batch_size,
                        verbose=1, sample_weight=rwi_train,
                        validation_data=(X_val, y_val, rwi_val),
                        callbacks=[chkpnt_ojb, early_stop, reduce_lr])
    '''
    loss is weighted by sample_weight but metrics=['accuracy'] is NOT
    https://github.com/fchollet/keras/issues/1642
    
    Custom metrics needs to take in (y_true, y_pred), same for train and val
    https://keras.io/metrics/

    difficult to get model.fit to use different weights for train and val
    '''
    model = keras.models.load_model(chkpnt_name)
    
    los, acc = pred(model, batch_size, X_val, y_val, rwi_val, .5, True)
    
    output_name = (current_file_name + '_' + str(regds) + '_' +
                   time.strftime("%Y_%m_%d_%H_%M") + '_' + str(acc))
    
    model.save(path_results+output_name+'_mod')
    save_obj(history.history, path_results, output_name+'_his')
    
    #os.rename(chkpnt_name, path_results + output_name+'_his')
    os.remove(chkpnt_name)
    
    filters = model.layers[0].get_weights()[0]
    plot_filters(filters, path_results, output_name, 10)
    plot_history_loss(history.history, False, path_results, output_name)
    plot_model(model, path_results, output_name)
    
    plt.show()
    
    pred_val = np.squeeze(model.predict(X_val, batch_size=batch_size, verbose=1))
    effPU = roc(rew_val, isPU_val, True,  pred_val)
    effHS = roc(rew_val, isPU_val, False, pred_val)
    np.save(path_results + output_name + '_effPU', effPU)
    np.save(path_results + output_name + '_effHS', effHS)
    print(time.time()-ttt)
