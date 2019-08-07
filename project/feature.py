import numpy as np

import matplotlib.pyplot as plt
import keras, time, os
from keras.layers import (Dense, Dropout, Activation, Flatten, Conv2D,
                          AveragePooling2D, MaxPooling2D, BatchNormalization)
import sys
import cv2
logistic = __import__('logistic')

def frame2matrix(frame):
    ss = np.sum(frame)  # code from 19122690
    if ss == 0:
        return (None, None, None, 0)
    ny, nx = frame.shape
    x = np.arange(nx)
    y = np.arange(ny)
    fy = np.sum(frame, axis=1)
    fx = np.sum(frame, axis=0)
    xb = np.dot(x, fx)/ss
    yb = np.dot(y, fy)/ss

    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    xx = xx.astype('float64') - xb
    yy = yy.astype('float64') - yb
    xx *= frame
    yy *= frame
    mat = np.concatenate(([yy.ravel()], [xx.ravel()]), axis=0).T
    return (mat, xb, yb, ss)  # xb, yb is centroid. ss is total

def featureEng(inp):
    '''
    input: np array. #samples x #channels x widht and height
    return: for each sample,
            total brightness and std along the principle component
    '''
    Nf = 2  # num of features
    (NN, Nl) = inp.shape[:2]
    out = np.zeros((NN, Nf*Nl))
    for i in range(NN):
        if not (i%10000):
            print('Processing', i, '/', NN)
        for j in range(Nl):
            mat, _, _, ss = frame2matrix(inp[i, j, :, :])
            if ss == 0:
                out[i, j*Nf:j*Nf+Nf] = (0, 0)
                continue
            covar, _ = cv2.calcCovarMatrix(mat,
                                           np.array([0., 0.]),
                                           cv2.COVAR_ROWS |
                                           cv2.COVAR_NORMAL)
            _, _, eVec = cv2.eigen(covar)
            proj = np.dot(eVec[0], mat.T)
            out[i, j*Nf:j*Nf+Nf] = (ss, proj.std())
    return out

if __name__ == "__main__":
    ttt = time.time()

    path_file    = 'data/'
    path_results = 'results/'
    
    (isPU_train, isPU_val, _,
     image_train, image_val, _,
     rwi_train, rwi_val, _,
     rew_train, rew_val, _,
     Rpt_train, Rpt_val, _,
     j0pt_train, j0pt_val, _) = logistic.load_data(path_file)
    
    y_train = isPU_train.astype(int)
    y_val   = isPU_val.astype(int)
    
    X_train = image_train
    X_val   = image_val

    # test fcns
    '''
    frame = X_train[2, 0, :, :]
    mat, xb, yb, ss = frame2matrix(frame)
    covar, _ = cv2.calcCovarMatrix(mat,
                                   np.array([0., 0.]),  # mat is centered
                                   # np.empty((0, 0), dtype=mat.dtype),
                                   # cv2.COVAR_SCALE |
                                   cv2.COVAR_ROWS |
                                   # cv2.COVAR_SCRAMBLED)
                                   cv2.COVAR_NORMAL)
    _, eVal, eVec = cv2.eigen(covar)
    
    plt.imshow(frame, cmap='gray')
    lgth = 3
    plt.plot((xb, xb+eVec[0, 1]*lgth), (yb, yb+eVec[0, 0]*lgth))
    plt.show()
    proj = np.dot(eVec[0], mat.T)
    feature = (ss, proj.std())
    raise Exception('ha')
    '''
    # extract or load feature
    try:
        X_train = np.load(path_results + 'feature_train.npy')
        X_val = np.load(path_results + 'feature_val.npy')
    except (IOError, ValueError):
        X_train = featureEng(X_train)
        X_val = featureEng(X_val)
        np.save(path_results + "feature_train.npy", X_train)
        np.save(path_results + "feature_val.npy", X_val)

    seed = 7
    np.random.seed(seed)
    batch_size = 10000
    
    model = keras.models.Sequential()
    numfil = 1000
    
    regds = 0#1e-8
    
    model.add(Dense(1, input_shape=(6,),
                    kernel_regularizer=keras.regularizers.l2(regds)))
    model.add(Activation('sigmoid'))
    
    model.compile(loss="binary_crossentropy",
                  optimizer='adam', metrics=['accuracy'])
    
    #sanity check
    los, acc = logistic.pred(model, batch_size, X_val, y_val, rwi_val, .5, True)
    
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
    
    model = keras.models.load_model(chkpnt_name)
    
    los, acc = logistic.pred(model, batch_size, X_val, y_val, rwi_val, .5, True)
    
    output_name = (current_file_name + '_' + str(regds) + '_' +
                   time.strftime("%Y_%m_%d_%H_%M") + '_' + str(acc))
    
    model.save(path_results+output_name+'_mod')
    logistic.save_obj(history.history, path_results, output_name+'_his')
    
    #os.rename(chkpnt_name, path_results + output_name+'_his')
    os.remove(chkpnt_name)
    
    #filters = model.layers[0].get_weights()[0]
    #logistic.plot_filters(filters, path_results, output_name, 10)
    logistic.plot_history_loss(history.history, False, path_results, output_name)
    logistic.plot_model(model, path_results, output_name)
    
    plt.show()
    
    pred_val = np.squeeze(model.predict(X_val, batch_size=batch_size, verbose=1))
    effPU = logistic.roc(rew_val, isPU_val, True,  pred_val)
    effHS = logistic.roc(rew_val, isPU_val, False, pred_val)
    np.save(path_results + output_name + '_effPU', effPU)
    np.save(path_results + output_name + '_effHS', effHS)
    print(time.time()-ttt)
