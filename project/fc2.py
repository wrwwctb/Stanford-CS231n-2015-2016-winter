import numpy as np

import matplotlib.pyplot as plt
import keras, time, os
from keras.layers import (Dense, Dropout, Activation, Flatten, Conv2D,
                          AveragePooling2D, MaxPooling2D, BatchNormalization)
import sys
logistic = __import__('logistic')

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
    
    seed = 7
    np.random.seed(seed)
    batch_size = 10000
    
    model = keras.models.Sequential()
    numfil = 1000
    
    regds = 0#1e-8
    
    model.add(Conv2D(filters=numfil, kernel_size=10, strides=(1, 1),
                     padding='valid', data_format='channels_first',
                     input_shape=(3, 10, 10),
                     kernel_regularizer=keras.regularizers.l2(regds)))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(numfil, kernel_regularizer=keras.regularizers.l2(regds)))
    model.add(Activation('relu'))
    model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(regds)))
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
    
    filters = model.layers[0].get_weights()[0]
    logistic.plot_filters(filters, path_results, output_name, 10)
    logistic.plot_history_loss(history.history, False, path_results, output_name)
    logistic.plot_model(model, path_results, output_name)
    
    plt.show()
    
    pred_val = np.squeeze(model.predict(X_val, batch_size=batch_size, verbose=1))
    effPU = logistic.roc(rew_val, isPU_val, True,  pred_val)
    effHS = logistic.roc(rew_val, isPU_val, False, pred_val)
    np.save(path_results + output_name + '_effPU', effPU)
    np.save(path_results + output_name + '_effHS', effHS)
    print(time.time()-ttt)
