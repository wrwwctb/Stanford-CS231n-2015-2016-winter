import numpy as np

import matplotlib.pyplot as plt
import keras, time, os
from keras.layers import (Dense, Dropout, Activation, Flatten, Conv2D,
                          AveragePooling2D, MaxPooling2D, BatchNormalization)
import sys
from keras.layers import Input
from keras.models import Model
logistic = __import__('logistic')
circular = __import__('fc2_circular_reg')

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
    
    #numfil = 1000
    #regds = 0#1e-8
    #regth = 1e-4
    #regthobj = circular.reg_grad_theta(eta_range=10, phi_range=10,
    #                                   num_filters=numfil,
    #                                   num_channels=3, regth=regth, regl2=regds)

    regth = 1e-4
    regds = 1e-5
    regth1 = circular.reg_grad_theta(eta_range=10, phi_range=10,
                                    num_filters=100,
                                    num_channels=3,
                                    regth=regth*.1, regl2=regds*.1)
    regth2 = circular.reg_grad_theta(eta_range=5, phi_range=5,
                                    num_filters=40,
                                    num_channels=20, regth=regth, regl2=regds)
    regth3 = circular.reg_grad_theta(eta_range=8, phi_range=8,
                                    num_filters=20,
                                    num_channels=10, regth=regth*10, regl2=regds*10)
    
    main_in = keras.layers.Input(shape=(3, 10, 10), name='main_in')
    
    x1 = Conv2D(filters=100, kernel_size=10, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=regth1)(main_in)
    x1 = Activation('relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(5)(x1)
    x1 = Activation('relu')(x1)
    
    x2 = Conv2D(filters=20, kernel_size=2, strides=(2, 2), padding='valid',
                data_format='channels_first')(main_in)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(filters=40, kernel_size=5, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=regth2)(x2)
    x2 = Activation('relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(3)(x2)
    x2 = Activation('relu')(x2)
    
    x3 = Conv2D(filters=10, kernel_size=3, strides=(1, 1), padding='valid',
                data_format='channels_first')(main_in)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(filters=20, kernel_size=8, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=regth3)(x3)
    x3 = Activation('relu')(x3)
    x3 = Flatten()(x3)
    x3 = Dense(2)(x3)
    x3 = Activation('relu')(x3)
    
    x = keras.layers.concatenate([x1, x2, x3])
    
    x = Dense(1)(x)
    main_out = Activation('sigmoid', name='out_prob_PU')(x)

    model = keras.models.Model(inputs=[main_in], outputs=[main_out])

    model.compile(optimizer='adam',
                  loss={'out_prob_PU': 'binary_crossentropy'},
                  loss_weights={'out_prob_PU': 1.}, metrics=['accuracy'])
    
    #sanity check
    los, acc = logistic.pred(model, batch_size,
                             {'main_in': X_val},
                             {'out_prob_PU': y_val},
                             rwi_val, .5, True)
    
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
    
    history = model.fit({'main_in': X_train},
                        {'out_prob_PU': y_train},
                        epochs=1000, batch_size=batch_size,
                        verbose=1, sample_weight=rwi_train,
                        validation_data=([X_val], [y_val], rwi_val),
                        callbacks=[chkpnt_ojb, early_stop, reduce_lr])
    
    #keras.regularizers.reg_grad_theta = circular.reg_grad_theta
    model = keras.models.load_model(chkpnt_name)
    
    los, acc = logistic.pred(model, batch_size,
                             {'main_in': X_val},
                             {'out_prob_PU': y_val},
                             rwi_val, .5, True)
    
    output_name = (current_file_name + '_' + str(regds) + '_' +
                   time.strftime("%Y_%m_%d_%H_%M") + '_' + str(acc))
    
    model.save(path_results+output_name+'_mod')
    logistic.save_obj(history.history, path_results, output_name+'_his')
    
    #os.rename(chkpnt_name, path_results + output_name+'_his')
    os.remove(chkpnt_name)
    
    filters5 = model.layers[5].get_weights()[0]
    logistic.plot_filters(filters5, path_results, output_name+'_5', 10)
    filters6 = model.layers[6].get_weights()[0]
    logistic.plot_filters(filters6, path_results, output_name+'_6', 10)
    filters7 = model.layers[7].get_weights()[0]
    logistic.plot_filters(filters7, path_results, output_name+'_7', 10)
    logistic.plot_history_loss(history.history, False, path_results, output_name)
    logistic.plot_model(model, path_results, output_name)
    
    plt.show()
    
    pred_val = np.squeeze(model.predict({'main_in': X_val},
                                        batch_size=batch_size, verbose=1))
    effPU = logistic.roc(rew_val, isPU_val, True,  pred_val)
    effHS = logistic.roc(rew_val, isPU_val, False, pred_val)
    np.save(path_results + output_name + '_effPU', effPU)
    np.save(path_results + output_name + '_effHS', effHS)
    print(time.time()-ttt)
