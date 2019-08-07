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
    regds = 0#1e-8
    #regth = 1e-4
    #regthobj = circular.reg_grad_theta(eta_range=10, phi_range=10,
    #                                   num_filters=numfil,
    #                                   num_channels=3, regth=regth, regl2=regds)

    main_in = Input(shape=(3, 10, 10), name='main_in')
    
    reg1 = 1e-5
    x1 = Conv2D(filters=128, kernel_size=9, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=keras.regularizers.l2(reg1))(main_in)
    x1 = BatchNormalization(axis=1)(x1)
    x1 = Activation('relu')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid',
                          data_format='channels_first')(x1)
    x1 = Flatten()(x1)
    x1 = Dropout(rate=.5, seed=seed)(x1)
    
    reg3 = 1e-6
    x3 = Conv2D(filters=4, kernel_size=3, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=keras.regularizers.l2(reg3))(main_in)
    x3 = BatchNormalization(axis=1)(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=keras.regularizers.l2(reg3))(x3)
    x3 = BatchNormalization(axis=1)(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=keras.regularizers.l2(reg3))(x3)
    x3 = BatchNormalization(axis=1)(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=keras.regularizers.l2(reg3))(x3)
    x3 = BatchNormalization(axis=1)(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',
                      data_format='channels_first')(x3)
    x3 = Flatten()(x3)
    x3 = Dropout(rate=.5, seed=seed)(x3)
    
    reg5 = reg3
    x5 = Conv2D(filters=6, kernel_size=5, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=keras.regularizers.l2(reg5))(main_in)
    x5 = BatchNormalization(axis=1)(x5)
    x5 = Activation('relu')(x5)
    x5 = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=keras.regularizers.l2(reg5))(x5)
    x5 = BatchNormalization(axis=1)(x5)
    x5 = Activation('relu')(x5)
    x5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',
                      data_format='channels_first')(x5)
    x5 = Flatten()(x5)
    x5 = Dropout(rate=.5, seed=seed)(x5)
    
    reg7 = reg3
    x7 = Conv2D(filters=16, kernel_size=7, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=keras.regularizers.l2(reg7))(main_in)
    x7 = BatchNormalization(axis=1)(x7)
    x7 = Activation('relu')(x7)
    x7 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid',
                data_format='channels_first',
                kernel_regularizer=keras.regularizers.l2(reg3))(x7)
    x7 = BatchNormalization(axis=1)(x7)
    x7 = Activation('relu')(x7)
    x7 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',
                      data_format='channels_first')(x7)
    x7 = Flatten()(x7)
    x7 = Dropout(rate=.5, seed=seed)(x7)
    
    x = keras.layers.concatenate([x1, x3, x5, x7])
    x = Dense(256)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    main_out = Activation('sigmoid', name='out_prob_PU')(x)
    
    model = Model(inputs=[main_in], outputs=[main_out])
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
    
    filters8 = model.layers[8].get_weights()[0]
    logistic.plot_filters(filters8, path_results, output_name+'_8', 10)
    filters9 = model.layers[9].get_weights()[0]
    logistic.plot_filters(filters9, path_results, output_name+'_9', 10)
    filters16 = model.layers[16].get_weights()[0]
    logistic.plot_filters(filters16, path_results, output_name+'_16', 10)
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
