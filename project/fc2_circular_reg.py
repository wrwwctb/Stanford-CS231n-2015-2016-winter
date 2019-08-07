import numpy as np

import matplotlib.pyplot as plt
import keras, time, os
from keras.layers import (Dense, Dropout, Activation, Flatten, Conv2D,
                          AveragePooling2D, MaxPooling2D, BatchNormalization)
import sys
from keras import backend
logistic = __import__('logistic')

class reg_grad_theta(keras.regularizers.Regularizer):
    '''
    regularizer for gradient in the theta direction

    when re-loading the model, before load_model, add
        import keras.regularizers
        keras.regularizers.reg_grad_theta = reg_grad_theta

    ref:
    https://github.com/fchollet/keras/issues/4990
    ^ doesn't work. the workaround before his pull request works. ie the above

    https://github.com/fchollet/keras/blob/master/keras/regularizers.py
    ^ followed the format here
    a good design pattern is essential
    '''
    def __init__(self, eta_range, phi_range,
                 num_filters, num_channels, regth, regl2):
        self.eta_range = eta_range
        self.phi_range = phi_range
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.regth = regth
        (self.ke,
         self.kp,
         self.th_e,
         self.th_p) = self._prep_grad_theta(eta_range,
                                            phi_range,
                                            num_filters,
                                            num_channels)
        #self.__name__ = 'reg_grad_theta'
        self.l2instance = keras.regularizers.l2(regl2)
        self.regl2 = regl2

    def __call__(self, fil):
        #eg. return 0.01 * backend.sum(backend.abs(fil))

        # for tf and th
        # before: 10 10 num_ch num_fil. after: num_fil 10 10 num_ch
        fil = backend.permute_dimensions(fil, [3, 0, 1, 2])

        # for v2
        #fil_e = fil_temp[:, 1:-1, :, :]
        #fil_p = fil_temp[:, :, 1:-1, :]

        # v1: num_fil  10   10  num_ch *  3   3  num_ch 1 = num_fil 8 8 1
        # v2: num_fil 8/10 10/8 num_ch * 1/3 3/1 num_ch 1 = num_fil 8 8 1
        ge = backend.conv2d(fil, self.ke)
        gp = backend.conv2d(fil, self.kp)
        #gd = backend.conv2d(fil, self.kd)
        #ga = backend.conv2d(fil, self.ka)
        #ge += gd - ga
        #gp += gd + ga
        gt = ge * self.th_e + gp * self.th_p
        out = backend.sum(backend.pow(gt, 2))
        #print(fil.shape, self.ke.shape, ge.shape,
        #      self.th_e.shape, gt.shape, out.shape)
        return out * self.regth + self.l2instance(fil)

    def get_config(self):
        return {'eta_range': self.eta_range,
                'phi_range': self.phi_range,
                'num_filters': self.num_filters,
                'num_channels': self.num_channels,
                'regth': self.regth,
                'regl2': self.regl2}

    def _prep_grad_theta(self, eta_range, phi_range,
                         num_filters, num_channels):
        ecoor = np.arange(1, eta_range-1)-(eta_range-1)/2
        pcoor = np.arange(1, phi_range-1)-(phi_range-1)/2
        ecoor = np.outer(np.ones(eta_range-2), ecoor)
        pcoor = np.outer(pcoor, np.ones(phi_range-2))
        #rcoor = np.sqrt(ecoor**2 + pcoor**2)#ecoor**2 + pcoor**2
        th_e = -pcoor# / rcoor
        th_p =  ecoor# / rcoor
        th_e = np.expand_dims(th_e, axis=-1)
        th_p = np.expand_dims(th_p, axis=-1)
        th_e = np.expand_dims(th_e, axis=-1)
        th_p = np.expand_dims(th_p, axis=-1)
        th_e = np.tile(th_e, num_filters)
        th_p = np.tile(th_p, num_filters)
        th_e = np.moveaxis(th_e, -1, 0)  # num_fil 8 8 1
        th_p = np.moveaxis(th_p, -1, 0)
        th_e = backend.constant(th_e, name='th_e')
        th_p = backend.constant(th_p, name='th_p')
        #v1

        ke = np.array([[ 1,  0, -1],  # prewitt, sobel, etc
                       [ 1,  0, -1],
                       [ 1,  0, -1]])
        kp = np.array([[ 1,  1,  1],
                       [ 0,  0,  0],
                       [-1, -1, -1]])
        #kd = np.array([[ 2,  1,  0],
        #               [ 1,  0, -1],
        #               [ 0, -1, -2]])/2/np.sqrt(2)
        #ka = np.array([[ 0,  1,  2],
        #               [-1,  0,  1],
        #               [-2, -1,  0]])/2/np.sqrt(2)

        ke = np.expand_dims(ke, -1)
        kp = np.expand_dims(kp, -1)
        #kd = np.expand_dims(kd, -1)
        #ka = np.expand_dims(ka, -1)
        ke = np.tile(ke, num_channels)
        kp = np.tile(kp, num_channels)
        #kd = np.tile(kd, num_channels)
        #ka = np.tile(ka, num_channels)
        ke = np.expand_dims(ke, -1)  # 3 3 num_ch 1
        kp = np.expand_dims(kp, -1)
        #kd = np.expand_dims(kd, -1)
        #ka = np.expand_dims(ka, -1)
        ke = backend.constant(ke, name='ke')
        kp = backend.constant(kp, name='kp')
        #kd = backend.constant(kd, name='kd')
        #ka = backend.constant(ka, name='ka')
        return ke, kp, th_e, th_p

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
    regth = 1e-4
    regthobj = reg_grad_theta(eta_range=10, phi_range=10,
                              num_filters=numfil,
                              num_channels=3, regth=regth, regl2=regds)
        
    model.add(Conv2D(filters=numfil, kernel_size=10, strides=(1, 1),
                     padding='valid', data_format='channels_first',
                     input_shape=(3, 10, 10),
                     kernel_regularizer=regthobj))
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
    
    keras.regularizers.reg_grad_theta = reg_grad_theta
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
