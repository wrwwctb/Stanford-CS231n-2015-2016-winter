import numpy as np
import keras, time
import operator
import matplotlib.pyplot as pl
logistic = __import__('logistic')

def tp_fp_tn_fn_util(rew, isPU, TorF, op, prob, thresh_linspace):
    '''
    rew rawEventWeight: Nx1 (N is number of samples)
    isPU: Nx1 bool. 1 is PU
    TorF: True (PU) or False (non PU)
    op: operator. le or gt
    prob: Nx1, output of classifier
    NUM: number of points for roc curve
    MAXPROB: 1 or 1000, max of probability

    Return:
    ToF     op      return
    ------------------------------
    True    le      false neg
    True    gt      true pos
    False   le      true neg
    False   gt      false pos
    '''
    rew = np.array(rew[isPU==TorF])
    pro = prob[isPU==TorF].squeeze()
    out = []
    for thresh in thresh_linspace:
        out.append(rew[op(pro, thresh)].sum())
    return np.array(out)

def tp_fp_tn_fn(rew, isPU,  pred, thresh_linspace): # can optimize
    fn = tp_fp_tn_fn_util(rew, isPU, True, operator.le, pred, thresh_linspace)
    tp = tp_fp_tn_fn_util(rew, isPU, True, operator.gt, pred, thresh_linspace)
    tn = tp_fp_tn_fn_util(rew, isPU, False, operator.le, pred, thresh_linspace)
    fp = tp_fp_tn_fn_util(rew, isPU, False, operator.gt, pred, thresh_linspace)
    return tp, fp, tn, fn

def prec(tp, fp, tn, fn):
    '''precision'''
    prec_p = tp / (tp + fp)
    prec_n = tn / (tn + fn)
    return prec_p, prec_n

def reca(tp, fp, tn, fn):
    '''recall'''
    reca_p = tp / (tp + fn)
    reca_n = tn / (tn + fp)
    return reca_p, reca_n

def dezero(x):
    x[x <= 0] = 1e-6
    return x

def f1(tp, fp, tn, fn):
    '''f1'''
    prec_p, prec_n = prec(tp, fp, tn, fn)
    reca_p, reca_n = reca(tp, fp, tn, fn)
    prec_p = dezero(prec_p)
    prec_n = dezero(prec_n)
    reca_p = dezero(reca_p)
    reca_n = dezero(reca_n)
    f1p = 2 / (1/prec_p + 1/reca_p)
    f1n = 2 / (1/prec_n + 1/reca_n)
    return f1p, f1n

def load_model(path_results, file_name):
    return keras.models.load_model(path_results + file_name)

def get_pred(model, Xfeed, batch_size=10000):
    return np.squeeze(model.predict(Xfeed, batch_size=batch_size, verbose=1))

def plot_find_peak(rew, isPU, pred, name, num=5001):
    thresh_linspace = np.linspace(0, 1, num=num)
    tp, fp, tn, fn = tp_fp_tn_fn(rew, isPU, pred, thresh_linspace)

#    idx = num//2
#    print('                   Truth')
#    print('                 PU      HS')
#    print('predicted   PU', tp[idx], fp[idx])
#    print('            HS', fn[idx], tn[idx])
#    print((tp[idx]+fn[idx]) / (fp[idx]+tn[idx]))

    f1p, f1n = f1(tp, fp, tn, fn)
    prec_p, prec_n = prec(tp, fp, tn, fn)
    reca_p, reca_n = reca(tp, fp, tn, fn)

    f1n[np.isnan(f1n)] = 0
    f1nmp = np.argmax(f1n)
    thresh = np.linspace(0, 1, num)
    pl.figure()
    pl.plot(thresh, f1p, label='F1_p (pile up)')
    pl.plot(thresh, f1n, label='F1_n (hard scatter)')
    pl.plot(thresh, prec_p, label='Precision_p')
    pl.plot(thresh, prec_n, label='Precision_n')
    pl.plot(thresh, reca_p, label='Recall_p')
    pl.plot(thresh, reca_n, label='Recall_n')
    pl.plot(thresh[f1nmp], f1n[f1nmp], 'o')
    pl.legend()
    pl.title(name+'\n(Threshold, max F1_n)\n('+\
             str(thresh[f1nmp])+', '+str(f1n[f1nmp])+')')
    #pl.show()
    return thresh[f1nmp]

def f1n_test(rew, isPU, pred, thresh_list):
    tp, fp, tn, fn = tp_fp_tn_fn(rew, isPU, pred, thresh_list)
    f1p, f1n = f1(tp, fp, tn, fn)
    return f1n

if __name__ == "__main__":
    ttt = time.time()
    
    path_file    = '../../../data/'
    path_results = '../../../results/'
    
    (isPU_train, isPU_val, isPU_test,
     image_train, image_val, image_test,
     rwi_train, rwi_val, rwi_test,
     rew_train, rew_val, rew_test,
     Rpt_train, Rpt_val, Rpt_test,
     j0pt_train, j0pt_val, j0pt_test) = logistic.load_data(path_file)

    save_names = ['logistic.py_0_2017_11_18_10_05_0.691116214722_mod',
                  'simple.py_0_2017_11_18_13_52_0.707155809418_mod',
                  'simple_aux.py_0_2017_11_18_15_02_0.712382166975_mod',
                  'wide.py_0_2017_12_20_15_51_0.703289627721_mod']
    feeds_train = [image_train,
                   image_train,
                   {'main_in': image_train, 'aux_in': j0pt_train},
                   image_train]
    feeds_test = [image_test,
                  image_test,
                  {'main_in': image_test, 'aux_in': j0pt_test},
                  image_test]
    display_names = ['logistic',
                     'simple',
                     'simple_aux',
                     'wide']
    test_output = []
    for idx in range(len(save_names)):
        model = load_model(path_results, save_names[idx])
        # use train data to find best thresh
        pred = get_pred(model, feeds_train[idx])
        thresh = plot_find_peak(rew_train, isPU_train, pred, display_names[idx])
        # use optimal thresh to test
        pred = get_pred(model, feeds_test[idx])
        f1n = f1n_test(rew_test, isPU_test, pred, [thresh])
        test_output.append('%10s, test f1_n %.6f' % (display_names[idx], f1n))

    print('')
    for idx in range(len(save_names)):
        print(test_output[idx])

    pl.show()
    print(time.time()-ttt)

