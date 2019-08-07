import numpy as np
import matplotlib.pyplot as plt

path_data  = 'data/'

#294573
def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))

def rescale_array(x):
    ma = np.amax(x)
    mi = np.amin(x)
    return (x - mi)/(ma - mi)

def acc_fcn(y_pred, y_train, weights, p_thresh = 0.5):
    acc = np.sum(((y_pred > p_thresh) == y_train) * weights)
    return acc

isPU    = np.load(path_data + "isPU_j0_EM_val.npy") 
jet_Rpt = np.load(path_data + "jet_Rpt_j0_EM_val.npy") 
j0pt    = np.load(path_data + "recopts_j0_EM_val.npy") 
rew     = np.load(path_data + "rawEventWeights_j0_EM_val.npy")
rwi     = np.load(path_data + "revisedWeights_j0_EM_val.npy")

path_results = 'results/'

# from previous runs
prob_base    = np.load(path_results + "classPredictions_CV_baselineNN_23.npy"  ).squeeze()/1e3
prob_CNN     = np.load(path_results + "classPredictions_CV_simpleCNN_23.npy"   ).squeeze()/1e3
prob_CNNr    = np.load(path_results + "classPredictions_CV_simpleCNNr_23.npy"  ).squeeze()/1e3
prob_CNNv3   = np.load(path_results + "classPredictions_CV_simpleCNNv3_23.npy" ).squeeze()/1e3
prob_CNNv2   = np.load(path_results + "classPredictions_CV_simpleCNNv2_23.npy" ).squeeze()/1e3
prob_auxCNN  = np.load(path_results + "classPredictions_CV_simpleAuxCNN_23.npy").squeeze()/1e3
prob_wCNN    = np.load(path_results + "classPredictions_CV_wideCNN_23.npy"     ).squeeze()/1e3
prob_wCNNv2  = np.load(path_results + "classPredictions_CV_wideCNNv2_23.npy"   ).squeeze()/1e3
prob_wCNNv3  = np.load(path_results + "classPredictions_CV_wideCNNv3_23.npy"   ).squeeze()/1e3
prob_wauxCNN = np.load(path_results + "classPredictions_CV_wideAuxCNN_23.npy"  ).squeeze()/1e3


print('Rpt Sigmoid    ### acc %.4f ###' % acc_fcn(sigmoid(jet_Rpt), isPU, rwi))
print('Rpt Scaled     ### acc %.4f ###' % acc_fcn(rescale_array(jet_Rpt), isPU, rwi))
print('j0pt Sigmoid   ### acc %.4f ###' % acc_fcn(sigmoid(j0pt), isPU, rwi))
print('j0pt Scaled    ### acc %.4f ###' % acc_fcn(rescale_array(j0pt), isPU, rwi))
print('Baseline NN    ### acc %.4f ###' % acc_fcn(prob_base, isPU, rwi))
print('simple CNN     ### acc %.4f ###' % acc_fcn(prob_CNN, isPU, rwi))
print('simple CNN r   ### acc %.4f ###' % acc_fcn(prob_CNNr, isPU, rwi))
print('simple CNN v2  ### acc %.4f ###' % acc_fcn(prob_CNNv2, isPU, rwi))
print('simple CNN v3  ### acc %.4f ###' % acc_fcn(prob_CNNv3, isPU, rwi))
print('simple Aux CNN ### acc %.4f ###' % acc_fcn(prob_auxCNN, isPU, rwi))
print('wide CNN       ### acc %.4f ###' % acc_fcn(prob_wCNN, isPU, rwi))
print('wide CNN v2    ### acc %.4f ###' % acc_fcn(prob_wCNNv2, isPU, rwi))
print('wide CNN v3    ### acc %.4f ###' % acc_fcn(prob_wCNNv3, isPU, rwi))
print('wide Aux CNN   ### acc %.4f ###' % acc_fcn(prob_wauxCNN, isPU, rwi))

jet_Rpt_PU      = jet_Rpt[isPU == True ]
jet_Rpt_HS      = jet_Rpt[isPU == False]
j0pt_PU         = j0pt[isPU == True ]
j0pt_HS         = j0pt[isPU == False]
prob_base_PU    = prob_base[isPU == True ]
prob_base_HS    = prob_base[isPU == False]
prob_CNN_PU     = prob_CNN[isPU == True ]
prob_CNN_HS     = prob_CNN[isPU == False]
prob_CNNr_PU    = prob_CNNr[isPU == True ]
prob_CNNr_HS    = prob_CNNr[isPU == False]
prob_CNNv2_PU   = prob_CNNv2[isPU == True ]
prob_CNNv2_HS   = prob_CNNv2[isPU == False]
prob_CNNv3_PU   = prob_CNNv3[isPU == True ]
prob_CNNv3_HS   = prob_CNNv3[isPU == False]
prob_auxCNN_PU  = prob_auxCNN[isPU == True ]
prob_auxCNN_HS  = prob_auxCNN[isPU == False]
prob_wCNN_PU    = prob_wCNN[isPU == True ]
prob_wCNN_HS    = prob_wCNN[isPU == False]
prob_wCNNv2_PU  = prob_wCNNv2[isPU == True ]
prob_wCNNv2_HS  = prob_wCNNv2[isPU == False]
prob_wauxCNN_PU = prob_wauxCNN[isPU == True ]
prob_wauxCNN_HS = prob_wauxCNN[isPU == False]
rew_PU          = rew[isPU == True ]
rew_HS          = rew[isPU == False]


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(jet_Rpt_PU, bins=100, range=(0,3), alpha=0.5, label='PU')
ax1.hist(jet_Rpt_HS, bins=100, range=(0,3), alpha=0.5, label='HS')
ax1.legend(loc='upper right')
ax1.set_xlabel("Jet Rpt")
ax1.set_ylabel("Frequency")
plt.suptitle('PU vs NON PU ')

fig1 = plt.figure()
ax11 = fig1.add_subplot(111)
ax11.hist(j0pt_PU, bins=100, range=(0,20), alpha=0.5, label='PU')
ax11.hist(j0pt_HS, bins=100, range=(0,20), alpha=0.5, label='HS')
ax11.legend(loc='upper right')
ax11.set_xlabel("Jet j0pT")
ax11.set_ylabel("Frequency")
plt.suptitle('PU vs NON PU ')

fig11 = plt.figure()
ax111 = fig11.add_subplot(111)
def hist_fcn(ax, var, label):
    ax.hist(var, bins=100, range=(0, 1), histtype='step', label=label)
hist_fcn(ax111, prob_base_PU,    'PU baseline NN')
hist_fcn(ax111, prob_base_HS,    'HS baseline NN')
hist_fcn(ax111, prob_CNNr_PU,    'PU seq CNN')
hist_fcn(ax111, prob_CNNr_HS,    'HS seq CNN')
hist_fcn(ax111, prob_auxCNN_PU,  'PU Aux CNN')
hist_fcn(ax111, prob_auxCNN_HS,  'HS Aux CNN')
hist_fcn(ax111, prob_wCNNv2_PU,  'PU wide CNN')
hist_fcn(ax111, prob_wCNNv2_HS,  'HS wide CNN')
hist_fcn(ax111, prob_wauxCNN_PU, 'PU wide Aux CNN')
hist_fcn(ax111, prob_wauxCNN_HS, 'HS wide Aux CNN')
ax111.legend(loc='upper center')
ax111.set_xlabel("Jet Prob")
ax111.set_ylabel("Frequency")
plt.suptitle('PU vs NON PU ')


effPU = []
effHS = []
effPU_pt = []
effHS_pt = []
for w in np.linspace(0.0, 3.0, num=5000):
    effHS.append(rew_HS[jet_Rpt_HS >= w].sum() / np.float32(rew_HS.sum()))
    effPU.append(rew_PU[jet_Rpt_PU >= w].sum() / np.float32(rew_PU.sum()))

for pt in np.linspace(8.18, 15.13, num=5000):
    effHS_pt.append(rew_HS[j0pt_HS >= pt].sum() / np.float32(rew_HS.sum()))
    effPU_pt.append(rew_PU[j0pt_PU >= pt].sum() / np.float32(rew_PU.sum()))

# from previous runs
effHS_base    = np.load(path_results + "effHS_baselineNN_23.npy")
effPU_base    = np.load(path_results + "effPU_baselineNN_23.npy")
effHS_CNN     = np.load(path_results + "effHS_simpleCNN_23.npy")
effPU_CNN     = np.load(path_results + "effPU_simpleCNN_23.npy")
effHS_CNNr    = np.load(path_results + "effHS_simpleCNNr_23.npy")
effPU_CNNr    = np.load(path_results + "effPU_simpleCNNr_23.npy")
effHS_CNNv2   = np.load(path_results + "effHS_simpleCNNv2_23.npy")
effPU_CNNv2   = np.load(path_results + "effPU_simpleCNNv2_23.npy")
effHS_CNNv3   = np.load(path_results + "effHS_simpleCNNv3_23.npy")
effPU_CNNv3   = np.load(path_results + "effPU_simpleCNNv3_23.npy")
effHS_auxCNN  = np.load(path_results + "effHS_simpleAuxCNN_23.npy")
effPU_auxCNN  = np.load(path_results + "effPU_simpleAuxCNN_23.npy")
effHS_wCNN    = np.load(path_results + "effHS_wideCNN_23.npy")
effPU_wCNN    = np.load(path_results + "effPU_wideCNN_23.npy")
effHS_wCNNv2  = np.load(path_results + "effHS_wideCNNv2_23.npy")
effPU_wCNNv2  = np.load(path_results + "effPU_wideCNNv2_23.npy")
effHS_wCNNv3  = np.load(path_results + "effHS_wideCNNv3_23.npy")
effPU_wCNNv3  = np.load(path_results + "effPU_wideCNNv3_23.npy")
effHS_wauxCNN = np.load(path_results + "effHS_wideAuxCNN_23.npy")
effPU_wauxCNN = np.load(path_results + "effPU_wideAuxCNN_23.npy")


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

l1,  = ax2.plot(effHS_pt,      effPU_pt,      '-', label='j0pt ROC')
l2,  = ax2.plot(effHS,         effPU,         '-', label='Rpt ROC')
l3,  = ax2.plot(effHS_base,    effPU_base,    '-', label='Baseline NN ROC')
l4,  = ax2.plot(effHS_CNN,     effPU_CNN,     ':', label='CNN ROC')
l41, = ax2.plot(effHS_CNNv2,   effPU_CNNv2,   ':', label='CNN v2 ROC')
l42, = ax2.plot(effHS_CNNv3,   effPU_CNNv3,   ':', label='CNN v3 ROC')
l5,  = ax2.plot(effHS_auxCNN,  effPU_auxCNN,  ':', label='Aux CNN ROC')
l6,  = ax2.plot(effHS_wCNN,    effPU_wCNN,    ':', label='wide CNN ROC')
l61, = ax2.plot(effHS_wCNNv2,  effPU_wCNNv2,  ':', label='wide v2 CNN ROC')
l7,  = ax2.plot(effHS_wauxCNN, effPU_wauxCNN, ':', label='wide Aux CNN ROC')
ax2.legend(loc='lower right')
ax2.set_xlabel("effeciency Hard Scatter")
ax2.set_ylabel("effecieny Pile Up")
plt.yscale('log', nonposy='clip')
plt.suptitle('ROC curves')

# new runs
effHS_feature    = np.load(path_results + 'feature.py_0_2017_11_18_15_59_0.688491968558_effHS.npy')
effPU_feature    = np.load(path_results + 'feature.py_0_2017_11_18_15_59_0.688491968558_effPU.npy')
effHS_logistic   = np.load(path_results + 'logistic.py_0_2017_11_18_10_05_0.691116214722_effHS.npy')
effPU_logistic   = np.load(path_results + 'logistic.py_0_2017_11_18_10_05_0.691116214722_effPU.npy')
effHS_simple     = np.load(path_results + 'simple.py_0_2017_11_18_13_52_0.707155809418_effHS.npy')
effPU_simple     = np.load(path_results + 'simple.py_0_2017_11_18_13_52_0.707155809418_effPU.npy')
effHS_simple_aux = np.load(path_results + 'simple_aux.py_0_2017_11_18_15_02_0.712382166975_effHS.npy')
effPU_simple_aux = np.load(path_results + 'simple_aux.py_0_2017_11_18_15_02_0.712382166975_effPU.npy')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)

ax3.plot(effHS_pt,         effPU_pt,         ':', label='j0pt ROC')
ax3.plot(effHS,            effPU,            ':', label='Rpt ROC')
ax3.plot(effHS_base,       effPU_base,       ':', label='Baseline NN ROC')
ax3.plot(effHS_CNNr,       effPU_CNNr,       ':', label='Pseudo CNN ROC')
ax3.plot(effHS_CNNv2,      effPU_CNNv2,      ':', label='Seq CNN ROC')
ax3.plot(effHS_CNNv3,      effPU_CNNv3,      ':', label='Seq downscaling CNN ROC')
ax3.plot(effHS_auxCNN,     effPU_auxCNN,     ':', label='Seq Aux CNN ROC')
ax3.plot(effHS_wCNN,       effPU_wCNN,       ':', label='wide CNN ROC')
ax3.plot(effHS_wCNNv2,     effPU_wCNNv2,     ':', label='Wide CNN ROC')
ax3.plot(effHS_wauxCNN,    effPU_wauxCNN,    ':', label='Wide Aux CNN ROC')
ax3.plot(effHS_feature,    effPU_feature,    '-', label='feature')
ax3.plot(effHS_logistic,   effPU_logistic,   '-', label='logistic')
ax3.plot(effHS_simple,     effPU_simple,     '-', label='simple')
ax3.plot(effHS_simple_aux, effPU_simple_aux, '-', label='simple_aux')

ax3.legend(loc='upper left')
ax3.set_xlabel("effeciency Hard Scatter")
ax3.set_ylabel("effecieny Pile Up")
ax3.set_xlim([0.8,1])
ax3.set_ylim([0.4,1.05])
plt.suptitle('ROC curves zoomed into region of interest')

plt.show()
