# The following code trains a recurrent neural network (RNN) on a task with inputs and desired outputs defined in the function generateINandTARGETOUT.py
# To train a RNN on a different task, simply adjust the inputs and desired outputs by using a different version of generateINandTARGETOUT
# CJ Cueva 8.23.2024

import os
root_dir = os.path.dirname(__file__) + '/'# return the folder path for the file we're currently executing
os.chdir(root_dir)# print(f'current working direction is {os.getcwd()}')
import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from parula import parula_map# from file import variable

from model_architectures import CTRNN# from file import function
from generateINandTARGETOUT_CuevaWei2019 import generateINandTARGETOUT_CuevaWei2019# from file import function
from compute_normalized_error import compute_normalized_error# from file import function



#%%###########################################################################
task_name = 'CuevaWei2019'# task_name determines which version of generateINandTARGETOUT is used, task_name is also used to name the folder where the RNN parameters are stored
optimizer_name = 'AdamW'# 'Adam' or 'AdamW', optimizer_name determines which optimizer to use when updating the RNN parameters
learning_rate = 1e-3# learning rate for optimizer
CLIP_GRADIENT_NORM = 1# 0 or 1, if CLIP_GRADIENT_NORM = 1 then clip the norm of the gradient
max_gradient_norm = 10# If CLIP_GRADIENT_NORM = 1 then the norm of the gradient is clipped to have a maximum value of max_gradient_norm. This is only used if CLIP_GRADIENT_NORM = 1
n_parameter_updates = 5000# number of parameter updates
model_class = 'CTRNN'# continuous time recurrent neural network (CTRNN). The model architecture is specified in model_architectures.py
activation_function = 'retanh'# Options for the activation function are listed in model_architectures.py
n_recurrent = 100# number of units in RNN
regularization_activityL2 = 0.01# L2 regularization on h - "firing rate" of units, larger regularization_activityL2 = more regularization = smaller absolute firing rates
activity_noise_std = 0.1# add noise to firing rates of RNN model with standard deviation activity_noise_std
random_seed = 1# set random seed for reproducible results
np.random.seed(random_seed); torch.manual_seed(random_seed)# set random seed for reproducible results 



#%%############################################################################
#         specify the inputs and desired output used to train the RNN
if task_name=='CuevaWei2019':
    generateINandTARGETOUT = generateINandTARGETOUT_CuevaWei2019
    n_input = 3# there are 3 inputs: angular velocity/angle to integrate, sin(angle0), cos(angle0)
    n_output = 2# there are 2 outputs: sin(integrated-angle) and cos(integrated-angle)
    n_T = 500
    n_trials = 100
    angle0_duration = 10# 10
    angular_momentum = 0.8# 0.8
    ANGULARSPEED = 1# 0.1, 1, 3 for Figures 5c, 5f, 5i of "Emergence of functional and structural properties of the head direction system by optimization of recurrent neural networks"
    angular_sd = ANGULARSPEED * 0.03
    T = np.arange(0,n_T)# (n_T,)
    task_input_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials, 'random_seed':random_seed, 'angle0_duration':angle0_duration, 'angular_momentum':angular_momentum, 'angular_sd':angular_sd}
  

#%%##############################################################################
if model_class == 'CTRNN':
    # continuous time recurrent neural network
    # Tau * dah/dt = -ah + Wahh @ f(ah) + Wahx @ x + bah
    #
    # ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)􏰩    
    # h[t] = f(ah[t]) + activity_noise[t], if t > 0
    # y[t] = Wyh @ h[t] + by  output
    #
    # constants that are not learned: dt, Tau, activity_noise
    # parameters to be learned: Wahh, Wahx, Wyh, bah, by, ah0(optional)
    
    # initialize network
    ah0 = torch.zeros(n_recurrent)
    bah = torch.zeros(n_recurrent)
    bah = 0.1 + 0.01*torch.randn(n_recurrent)# initialize with small positive values so there are fewer "dead" units
    by = torch.zeros(n_output)
    
    
    np.random.seed(random_seed+2); torch.manual_seed(random_seed+2)# set random seed for reproducible results 
    Wahx = torch.randn(n_recurrent,n_input) / np.sqrt(n_input)
    Wahh = 1.5 * torch.randn(n_recurrent,n_recurrent) / np.sqrt(n_recurrent)
    Wahh[np.eye(n_recurrent,n_recurrent)==1] = 0# initialize recurrent connectivity matrix to have no self connections
    Wyh = torch.zeros(n_output,n_recurrent)
    
    
    LEARN_OUTPUTWEIGHT = True# Wyh
    LEARN_OUTPUTBIAS = False# by
    model = CTRNN(n_input, n_recurrent, n_output, activation_function=activation_function, ah0=ah0, LEARN_ah0=True, Wahx=Wahx, Wahh=Wahh, Wyh=Wyh, bah=bah, by=by, LEARN_OUTPUTWEIGHT=LEARN_OUTPUTWEIGHT, LEARN_OUTPUTBIAS=LEARN_OUTPUTBIAS); model_class = 'CTRNN'; 


#---------------check number of learned parameters---------------
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)# model.parameters include those defined in __init__ even if they are not used in forward pass
assert np.allclose(model.n_parameters, n_parameters), "Number of learned parameters don't match!"
#import sys; sys.exit()# stop script at current line

#---------------make folder to store files---------------
figure_dir = root_dir + f'/store_trained_models/{task_name}_{model_class}_{activation_function}_nin{n_input}nout{n_output}nrecurrent{n_recurrent}_{n_parameter_updates}parameterupdates_regularizationactivityL2{regularization_activityL2}_activitynoisestd{activity_noise_std}_rng{random_seed}'
if not os.path.exists(figure_dir):# if folder doesn't exist then make folder
    os.makedirs(figure_dir)

#---------------save pset_saveparameters---------------
# During training sample pset_saveparameters more densely during the first 1500 parameter updates, e.g. np.round(np.linspace(200,1500,num=50,endpoint=True)) because this is where most of the gains occur. This will help if I want to compare models at some performance cutoff and actually find models with a performance near this cutoff.
pset_saveparameters = np.unique(np.concatenate((np.arange(0,6), np.round(np.linspace(25,150,num=20,endpoint=True)), np.round(np.linspace(0,n_parameter_updates,num=20,endpoint=True))  ))).astype(int)# save parameters when parameter update p is a member of pset_saveparameters, save as int so we can use elements to load model parameters: for example, model_parameter_update211.pth versus model_parameter_update211.0.pth   
pset_saveparameters = np.delete(pset_saveparameters, pset_saveparameters>n_parameter_updates)
np.save(f'{figure_dir}/pset_saveparameters.npy', pset_saveparameters)#pset_saveparameters = np.load(f'{figure_dir}/pset_saveparameters.npy')

#---------------save entire model, not just model parameters---------------
torch.save(model, f'{figure_dir}/model.pth')# model = torch.load(f'{figure_dir}/model.pth')# save entire model, not just model parameters
# This save/load process uses the most intuitive syntax and involves the least amount of code. 
# Saving a model in this way will save the entire module using Python’s pickle module. 
# The disadvantage of this approach is that the serialized data is bound to the specific classes and the exact directory structure used when the model is saved. The reason for this is because pickle does not save the model class itself. Rather, it saves a path to the file containing the class, which is used during load time. Because of this, your code can break in various ways when used in other projects or after refactors.
# https://pytorch.org/tutorials/beginner/saving_loading_models.html

#---------------plot initial eigenvalue spectrum---------------
plt.figure()# initial eigenvalue spectrum of recurrent weight matrix  
if model_class=='CTRNN': W = model.fc_h2ah.weight.detach().numpy() 
plt.clf()
eigVal = np.linalg.eigvals(W)
plt.plot(eigVal.real, eigVal.imag, 'k.', markersize=10)
plt.xlabel('real(eig(W))')
plt.ylabel('imag(eig(W))')
plt.title(model_class)
plt.axis('equal')# plt.axis('scaled') 
plt.savefig('%s/eigenvaluesW_beforelearning_%s.pdf'%(figure_dir,model_class.replace(" ", "")), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
#import sys; sys.exit()# stop script at current line


#%%############################################################################
#                                train network
np.random.seed(random_seed+3); torch.manual_seed(random_seed+3)# set random seed for reproducible results 
if optimizer_name=='Adam': optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)# learning_rate = 1e-3 default
if optimizer_name=='AdamW': optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)# learning_rate = 1e-3 default
error_store = -700*np.ones(n_parameter_updates+1)# error_store[0] is the error before any parameter updates have been made, error_store[j] is the error after j parameter updates
errormain_store = -700*np.ones(n_parameter_updates+1)# error_store[0] is the error before any parameter updates have been made, error_store[j] is the error after j parameter updates
error_activityL2_store = -700*np.ones(n_parameter_updates+1)# error_store[0] is the error before any parameter updates have been made, error_store[j] is the error after j parameter updates
gradient_norm = -700*np.ones(n_parameter_updates+1)# gradient_norm[0] is norm of the gradient before any parameter updates have been made, gradient_norm[j] is the norm of the gradient after j parameter updates
figure_suffix = ''
for p in range(n_parameter_updates+1):# 0, 1, 2, ... n_parameter_updates
    activity_noise = activity_noise_std*torch.randn(n_trials, n_T, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
    IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
    
    model_output_forwardpass = model({'input':IN, 'activity_noise':activity_noise})# model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
    output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']# (n_trials, n_T, n_output/n_recurrent)
    errormain = torch.sum((output[output_mask==1] - TARGETOUT[output_mask==1])**2) / torch.sum(output_mask==1)# output_mask: n_trials x n_T x n_output tensor, elements 0(timepoint does not contribute to this term in the error function), 1(timepoint contributes to this term in the error function) 
    error_activityL2 = regularization_activityL2*torch.mean(activity.flatten()**2) 
    error = errormain + error_activityL2 
    
    if error_activityL2 >= errormain/2: regularization_activityL2 = regularization_activityL2*2/3# bound error_activityL2 from getting too large
    error_store[p] = error.item(); errormain_store[p] = errormain.item(); error_activityL2_store[p] = error_activityL2.item()# error.item() gets the scalar value held in error
    
    
    if np.isin(p,pset_saveparameters):# model_parameter_update{p}.pth stores the parameters after p parameter updates, model_parameter_update0.pth are the initial parameters
        print(f'{p} parameter updates: error = {error.item():.4g}')
        torch.save({'model_state_dict':model.state_dict(), 'figure_suffix':figure_suffix, 'regularization_activityL2':regularization_activityL2}, figure_dir + f'/model_parameter_update{p}.pth')# save the trained model’s learned parameters
    
    if p==10 or np.isin(p, np.linspace(1,n_parameter_updates,num=4,endpoint=True).astype(int)):# only plot and save 5 times during training
        plt.figure()# training error vs number of parameter updates
        fontsize = 12
        plt.semilogy(np.arange(0,p+1), errormain_store[0:p+1], 'k-', linewidth=1, label=f'{model_class} main {errormain.item():.4g}')
        plt.semilogy(np.arange(0,p+1), error_activityL2_store[0:p+1], 'k--', linewidth=1, label=f'{model_class} hL2 {error_activityL2.item():.4g}')
        plt.xlabel('Number of parameter updates', fontsize=fontsize)
        plt.ylabel('Error during training', fontsize=fontsize)
        plt.legend()
        plt.title(f"{p} parameter updates, error = {error_store[p]:.4g}\n"
                  f"errormain = {errormain_store[p]:.4g}, error_activityL2 = {error_activityL2_store[p]:.4g}", fontsize=fontsize)
        plt.xlim(left=0)
        #plt.ylim(bottom=0) 
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.savefig('%s/error_trainingerrorVSparameterupdates%s_.pdf'%(figure_dir,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        plt.pause(0.05)# https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
    
    
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the Tensors it will update (which are the learnable weights of the model)
    optimizer.zero_grad()
    
    # Backward pass: compute gradient of the error with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    error.backward()
     
    # clip the norm of the gradient
    if 'CLIP_GRADIENT_NORM':
        figure_suffix = f'_maxgradientnorm{max_gradient_norm}'
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    
    # Calling the step function on an Optimizer makes an update to its parameters
    if p<=n_parameter_updates-1:# only update the parameters n_parameter_updates times. note that the parameters are first updated when p is 0
        optimizer.step()# parameters that have a gradient of zero may still be updated due to weight decay or momentum (if previous gradients were nonzero)
    
 
    gradient = []# store all gradients
    for param in model.parameters():# model.parameters include those defined in __init__ even if they are not used in forward pass
        if param.requires_grad is True:# model.parameters include those defined in __init__ even if param.requires_grad is False (in this case param.grad is None)
            gradient.append(param.grad.detach().flatten().numpy())
    gradient = np.concatenate(gradient)# gradient = torch.cat(gradient)
    assert np.allclose(gradient.size,model.n_parameters), "size of gradient and number of learned parameters don't match!"
    gradient_norm[p] = np.sqrt(np.sum(gradient**2))
    #print(f'{p} parameter updates: gradient norm = {gradient_norm[p]:.4g}')
    
    
    

        
#import sys; sys.exit()# stop script at current line    
#%%############################################################################
#                        plot training figures
###############################################################################
# normalized error, if the RNN output is constant for each n_output and this constant value is the mean of the target output (each n_output can be a different constant) then errornormalized = 1
# outputforerror = output(output_mask==1)
# TARGETOUTforerror = TARGETOUT(output_mask==1)
# errornormalized = ((outputforerror(:) - TARGETOUTforerror(:))' @ (outputforerror(:) - TARGETOUTforerror(:))) / ((mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))' @ (mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))), normalized error when using outputs for which output_mask = 1
errornormalized = compute_normalized_error(TARGETOUT.detach().numpy(), output.detach().numpy(), output_mask.detach().numpy())# all inputs are arrays with shape (n_trials, n_T, n_output)


plt.figure()# norm of the gradient vs number of parameter updates
fontsize = 12
plt.plot(np.arange(0,n_parameter_updates+1), gradient_norm, 'k-', label=model_class)
plt.xlabel('Number of parameter updates', fontsize=fontsize)
plt.ylabel('Gradient norm during training', fontsize=fontsize)
plt.legend()
plt.title(f"{model_class}\n{n_parameter_updates} parameter updates, error = {error_store[-1]:.4g}, normalized error = {errornormalized:.4g}\nmax = {np.max(gradient_norm):.2g}, min = {np.min(gradient_norm):.2g}, median = {np.median(gradient_norm):.2g}", fontsize=fontsize)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.tick_params(axis='both', labelsize=fontsize)
plt.savefig('%s/gradient_norm%s.pdf'%(figure_dir,figure_suffix), bbox_inches='tight')


plt.figure()# training error vs number of parameter updates
fontsize = 12
plt.plot(np.arange(0,n_parameter_updates+1), error_store, 'k-', linewidth=1, label=f'{model_class} {error_store[n_parameter_updates]:.4g}')
plt.xlabel('Number of parameter updates', fontsize=fontsize)
plt.ylabel('Mean squared error during training', fontsize=fontsize)
plt.legend()
plt.title('%s\n%.4g parameter updates, error = %.4g, normalized error = %.4g\nerror i%g = %.4g, i%g = %.4g, i%g = %.4g'\
          %(model_class,n_parameter_updates,error_store[-1],errornormalized,5,error_store[5],round(n_parameter_updates/2),error_store[round(n_parameter_updates/2)],n_parameter_updates,error_store[n_parameter_updates]), fontsize=fontsize)
plt.xlim(left=0)
plt.ylim(bottom=0)  
plt.tick_params(axis='both', labelsize=fontsize)
plt.savefig('%s/error_trainingerrorVSparameterupdates%s.pdf'%(figure_dir,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff


plt.figure()# training error vs number of parameter updates, semilogy
fontsize = 12
plt.semilogy(np.arange(0,n_parameter_updates+1), error_store, 'k-', linewidth=1, label=f'{model_class} {error_store[n_parameter_updates]:.4g}')
plt.xlabel('Number of parameter updates', fontsize=fontsize)
plt.ylabel('Mean squared error during training', fontsize=fontsize)
plt.legend()
plt.title('%s\n%.4g parameter updates, error = %.4g, normalized error = %.4g\nerror i%g = %.4g, i%g = %.4g, i%g = %.4g'\
          %(model_class,n_parameter_updates,error_store[-1],errornormalized,5,error_store[5],round(n_parameter_updates/2),error_store[round(n_parameter_updates/2)],n_parameter_updates,error_store[n_parameter_updates]), fontsize=fontsize)
plt.xlim(left=0); 
plt.tick_params(axis='both', labelsize=fontsize)
plt.savefig('%s/error_trainingerrorVSparameterupdates_semilogy%s.pdf'%(figure_dir,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff

  
plt.figure()# final eigenvalue spectrum of recurrent weight matrix 
fontsize = 12
if model_class=='CTRNN': W = model.fc_h2ah.weight.detach().numpy(); 
plt.clf()
eigVal = np.linalg.eigvals(W)
plt.plot(eigVal.real, eigVal.imag, 'k.', markersize=10)
plt.xlabel('real(eig(W))', fontsize=fontsize)
plt.ylabel('imag(eig(W))', fontsize=fontsize)
plt.title(f"{model_class}\n{n_parameter_updates} parameter updates, error = {error_store[-1]:.4g}, normalized error = {errornormalized:.4g}", fontsize=fontsize)
plt.axis('equal')# plt.axis('scaled')
plt.tick_params(axis='both', labelsize=fontsize)
plt.savefig('%s/eigenvaluesW_%gparameterupdates_%s%s.pdf'%(figure_dir,n_parameter_updates,model_class.replace(" ", ""),figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff




#%%############################################################################
#                            Test data
##############################################################################
np.random.seed(random_seed); torch.manual_seed(random_seed)# set random seed for reproducible results
n_trials_test = 1000
n_T_test = n_T
activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
#activity_noise = activity_noise_std*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
task_input_dict['activity_noise'] = activity_noise
task_input_dict['n_trials'] = n_trials_test# method2: task_input_dict.update({'n_trials':n_trials_test})
IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
#n_T_test = task_output_dict['n_T']
#n_trials_test = task_output_dict['n_trials']
TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy()
# IN:        (n_trials_test, n_T_test, n_input) tensor
# TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
# activity:  (n_trials_test, n_T_test, n_recurrent) tensor


#%%############################################################################
#         normalized test error as a function of training iteration
pset = pset_saveparameters[pset_saveparameters>=0]
#pset = pset_saveparameters[pset_saveparameters>=0]
errornormalized_store = -700*np.ones((pset.shape[0]))
for ip, p in enumerate(pset):
    # load models, first re-create the model structure and then load the state dictionary into it
    checkpoint = torch.load(figure_dir + f'/model_parameter_update{p}.pth'); model.load_state_dict(checkpoint['model_state_dict']); 
    
    model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
    model_output_forwardpass = model(model_input_forwardpass)
    output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']# (n_trials, n_T, n_output/n_recurrent)
    output = output.detach().numpy(); activity = activity.detach().numpy()
    # output:   (n_trials_test, n_T_test, n_output) tensor
    # activity: (n_trials_test, n_T_test, n_recurrent) tensor

    # normalized error, if the RNN output is constant for each n_output and this constant value is the mean of the target output (each n_output can be a different constant) then errornormalized = 1
    # outputforerror = output(output_mask==1)
    # TARGETOUTforerror = TARGETOUT(output_mask==1)
    # errornormalized = ((outputforerror(:) - TARGETOUTforerror(:))' @ (outputforerror(:) - TARGETOUTforerror(:))) / ((mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))' @ (mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))), normalized error when using outputs for which output_mask = 1
    errornormalized = compute_normalized_error(TARGETOUT, output, output_mask)# all inputs are arrays with shape (n_trials, n_T, n_output)
    errornormalized_store[ip] = errornormalized 
np.save(f'{figure_dir}/pset.npy', pset)# pset = np.load(f'{figure_dir}/pset.npy')
np.save(f'{figure_dir}/errornormalized_store.npy', errornormalized_store)# errornormalized_store = np.load(f'{figure_dir}/errornormalized_store.npy')

fig, ax = plt.subplots()# normalized error versus number of parameter updates
fontsize = 14
handle = ax.plot(pset, errornormalized_store, 'k-', linewidth=3)  
ax.legend(handles=handle, labels=[f'{model_class} {errornormalized_store[-1]:.6g}'], loc='best', frameon=True)
ax.set_xlabel('Number of parameter updates', fontsize=fontsize)
ax.set_ylabel('Normalized error', fontsize=fontsize)
imin = np.argmin(errornormalized_store)# index of minimum normalized error
if imin==(pset.size-1):# if the minimum normalized error occurs after the last parameter update only put the error after parameter updates pset[0] and pset[-1] in the title, remember pset[pset.size-1] gives the last element of pset 
    ax.set_title(f'{n_trials_test} test trials, {n_T_test} timesteps, {n_parameter_updates} parameter updates\nError after {pset[0]} parameter updates = {errornormalized_store[0]:.6g}\nError after {pset[imin]} parameter updates = {errornormalized_store[imin]:.6g}', fontsize=fontsize)
else:
    ax.set_title(f'{n_trials_test} test trials, {n_T_test} timesteps, {n_parameter_updates} parameter updates\nError after {pset[0]} parameter updates = {errornormalized_store[0]:.6g}\nError after {pset[imin]} parameter updates = {errornormalized_store[imin]:.6g}\nError after {pset[-1]} parameter updates = {errornormalized_store[-1]:.6g}', fontsize=fontsize)
ax.tick_params(axis='both', labelsize=fontsize)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
fig.savefig('%s/errornormalized_test_nTtest%g%s.pdf'%(figure_dir,n_T_test,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
  
  
#%%############################################################################
#                     load model with lowest test error
# load models, first re-create the model structure and then load the state dictionary into it
n_parameter_updates_model = n_parameter_updates
n_parameter_updates_model = pset[np.argmin(errornormalized_store)]
checkpoint = torch.load(figure_dir + f'/model_parameter_update{n_parameter_updates_model}.pth'); model.load_state_dict(checkpoint['model_state_dict']); figure_suffix = checkpoint['figure_suffix']

#%%############################################################################
#                   compute normalized error on test set
model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
model_output_forwardpass = model(model_input_forwardpass)
output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']# (n_trials, n_T, n_output/n_recurrent)
output = output.detach().numpy(); activity = activity.detach().numpy()
# output:   (n_trials_test, n_T_test, n_output) tensor
# activity: (n_trials_test, n_T_test, n_recurrent) tensor
#loss = loss_fn(output[output_mask==1], TARGETOUT[output_mask==1])


# normalized error, if the RNN output is constant for each n_output and this constant value is the mean of the target output (each n_output can be a different constant) then errornormalized = 1
# outputforerror = output(output_mask==1)
# TARGETOUTforerror = TARGETOUT(output_mask==1)
# errornormalized = ((outputforerror(:) - TARGETOUTforerror(:))' @ (outputforerror(:) - TARGETOUTforerror(:))) / ((mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))' @ (mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))), normalized error when using outputs for which output_mask = 1
errornormalized = compute_normalized_error(TARGETOUT, output, output_mask)# all inputs are arrays with shape (n_trials, n_T, n_output)


fontsize = 12
T = np.arange(0,n_T_test)# (n_T_test,)

plt.figure()# RNN input and output on test trials
#----colormaps----
cool = cm.get_cmap('cool', n_input)
colormap_input = cool(range(n_input))# (n_input, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
#-----------------
n_curves = n_output# number of curves to plot
blacks = cm.get_cmap('Greys', n_curves+3) 
colormap = blacks(range(n_curves+3));# (n_curves+3, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha
# colormap[0,:] = white, first row, check: plt.figure(); plt.plot(np.arange(0,10), -13*np.ones(10), c=colormap[0,:], linewidth=3)
# colormap[-1,:] = black, last row, check: plt.figure(); plt.plot(np.arange(0,10), -13*np.ones(10), c=colormap[-1,:], linewidth=3)
colormap = colormap[3:,:]# (n_curves, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha
colormap_outputtarget = colormap
#-----------------
n_curves = n_output# number of curves to plot
reds = cm.get_cmap('Reds', n_curves+3) 
colormap = reds(range(n_curves+3));# (n_curves+3, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha
# colormap[0,:] = almost white, first row, check: plt.figure(); plt.plot(np.arange(0,10), -13*np.ones(10), c=colormap[0,:], linewidth=3)
# colormap[-1,:] = dark red, last row, check: plt.figure(); plt.plot(np.arange(0,10), -13*np.ones(10), c=colormap[-1,:], linewidth=3)
colormap = colormap[2:,:]# (n_curves+1, 4) array, remove first two rows because they are too light
colormap = colormap[:-1,:]# (n_curves, 4) array, remove last row because it is too similar to black
colormap_outputrnn = colormap
if n_curves==1: colormap_outputrnn = np.array([1, 0, 0, 1])[None,:]# (1, 4) array, red
#-----------------
#for itrial in range(n_trials_test):
for itrial in range(5):  
    plt.clf()
    #----plot single input and output for legend----
    plt.plot(T, IN[itrial,:,0], c=colormap_input[0,:], linewidth=3, label='Input'); 
    plt.plot(T[output_mask[itrial,:,0]==1], TARGETOUT[itrial,output_mask[itrial,:,0]==1,0], '-', c=colormap_outputtarget[-1,:], linewidth=3, label='Output: target'); 
    plt.plot(T, output[itrial,:,0], '--', c=colormap_outputrnn[-1,:], linewidth=3, label='Output: RNN')# for legend
    #----plot all inputs and outputs----
    for i in range(n_input):
        plt.plot(T, IN[itrial,:,i], c=colormap_input[i,:], linewidth=3)
    for i in range(n_output):
        plt.plot(T[output_mask[itrial,:,i]==1], TARGETOUT[itrial,output_mask[itrial,:,i]==1,i], c=colormap_outputtarget[i,:], linewidth=3)# black
        plt.plot(T, output[itrial,:,i], '--', c=colormap_outputrnn[i,:], linewidth=3)# red
    plt.xlabel('Timestep', fontsize=fontsize)
    plt.legend()
    plt.title(f'{model_class}, trial {itrial}\n{n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
    plt.xlim(left=0)
    plt.tick_params(axis='both', labelsize=fontsize)
    #plt.show(); input("Press Enter to continue...")# pause the program until the user presses Enter, https://stackoverflow.com/questions/21875356/saving-a-figure-after-invoking-pyplot-show-results-in-an-empty-file
    plt.savefig('%s/testtrial%g_nTtest%g_%gparameterupdates_%s%s.pdf'%(figure_dir,itrial,n_T_test,n_parameter_updates_model,model_class.replace(" ", ""),figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff

    
plt.figure()# firing of all hidden units on a single trial
#for itrial in range(n_trials_test):
for itrial in range(5): 
    plt.clf()
    plt.plot(T, activity[itrial,:,:])
    plt.xlabel('Timestep', fontsize=fontsize)
    plt.ylabel('Firing of hidden units', fontsize=fontsize)
    #plt.legend()
    plt.title(f'{model_class}, trial {itrial}\n{n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
    plt.xlim(left=0)
    plt.tick_params(axis='both', labelsize=fontsize)
    #plt.show(); input("Press Enter to continue...")# pause the program until the user presses Enter, https://stackoverflow.com/questions/21875356/saving-a-figure-after-invoking-pyplot-show-results-in-an-empty-file
    plt.savefig('%s/testtrial%g_nTtest%g_%gparameterupdates_h_%s%s.pdf'%(figure_dir,itrial,n_T_test,n_parameter_updates_model,model_class.replace(" ", ""),figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff

    
plt.figure()# firing of a single hidden unit across all trials
#for iunit in range(n_recurrent):
#for iunit in range(10): 
for iunit in range(1):
    plt.clf()
    plt.plot(T, activity[:,:,iunit].T)
    plt.xlabel('Timestep', fontsize=fontsize)
    plt.ylabel(f'Firing rate of unit {iunit}', fontsize=fontsize)
    #plt.legend()
    plt.title(f'{model_class}, unit {iunit}\n{n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
    plt.xlim(left=0)
    plt.tick_params(axis='both', labelsize=fontsize)
    #plt.show(); input("Press Enter to continue...")# pause the program until the user presses Enter, https://stackoverflow.com/questions/21875356/saving-a-figure-after-invoking-pyplot-show-results-in-an-empty-file
    plt.savefig('%s/unit%g_nTtest%g_%gparameterupdates_h_%s%s.pdf'%(figure_dir,iunit,n_T_test,n_parameter_updates_model,model_class.replace(" ", ""),figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff







#%%############################################################################
#                            tuning curves
#  plot the firing rates of each unit as a function of the stored angle (integrated input)
#  integrated angle is computed in two ways: 1) based on integrated inputs (ground truth) and 2) based on RNN outputs (internal estimate of heading direction)
###############################################################################
angle_radians = task_output_dict['angle_radians']# (n_trials_test, n_T_test) array

anglegrid = np.linspace(0,360,num=361,endpoint=True)# (361,) array, discretize the angle
anglegrid = np.linspace(0,360,num=100,endpoint=True)# (100,) array, discretize the angle
dangle = anglegrid[1] - anglegrid[0] 
firingrate_store = -700*np.ones((n_recurrent, anglegrid.size-1))
firingrate_integratedanglefromRNNoutput_store = -700*np.ones((n_recurrent, anglegrid.size-1))

for iunit in range(n_recurrent):
    firingrate = np.zeros((anglegrid.size-1))
    firingrate_integratedanglefromRNNoutput = np.zeros((anglegrid.size-1))
    n_visits = np.zeros((anglegrid.size-1))# number of times RNN stores a specific integrated-angle
    n_visits_integratedanglefromRNNoutput = np.zeros((anglegrid.size-1))# number of times RNN stores a specific integrated-angle
    for itrial in range(n_trials_test):
        RNNoutputangle = np.arctan2(output[itrial,:,0], output[itrial,:,1]) * 180/np.pi# (n_T_test,) array, angle between -180 and +180 degrees
        iswitch = (RNNoutputangle < 0); RNNoutputangle[iswitch] = 360 + RNNoutputangle[iswitch]# (n_T_test,) array, angle between 0 and 360 degrees
        for t in range(n_T_test):
            angle = angle_radians[itrial, t] * 180/np.pi# stored angle in degrees, between 0 and 360
            # iposition = find(angle < anglegrid ,1,'first') - 1;
            indices = angle < anglegrid# anglegrid.size array of bool
            indices = np.arange(indices.size)[indices==True]# indices where condition is True
            iposition = indices[0] - 1
            n_visits[iposition] = n_visits[iposition] + 1
            firingrate[iposition] = firingrate[iposition] + activity[itrial,t,iunit]
            
            angle = RNNoutputangle[t]# stored angle in degrees, between 0 and 360
            # iposition = find(angle < anglegrid ,1,'first') - 1;
            indices = angle < anglegrid# anglegrid.size array of bool
            indices = np.arange(indices.size)[indices==True]# indices where condition is True
            iposition = indices[0] - 1
            n_visits_integratedanglefromRNNoutput[iposition] = n_visits_integratedanglefromRNNoutput[iposition] + 1
            firingrate_integratedanglefromRNNoutput[iposition] = firingrate_integratedanglefromRNNoutput[iposition] + activity[itrial,t,iunit]
    #------------------------
    assert np.sum(n_visits) == n_trials_test*n_T_test, "Error: missing visits"
    assert np.sum(n_visits_integratedanglefromRNNoutput) == n_trials_test*n_T_test, "Error: missing visits"
    firingrate = firingrate / n_visits# elementwise division
    firingrate_integratedanglefromRNNoutput = firingrate_integratedanglefromRNNoutput / n_visits_integratedanglefromRNNoutput# elementwise division
    firingrate_store[iunit,: ] = firingrate
    firingrate_integratedanglefromRNNoutput_store[iunit,: ] = firingrate_integratedanglefromRNNoutput
    print(f"tuning curve computed for unit {iunit+1}/{n_recurrent}")
    
    
# figure showing tuning curve for each unit
for iteration in range(2):
    if iteration==0: xplot = anglegrid[0:-1] + dangle/2; yplot = firingrate_store; figure_suffix_plot = ''
    if iteration==1: xplot = anglegrid[0:-1] + dangle/2; yplot = firingrate_integratedanglefromRNNoutput_store; figure_suffix_plot = '_integratedanglefromRNNoutput'
    
    SAMEYAXISLIMITSFORALLUNITS = 1# 0 or 1
    xlabel = f'Integrated angular input ({np.min(anglegrid):.0f} to {np.max(anglegrid):.0f} degrees)'
    if SAMEYAXISLIMITSFORALLUNITS==0: ylabel = 'Activity of unit (different scale for each unit)'
    n_rowsinfigure = np.ceil(np.sqrt(n_recurrent)).astype(int)# number of rows must be a positive integer
    n_columnsinfigure = np.ceil(n_recurrent/n_rowsinfigure).astype(int)# number of columns must be a positive integer
    fig, ax = plt.subplots(n_rowsinfigure, n_columnsinfigure, sharex=True)# tuning curve for each unit
    for irow in range(n_rowsinfigure):# remove ticks and labels
        for icolumn in range(n_columnsinfigure):
            ax[irow, icolumn].set_xticks([]);# ax[irow, icolumn].set_xticklabels([])
            ax[irow, icolumn].set_yticks([]);# ax[irow, icolumn].set_yticklabels([])   
            ax[irow, icolumn].axis('off')
    fontsize = 10
    for iunit in range(n_recurrent):
        irow = np.floor(iunit / n_columnsinfigure).astype(int)# fill row 0, left to right, and then row 1, etc., convert to integer to use as index
        icolumn = np.remainder(iunit, n_columnsinfigure)# fill row 0, left to right, and then row 1, etc.
        ax[irow, icolumn].axis('on')# plot bounding box for units in RNN
        ax[irow, icolumn].plot(xplot, yplot[iunit,:], 'k-', linewidth=3)
        #ax[irow, icolumn].plot(anglegrid[0:-1] + dangle/2, firingrate_smoothed[:,iunit], 'r-', linewidth=0.5)
        #ax[irow, icolumn].plot(anglepreferred[iunit], firingrate_smoothed[indexpreferredangle[iunit],iunit], 'r.', markersize=3)
        #ax[irow, icolumn].set_title(f"{anglepreferred[iunit]:.1f}\N{DEGREE SIGN}", fontsize=3, pad=1)
        
        if SAMEYAXISLIMITSFORALLUNITS: 
            if activation_function=='tanh': ax[irow, icolumn].set_ylim(-1, 1); ylabel = 'Activity of unit (-1 to 1)'
            if activation_function=='retanh': ax[irow, icolumn].set_ylim(0, 1); ylabel = 'Activity of unit (0 to 1)'
            if activation_function=='logistic': ax[irow, icolumn].set_ylim(0, 1); ylabel = 'Activity of unit (0 to 1)'
            if activation_function=='ReLU': ax[irow, icolumn].set_ylim(0, np.max(firingrate_store)); ylabel = 'Activity of unit (0 to max(firingrate))'
            if activation_function=='smoothReLU': ax[irow, icolumn].set_ylim(0, np.max(firingrate_store)); ylabel = 'Activity of unit (0 to max(firingrate))'
    fig.supxlabel(f'{xlabel}', fontsize=fontsize)
    fig.supylabel(f'{ylabel}', fontsize=fontsize)
    fig.suptitle(f'{n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
    if SAMEYAXISLIMITSFORALLUNITS==1: fig.savefig('%s/allunits_meanhVSintegratedangularinput_%gparameterupdates_sameyaxislimitsforallunits%s.pdf'%(figure_dir,n_parameter_updates_model,figure_suffix_plot), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    if SAMEYAXISLIMITSFORALLUNITS==0: fig.savefig('%s/allunits_meanhVSintegratedangularinput_%gparameterupdates_differentyaxislimitsforeachunit%s.pdf'%(figure_dir,n_parameter_updates_model,figure_suffix_plot), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff



#%%##############################################################################
#                            tuning curves
#    plot the firing rates of each unit as a function of integrated angle and angular input
#    integrated angle is computed based on integrated inputs (ground truth) 
###############################################################################
figure_suffix_plot = ''
#angle1grid = np.linspace(0,360,num=361,endpoint=True)# (361,) array, discretize the angle, imshow seems to not show many datapoints when angle1grid is discretized more
angle1grid = np.linspace(0,360,num=100,endpoint=True)# (100,) array, discretize the angle
dangle1 = angle1grid[1] - angle1grid[0] 
minangularinput = np.min(IN[:,:,0].numpy()) * 180/np.pi; maxangularinput = np.max(IN[:,:,0].numpy()) * 180/np.pi
eps = 10**-4# add epsilon to largest element of anglegrid so when angle equals max(anglegrid) then iangle is not empty 
angle2grid = np.linspace(minangularinput, maxangularinput+eps,num=101,endpoint=True)# (101,) array, discretize the angle
dangle2 = angle2grid[1] - angle2grid[0] 

firingrate_store = -700*np.ones((n_recurrent, angle2grid.size-1, angle1grid.size-1))
for iunit in range(n_recurrent):
    firingrate = np.zeros((angle2grid.size-1, angle1grid.size-1))
    n_visits = np.zeros((angle2grid.size-1, angle1grid.size-1))# number of times agent visits bin
    for itrial in range(n_trials_test):
        for t in range(n_T_test):
            angle1 = angle_radians[itrial, t] * 180/np.pi# stored angle in degrees, between 0 and 360
            angle2 = IN[itrial,t,0].numpy() * 180/np.pi# angular input in degrees
            
            # iangle1 = find(angle1 < angle1grid ,1,'first') - 1
            indices = angle1 < angle1grid# anglegrid.size array of bool
            indices = np.arange(indices.size)[indices==True]# indices where condition is True
            iangle1 = indices[0] - 1
            # iangle2 = find(angle2 < angle2grid ,1,'first') - 1
            indices = angle2 < angle2grid# anglegrid.size array of bool
            indices = np.arange(indices.size)[indices==True]# indices where condition is True
            iangle2 = indices[0] - 1
            
            n_visits[iangle2, iangle1] = n_visits[iangle2, iangle1] + 1
            firingrate[iangle2, iangle1] = firingrate[iangle2, iangle1] + activity[itrial,t,iunit]
    #------------------------
    assert np.sum(n_visits) == n_trials_test*n_T_test, "Error: missing visits"
    firingrate = firingrate / n_visits# if numvisits is 0 firingrate is NaN
    firingrate_store[iunit,: ] = firingrate
    print(f"tuning curve computed for unit {iunit+1}/{n_recurrent}")
    

# figure showing tuning curve for each unit
angle2grid_plot = angle2grid[0:-1] + dangle2/2
ikeep_angle2grid_plot = np.logical_and((minangularinput - minangularinput/5) < angle2grid_plot, angle2grid_plot < (maxangularinput - maxangularinput/5))# array of boolean indices where both conditions are true, plot only a subset of angle2grid values to remove some whitespace
SAMEYAXISLIMITSFORALLUNITS = 0
n_rowsinfigure = np.ceil(np.sqrt(n_recurrent)).astype(int)# number of rows must be a positive integer
n_columnsinfigure = np.ceil(n_recurrent/n_rowsinfigure).astype(int)# number of columns must be a positive integer
fig, ax = plt.subplots(n_rowsinfigure, n_columnsinfigure, sharex=True)# firing of all hidden units as a function of angle1 and angle2
for irow in range(n_rowsinfigure):# remove ticks and labels
    for icolumn in range(n_columnsinfigure):
        ax[irow, icolumn].set_xticks([]);# ax[irow, icolumn].set_xticklabels([])
        ax[irow, icolumn].set_yticks([]);# ax[irow, icolumn].set_yticklabels([])  
        ax[irow, icolumn].axis('off')
fontsize = 10
for iunit in range(n_recurrent):  
    irow = np.floor(iunit / n_columnsinfigure).astype(int)# fill row 0, left to right, and then row 1, etc., convert to integer to use as index
    icolumn = np.remainder(iunit, n_columnsinfigure)# fill row 0, left to right, and then row 1, etc.
    ax[irow, icolumn].axis('on')# plot bounding box for units in RNN
    #colormap = cm.get_cmap('cet_gouldian')
    #colormap = 'viridis'
    colormap = parula_map
    if SAMEYAXISLIMITSFORALLUNITS==0:
        ax[irow, icolumn].imshow(firingrate_store[iunit,ikeep_angle2grid_plot,:], cmap=colormap, aspect='auto', origin='lower')
        #ax[irow, icolumn].imshow(firingrate_store[iunit,:,:], cmap=colormap, apsect='equal', origin='upper')
    else: 
        if activation_function=='tanh': ax[irow, icolumn].imshow(firingrate_store[iunit,:,:], cmap=colormap, aspect='auto', origin='lower', vmin=-1, vmax=1)  
        if activation_function=='retanh': ax[irow, icolumn].imshow(firingrate_store[iunit,:,:], cmap=colormap, aspect='auto', origin='lower', vmin=0, vmax=1)        
        if activation_function=='ReLU': ax[irow, icolumn].imshow(firingrate_store[iunit,:,:], cmap=colormap, aspect='auto', origin='lower', vmin=0, vmax=np.max(firingrate_store))      
        if activation_function=='smoothReLU': ax[irow, icolumn].imshow(firingrate_store[iunit,:,:], cmap=colormap, aspect='auto', origin='lower', vmin=0, vmax=np.max(firingrate_store))         
    
fig.supxlabel(f'Integrated angular input ({np.min(angle1grid):.0f} to {np.max(angle1grid):.0f} degrees)', fontsize=fontsize)
fig.supylabel(f'Angular input ({np.min(angle2grid_plot[ikeep_angle2grid_plot]):.0f} to {np.max(angle2grid_plot[ikeep_angle2grid_plot]):.0f} degrees)', fontsize=fontsize)
if SAMEYAXISLIMITSFORALLUNITS==1: fig.suptitle(f'Same scale for all units, {n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
if SAMEYAXISLIMITSFORALLUNITS==0: fig.suptitle(f'Different scale for each unit, {n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
if SAMEYAXISLIMITSFORALLUNITS==1: fig.savefig('%s/allunits_meanhVSangularinputandintegratedangle_%gparameterupdates_samecolorscaleforallunits%s.pdf'%(figure_dir,n_parameter_updates_model,figure_suffix_plot), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
if SAMEYAXISLIMITSFORALLUNITS==0: fig.savefig('%s/allunits_meanhVSangularinputandintegratedangle_%gparameterupdates_differentcolorscaleforeachunit%s.pdf'%(figure_dir,n_parameter_updates_model,figure_suffix_plot), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff


       



