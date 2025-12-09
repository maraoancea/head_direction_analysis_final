# The function generateINandTARGETOUT_CuevaWei2019 generates the inputs and target outputs for training a recurrent neural network 
# to estimate head direction, between 0 and 360 degrees, by integrating an angular velocity input, as in Cueva et al. 2019 "Emergence of functional and structural properties of the head direction system by optimization of recurrent neural networks"
# CJ Cueva 8.21.2024

import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch


   


###############################################################################
#%% generate inputs and target outputs for training a recurrent neural network 
def generateINandTARGETOUT_CuevaWei2019(task_input_dict={}):    
    #--------------------------------------------------------------------------
    #                 INPUTS to generateINandTARGETOUT
    #--------------------------------------------------------------------------
    # n_input:     number of inputs into the RNN
    # n_output:    number of outputs from the RNN
    # n_T:         number of timesteps in a trial
    # n_trials:    number of trials 
    # random_seed: seed for random number generator
    # angle0_duration:  angle0 input (sin(angle0) and cos(angle0)) is nonzero for angle0duration timesteps at beginning of trial
    # angular_momentum: angular_velocity(t) = angular_sd*randn + angular_momentum * angular_velocity(t-1)
    # angular_sd:       at each timestep the new angular_velocity is a gaussian random variable with mean 0 and standard deviation angular_sd
    
    #--------------------------------------------------------------------------
    #                OUTPUTS from generateINandTARGETOUT
    #--------------------------------------------------------------------------
    # IN:           n_trials x n_T x n_input tensor
    # TARGETOUT:    n_trials x n_T x n_output tensor
    # output_mask:  n_trials x n_T x n_output tensor, elements 0(timepoint does not contribute to cost function), 1(timepoint contributes to cost function)
    # angle:        n_trials x n_T x n_output tensor
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------
    #               IN, TARGETOUT, output_mask, angle
    #--------------------------------------------------------------------------
    # there are 3 inputs, 
    # 1) angular velocity, angle to integrate
    # 2) sin(angle0)
    # 3) cos(angle0)
    # there are 2 outputs, sin(integrated-angle) and cos(integrated-angle)
    #--------------------------------------------------------------------------
    n_input = task_input_dict['n_input']
    n_output = task_input_dict['n_output']
    n_T = task_input_dict['n_T']
    n_trials = task_input_dict['n_trials']
    random_seed = task_input_dict['random_seed']
    angle0_duration = task_input_dict['angle0_duration']
    angular_momentum = task_input_dict['angular_momentum']
    angular_sd = task_input_dict['angular_sd']
    assert n_input==3, "Error: input should be 3 numbers. 2 for the intial angle and 1 for the angle to integrate."
    np.random.seed(random_seed)# set random seed for reproducible results 
    
    IN = np.zeros((n_trials, n_T, n_input))
    TARGETOUT = np.zeros((n_trials, n_T, n_output))
    output_mask = np.ones((n_trials, n_T, n_output))
    angle0 = 2*np.pi*np.random.rand(n_trials)# (n_trials,) array, initial angle in radians
    angle = -700*np.ones((n_trials, n_T))# integrated angular_velocity, target output of RNN (technically target output is sin(angle), cos(angle))
    for itrial in range(0,n_trials):# 0, 1, 2, ... n_trials-1
        # timesteps for important events in the trial, timestep are 1,2,3,...n_T, these ultimately need to be translated to indices 0,1,2,...n_T-1
        if itrial < n_trials/2: tstart0 = np.random.randint(np.round(n_T/3)+1); tend0 = np.minimum(n_T, tstart0 + np.random.randint(np.round(2*n_T/3)+1))# set angular_velocity for middle third of trial to 0
        
        angular_velocity = np.zeros(n_T)# input to RNN
        tstart = angle0_duration + 1# timestep to start inputting nonzero angular_velocity
        angle[itrial,0:angle0_duration] = angle0[itrial]
        
        # all timesteps from 1,2,3,...n_T are translated to indices 0,1,2,...n_T-1
        for i in range(angle0_duration,n_T):# start inputting nonzero angular_velocity
            if itrial < (n_trials/2):
                if i>=(tstart0-1) and i<=(tend0-1):
                    angular_velocity[i] = 0
                else:
                    angular_velocity[i] = angular_sd*np.random.randn() + angular_momentum * angular_velocity[i-1]
            else:
                angular_velocity[i] = angular_sd*np.random.randn() + angular_momentum * angular_velocity[i-1]
                
            anglenew = angle[itrial,i-1] + angular_velocity[i]
            angle[itrial,i] = np.mod(anglenew,2*np.pi)# integrated angularvelocity, target output of RNN (technically target output is sin(angle), cos(angle))
        
        IN[itrial,:,0] = angular_velocity# angle in radians
        IN[itrial,0:angle0_duration,1] = np.sin(angle0[itrial])
        IN[itrial,0:angle0_duration,2] = np.cos(angle0[itrial])
        
    TARGETOUT[:,:,0] = np.sin(angle)# (n_trials, n_T) array
    TARGETOUT[:,:,1] = np.cos(angle)# (n_trials, n_T) array
    output_mask[:,0:angle0_duration,:] = 0# elements 0(timepoint does not contribute to cost function), 1(timepoint contributes to cost function)

    # convert to pytorch tensors 
    dtype = torch.float32
    IN = torch.tensor(IN, dtype=dtype); TARGETOUT = torch.tensor(TARGETOUT, dtype=dtype); output_mask = torch.tensor(output_mask, dtype=dtype); 
    task_output_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials, 'angle_radians':angle}
    return IN, TARGETOUT, output_mask, task_output_dict






#%%############################################################################
#                       test generateINandTARGETOUT
###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import os
    figure_dir = os.path.dirname(__file__)# return the folder path for the file we're currently executing

    np.random.seed(123); torch.manual_seed(123)# set random seed for reproducible results
    n_input = 3# there are 3 inputs: angular velocity/angle to integrate, sin(angle0), cos(angle0)
    n_output = 2# there are 2 outputs: sin(integrated-angle) and cos(integrated-angle)
    n_T = 500
    n_trials = 100
    random_seed = 11
    angle0_duration = 10# 10
    angular_momentum = 0.8# 0.8
    ANGULARSPEED = 1# 0.1, 1, 3 for Figures 5c, 5f, 5i of "Emergence of functional and structural properties of the head direction system by optimization of recurrent neural networks"
    angular_sd = ANGULARSPEED * 0.03
    T = np.arange(0,n_T)# (n_T,)

   
    task_input_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials, 'random_seed':random_seed, 'angle0_duration':angle0_duration, 'angular_momentum':angular_momentum, 'angular_sd':angular_sd}
    IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT_CuevaWei2019(task_input_dict)   
    angle_radians = task_output_dict['angle_radians']# (n_trials, n_T) array
    angle_degrees = task_output_dict['angle_radians']*180/np.pi# (n_trials, n_T) array
    
    
    plt.figure()# inputs and target outputs for RNN
    fontsize = 12
    for itrial in range(n_trials):
        plt.clf()
        #----colormaps----
        cool = cm.get_cmap('cool', n_input)
        colormap_input = cool(range(n_input))# (n_input, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
        copper = cm.get_cmap('copper_r', n_output)
        colormap_output = copper(range(n_output))# (n_output, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
        #----plot all inputs and outputs----
        ilabelsinlegend = np.round(np.linspace(0,n_input-1,5,endpoint=True))# if there are many inputs only label 5 of them in the legend
        for i in range(n_input):# 0,1,2,...n_input-1
            if np.isin(i,ilabelsinlegend):
                plt.plot(T, IN[itrial,:,i], c=colormap_input[i,:], linewidth=3, label=f'Input {i+1}')# label inputs 1,2,3,..
            else:
                plt.plot(T, IN[itrial,:,i], c=colormap_input[i,:], linewidth=3)# don't label these inputs
        ilabelsinlegend = np.round(np.linspace(0,n_output-1,5,endpoint=True))# if there are many outputs only label 5 of them in the legend
        for i in range(n_output):# 0,1,2,...n_output-1
            if np.isin(i,ilabelsinlegend):
                plt.plot(T[output_mask[itrial,:,i]==1], TARGETOUT[itrial,output_mask[itrial,:,i]==1,i], c=colormap_output[i,:], linewidth=3, label=f'Output {i+1}')# label outputs 1,2,3,..
            else:
                plt.plot(T[output_mask[itrial,:,i]==1], TARGETOUT[itrial,output_mask[itrial,:,i]==1,i], c=colormap_output[i,:], linewidth=3)# don't label these outputs
        #---------------------
        plt.legend(loc='best', fontsize=fontsize)
        plt.xlabel('Time', fontsize=fontsize)
        plt.title(f'Trial {itrial}', fontsize=fontsize)
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.xlim(left=0)
        #plt.savefig('%s/generateINandTARGETOUT_trial%g.pdf'%(figure_dir,itrial), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.show()
        input("Press Enter to continue...")# pause the program until the user presses Enter
     


    plt.figure()# input and target output, with integrated-angle in degrees
    fontsize = 12
    for itrial in range(n_trials):
    #for itrial in range(9,10):
        plt.clf()
        #----colormaps----
        cool = cm.get_cmap('cool', n_input)
        colormap_input = cool(range(n_input))# (n_input, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
        copper = cm.get_cmap('copper_r', n_output)
        colormap_output = copper(range(n_output))# (n_output, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
        #----plot all inputs and outputs----
        for i in range(n_input):# 0,1,2,...n_input-1
            plt.plot(T, IN[itrial,:,i], c=colormap_input[i,:], linewidth=3, label=f'Input {i+1}')# label inputs 1,2,3,..
        for i in range(n_output):# 0,1,2,...n_output-1
            plt.plot(T[output_mask[itrial,:,i]==1], TARGETOUT[itrial,output_mask[itrial,:,i]==1,i], c=colormap_output[i,:], linewidth=3, label=f'Output {i+1}')# label outputs 1,2,3,..
        plt.plot(T, angle_degrees[itrial,:], 'g.', markersize=20, label='Integrated angle (degrees)')# integrated angle in degrees
        plt.legend(loc='best', fontsize=fontsize)
        plt.xlabel('Time', fontsize=fontsize)
        plt.title(f'Trial {itrial}', fontsize=fontsize)
        plt.xlim(left=0); plt.ylim(bottom=0, top=360)
        plt.yticks(ticks=[0 ,180, 360], labels=[0, 180, 360])
        #ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.show()
        input("Press Enter to continue...")# pause the program until the user presses Enter


    
