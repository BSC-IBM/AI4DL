import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_timeseries(timeseries:            pd.DataFrame or np.ndarray,
                    column_names:          list=[],
                    fig_size:              tuple=(20,10),
                    color:                 str="red", 
                    font_size:             int=20,         # size of title
                    label_size:            int=15,         # size of the x and y numbers
                    dist_between_subplots: float=3.0,      # distance between each plot
                    y_lim:                 tuple=(0,1),  
                    y_ticks_delta:         float=0.25,
                    file_to_save:          str="",         # save if this is not an empy string
                   ):
    '''
    Function for plotting a multidimensional timeseries passed as dataframe or numpy array.

    The function makes a plot for each column of the dataframe or the numpy array.

    - If `timeseries` is a `np.ndarray` and `column_names` is passed then title of timeseries[:,j] is column_names[j]. 

    - If `timeseries` is a dataframe it will use as title for each column it's own colname. If `column_names` is provided
      it will use by default what is in `column_names` instead of the original column name in the dataframe.

    '''

    assert isinstance(timeseries, pd.DataFrame) or isinstance(timeseries, np.ndarray),\
           "\ntimeseries is not pd.DataFrame or np.ndarray.\n\ttimeseries is {}".format(type(timeseries))

    if isinstance(timeseries, np.ndarray):
        if timeseries.ndim == 1:
            timeseries = np.array([timeseries]).T

        if column_names:
            assert len(column_names) == timeseries.shape[1], \
                "\ntimeseries has {} columns, column_names has {} columns. They should match.".format(len(column_names),
                                                                                                      timeseries.shape[1])

            n_plots = len(column_names)
        else:
            n_plots = timeseries.shape[1]
            column_names = ["Column " + str(j) for j in range(n_plots)]

    
    if isinstance(timeseries, pd.DataFrame):
        column_names = timeseries.columns if not column_names else column_names
        n_plots = len(column_names)
        
    fig, axs = plt.subplots(nrows=n_plots, ncols=1, figsize=fig_size)
    
    if type(axs) is not np.ndarray:
        axs = np.array([axs])

    if isinstance(timeseries, pd.DataFrame):
        for j,axis in enumerate(axs):
            #axs.set_ylabel("unit column j")
            axis.plot(timeseries[column_names[j]])
            axis.set_title(column_names[j] , color=color, fontsize=font_size)
            axis.tick_params(labelsize=label_size) # change number sizes in x and y axis
            axis.set_yticks(np.arange(0, 1.04, y_ticks_delta))

    if isinstance(timeseries, np.ndarray):
        for j,axis in enumerate(axs):
            #axs.set_ylabel("unit column j")
            axis.plot(timeseries[:, j])
            axis.set_title(column_names[j], color=color, fontsize=font_size)
            axis.tick_params(labelsize=label_size) # change number sizes in x and y axis
            axis.set_yticks(np.arange(0, 1.04, y_ticks_delta))
            axis.set_ylim(y_lim)

    fig.tight_layout(pad=2., w_pad=0., h_pad=dist_between_subplots) 

    if len(file_to_save)>0:
        plt.savefig(file_to_save)


def plot_timeseries_overlaped(timeseries_1:            pd.DataFrame or np.ndarray,
                              timeseries_2:            pd.DataFrame or np.ndarray,
                              column_names:          list=[],
                              fig_size:              tuple=(20,10),
                              color:                 str="red", 
                              font_size:             int=20,         # size of title
                              label_size:            int=15,         # size of the x and y numbers
                              dist_between_subplots: float=3.0,      # distance between each plot
                              y_lim:                 tuple=(0,1),  
                              y_ticks_delta:         float=0.25,
                              file_to_save:          str="",         # save if this is not an empy string
                             ):
    '''
    Function for plotting a multidimensional timeseries passed as dataframe or numpy array.

    The function makes a plot for each column of the dataframe or the numpy array.

    - If `timeseries` is a `np.ndarray` and `column_names` is passed then title of timeseries[:,j] is column_names[j]. 

    - If `timeseries` is a dataframe it will use as title for each column it's own colname. If `column_names` is provided
      it will use by default what is in `column_names` instead of the original column name in the dataframe.

    '''
    assert type(timeseries_1) == type(timeseries_2), \
           "\ntimeseries_1 has type {}, timeseries_2 has type {}. Types should match".format(type(timeseries_1),
                                                                                             type(timeseries_2))
    
    assert isinstance(timeseries_1, pd.DataFrame) or isinstance(timeseries_1, np.ndarray),\
           "\ntimeseries_1 is not pd.DataFrame or np.ndarray.\n\ttimeseries is {}".format(type(timeseries_1))

    assert isinstance(timeseries_2, pd.DataFrame) or isinstance(timeseries_2, np.ndarray),\
           "\ntimeseries_2 is not pd.DataFrame or np.ndarray.\n\ttimeseries is {}".format(type(timeseries_2))

        
    if isinstance(timeseries_1, np.ndarray):
        if timeseries_1.ndim == 1:
            timeseries_1 = np.array([timeseries_1]).T
            timeseries_2 = np.array([timeseries_2]).T
            
        if column_names:
            assert len(column_names) == timeseries_1.shape[1], \
                "\ntimeseries has {} columns, column_names has {} columns. They should match.".format(len(column_names),
                                                                                                      timeseries_1.shape[1])

            n_plots = len(column_names)
        else:
            n_plots = timeseries_1.shape[1]
            column_names = ["Column " + str(j) for j in range(n_plots)]

    
    if isinstance(timeseries_1, pd.DataFrame):
        column_names = timeseries_1.columns if not column_names else column_names
        n_plots = len(column_names)
        
        
    fig, axs = plt.subplots(nrows=n_plots, ncols=1, figsize=fig_size)
    
    if type(axs) is not np.ndarray:
        axs = np.array([axs])
        
    if isinstance(timeseries_1, pd.DataFrame):
        for j,axis in enumerate(axs):
            #axs.set_ylabel("unit column j")
            axis.plot(timeseries_1[column_names[j]])
            axis.plot(timeseries_2[column_names[j]], linestyle='dashed')
            axis.set_title(column_names[j] , color=color, fontsize=font_size)
            axis.tick_params(labelsize=label_size) # change number sizes in x and y axis
            axis.set_yticks(np.arange(0, 1.04, y_ticks_delta))

    if isinstance(timeseries_1, np.ndarray):
        for j,axis in enumerate(axs):
            #axs.set_ylabel("unit column j")
            axis.plot(timeseries_1[:, j])
            axis.plot(timeseries_2[:, j], linestyle='dashed')
            axis.set_title(column_names[j], color=color, fontsize=font_size)
            axis.tick_params(labelsize=label_size) # change number sizes in x and y axis
            axis.set_yticks(np.arange(0, 1.04, y_ticks_delta))
            axis.set_ylim(y_lim)

    fig.tight_layout(pad=2., w_pad=0., h_pad=dist_between_subplots) 

    if len(file_to_save)>0:
        plt.savefig(file_to_save)

def build_duplicated_workloads(workloads: list):
    """
    build target_workloads that "stretch" the original data in the time axis
    """
    assert isinstance(workloads[0], pd.DataFrame)
    target_workloads = []

    for workload in workloads:
        target_workloads.append(np.concatenate((workload, workload)))

    return target_workloads



def build_streched_workloads(workloads: list):
    """
    build target_workloads that "stretch" the original data in the time axis
    """
    assert isinstance(workloads[0], pd.DataFrame)
    target_workloads = []

    for workload in workloads:
        w = np.array(workload.as_matrix(), dtype="float32")

        streched_workload = np.zeros((2 * w.shape[0], w.shape[1]))
        n_time_steps = w.shape[0]
       
        for t in range(n_time_steps):
            k = t*2
            streched_workload[k:k + 2, :] = w[[t], :]

        target_workloads.append(streched_workload)

    return target_workloads



def build_compressed_workloads(workloads: list, compression_rate=4, add_noise=False):
    """
    build target_workloads that compresses the original data in the time axis.
    Every compression_rate values are aggregated into a single value. 
    For example, if compression_rate is 2 then the compressed timeseries
    should have half the length of the original one. 
    
    Examples:
    
    In the previous examples x
    Given x = [x1,x2,x3,x4,x5,x6,x7,x8] if  compression_rate = 2 then the 
    output of should be y = [y1,y2,y3,y4] where y1 = mean(x1,x2), 
    y2 = mean(x3,x4), y3 = mean(x5,x6), y4 = mean(x7, x8). 

    If x is not divisible by compression_rate then the last value of y will be 
    mean across the "left values". In the previous example if compression_rate=3
    then y1 = mean(x1,x2,x3), y2 = mean(x4,x5,x6), y3 = mean(x7,x8).
    
    If add_noise = True then a bit of gaussian noise is added at every time step.
    """
    assert isinstance(workloads[0], pd.DataFrame)
    target_workloads = []

    for workload in workloads:
        w_ = np.array(workload.as_matrix(), dtype="float32")
        new_length = w_.shape[0]//compression_rate 
        compressed_workload = np.zeros((new_length, w_.shape[1]))
       
        j = 0
        for t in range(new_length):
            # write the mean of the 4 feature values across compression_rate timesteps
            #import pdb; pdb.set_trace()
            if add_noise:
                sigma_t = w_[j:(j+compression_rate), :].std(axis=0)
                mean_t =  w_[j:(j+compression_rate), :].mean(axis=0) 
                compressed_workload[t, :] = 0.5*sigma_t * np.random.randn(len(mean_t)) + mean_t
            else:
                compressed_workload[t, :] = w_[j:(j+compression_rate), :].mean(axis=0)
            
            j = j + compression_rate

        target_workloads.append(compressed_workload)

    return target_workloads

def read_triplets(path: str):
    """
    Path: path to the directory of the dataset.
    It returns a list L of triplets of df, where `L[i][0]` and `L[i][1]` are 
    two executions in isolation and `L[i][2]` corresponds to  `L[i][0] + L[i][1]` executed
    concurrently.

    ATENTION: THIS FUNCTION ASSUMES THAT A ".Directory" exists in path (dolphin stuff)
    """
    file_names = os.listdir(path)
    file_names.sort()
    file_names = file_names[1:] #remove .directory (dolphin stuff)
    datasets = []
    for i in range(0, len(file_names),3):
        a  = pd.read_csv(path + file_names[i])
        b  = pd.read_csv(path + file_names[i+1])
        ab = pd.read_csv(path + file_names[i+2])
        datasets.append([a, b, ab])
    return datasets


def plot_barplots_triplets(datasets: list, resource: str, function:str):
    """
    datasets: list of dataset triplets.
    resource: column to plot.
    fuction: function to apply to the column. Supported functions are: max, mean 
    """
    for i in range(0, len(datasets)):
        a  = datasets[i][0]
        b  = datasets[i][1]
        ab = datasets[i][2]
        if (function == "max"):
            df_plot = pd.DataFrame({resource + "_A": [], resource + "_B":[], resource + "_AB":[] })
            df_plot = df_plot.append({resource + "_A":a[resource].max(),resource + "_B":b[resource].max(),\
                                      resource + "_AB":ab[resource].max()}, ignore_index=True)
        if (function == "mean"):
            df_plot = pd.DataFrame({resource + "_A": [], resource + "_B":[], resource + "_AB":[] })
            df_plot = df_plot.append({resource + "_A":a[resource].mean(),resource + "_B":b[resource].mean(),\
                                      resource + "_AB":ab[resource].mean()}, ignore_index=True)
        df_plot.plot(kind="bar")


def hstack_with_padding(timeseries_a: np.ndarray, timeseries_b: np.ndarray):
    """
    Function for concatenating two timeseries. It creates a new timeseries 
    with the number of features beein sum of the number of columns in the arrays
    of the input.
    To make this possible the function appends new rows containing 0's to the shortest
    timeseries so that both match in number of timesteps.
    """

    assert isinstance(timeseries_a, np.ndarray) and isinstance(timeseries_b, np.ndarray),\
         "\nBoth inputs must be np.ndarray objects."

    assert (timeseries_a.ndim==2 and timeseries_b.ndim==2),\
         "\nOne of the timeseries has not ndim = 2. Both inputs have to be 2D matrices."

    n_time_steps_a, n_features_a = timeseries_a.shape
    n_time_steps_b, n_features_b = timeseries_b.shape
    
    if  n_time_steps_b == n_time_steps_a:
        return np.hstack( (timeseries_a, timeseries_b))

    n_difference = abs(n_time_steps_a - n_time_steps_b)

    if n_time_steps_a < n_time_steps_b:
        timeseries_a = np.vstack((timeseries_a, np.zeros((n_difference, n_features_a))))

    if n_time_steps_b < n_time_steps_a:
        timeseries_b = np.vstack((timeseries_b, np.zeros((n_difference, n_features_b))))

    return np.hstack((timeseries_a, timeseries_b))



def plot_timeseries_with_phases(timeseries:            pd.DataFrame or np.ndarray,
                                phases:                pd.DataFrame or np.ndarray,
                                num_to_color:          dict=None,
                                y_phase_values:        list=[],        # y axis values at the phase descriptor
                                column_names:          list=[],
                                fig_size:              tuple=(20,10),
                                color:                 str="red", 
                                font_size:             int=20,         # size of title
                                label_size:            int=15,         # size of the x and y numbers
                                dist_between_subplots: float=3.0,      # distance between each plot
                                y_lim:                 tuple=(0,1),  
                                y_ticks_delta:         float=0.25,
                                file_to_save:          str="",         # save if this is not an empy string
                               	merge_phases:          bool=False
                               ):
    '''
    Function for plotting a multidimensional timeseries passed as dataframe or numpy array.

    The function makes a plot for each column of the dataframe or the numpy array.

    - If `timeseries` is a `np.ndarray` and `column_names` is passed then title of timeseries[:,j] is column_names[j]. 

    - If `timeseries` is a dataframe it will use as title for each column it's own colname. If `column_names` is provided
      it will use by default what is in `column_names` instead of the original column name in the dataframe.

    '''

    assert isinstance(timeseries, pd.DataFrame) or isinstance(timeseries, np.ndarray),\
           "\ntimeseries is not pd.DataFrame or np.ndarray.\n\ttimeseries is {}".format(type(timeseries))

    assert isinstance(phases, pd.DataFrame) or isinstance(phases, np.ndarray),\
           "\nphases is not pd.DataFrame or np.ndarray.\n\tphases is is {}".format(type(timeseries))


    if isinstance(timeseries, np.ndarray):
        if timeseries.ndim == 1:
            timeseries = np.array([timeseries]).T

        if column_names:
            assert len(column_names) == timeseries.shape[1], \
                "\ntimeseries has {} columns, column_names has {} columns. They should match.".format(len(column_names),
                                                                                                      timeseries.shape[1])

            n_plots = len(column_names)
        else:
            n_plots = timeseries.shape[1]
            column_names = ["Column " + str(j) for j in range(n_plots)]

    
    if isinstance(timeseries, pd.DataFrame):
        column_names = timeseries.columns if not column_names else column_names
        n_plots = len(column_names)

    if num_to_color ==None:
        #num_to_color = {0:'black', 1:'red', 2:'green', 3:'blue', 4:'cyan'}
        unique_colors = np.unique(phases)
        num_to_color = {c: c/np.max(unique_colors) for c in unique_colors}
        

    # the n_plots + 1 is because we add the first row showing the phases
    fig, axs = plt.subplots(nrows=n_plots + 1, ncols=1, figsize=fig_size)
    x_positions = list(range(len(phases)))

    if type(axs) is not np.ndarray:
        axs = np.array([axs])

    if isinstance(timeseries, pd.DataFrame):
        axs[0].bar(x_positions, phases+1, color=[num_to_color[num] for num in phases],width=1)
        axs[0].set_title("phases", color=color, fontsize=font_size)
        axs[0].tick_params(labelsize=label_size) # change number sizes in x and y axis
        axs[0].set_yticks(y_phase_values)

        for j,axis in enumerate(axs[1:]):
            #axs.set_ylabel("unit column j")
            if merge_phases:
                phases_placeholder=np.ones(len(phases))*10
                axis.bar(x_positions, phases_placeholder, color=[num_to_color[num] for num in phases],
                        width=1,alpha=0.1, bottom=-4)

            axis.plot(timeseries[column_names[j]])
            axis.set_title(column_names[j] , color=color, fontsize=font_size)
            axis.tick_params(labelsize=label_size) # change number sizes in x and y axis
            axis.set_yticks(np.arange(y_lim[0], y_lim[1], y_ticks_delta))

    if isinstance(timeseries, np.ndarray):
        axs[0].bar(x_positions, phases+1, color=[num_to_color[num] for num in phases],width=1)
        axs[0].set_title("phases" , color=color, fontsize=font_size)
        axs[0].tick_params(labelsize=label_size) # change number sizes in x and y axis
        axs[0].set_yticks(y_phase_values)

        for j,axis in enumerate(axs[1:]):
            #axs.set_ylabel("unit column j")
            if merge_phases:
                phases_placeholder=np.ones(len(phases))*10
                axis.bar(x_positions, phases_placeholder, color=[num_to_color[num] for num in phases],
                        width=1,alpha=0.1, bottom=-4)

            axis.plot(timeseries[:, j])

            axis.set_title(column_names[j], color=color, fontsize=font_size)
            axis.tick_params(labelsize=label_size) # change number sizes in x and y axis
            axis.set_yticks(np.arange(y_lim[0], y_lim[1], y_ticks_delta))
            axis.set_ylim(y_lim)

    fig.tight_layout(pad=2., w_pad=0., h_pad=dist_between_subplots) 

    if len(file_to_save)>0:
        plt.savefig(file_to_save)



def plot_timeseries_overlaped_with_phases(timeseries_1:          pd.DataFrame or np.ndarray,
                                          timeseries_2:          pd.DataFrame or np.ndarray,
                                          phases_1:              pd.DataFrame or np.ndarray,
                                          phases_2:              pd.DataFrame or np.ndarray,
                                          num_to_color:          dict,
                                          column_names:          list=[],
                                          y_phase_values:        list=[],
                                          fig_size:              tuple=(20,10),
                                          color:                 str="red", 
                                          font_size:             int=20,         # size of title
                                          label_size:            int=15,         # size of the x and y numbers
                                          dist_between_subplots: float=3.0,      # distance between each plot
                                          y_lim:                 tuple=(0,1),  
                                          y_ticks_delta:         float=0.25,
                                          file_to_save:          str="",         # save if this is not an empy string
                                         ):
    '''
    Function for plotting a multidimensional timeseries passed as dataframe or numpy array.

    The function makes a plot for each column of the dataframe or the numpy array.

    - If `timeseries` is a `np.ndarray` and `column_names` is passed then title of timeseries[:,j] is column_names[j]. 

    - If `timeseries` is a dataframe it will use as title for each column it's own colname. If `column_names` is provided
      it will use by default what is in `column_names` instead of the original column name in the dataframe.

    '''
    assert type(timeseries_1) == type(timeseries_2), \
           "\ntimeseries_1 has type {}, timeseries_2 has type {}. Types should match".format(type(timeseries_1),
                                                                                             type(timeseries_2))
    
    assert isinstance(timeseries_1, pd.DataFrame) or isinstance(timeseries_1, np.ndarray),\
           "\ntimeseries_1 is not pd.DataFrame or np.ndarray.\n\ttimeseries is {}".format(type(timeseries_1))

    assert isinstance(timeseries_2, pd.DataFrame) or isinstance(timeseries_2, np.ndarray),\
           "\ntimeseries_2 is not pd.DataFrame or np.ndarray.\n\ttimeseries is {}".format(type(timeseries_2))

        
    if isinstance(timeseries_1, np.ndarray):
        if timeseries_1.ndim == 1:
            timeseries_1 = np.array([timeseries_1]).T
            timeseries_2 = np.array([timeseries_2]).T
            
        if column_names:
            assert len(column_names) == timeseries_1.shape[1], \
                "\ntimeseries has {} columns, column_names has {} columns. They should match.".format(len(column_names),
                                                                                                      timeseries_1.shape[1])

            n_plots = len(column_names)
        else:
            n_plots = timeseries_1.shape[1]
            column_names = ["Column " + str(j) for j in range(n_plots)]

    
    if isinstance(timeseries_1, pd.DataFrame):
        column_names = timeseries_1.columns if not column_names else column_names
        n_plots = len(column_names)
        
        
    fig, axs = plt.subplots(nrows=n_plots+2, ncols=1, figsize=fig_size)
    x_positions = list(range(len(phases_1)))

    if type(axs) is not np.ndarray:
        axs = np.array([axs])
        
    if isinstance(timeseries_1, pd.DataFrame):
        axs[0].bar(x_positions, phases_1+1, color=[num_to_color[num] for num in phases_1],width=1)
        axs[0].set_title("phases from true trace", color=color, fontsize=font_size)
        axs[0].tick_params(labelsize=label_size) # change number sizes in x and y axis
        axs[0].set_yticks(y_phase_values)

        axs[1].bar(x_positions, phases_2+1, color=[num_to_color[num] for num in phases_2],width=1)
        axs[1].set_title("phases from generated trace", color=color, fontsize=font_size)
        axs[1].tick_params(labelsize=label_size) # change number sizes in x and y axis
        axs[1].set_yticks(y_phase_values)


        for j,axis in enumerate(axs[2:]):
            #axs.set_ylabel("unit column j")
            axis.plot(timeseries_1[column_names[j]])
            axis.plot(timeseries_2[column_names[j]], linestyle='dashed')
            axis.set_title(column_names[j] , color=color, fontsize=font_size)
            axis.tick_params(labelsize=label_size) # change number sizes in x and y axis
            axis.set_yticks(np.arange(0, 1.04, y_ticks_delta))

    if isinstance(timeseries_1, np.ndarray):
        axs[0].bar(x_positions, phases_1+1, color=[num_to_color[num] for num in phases_1],width=1)
        axs[0].set_title("phases from true trace", color=color, fontsize=font_size)
        axs[0].tick_params(labelsize=label_size) # change number sizes in x and y axis
        axs[0].set_yticks(y_phase_values)

        axs[1].bar(x_positions, phases_2+1, color=[num_to_color[num] for num in phases_2],width=1)
        axs[1].set_title("phases from generated trace", color=color, fontsize=font_size)
        axs[1].tick_params(labelsize=label_size) # change number sizes in x and y axis
        axs[1].set_yticks(y_phase_values)

        for j,axis in enumerate(axs[2:]):
            #axs.set_ylabel("unit column j")
            axis.plot(timeseries_1[:, j])
            axis.plot(timeseries_2[:, j], linestyle='dashed')
            axis.set_title(column_names[j], color=color, fontsize=font_size)
            axis.tick_params(labelsize=label_size) # change number sizes in x and y axis
            axis.set_yticks(np.arange(0, 1.04, y_ticks_delta))
            axis.set_ylim(y_lim)

    fig.tight_layout(pad=2., w_pad=0., h_pad=dist_between_subplots) 

    if len(file_to_save)>0:
        plt.savefig(file_to_save)
        
