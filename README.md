# AI4DL

AI4DL - Deep-Learning Containter Auto-Scaling framework

## Introduction

AI4DL is a framework developed by BSC and IBM to analyze resource consumption traces from Cloud containers and discover and detect recurrent behaviors, focusing on Deep Learning workloads. AI4DL uses unsupervised learning methods like CRBMs and clustering to discover patterns in time-series from CPU/Memory/... traces, providing information about the workload behavior along time to be leveraged for resource planification. Here we present the implementation of the AI4DL framework in python, to fit a model for phase discovery and detection from traces in tabular format.


The principles and experiments for AI4DL can be found in ***AI4DL Mining Behaviors of Deep Learning Workloads for Resource Management*** from the *HotCloud'20: 12th USENIX Workshop on Hot Topics in Cloud Computing*, by Josep L. Berral (BSC), Chen Wang (IBM) and Alaa Youssef (IBM). Find the document and presentation [HERE](https://www.usenix.org/conference/hotcloud20/presentation/berral)

The potential actions towards expansion and discovery of new phases and behaviors, also prediction of phases towards preemptive placement, can be found in ***Proactive Container Auto-scaling for Cloud Native Machine Learning Services*** from the *The IEEE International Conference on Cloud Computing (CLOUD)*, by David Buchaca (BSC), Josep L. Berral (BSC), Chen Wang (IBM) and Alaa Youssef (IBM). Find the document and presentation HERE (TBP)

Abstract:

> The more we know about the resource usage patterns of workloads, the better we can allocate resources. Here we present a methodology to discover resource usage behaviors of containers training Deep Learning (DL) models. From monitoring, we can observe repeating patterns and similitude of resource usage among containers training different DL models. The repeating patterns observed can be leveraged by the scheduler or the resource autoscaler to reduce resource fragmentation and overall resource utilization in a dedicated DL cluster. Specifically, our approach combines Conditional Restricted Boltzmann Machines (CRBMs) and clustering techniques to discover common sequences of behaviors (phases) of containers running the DL training workloads in clusters providing IBM Deep Learning Services. By studying the resource usage pattern at each phase and the typical sequences of phases among different containers, we discover a reduced set of prototypical executions representing the majority of executions. We use statistical information from each phase to refine resource provisioning by dynamically tuning the amount of resource each container requires at each phase. Evaluation of our method shows that by leveraging typical resource usage patterns, we can auto-scale containers to reduce CPU and Memory allocation by 30% compared to statistics based reactive policies, which is close to having a-priori knowledge of resource usage while fulfilling resource demand over 95% of the time.

## Package Content

The AI4DL package contains the basic code to train and use the AI4Dl tool from CPU/Memory/etc... traces.

### Basic elements
 
 * **AI4DL.py**: Class containing the basic functions and elements for *training* and *predicting* traces using the AI4DL mechanism
 * **crbm_tools.py**: Library containing the CRBM mechanism in Python. It is used by AI4DL class 
 * **timeseries.py**: Library containing functions to plot the traces with and without the phases
 
### An example

 * **example.py** : An example loading a traces data-set, training the AI4DL (complete crbm + clustering pipeline) and printing an example trace with found phases 
 
### Sample Notebooks

 * **1 - detecting phases**: Here's a sample notebook with a trained model and sample traces
 * **2 - expanding phases**: Here's a sample notebook on updating the set of phases with new traces

## Using the Tool

### Load the Package

To load the package you only need to import the element **AI4DL**, present in the AI4DL folder.

```
import AI4DL
from AI4DL import AI4DL
```

It will automatically import the required packages and elements (either from the AI4DL folder, like the CRBM and libraries) and dependencies. AI4DL uses the following python packages:

```
copy, inspect, joblib, json, matplotlib, numexpr, numpy, os, pandas, pickle, random, sklearn, sys, timeit, time
```

### AI4DL package Functions

#### Constructor:
- **AI4DL** ( ) : Builder of the AI4DL Class. Creates an "AI4DL" python object.
 
#### Load and Transform Data:

- **LoadTrainingDataset** ( *data_file*, *features*, *exec_id* ) : 	Reads, Scales and Transforms a dataset (reading a Training-Set). This function is used to load the training dataset *data_file*, and automatically trains the scaler (to scale the selected *features*), and separate executions by an *exec_id*. The AI4DL object stores the "List of Timeseries" read and transformed from the training-set, also the scaler.
- **GetTrainingDataset** ( ) : Returns the "List of Timeseries" loaded from the training-set.
- **GetScaler** ( ) : Returns the scaler trained from the training-set (e.g. to denormalize data values after normalization).
- **TransformData** ( *data_file* ) : Reads and Transforms a dataset *data_file*, scaled by the trained scaler (e.g. reading a Test-Set). Returns a "List of Timeseries" (each execution a timeseries, all values scaled using the trained scaler).

#### Training / Loading Model:
- **TrainModel** ( *n_clusters*, *n_hidden*, *n_history*, *learning_rate*, *n_epochs*, *crbm_save*, *kmeans_save*, *scaler_save*, *seed* ) : Trains (fits) the full model CRBM + k-Means using the loaded training-set. The AI4DL object stores the trained pipeline, also can be stored in disk files.
- **LoadModel** ( *crbm_save*, *kmeans_save*, *scaler_save*, *crbm_sub*, *kmeans_sub*, *scaler_sub* ) : Loads the model (CRBM + k-Means + scaler, or parts of it) from disk files.

#### Predict / Evaluate:
- **Predict** ( *list_of_timeseries* ) : Predicts a "List of Timeseries". Returns a "List of Sequences (of phases)".
- **Evaluate** ( ) : Predicts the stored "List of Timeseries" loaded from the training-set.
- **LoadAndPredict** ( *data_file* ) : Loads a file *data_file*, transforms its data into a "List of Timeseries" and finally predicts it.

#### Print Traces:
- **PrintTrace** ( *list_of_timeseries*, *predicted_seq_phases*, *n_exec*, *cpu_idx*, *mem_idx*, *palette*, *col_names*, *f_name* ) : Print a selected trace *n_exec* from a "List of TimeSeries" with its detected phases from the "Predicted Sequence Phases", to screen or a file *f_name*, indicating the features *cpu_idx* and *mem_idx* to show used CPU and Memory.
- **PrintVarAnalysis** ( *list_of_timeseries*, *pred_seq_phases*, *cpu_idx*, *mem_idx*, *palette*, *col_names*, *f_name* ) : Print the variance analysis plots of a "List of TimeSeries" and its "Predicted Sequence Phases", indicating the features *cpu_idx* and *mem_idx* to show used CPU and Memory.
