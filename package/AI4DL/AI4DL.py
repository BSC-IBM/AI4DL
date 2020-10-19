
#-------------------------------------------------------------------------------
# General Libraries
#-------------------------------------------------------------------------------

import inspect
import json
import pickle

import random
import numpy as np 
import pandas as pd
import numexpr as ne
import sklearn
from sklearn import preprocessing
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import copy

from matplotlib import pyplot as plt 

from joblib import dump, load

import os,sys,inspect

#-------------------------------------------------------------------------------
# CRBM libraries and tools
#-------------------------------------------------------------------------------

from . import timeseries
from . import crbm_tools
from . crbm_tools import build_slices_from_list_of_arrays as build_slices

#-------------------------------------------------------------------------------
# Classes
#-------------------------------------------------------------------------------

class TimeseriesPipeline():
	
	def __init__(self, steps, memory = None, verbose = False):
		self.steps = steps
		self.memory = memory
		self.verbose = verbose
    
	def predict(self, list_of_timeseries):
		list_of_outputs = []
		for x in list_of_timeseries:
			step_output = x
			for step in self.steps:
				step_output = step.predict(step_output)
			list_of_outputs.append(step_output)
		return list_of_outputs
	
	def getStep (self, idx):
		return self.steps[idx]

class AI4DL():
	
	def __init__(self):
		
		# Configuration vars
		self.exec_id = None
		self.features = None
		
		# Data for Training
		self.list_of_timeseries = None
		
		# Trained Objects
		self.scaler = None
		self.pipeline = None
	
	'''
	Reads, Scales and Transforms a dataset (e.g. reading a Training-Set)
	'''
	def LoadTrainingDataset (self, data_file, features, exec_id):
		
		## Save configuration info
		self.exec_id = exec_id
		self.features = features
		self.n_features = len(features)
		
		list_of_timeseries = []
		
		## Load Dataset
		dataset = pd.read_csv(data_file)
		
		## Create Scaler from df_train
		scaler = StandardScaler()
		scaler.fit(dataset[self.features].values)
		dataset_scaled = copy.deepcopy(dataset)
		dataset_scaled[self.features] = scaler.transform(dataset[self.features].values.astype(np.float))
		
		## Prepare Timeseries
		unique_ids = dataset_scaled[self.exec_id].unique()
		for identifier in unique_ids:
			indices = dataset_scaled[self.exec_id]==identifier
			data = dataset_scaled[indices][self.features].values
			list_of_timeseries.append(data)
		
		self.scaler = scaler		
		self.list_of_timeseries = list_of_timeseries
	
	'''
	Reads and Transforms a dataset, scaled by the Training Data (e.g. reading a Test-Set)
	'''
	def TransformData (self, data_file):
		
		list_of_timeseries = []
		dataset  = pd.read_csv(data_file)
		
		dataset_scaled = copy.deepcopy(dataset)
		dataset_scaled[self.features] = self.scaler.transform(dataset[self.features].values.astype(np.float))
			
		unique_ids = dataset_scaled[self.exec_id].unique()
		for identifier in unique_ids:
			indices = dataset_scaled[self.exec_id]==identifier
			data = dataset_scaled[indices][self.features].values
			list_of_timeseries.append(data)
		
		return list_of_timeseries
	
	'''
	Returns the "List of Timeseries" from the Training-Set
	'''
	def GetTrainingDataset (self):
		return self.list_of_timeseries
	
	'''
	Returns the Scaler
	'''
	def GetScaler (self):
		return self.scaler
	
	'''
	Trains the full model CRBM + k-Means using the Training Data
	'''
	def TrainModel (self, n_clusters, n_hidden, n_history, learning_rate = 0.001, n_epochs = 100, crbm_save = None, kmeans_save = None, scaler_save = None, seed = 123):
		
		## Save configuration info
		self.n_clusters = n_clusters
		self.n_hidden = n_hidden
		self.n_history = n_history
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs
		
		## Train CRBM
		X_slices = build_slices(self.list_of_timeseries, self.n_history, self.n_features)
		crbm = crbm_tools.CRBM(n_vis = self.n_features, n_hid = self.n_hidden, n_his = self.n_history, learning_rate = self.learning_rate, n_epochs = self.n_epochs)
		crbm.fit(X_slices)
		
		## Train K-Means
		list_of_activations = []
		for l in self.list_of_timeseries:
			list_of_activations.append(crbm.predict(l))
		
		X_activations = np.vstack(list_of_activations)
		kmeans = sklearn.cluster.KMeans(n_clusters = n_clusters, random_state = seed)
		kmeans.fit(X_activations)
		
		## Save Sub_Models in files
		if not crbm_save is None:
			crbm.save("./", crbm_save)
		
		if not kmeans_save is None:
			dump(kmeans, kmeans_save)
		
		if not scaler_save is None:
			dump(self.scaler, scaler_save)
		
		## Storing the models and Pipeline
		steps = (crbm, kmeans)
		self.pipeline = TimeseriesPipeline(steps)
	
	'''
	Loads the model CRBM + k-Means (or parts of it) from files
	'''
	def LoadModel (self, crbm_save, kmeans_save, scaler_save, crbm_sub = None, kmeans_sub = None, scaler_sub = None):
		
		crbm = crbm_tools.CRBM(1,1,1)
		if not crbm_save is None:
			crbm = crbm.load(crbm_save)
		else:
			crbm = crbm_sub
		
		if not kmeans_save is None:
			kmeans = load(kmeans_save)
		else:
			kmeans = kmeans_sub
		
		if not scaler_save is None:
			self.scaler = load(scaler_save)
		else:
			self.scaler = scaler_sub		
		
		## Storing the models and Pipeline
		steps = (crbm, kmeans)
		self.pipeline = TimeseriesPipeline(steps)
	
	'''
	Predicts a "List of Timeseries"
	'''
	def Predict (self, list_of_timeseries):
		return self.pipeline.predict(list_of_timeseries)
	
	'''
	Predicts the Training-Set "List of Timeseries"
	'''
	def Evaluate (self):
		return self.Predict(self.list_of_timeseries)
	
	'''
	Loads a file and transforms into a "List of Timeseries", then predicts it
	'''
	def LoadAndPredict (self, data_file):
		list_of_timeseries = self.TransformData(data_file)
		return self.Predict(list_of_timeseries)
		
	'''
	Printingt Functions - Print the trace from a TimeSeries
	'''
	def PrintTrace (self, list_of_timeseries, predicted_seq_phases, n_exec, cpu_idx, mem_idx, palette, col_names, f_name):
		
		## Print into a File
		timeseries.plot_timeseries_with_phases(list_of_timeseries[n_exec][:,(cpu_idx, mem_idx)], phases = predicted_seq_phases[n_exec], num_to_color = palette, column_names = col_names, fig_size = (20, 10), y_lim = (-3, 3 + 0.1), y_ticks_delta = 2, file_to_save = f_name)
	
	'''
	Printingt Functions - Print the variance analysis plots of a "List of TimeSeries"
	'''
	def PrintVarAnalysis (self, list_of_timeseries, pred_seq_phases, cpu_idx, mem_idx, palette, col_names, f_name):
		
		## Get number of phases
		n_phases = max([item for sublist in pred_seq_phases for item in sublist]) + 1
		palette = {k: palette[k] for k in list(palette)[:n_phases]}
		
		## Get Phase Information
		phase_sets_cpu = []
		phase_sets_mem = []
		for k in range(0, n_phases):
			phase_sets_cpu.append([])
			phase_sets_mem.append([])
		
		for x in range(0, len(pred_seq_phases)):
			for y in range(0, len(pred_seq_phases[x])):
				curr_phase = pred_seq_phases[x][y]
				phase_sets_cpu[curr_phase].append(list_of_timeseries[x][y, cpu_idx])
				phase_sets_mem[curr_phase].append(list_of_timeseries[x][y, mem_idx])
		
		## Print into a File
		fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 6))
		axs[0].set_title(col_names[0])
		axs[0].boxplot(phase_sets_cpu, labels=palette.values())
		axs[1].set_title(col_names[1])
		axs[1].boxplot(phase_sets_mem, labels=palette.values())
		axs[2].set_title("Freq. phases")
		N, bins, patches = axs[2].hist(np.concatenate(pred_seq_phases), bins = n_phases, range=[0,n_phases])
		for x in range(0, len(patches)):
			patches[x].set_facecolor(palette[x])
		fig.tight_layout(pad = 2.0, w_pad = 0.0, h_pad = 3.0)
		fig.savefig(f_name)
