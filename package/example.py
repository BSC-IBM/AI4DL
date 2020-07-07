
#-------------------------------------------------------------------------------
# Import AI4DL
#-------------------------------------------------------------------------------

import AI4DL
from AI4DL import AI4DL

#-------------------------------------------------------------------------------
# Main Program
#-------------------------------------------------------------------------------

def main_test():
	
	## 0. Configuration
	training_file = "df.train.csv"
	testing_file = "df.test.csv"
	
	selected_features = ["cpu.usage", "cpu.usage.cores", "cpu.usage.pct", "cpu.usage.pct.container.requested", "mem.current", "mem.limit", "mem.usage.pct", "mem.working.set"]
	container_id = "pod.id"
	
	n_hidden  = 10
	n_history  = 3
	learning_rate = 0.001
	n_epochs = 100
	n_clusters = 5
	
	crbm_save = "ai4dl_crbm1"
	kmeans_save = "ai4dl_kmeans1.joblib"
	scaler_save = "ai4dl_scaler1.joblib"
	
	palette = {0:'grey', 1:'green', 2:'blue', 3:'red', 4:'orange'}
	cpu_idx = 0 # cpu.usage
	mem_idx = 4 # mem.current
	
	
	## 1. Load and Process Training Data
	print("Loading Datasets")
	
	ai4dl1 = AI4DL.AI4DL()
	ai4dl1.LoadTrainingDataset(training_file, selected_features, container_id)
	
	## 2. Train Model (CRBM + K-Means)
	print("Training the CRBM + K-Means")
	
	ai4dl1.TrainModel(n_clusters, n_hidden, n_history, learning_rate, n_epochs, crbm_save, kmeans_save, scaler_save, seed = 123)
	#ai4dl1.LoadModel(crbm_save, kmeans_save, scaler_save)
	
	## 3. Predict Sequences
	print("Inference of traces")
	
	list_of_timeseries_tr = ai4dl1.GetTrainingDataset()
	predicted_seq_phases_tr = ai4dl1.Predict(list_of_timeseries_tr)
	#predicted_seq_phases_tr = ai4dl1.Evaluate()
	
	list_of_timeseries_ts = ai4dl1.TransformData(testing_file)
	predicted_seq_phases_ts = ai4dl1.Predict(list_of_timeseries_ts)
	#predicted_seq_phases_ts = ai4dl1.LoadAndPredict(testing_file)
	
	## 4.1 Plot Example Execution
	print("Plotting Sample Execution")
	
	selected_exec = 11
	ai4dl1.PrintTrace(list_of_timeseries_tr, predicted_seq_phases_tr, selected_exec, cpu_idx, mem_idx, palette, col_names = ["CPU usage", "MEM usage"], f_name = "trace_11.png")
	
	## 4.2 Plot Variance of Phases
	print("Plotting Variance of Phases")
	
	ai4dl1.PrintVarAnalysis(list_of_timeseries_tr, predicted_seq_phases_tr, cpu_idx, mem_idx, palette, col_names = ["CPU variation", "MEM variation"], f_name = "variance.png")
	
	
	print("The End")

if __name__ == "__main__":
	main_test()
