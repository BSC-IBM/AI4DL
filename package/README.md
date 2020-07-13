
## Package Content

Here there is the code of AI4DL ready to use, with an example. The AI4DL package contains the basic code to train and use the AI4Dl tool from CPU/Memory/etc... traces.

### Libraries
 
 * **AI4DL.py**: Class containing the basic functions and elements for *training* and *predicting* traces using the AI4DL mechanism
 * **crbm_tools.py**: Library containing the CRBM mechanism in Python. It is used by AI4DL class 
 * **timeseries.py**: Library containing functions to plot the traces with and without the phases
 
### An example

 * **example.py** : An example of how load a traces data-set, training the AI4DL (complete crbm + clustering pipeline) and printing an example trace with found phases. Check the data format here proposed to prepare the "training" and "testing" data (e.g. you could use files named "df.train.csv" and "df.test.csv"), and feel free to play with the variables to be used from the traces (e.g. in our example we use { "cpu.usage", "cpu.usage.cores", "cpu.usage.pct", etc... }, also we use "pod.id" to identify different executions in our collection of traces). As the model is agnostic of the features, you can include those features/traces/metrics you target in your use case.
 
## Data Format for traces

Example of traces tabular data:

| Timestamp  | Service  | Pod.ID  | CPU.usage | Num.cores | Mem.current | Mem.limit   | ... |
| :--------- | :------- | :------ | --------: | --------: | ----------: | ----------: | :-: |
| 1564012810 | service1 | 0000001 | 55739308  | 16        | 11538403328 | 21474836480 | ... |
| 1564012830 | service1 | 0000001 | 857430760 | 16        | 11538952192 | 21474836480 | ... |
| 1564012840 | service1 | 0000001 | 38804371  | 16        | 11539546112 | 21474836480 | ... |
| 1564012860 | service1 | 0000001 | 36381764  | 16        | 11539546112 | 21474836480 | ... |
| ...        | ...      | ...     | ...       | ...       | ...         | ...         | ... |


The features used to identify containers are:

- "timestamp" : The timestamp of the registered metric
- "service" : Identifies the service. Here we focused on "training" (learning) containers from the DLaaS
- "pod.id" : Identifies the container instance, performing a training process.

The selected features from traces used for behavior modelling on the presented examples are:

- "cpu.usage" :	Indicates the amount CPU load in use by the instance
- "cpu.usage.cores" : Indicates the amount of cores provided to the instance (the maximum core usage)
- "cpu.usage.pct" : Indicates the usage of CPU in percentage
- "cpu.usage.pct.container.requested" : Indicates the CPU requested in percentage (from the maximum available)
- "mem.current" : Indicates the memory load in use by the instance
- "mem.limit" : Indicates the memory limit for the instance
- "mem.usage.pct" : Indicates the memory usage in percentage
- "mem.working.set" : Indicates the requested memory available

From those variables, we focus on *cpu.usage* and *mem.current* as absolute indicators of resource usage, although all the selected features are used for training by providing resource consumption and resource limits to detect stress scenarios.

The model is agnostic of the metrics, so equivalent CPU/Mem/IO metrics can be trained and predicted later.
