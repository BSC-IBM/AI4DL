
## Package Content

Here there is the code of AI4DL ready to use, with an example. The AI4DL package contains the basic code to train and use the AI4Dl tool from CPU/Memory/etc... traces.

### Libraries
 
 * **AI4DL.py**: Class containing the basic functions and elements for *training* and *predicting* traces using the AI4DL mechanism
 * **crbm_tools.py**: Library containing the CRBM mechanism in Python. It is used by AI4DL class 
 * **timeseries.py**: Library containing functions to plot the traces with and without the phases
 
### An example

 * **example.py** : An example loading a traces data-set, training the AI4DL (complete crbm + clustering pipeline) and printing an example trace with found phases 
 
## Data Format for traces

Example of traces tabular data:

| Timestamp  | Service | Pod.ID  | CPU.usage | Num.cores | Mem.current | Mem.limit   | ... |
| :--------- | :------ | :------ | --------: | --------: | ----------: | ----------: | :-: |
| 1564012810 | learner | 0000001 | 55739308  | 16        | 11538403328 | 21474836480 | ... |
| 1564012830 | learner | 0000001 | 857430760 | 16        | 11538952192 | 21474836480 | ... |
| 1564012840 | learner | 0000001 | 38804371  | 16        | 11539546112 | 21474836480 | ... |
| 1564012860 | learner | 0000001 | 36381764  | 16        | 11539546112 | 21474836480 | ... |
| ...        | ...     | ...     | ...       | ...       | ...         | ...         | ... |


The features used to identify containers are:

- "timestamp" : The timestamp of the registered metric
- "service" : Identifies the service. Here we focused on "training" (learning) containers from the DLaaS
- "pod.id" : Identifies the container instance, performing a training process.

The selected features from traces used for behavior modelling on the presented examples are:

- "cpu.usage" :	Indicates the amount of cores x machines in use by the instance
- "cpu.usage.cores" : Indicates the amount of cores provided to the instance (the maximum core usage)
- "cpu.usage.pct" : Indicates the usage of CPU in percentage
- "cpu.usage.pct.container.requested" : Indicates the CPU requested in percentage (from the maximum available)
- "mem.current" : Indicates the amount of memory in use by the instance
- "mem.limit" : Indicates the memory limit for the instance
- "mem.usage.pct" : Indicates the memory usage in percentage
- "mem.working.set" : Indicates the requested memory available

From those variables, we focus on *cpu.usage* and *mem.current* as absolute indicators of resource usage, although all the selected features are used for training by providing resource consumption and resource limits to detect stress scenarios.
