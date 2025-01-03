# kim_optimal_control

For the moment, I have put the code of the original prototype into separate python scripts and modified them to run them into the LAMSADE server.

**Modifications:**
 - Changed the scale of some of the features by changing some of the parameters of the base model
 - Parameters for the base model, as well as training parameters are now read from configuration files, to improve reproducibility and testing
 - Added checkpointing during training


**TODO**
 - Implement data parallelization to use multiple GPUS during training, currently using single GPU
 - Implement the "real" testing of the value network
