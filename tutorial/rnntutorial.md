## RNN misc notes




RNN in Keras

* by default, stateful=False that is stateless.
* Most of problems can be solved with stateless RNN.
* RNN state is reset at the end of a batch.
* if stateful set to True, then use model.reset_states() to reset previous states.
* in case of RNN: input matrix X shape is: nb_samples, timesteps, input_dim
* By default, Keras shuffles (permutes) the samples in X and the dependencies between X_i and X_(i+1) are lost. Let’s assume there’s no shuffling in our explanation. 



