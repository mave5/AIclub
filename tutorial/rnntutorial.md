## RNN misc notes




RNN in Keras

* by default, stateful=False that is stateless.
* Most of problems can be solved with stateless RNN.
* RNN state is reset at the end of a batch.
* if stateful set to True, then use model.reset_states() to reset previous states.




