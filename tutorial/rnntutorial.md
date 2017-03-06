## RNN misc notes


RNN in Keras
* by default, stateful is False
* Most of the problems can be solved with stateless LSTM
* RNN state is reset at the end of a batch.
* if stateful set to True, then use model.reset_states() to reset previous states.




