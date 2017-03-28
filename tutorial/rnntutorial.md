## RNN misc notes




#### RNN in Keras

* by default, stateful=False that is stateless.
* Most of problems can be solved with stateless RNN.
* If the model is stateless, the cell states are reset at each sequence.
* if stateful set to True, then use model.reset_states() to reset previous states.
* in case of RNN: input matrix X shape is: nb_samples, timesteps, input_dim
* By default, Keras shuffles (permutes) the samples in X and the dependencies between X_i and X_(i+1) are lost.
* With the stateful model, all the states are propagated to the next batch. It means that the state of the sample located at index i, X_i will be used in the computation of the sample X_(i+bs) in the next batch, where bs is the batch size (no shuffling).
* in case input is a sequence of frames, the extracted features can be re-used for the new video clip





#### links
* [Keras examples](https://keras.io/getting-started/sequential-model-guide/#examples)
* [Stateful LSTM in Keras](http://philipperemy.github.io/keras-stateful-lstm/)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
