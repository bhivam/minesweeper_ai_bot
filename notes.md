Currently we have a model which takes in a tensor of size 9x9x10. 

There are 9x9 slots in the board and each cell can be in one of 10 states known to the agent (neural network).

The one hot coding is pushed to a 1 dimensional embedding. This embedding is trained at the beginning of the model. The embedding is a linear layer with a tanh activation. This is to keep embeddings from -1 to 1. 

Then we use some convolutional layers to perform reasoning on the grid of embedded cells. Finally, some softmax is applied to an output of length 81. This is interpreted as the probability of having a mine at particular cell. It could be interesting to try this out with a Sigmoid doing a distribution on the cell level instead. 


Try limiting the target to only consider bombs that are on the edge of opened tiles. I initially thought this might be useful because I didn't want the model to focus on the wrong tiles. But looking at the outputs now, it seems like the model is able to make sense of which mines to use.
