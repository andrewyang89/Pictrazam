import mynn
from mynn.layers.dense import dense
from mynn.optimizers.sgd import SGD
from mygrad.nnet.losses.margin_ranking_loss import margin_ranking_loss
import mygrad as mg
from noggin import create_plot
import numpy as np

class Img2Caption:
    def __init__(self, dim_input: int, dim_output: int):
        """
        Initializes the Img2Caption Linear Encoder
        Parameters
        ----------
        dim_input the dimensions of the input descriptor
        dim_output the dimensions of the encoded value
        """
        self.layer = dense(dim_input, dim_output)

    def __call__(self, descriptor):
        """
        Runs the forward pass of the model
        Parameters
        ----------
        descriptor the descriptor vector to be processed

        Returns
        -------
        The shape-(dim_output) embedding

        """
        return self.layer(descriptor)

    @property
    def parameters(self):
        """
        Returns the parameters of the encoding layer
        Returns
        -------
        The parameters of the encoding layer
        """
        return self.layer.parameters


model = Img2Caption(512, 50)
lr = 1e-3
momentum = 0.9
optim = SGD(model.parameters, learning_rate = lr, momentum = momentum)

plotter, figs, axes = create_plot(metrics=["loss"])

epochs = 5
batch_size = 32
train, validation = rohan_func() # List(Tuple[]), Shape (N,)
for ep in range(epochs):
    for batch in range(len() // batch_size):

        d_img = train[::, 0]
        w_good = train[:: 1]
        d_bad = train[::, 2]

        w_bad = model(d_bad) # Shape (32, 50)
        w_img = model(d_img) # Shape (32, 50)

        dot_good = mg.sum(w_img * w_good, axis=1) # Shape (32,)
        dot_bad = mg.sum(w_img * w_bad, axis=1) # Shape (32,)
        loss = margin_ranking_loss(dot_good, dot_bad, margin=0.1)

        loss.backward()

        optim.step()

        loss.null_gradients()

        plotter.set_train_batch(metrics={"loss": loss}, batch_size=batch_size)

    plotter.set_train_epoch()
