# Simple Quantum Kitchen Sink Implementation

A simple implementation of the approach presented by Christopher M. Wilson, J. S. Otterbach, Nikolas Tezak, Robert S. Smith, Gavin E. Crooks, and Marcus P. da Silva in their paper called Quantum Kitchen Sinks (arXiv:1806.08321) used on a toy example. 

The core idea of this approach is to have a certain amount of quantum circuits with gates that are parametrized by classical input. The circuits are each applied to the state <img src="https://latex.codecogs.com/svg.latex?|0>^{\otimes n}"/> and the results are then measured. The array of measurements is processed such that the measured state is one-hot encoded and this sparse object is used as input to a feedforward neural network with linear activations. 

(PLEASE NOTE: Currently in this implementation the measurements are not implemented according to the Born rule, but rather the result is always the possibility with the highest probability.) 

The classical inputs <img src="https://latex.codecogs.com/svg.latex?x"/> that determine the quantum gates are preprocessed linearly with a mainly random (normal) matrix <img src="https://latex.codecogs.com/svg.latex?\Omega"/> and a random (uniform) vector <img src="https://latex.codecogs.com/svg.latex?\beta"/> such that circuit parameters can be calculated as

<img src="https://latex.codecogs.com/svg.latex?\theta=\Omega\,x+\beta"/>.

Therefore all classical calculations of this approach are linear, whereas non-linearities are put onto the quantum kernels. 

The figure below shows the output of a quantum kitchen sink from this implementation over a given 2D input space for binary classification in addition to the training data that was used. The distributions for the two classes <img src="https://latex.codecogs.com/svg.latex?t\in\{0,1\}"/> in this example are 

<img src="https://latex.codecogs.com/svg.latex?D_t=(\mathcal{N}(0,0.2)+t)\begin{pmatrix}\cos(\phi)}\\\sin(\phi)\end{pmatrix}"/>,

where <img src="https://latex.codecogs.com/svg.latex?\mathcal{N}(\mu,\sigma)"/> is a normal distribution and the <img src="https://latex.codecogs.com/svg.latex?\phi\in[0,2\pi]"/> are uniformly random.

![result](figures/result.png "Results")