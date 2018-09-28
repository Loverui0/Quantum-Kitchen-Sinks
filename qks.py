import numpy as np
import tensorflow as tf
import sklearn
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.engine.topology import Layer
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
sigma_squared = 1
learning_rate = 0.01
episodes = 400

class preprocessor(Layer):
    # Linear preprocessing layer with preinitialized random episodes.

    def __init__(self, q, p, r, eps, **kwargs):
        # Set class variables
        self.episodes = eps
        self.q = q
        self.p = p
        self.r = r
        super(preprocessor, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize (q,p)-dimensional random (normal distribution) matrix for each episode.
        self.omega = np.random.normal(0, sigma_squared, (self.episodes, self.q, self.p))

        # Iterate over all episodes.
        for e in range(self.episodes):
            # Choose for each column vector r random indices between 0 and p.
            zero_choices = np.random.choice(np.arange(0,self.p,1),(self.q,self.r))

            # Iterate over column vectors
            for j in range(self.q):
                # Set appropriate elements of omega to 0.
                self.omega[e,j,zero_choices[j]] = 0

        # Initialize q-dimensional random (uniform) vector.
        self.beta  = np.random.rand(self.episodes, self.q)*2*np.pi

        # Expand dimensions of omega and beta.
        self.omega = tf.constant(self.omega, dtype=tf.float32)
        self.omega = tf.expand_dims(self.omega, 0)
        self.omega = tf.tile(self.omega, (1, 1, 1, 1))
        self.beta = tf.constant(self.beta, dtype=tf.float32)
        self.beta = tf.expand_dims(self.beta, 0)
        self.beta = tf.tile(self.beta, (1, 1, 1))
        super(preprocessor, self).build(input_shape)

    def call(self, x):
        # Multiply matrix omega into input vector x and add beta for each episode.
        return tf.einsum('beqp,bp->beq', self.omega, x) + self.beta

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.episodes, self.q)

class kernel(Layer):
    # Quantum Kernel consisting of a circuit for each episode.

    def __init__(self, **kwargs):
        # Initialize CNOT gate.
        self.cnot = tf.constant(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),dtype=tf.complex64)

        # Initialize both permutations of tensor products of sigma_x and identity.
        self.sx1 = np.reshape(np.kron(np.array([[0,1],[1,0]]),np.eye(2)),(1,1,4,4))
        self.sx2 = np.reshape(np.kron(np.eye(2),np.array([[0,1],[1,0]])),(1,1,4,4))
        super(kernel, self).__init__(**kwargs)

    def build(self, input_shape):
        super(kernel, self).build(input_shape)

    def call(self, x):
        # Reshape input and cast to complex numbers.
        x = tf.reshape(x, (1,episodes,2,1,1))
        x = tf.cast(x, dtype=tf.complex64)

        # Generate effective single-qubit gates from input.
        Rx = tf.linalg.expm(-1.j*(self.sx1*x[:,:,0]+self.sx2*x[:,:,1]))

        # Calculate propagator of circuit with given input.
        propagator = tf.einsum('iq,beqp->beip', self.cnot, Rx)

        # Take the first column vector of the propagator.
        # This is equivalent to acting it on the zero state.
        state_out = propagator[:,:,:,0]

        # Calculate transition probabilities.
        probabilities = tf.cast(tf.square(tf.abs(state_out)),dtype=tf.float32)

        # Prepare list of collapsed states
        measurements = []

        # Iterate over quantum kernels
        for e in range(episodes):

            # Get a weighted random choice from the state probabilities
            choice = tf.multinomial(tf.log(probabilities[:,e]), 1)

            # Append collapsed (one hot) state to measurements
            measurements.append(tf.one_hot(choice, 4))

        # Stack measurements of individual kernels to full tensor
        measurement = tf.stack(measurements, axis=1)
        return measurement

    def compute_output_shape(self, input_shape):
        return (1,episodes,4)

class quantum_kitchen_sink():
    # The model containing the QKS algorithm.

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        # Input to the network.
        input_layer = Input(shape=(2,))

        # Classical linear preprocessing
        pre = preprocessor(2,2,1,episodes)(input_layer)

        # Apply set of quantum kernels.
        post = kernel()(pre)
        
        # Evaluate measurements with linear feedforward network.
        interm = Flatten()(post)
        interm2 = Dense(200, activation='linear')(interm)
        outp = Dense(1, activation='sigmoid')(interm2)

        # Compile model.
        model = Model(inputs=input_layer, outputs=outp)
        model.compile(optimizer=SGD(learning_rate), loss='mse')
        return model

    def train(self, steps=1000):
        # Initialize array for used training data.
        teachers = [[],[]]

        # Perform training steps.
        for k in range(steps):
            # Initialize random target.
            t = np.random.randint(2)

            # Take corresponding input from distribution.
            phi = np.random.rand()*2*np.pi
            u = (np.random.normal(0,0.2,1)+t)*np.array([np.cos(phi),np.sin(phi)])
            u = np.reshape(u, (1,2))

            # Append labeled input to training data.
            teachers[t].append(u[0])

            # Fit model on generated data.
            QKS.model.fit(u,np.array([t*1.0]),verbose=1)

        return teachers

# Train a quantum kitchen sink.
QKS = quantum_kitchen_sink()
teachers = QKS.train(2000)

# Initialize plotting arrays.
X,Y,C = [],[],[]

# Iterate over grid of possible inputs.
for i in np.linspace(-1.25,1.25,30):
    row = []
    for j in np.linspace(-1.25,1.25,30):
        # Get input at grid node.
        u = np.array([[i,j]])

        # Calculate network prediction for input.
        guess = QKS.model.predict(u)

        # Append X and Y with grid points.
        X.append(i)
        Y.append(j)

        # Append prediction to current row.
        row.append(guess[0,0])
    C.append(row)

# Rearrange teacher arrays.
zeros = teachers[0]
ones = teachers[1]
zeros=np.array(zeros).transpose()
ones=np.array(ones).transpose()

# Plotting stuff.
fig = plt.figure()
ax = Axes3D(fig)
C = np.array(C)
X,Y = np.meshgrid(np.linspace(-1.25,1.25,30), np.linspace(-1.25,1.25,30))
ax.plot_wireframe(X, Y, C, color='black', alpha=0.5)
ax.set_xlabel('Input x')
ax.set_ylabel('Input y')
ax.set_zlabel('Output')
ax.scatter(zeros[0],zeros[1],0,color='red',alpha=1.0)
ax.scatter(ones[0],ones[1],1,color='green',alpha=1.0)
plt.show()
