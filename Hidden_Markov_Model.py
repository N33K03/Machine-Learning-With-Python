import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf
import numpy

tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # The first day in our sequence has an 80% chance of being cold.
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                 [0.2, 0.8]])  # Cold day has 30% chance of being followed by a hot day, hot day has 20% chance of being followed bby a cold day
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # Each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day

# the loc argument represents the mean and the scale is the standard devitation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:
  print(mean.numpy())