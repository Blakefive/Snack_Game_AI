import numpy as np
import tensorflow as tf

class Genome():
  def __init__(self):
    self.fitness = 0

    hidden_layer = 10
    self.w1 = np.random.randn(6, hidden_layer)
    self.w2 = np.random.randn(hidden_layer, 20)
    self.w3 = np.random.randn(20, hidden_layer)
    self.w4 = np.random.randn(hidden_layer, 3)

    self.b1 = np.random.randn(hidden_layer)
    self.b2 = np.random.randn(20)
    self.b3 = np.random.randn(hidden_layer)
    self.b4 = np.random.randn(3)
    
  def forward(self, inputs):
    net = np.matmul(inputs, self.w1) + self.b1
    net = tf.keras.activations.selu(net)

    net = np.matmul(net, self.w2) + self.b2
    net = tf.keras.activations.elu(net)

    net = np.matmul(net, self.w3) + self.b3
    net = tf.keras.activations.gelu(net)

    net = np.matmul(net, self.w4) + self.b4
    net = tf.keras.activations.tanh(net)
    return net

      
