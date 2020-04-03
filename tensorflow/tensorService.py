import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.random.set_seed(101)

rand_a = np.random.uniform(0,100,(5,5))
rand_b = np.random.uniform(0,100,(5,1))

@tf.function
def foo(a, b):
    return a + b

@tf.function
def foo2(a, b):
    return a * b

#print(foo(rand_a, rand_b))
# print(foo2(10, 20))



#W = tf.Variable(tf.random_normal({n_features, n_dense_neurons}))
#b = tf.Variable(tf.ones([n_dense_neurons]))


def a():
    n_features = 10
    n_dense_neurons = 3
    W = tf.Variable(tf.random.normal([n_features, n_dense_neurons]))
    b = tf.Variable(tf.ones([n_dense_neurons]))
    @tf.function
    def model():
        x = np.random.random_sample((1, n_features))
        x_64 = tf.cast(x, tf.float32)
        xW = tf.matmul(x_64,W)
        z = tf.add(xW,b)
        return tf.sigmoid(z)
    return model

#init = tf.global_variables_initializer()

a_result = a()

#print(a_result())
def train():
    x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
    y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

    # np.random.rand(2)
    m = tf.Variable(0.44)
    b = tf.Variable(0.87)
    def loss():#mean squared error
        error = 0
        for x,y in zip(x_data,y_label):
            y_hat = m * x + b
            error += (y-y_hat)**2
        return tf.cast(error, tf.float32)
    optimizer = tf.optimizers.SGD (learning_rate=0.001, momentum=0.0, nesterov=False, name='SGD')

    train = optimizer.minimize(loss, var_list=[m,b])
    @tf.function
    def model():
        # error = 0

        

        # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
        
        print(train)
    return model
# optimizer = tf.optimizers.SGD (learning_rate=0.001, lr_decay=0.0, decay_step=100, staircase=False, use_locking=False, name='SGD')
train()()
# plt.plot(x_data,y_label,'*')
# plt.show()
