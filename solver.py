import logging
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

DELTA_CLIP = 50.0


class BSDESolver(object):
    """The fully connected neural network model."""
    def __init__(self, T,L,nbStep,num_hiddens, dtype, bs, bsV, numiT, bsde):
        #self.eqn_config = config.eqn_config
        #self.net_config = config.net_config
        self.batch_size=bs
        self.valid_size=bsV
        self.num_iterations=numiT
        self.bsde = bsde
        self.TStep = T / nbStep
        self.dtype = dtype
        ############################################
        self.lr_values = [1e-2, 1e-2]
        self.lr_boundaries = [1000]
        self.y_init_range = [0, 1]
        self.logging_frequency = 10
        ############################################
        self.model = NonsharedModel(L,nbStep,num_hiddens,self.y_init_range, self.TStep, bsde)
        self.y_init = self.model.y_init
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.lr_boundaries, self.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        y_init_history=[]
        loss_history=[]
        xptsT = self.bsde.xpointsintg(self.batch_size)  # (L+1, batchsize, intgstep)
        xptsV = self.bsde.xpointsintg(self.valid_size)  # (L+1, batchsizeVal, intgstep)
        LbPT, RbPT = self.bsde.basisxpoints(self.batch_size)  # (batchsize, intgstep) these are same points for all intervals
        LbPV, RbPV= self.bsde.basisxpoints(self.valid_size)  # (batchsizeval, intgstep)  these are same points for all intervals
        dWVal, x0Val = self.bsde.sample(self.valid_size)

        # begin sgd iteration
        for step in range(self.num_iterations+1):
            if step % self.logging_frequency == 0:
                loss = self.loss_fn(dWVal, x0Val, xptsV, LbPV, RbPV, training=False).numpy()
                loss_history.append(loss)
                y_init = self.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, elapsed_time])
                y_init_history.append(y_init)
                #if self.net_config.verbose:
                print("step: ", step, "  loss: ", loss, "   elapsed time: ", elapsed_time)
                print(" ........... Y0: ", y_init)
                    #logging.info("step: ", step, "  loss: ", loss, " Y0: ", y_init, "   elapsed time: ", elapsed_time)
                    #logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (step, loss, y_init, elapsed_time))
            dWT, x0T=self.bsde.sample(self.batch_size)
            self.train_step(dWT, x0T, xptsT,LbPT,RbPT)
        return np.array(training_history), np.array(y_init_history), np.array(loss_history)

    def loss_fn(self, dw, x0, xpts, LbP, RbP, training):
        #dw = inputs
        y_terminal, x_terminal, cumloss = self.model(dw, x0, xpts, training)
        delta = cumloss+ tf.square(y_terminal - self.bsde.YT(x_terminal, LbP, RbP))
        # use linear approximation outside the clipped range
        #loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        loss=tf.reduce_mean(delta)
        return loss

    def grad(self, dw, x0, xpts, LbP, RbP,training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(dw,x0,  xpts, LbP, RbP, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, dw, x0, xpts, LbP, RbP,):
        grad = self.grad(dw,x0,  xpts, LbP, RbP,training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))


class NonsharedModel(tf.keras.Model):
    def __init__(self, L,nbStep,num_hiddens,y_init_range,TStep, bsde):
        super(NonsharedModel, self).__init__()
        #self.eqn_config = config.eqn_config
        self.L=L
        self.y_init_range=y_init_range
        self.num_time_interval=nbStep
        self.TStep=TStep
        self.num_hiddens=num_hiddens

        #self.net_config = config.net_config
        self.bsde = bsde
        self.y_init = tf.Variable(np.random.uniform(low=self.y_init_range[0],high=self.y_init_range[1],size=[1,self.L]))
        #self.y_init = tf.Variable(np.random.normal(size=[1, self.eqn_config.dim]))
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.L]))

        self.subnet = [FeedForwardSubNet(self.L,self.num_hiddens) for _ in range(self.num_time_interval-1)]

    def call(self, dw, x0, xpts, training):
        #dw = inputs
        time_stamp = np.arange(0, self.num_time_interval) * self.TStep
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[1], 1]), dtype=self.dtype)
        #y = all_one_vec * self.y_init
        y = tf.matmul(all_one_vec, self.y_init)
        #y=x[0,:,:]
        z = tf.matmul(all_one_vec, self.z_init)
        x = x0
        cum_loss=0

        for t in range(0, self.num_time_interval-1):
            Wt=tf.math.reduce_sum(dw[:t,:,:],axis=0)
            x_next=self.bsde.generateX1stepFw( t, x, y, Wt, xpts, dw[t,:,:])
            y_next = y - (self.bsde.F( y) + self.bsde.F2(t, x, y, Wt, xpts)) * self.TStep \
                 + z * tf.tile(dw[t,:,:], [1, self.L])
            #y = y - (self.bsde.F(time_stamp[t], x[t, :, :], y, z) +self.bsde.F2(t,x[t,:,:],DRo[:,:,t],DDRo[:,:,t],dw)) * self.bsde.TStep + z * tf.tile(dw[t, :, :], [1, self.bsde.L])
            # y = y - self.bsde.delta_t * (self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)

            ## (Xt, Wt ) goes as input of NN
            Xinp=tf.concat([x_next, Wt+dw[t,:,:]], 1)
            y, z = self.subnet[t](Xinp, training)
            x = x_next
            cum_loss=cum_loss+self.TStep*tf.square(y-y_next)
        # terminal time
        Wt = tf.math.reduce_sum(dw[:t+1, :, :], axis=0)
        x_T = self.bsde.generateX1stepFw(t+1, x, y, Wt, xpts, dw[t+1, :, :])
        y_T = y - (self.bsde.F( y) + self.bsde.F2(t+1, x, y, Wt, xpts)) * self.TStep \
                 + z * tf.tile(dw[t+1, :, :], [1, self.L])
        #y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
        return y_T, x_T, cum_loss


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, L,num_hiddens):
        super(FeedForwardSubNet, self).__init__()
        self.dim = L
        #num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=True,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim+dim
        self.dense_layers.append(tf.keras.layers.Dense(2*self.dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            #x = self.bn_layers[i+1](x, training)
            x = tf.nn.tanh(x)
        x = self.dense_layers[-1](x)
        #x = self.bn_layers[-1](x, training)
        return x[:,0:self.dim], x[:,self.dim:]
