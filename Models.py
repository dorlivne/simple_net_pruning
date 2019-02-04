import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning
from networks.network_base import BaseNetwork
from utils.tensorflow_utils import get_batch, load_mnist, load_cifar_10
from configs import ConfigSimpleNetDense as simple_config
from tqdm import tqdm
import numpy as np
from utils.logger_utils import get_logger

ADD_EPISODES = 1


class ConvNet(BaseNetwork):

    def __init__(self, input_size,
                 output_size,
                 model_path: str,
                 momentum=0.9,
                 reg_str=0.0005,
                 scope='ConvNet',
                 pruning_start=int(10e4),
                 pruning_end=int(10e5),
                 pruning_freq=int(10),
                 sparsity_start=0,
                 sparsity_end=int(10e5),
                 target_sparsity=0.0,
                 dropout=0.5,
                 initial_sparsity=0,
                 wd=0.0):
        super(ConvNet, self).__init__(input_size=input_size,
                                             output_size=output_size,
                                             model_path=model_path)
        self.scope = scope
        self.momentum = momentum
        self.reg_str = reg_str
        self.dropout = dropout
        self.logger = get_logger(scope)
        self.wd = wd
        self.logger.info("creating graph...")
        with self.graph.as_default():
                self.global_step = tf.Variable(0, trainable=False)
                self._build_placeholders()
                self.logits = self._build_model()
                self.weights_matrices = pruning.get_masked_weights()
                self.sparsity = pruning.get_weight_sparsity()
                self.loss = self._loss()
                self.train_op = self._optimizer()
                self._create_metrics()
                self.saver = tf.train.Saver(var_list=tf.global_variables())
                self.hparams = pruning.get_pruning_hparams()\
                    .parse('name={}, begin_pruning_step={}, end_pruning_step={}, target_sparsity={},'
                           ' sparsity_function_begin_step={},sparsity_function_end_step={},'
                           'pruning_frequency={},initial_sparsity={},'
                           ' sparsity_function_exponent={}'.format(scope,
                                                                   pruning_start,
                                                                   pruning_end,
                                                                   target_sparsity,
                                                                   sparsity_start,
                                                                   sparsity_end,
                                                                   pruning_freq,
                                                                   initial_sparsity,
                                                                   3))
                # note that the global step plays an important part in the pruning mechanism,
                # the higher the global step the closer the sparsity is to sparsity end
                self.pruning_obj = pruning.Pruning(self.hparams, global_step=self.global_step)
                self.mask_update_op = self.pruning_obj.conditional_mask_update_op()
                # the pruning objects defines the pruning mechanism, via the mask_update_op the model gets pruned
                # the pruning takes place at each training epoch and it objective to achieve the sparsity end HP
                self.init_variables(tf.global_variables())  # initialize variables in graph


    def get_num_of_params(self):
        with self.graph.as_default():
            weights = self.get_flat_weights()
            num_of_params = 0
            for layer in weights:
                values = layer[np.abs(layer) != 0]
                num_of_params += np.size(values)
        return num_of_params

    def get_flat_weights(self):
        weights_matrices = self.sess.run(self.weights_matrices)
        flatten_matrices = []
        for matrix in weights_matrices:
            flatten_matrices.append(np.ndarray.flatten(matrix))
        return flatten_matrices

    def get_model_sparsity(self):
        sparsity = self.sess.run(self.sparsity)
        return np.mean(sparsity)

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.

        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable

        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, wd, initialization):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """
        var = self._variable_on_cpu(
            name,
            shape,
            initialization)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _build_placeholders(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=self.input_size, name='input')
        self.labels = tf.placeholder(dtype=tf.int64, shape=None, name='label')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name='keep_prob')
        self.lr = tf.placeholder(dtype=tf.float32, shape=None, name='learning_rate')

    def _build_conv_layer(self, input, scope, weight_init, filter_hight, filter_width, channel_in, channel_out, activation=None):
        kernel = self._variable_with_weight_decay(
            'weights', shape=[filter_hight, filter_width, channel_in, channel_out], initialization=weight_init, wd=self.wd)
        conv = tf.nn.conv2d(
           input, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
        biases = self._variable_on_cpu('biases', channel_out, tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        if activation:
            return activation(pre_activation, name=scope.name)
        else:
            return pre_activation

    def _build_fc_layer(self, input, scope, weight_init, shape, activation=None):
        weights = self._variable_with_weight_decay(
            'weights', shape=shape, initialization=weight_init, wd=0.0)
        biases = self._variable_on_cpu('biases', shape[1], tf.constant_initializer(0.0))
        if activation is not None:
            return activation(
                tf.matmul(input, pruning.apply_mask(weights, scope)) + biases,
                name=scope.name)
        else:
            return tf.matmul(input, pruning.apply_mask(weights, scope)) + biases

    def _build_model(self):
        net = self.input
        self.biases = []
        self.weights_init = tf.keras.initializers.lecun_uniform()
        self.bias_init = tf.constant_initializer(0.1)
        with tf.variable_scope('conv1') as scope:
                net = self._build_conv_layer(input=net, weight_init=self.weights_init, filter_width=3, filter_hight=3,
                                             channel_in=1, channel_out=32, scope=scope)
        with tf.variable_scope('conv2') as scope:
                net = self._build_conv_layer(input=net, weight_init=self.weights_init, filter_width=3, filter_hight=3,
                                             channel_in=32, channel_out=64, scope=scope)

        net = tf.layers.average_pooling2d(inputs=net, pool_size=(2, 2), strides=1)
        net = tf.layers.dropout(inputs=net, rate=self.keep_prob)
        net = tf.layers.flatten(net)
        with tf.variable_scope('dense_1') as scope:
             net = self._build_fc_layer(input=net, scope=scope, weight_init=self.weights_init, shape=[46656, 128],
                                        activation=tf.nn.relu)
        # the output of the convlutions is 28 by width and hight, the pooling layer reduece the spatial area to 27
        with tf.variable_scope('dense_2') as scope:
             net = self._build_fc_layer(input=net, scope=scope, weight_init=self.weights_init,
                                        shape=[128, self.output_size[-1]])
        return net

    def _loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        loss = tf.reduce_mean(loss)  # averaged out
        return loss + self.reg_str * tf.losses.get_regularization_loss()

    def _optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum).\
                                           minimize(self.loss, global_step=self.global_step)

    def evaluate(self, set_x, set_y, batch_size, num_of_iter=None):
        if not num_of_iter:
             num_of_iter = set_x.shape[0] // batch_size
        averaged_acc, averaged_loss = 0, 0
        for i in range(num_of_iter):
            x, y = get_batch(batch_size=batch_size, x_set=set_x, y_set=set_y)
            accuracy, loss = self.sess.run([self.accuracy, self.loss], feed_dict={
                self.input: x,
                self.keep_prob: 1,
                self.labels: y})
            averaged_acc += accuracy
            averaged_loss += loss
        return averaged_acc / num_of_iter, averaged_loss / num_of_iter

    def _create_metrics(self):
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def reset_global_step(self):
        self.init_variables(var_list=[self.global_step])

    def maybe_update_best(self, accuracy, config):
        if accuracy > 0.8:
            self.save_model(config.best_path)

    def fit(self, config,
            n_epochs: int,
            batch_size: int,
            learning_rate_schedule: callable,
            verbose=True,
            prune=False):
        if prune:
            self.logger.info("----------Iterative Pruning----------")
        sparsity_vs_accuracy = [[], []]
        x_train, y_train, x_test, y_test, x_val, y_val = load_mnist(num_train=55000, num_val=5000)
        num_of_iter = x_train.shape[0] // batch_size
        for e in range(n_epochs):
            train_acc_avg = 0
            print('Starting epoch {}.\n'.format(e + 1))
            for iter in tqdm(range(num_of_iter)):
                x_batch, y_batch = get_batch(batch_size=batch_size, x_set=x_train, y_set=y_train)
                _, train_acc = self.sess.run([self.train_op, self.accuracy], feed_dict={self.input: x_batch, self.labels: y_batch,
                                                        self.keep_prob: (1 - self.dropout),
                                                        self.lr: learning_rate_schedule(e)})
                train_acc_avg += train_acc
                if prune:
                    self.sess.run([self.mask_update_op])
            train_acc_avg /= num_of_iter
            test_acc, test_loss = self.evaluate(x_train, y_train, batch_size)
            self.maybe_prune_and_maybe_print_results(config, train_acc_avg, test_acc, test_loss,
                                                     sparsity_vs_accuracy, n_epochs, e, prune=prune)

        if verbose:
            test_acc, test_loss = self.evaluate(x_test, y_test, batch_size)
            print('\nOptimization finished.')
            print(' Final Accuracy on test: {accuracy},  Final loss on test: {loss}'.format(
                accuracy=test_acc, loss=test_loss))
        self.save_model()
        return sparsity_vs_accuracy

    def maybe_prune_and_maybe_print_results(self, config, train_acc_avg, val_acc, val_loss,
                                            sparsity_vs_accuracy, n_epochs, e, prune=False, verbose=True):
        """
        this function does two things, either prunes ,prints info or both
        :param config: list of HP
        :param train_acc_avg: info for printing
        :param val_acc: info for printing
        :param val_loss: info for printing
        :param sparsity_vs_accuracy: info for graph that depicts sparsity vs accuracy of pruned model
        :param n_epochs: total number of episodes (for printing)
        :param e: current episode(for printing)
        :param prune: if true , then this function initiates pruning via mask_update_op
        :param verbose: to print or not to print
        :return: VOID
        """
        if prune:
            self.sess.run(self.mask_update_op) # the only complicated thing here
            sparsity = self.get_model_sparsity()
            print('Sparsity for epoch {} / {} is {} \n'.format(e, n_epochs, sparsity))
            self.logger.info('Sparsity for epoch {} / {} is {} \n'.format(e, n_epochs, sparsity))
            if e % 5 == 0:
                sparsity_vs_accuracy[0].append(sparsity)
                sparsity_vs_accuracy[1].append(val_acc)
                self.maybe_update_best(val_acc, config)  # only when pruning
        if verbose:
            print('\nEpoch {} / {} completed.'.format(e + 1, n_epochs))
            print('Accuracy on train: {accuracy}'.format(
                accuracy=train_acc_avg))
            print('Accuracy on val: {accuracy}, loss on val: {loss}'.format(
                accuracy=val_acc, loss=val_loss))
            self.logger.info(" Accuracy on train : {} \n Accuracy on test : {}".format(train_acc_avg, val_acc))
        return


class SimpleNet(ConvNet):

    def __init__(self, input_size,
                 output_size,
                 model_path:str,
                 reg_str=0.0005,
                 scope='SimpleNet',
                 pruning_start=int(10e4),
                 pruning_end=int(10e5),
                 pruning_freq=int(10),
                 sparsity_start=0,
                 sparsity_end=int(10e5),
                 target_sparsity=0.0,
                 dropout=0.2,
                 initial_sparsity=0,
                 wd=0.0):
        super(SimpleNet, self).__init__(input_size=input_size,
                                             output_size=output_size,
                                             model_path=model_path,
                                             reg_str=reg_str,
                                             scope=scope,
                                             pruning_start=pruning_start,
                                             pruning_end=pruning_end,
                                             pruning_freq=pruning_freq,
                                             sparsity_start=sparsity_start,
                                             sparsity_end=sparsity_end,
                                             target_sparsity=target_sparsity,
                                             dropout=dropout,
                                             initial_sparsity=initial_sparsity,
                                             wd=wd
                                             )

    def _build_model(self):
        # Block 1
        with tf.variable_scope('conv_1') as scope:
            conv_1 = self._build_conv_layer(self.input, filter_hight=3, filter_width=3,
                                            channel_in=3, channel_out=64,
                                            weight_init=tf.keras.initializers.glorot_normal(), scope=scope )
            # Batch_Normalization
            conv_1 = tf.layers.batch_normalization(inputs=conv_1, training=self.in_train)
            # Relu
            conv_1 = tf.nn.relu(conv_1)
            # Dropout
            conv_1 = tf.nn.dropout(x=conv_1, keep_prob=self.keep_prob)

        # Block 2
        with tf.variable_scope('conv_2') as scope:
            conv_2 = self._build_conv_layer(conv_1, filter_hight=3, filter_width=3,
                                            channel_in=64, channel_out=128,
                                            weight_init=tf.keras.initializers.glorot_normal(), scope=scope)
            # Batch_Normalization
            conv_2 = tf.layers.batch_normalization(inputs=conv_2, training=self.in_train)
            # Relu
            conv_2 = tf.nn.relu(conv_2)
            # Dropout
            conv_2 = tf.nn.dropout(x=conv_2, keep_prob=self.keep_prob)
        # Block 3
        with tf.variable_scope('conv_3') as scope:
            conv_3 = self._build_conv_layer(conv_2, filter_hight=3, filter_width=3,
                                            channel_in=128, channel_out=128,
                                            weight_init=tf.keras.initializers.random_normal(stddev=0.01), scope=scope) # used to be normal
            # Batch_Normalization
            conv_3 = tf.layers.batch_normalization(inputs=conv_3, training=self.in_train)
            # Relu
            conv_3 = tf.nn.relu(conv_3)
            # Dropout
            conv_3 = tf.nn.dropout(x=conv_3, keep_prob=self.keep_prob)
        # Block 4
        with tf.variable_scope('conv_4') as scope:
            conv_4 = self._build_conv_layer(conv_3, filter_hight=3, filter_width=3,
                                            channel_in=128, channel_out=128,
                                            weight_init=tf.keras.initializers.random_normal(stddev=0.01), scope=scope) # use to be normal
            # Batch_Normalization
            conv_4 = tf.layers.batch_normalization(inputs=conv_4, training=self.in_train)
            # Relu
            conv_4 = tf.nn.relu(conv_4)
            # Max Pooling
            conv_4 = tf.layers.max_pooling2d(inputs=conv_4, pool_size=(2, 2), strides=2)
            # Dropout
            conv_4 = tf.nn.dropout(x=conv_4, keep_prob=self.keep_prob)
        # Block 5
        with tf.variable_scope('conv_5') as scope:
            conv_5 = self._build_conv_layer(conv_4, filter_hight=3, filter_width=3,
                                            channel_in=128, channel_out=128,
                                            weight_init=tf.keras.initializers.random_normal(stddev=0.01), scope=scope)
            # Batch_Normalization
            conv_5 = tf.layers.batch_normalization(inputs=conv_5, training=self.in_train)
            # Relu
            conv_5 = tf.nn.relu(conv_5)

            # Dropout
            conv_5 = tf.nn.dropout(x=conv_5, keep_prob=self.keep_prob)
        # Block 6
        with tf.variable_scope('conv_6') as scope:
            conv_6 = self._build_conv_layer(conv_5, filter_hight=3, filter_width=3,
                                            channel_in=128, channel_out=128,
                                            weight_init=tf.keras.initializers.glorot_normal(), scope=scope)
            # Batch_Normalization
            conv_6 = tf.layers.batch_normalization(inputs=conv_6,training=self.in_train)
            # Relu
            conv_6 = tf.nn.relu(conv_6)

            # Dropout
            conv_6 = tf.nn.dropout(x=conv_6, keep_prob=self.keep_prob)
        # Block 7
        with tf.variable_scope('conv_7') as scope:
            conv_7 = self._build_conv_layer(conv_6, filter_hight=3, filter_width=3,
                                            channel_in=128, channel_out=256,
                                            weight_init=tf.keras.initializers.glorot_normal(), scope=scope)
            # Batch_Normalization
            conv_7 = tf.layers.batch_normalization(inputs=conv_7, training=self.in_train)
            # Max Pooling
            conv_7 = tf.layers.max_pooling2d(inputs=conv_7, pool_size=(2, 2), strides=2)
            # Relu
            conv_7 = tf.nn.relu(conv_7)
            # Dropout
            conv_7 = tf.nn.dropout(x=conv_7, keep_prob=self.keep_prob)
        # Block 8
        with tf.variable_scope('conv_8') as scope:
            conv_8 = self._build_conv_layer(conv_7, filter_hight=3, filter_width=3,
                                            channel_in=256, channel_out=256,
                                            weight_init=tf.keras.initializers.glorot_normal(), scope=scope)
            # Batch_Normalization
            conv_8 = tf.layers.batch_normalization(inputs=conv_8, training=self.in_train)
            # Max Pooling
            # Relu
            conv_8 = tf.nn.relu(conv_8)
            # Dropout
            conv_8 = tf.nn.dropout(x=conv_8, keep_prob=self.keep_prob)
        # Block 9
        with tf.variable_scope('conv_9') as scope:
            conv_9 = self._build_conv_layer(conv_8, filter_hight=3, filter_width=3,
                                            channel_in=256, channel_out=256,
                                            weight_init=tf.keras.initializers.glorot_normal(), scope=scope)
            # Batch_Normalization
            conv_9 = tf.layers.batch_normalization(inputs=conv_9, training=self.in_train)
            # Relu
            conv_9 = tf.nn.relu(conv_9)
            # Dropout
            conv_9 = tf.nn.dropout(x=conv_9, keep_prob=self.keep_prob)
            # Max Pooling
            conv_9 = tf.layers.max_pooling2d(inputs=conv_9, pool_size=(2, 2), strides=2)
        # Block 10
        with tf.variable_scope('conv_10') as scope:
            conv_10 = self._build_conv_layer(conv_9, filter_hight=3, filter_width=3,
                                             channel_in=256, channel_out=512,
                                             weight_init=tf.keras.initializers.glorot_normal(), scope=scope)
            # Batch_Normalization
            conv_10 = tf.layers.batch_normalization(inputs=conv_10, training=self.in_train)
            # Relu
            conv_10 = tf.nn.relu(conv_10)
            # Dropout
            conv_10 = tf.nn.dropout(x=conv_10, keep_prob=self.keep_prob)
        # Block 11
        with tf.variable_scope('conv_11') as scope:
            conv_11 = self._build_conv_layer(conv_10, filter_hight=1, filter_width=1,
                                             channel_in=512, channel_out=2048,
                                             weight_init=tf.keras.initializers.glorot_normal(), scope=scope)
            # Relu
            conv_11 = tf.nn.relu(conv_11)
            # Dropout
            conv_11 = tf.nn.dropout(x=conv_11, keep_prob=self.keep_prob)
        # Block 12
        with tf.variable_scope('conv_12') as scope:
            conv_12 = self._build_conv_layer(conv_11, filter_hight=1, filter_width=1,
                                             channel_in=2048, channel_out=256,
                                             weight_init=tf.keras.initializers.glorot_normal(), scope=scope)
            # Relu
            conv_12 = tf.nn.relu(conv_12)
            # Max Pooling
            conv_12 = tf.layers.max_pooling2d(inputs=conv_12, pool_size=(2, 2), strides=2)
            # Dropout
            conv_12 = tf.nn.dropout(x=conv_12, keep_prob=self.keep_prob)
        # Block 13
        with tf.variable_scope('conv_13') as scope:
            conv_13 = self._build_conv_layer(conv_12, filter_hight=3, filter_width=3,
                                             channel_in=256, channel_out=256,
                                             weight_init=tf.keras.initializers.glorot_normal(), scope=scope)
            # Relu
            conv_13 = tf.nn.relu(conv_13)
            # Max Pooling
            conv_13 = tf.layers.max_pooling2d(inputs=conv_13, pool_size=(2, 2), strides=2)
        # Logits
        with tf.variable_scope('output') as scope:
            logits = tf.layers.flatten(conv_13)
            logits = self._build_fc_layer(input=logits, scope=scope, weight_init=tf.keras.initializers.glorot_normal(),
                                          shape=[256, self.output_size[1]])

        return logits


    def _build_placeholders(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=self.input_size, name='input')
        self.labels = tf.placeholder(dtype=tf.int64, shape=None, name='label')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name='keep_prob')
        self.lr = tf.placeholder(dtype=tf.float32, shape=None, name='learning_rate')
        self.rho = tf.placeholder(dtype=tf.float32, shape=None, name='rho')
        self.batch_size = tf.placeholder(dtype=tf.int8, shape=None, name='batch_size')
        self.in_train = tf.placeholder(dtype=tf.bool, shape=None, name='in_train')

    def _optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.AdadeltaOptimizer(learning_rate=self.lr, rho=self.rho)\
                                              .minimize(global_step=self.global_step, loss=self.loss)

    def fit(self, config,
            n_epochs: int,
            batch_size: int,
            learning_rate_schedule: callable,
            verbose=True,
            prune=False,
            num_of_iteration=100):
        """
        :param config: a list of HP we work with
        :param n_epochs:  number of episodes
        :param batch_size: batch size for training
        :param learning_rate_schedule: changing learning rate according to current episode count
        :param verbose: print commments
        :param prune: if true the model will prune its weights each training epoch
        :param num_of_iteration: iterations per epoch
        :return: info about convergence and pruning accuracy
        """
        self.logger.info("SimpleNet : dropout ALL, with data aug, bias are zero and BN with moving average of 0.95 No Regularization - test 3 \n")
        sparsity_vs_accuracy = [[], []]
        train_vs_episodes = [[], []]
        val_vs_episodes = [[], []]
        x_train, y_train, x_val, y_val, x_test, y_test = load_cifar_10()
        best_acc = 0
        for e in range(n_epochs):
            train_acc_avg = 0
            for iteration in tqdm(range(num_of_iteration)):
                x_batch, y_batch = get_batch(batch_size, x_train, y_train)
                _, train_acc = self.sess.run([self.train_op, self.accuracy],
                                              feed_dict={self.input: x_batch,
                                              self.labels: y_batch,
                                              self.keep_prob: (1-self.dropout),
                                              self.lr: learning_rate_schedule(e),
                                              self.rho: simple_config.rho_rate_schedule(e),
                                              self.in_train: True})
                train_acc_avg += train_acc
            train_acc_avg = train_acc_avg / num_of_iteration
            val_acc, val_loss = self.evaluate(batch_size=batch_size, set_x=x_val, set_y=y_val)

            self.maybe_prune_and_maybe_print_results(config, train_acc_avg, val_acc, val_loss,
                                                     sparsity_vs_accuracy, n_epochs, e, prune, verbose)
            train_vs_episodes[0].append(e)
            train_vs_episodes[1].append(train_acc_avg)
            val_vs_episodes[0].append(e)
            val_vs_episodes[1].append(val_acc)
            if best_acc < val_acc:
                self.save_model()  # saves the model
                best_acc = val_acc
                print("Best Accuracy so Far is : {}".format(best_acc))
        print(" Best Accuracy is: {}".format(best_acc))
        return sparsity_vs_accuracy, val_vs_episodes, train_vs_episodes

    def evaluate(self, set_x, set_y, batch_size, num_of_iter=100, in_train=True):
        averaged_acc, averaged_loss = 0, 0
        for i in range(num_of_iter):
            x, y = get_batch(batch_size=batch_size, x_set=set_x, y_set=set_y)
            accuracy, loss = self.sess.run([self.accuracy, self.loss], feed_dict={
                self.input: x,
                self.keep_prob: 1,
                self.labels: y,
                self.in_train: in_train})
            averaged_acc += accuracy
            averaged_loss += loss
        return averaged_acc / num_of_iter, averaged_loss / num_of_iter