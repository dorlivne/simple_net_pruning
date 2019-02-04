
class ConfigNetworkDense:
    model_path = 'saved_models/network_dense'
    target_path = 'saved_models/network_dense_target'
    ready_model = 'saved_models/network_dense_ready'
    n_epochs = 10000
    batch_size = 128
    learning_rate = 5e-5

class ConfigConvNet:
    input_size = (None, 28, 28, 1)
    output_size = (None, 10)
    dropout = 0.5
    model_path = 'saved_models/network_dense_CONV'
    n_epochs = 25
    batch_size = 128


    @staticmethod
    def learning_rate_schedule(epoch):

        if epoch < 10:
            return 1e-2
        elif epoch < 20:
            return 1e-3
        else:
            return 1e-4


class ConfigConvNetPruned:
    input_size = (None, 28, 28, 1)
    output_size = (None, 10)
    dropout = 0.0
    n_epochs = 300
    batch_size = 128
    model_path = 'saved_models/network_dense_pruned_Conv'
    best_path = 'saved_models/network_dense_pruned_Conv_best'
    sparse_output_dir = 'saved_models/network_dense_sparse_Conv'

    @staticmethod
    def no_train(epoch):
        return 0.0

    @staticmethod
    def learning_rate_schedule(epoch):

        if epoch < 10:
            return 1e-3
        elif  10 <= epoch < 20:
            return 1e-4
        else:
            return 1e-5


class ConfigSimpleNetDense:
    input_size = (None, 32, 32, 3)
    output_size = (None, 10)
    dropout = 0.2
    n_epochs = 2000
    batch_size = 64
    ready_path = 'saved_models/network_dense_SimpleNet_ready'
    model_path = 'saved_models/network_dense_SimpleNet'
    best_path = 'saved_models/network_dense_SimpleNet_best'
    sparse_output_dir = 'saved_models/network_dense_sparse_SimpleNet'
    @staticmethod
    def learning_rate_schedule(epoch):

        if epoch < 50:
            return 0.7
        elif 50 <= epoch < 125:
            return 0.5
        elif 125 <= epoch < 150:
            return 0.25
        elif 150 <= epoch < 175:
            return 0.1
        elif 175 <= epoch < 250:
            return 1e-2
        else:
            return 5e-3



    @staticmethod
    def rho_rate_schedule(epoch):

        if epoch < 75:
            return 0.95
        elif 75 <= epoch < 100:
            return 0.9
        elif 100 <= epoch < 125:
            return 0.85
        elif 125 <= epoch:
            return 0.7

class ConfigSimpleNetPruned:
    input_size = (None, 32, 32, 3)
    output_size = (None, 10)
    dropout = 0.0
    n_epochs = 500
    batch_size = 64
    ready_path = 'saved_models/network_dense_SimpleNet_ready'
    model_path = 'saved_models/network_pruned_SimpleNet'
    best_path = 'saved_models/network_pruned_SimpleNet_best'
    output_dir = 'saved_models/network_pruned_sparse_SimpleNet'
    pruning_state = 0
    pruning_end = int(1e5)
    target_sparsity = 0.9999
    pruning_freq = 100
    initial_sparsity = 0
    sparsity_start = 0
    sparsity_end = 100000

    def no_train(epoch):
        return 0.0

    @staticmethod
    def learning_rate_schedule(epoch):

        if epoch < 10:
            return 1e-3
        elif epoch < 20:
            return 1e-4
        else:
            return 1e-5

