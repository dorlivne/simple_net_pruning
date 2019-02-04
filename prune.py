from Models import ConvNet, SimpleNet
from configs import ConfigSimpleNetPruned as prune_config
from utils.plot_utils import plot_conv_weights, plot_graph
from argparse import ArgumentParser


def main():
    prune_model = SimpleNet(input_size=prune_config.input_size, output_size=prune_config.output_size,
                          model_path=FLAGS.model_path, pruning_start=0, pruning_end=100000,
                          target_sparsity=FLAGS.goal_sparsity, dropout=FLAGS.dropout, pruning_freq=FLAGS.prune_freq, initial_sparsity=prune_config.initial_sparsity,
                          sparsity_start=prune_config.sparsity_start, sparsity_end=FLAGS.end_sparsity, scope='SimpleNetPruned')
    prune_model.load_model(FLAGS.ready_path)
    prune_model.reset_global_step()
    # important to allow a stable pruning, the pruning mechanism takes the global step as a parameter
    sparsity_vs_accuracy = prune_model.fit(n_epochs=FLAGS.n_epochs, learning_rate_schedule=prune_config.learning_rate_schedule,
                                           batch_size=FLAGS.batch_size, prune=True, config=prune_config)
    # the fit function allow gentle pruning along with fine-tuning to reach maximum sparsity and accuracy possible
    plot_graph(sparsity_vs_accuracy, "sparsity_vs_accuracy_simplenet")
    prune_model.load_model(FLAGS.best_path)
    plot_conv_weights(prune_model, title='after')
    prune_model.sess.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=prune_config.model_path,
        help=' Directory where to save model checkpoint.')
    parser.add_argument(
        '--ready_path',
        type=str,
        default=prune_config.ready_path,
        help=' path that depicts the location of the ready model.')
    parser.add_argument(
        '--best_path',
        type=str,
        default=prune_config.best_path,
        help=' Directory where to save the best model(highest sparsity with satisfying accuracy) checkpoint .')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=prune_config.n_epochs,
        help='number of epoches')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=prune_config.batch_size,
        help='batch_size')
    parser.add_argument(
        '--dropout',
        type=int,
        default=prune_config.dropout,
        help='dropout, for pruning it is recommended to leave it at 0')
    parser.add_argument(
        '--goal_sparsity',
        type=float,
        default=prune_config.target_sparsity,
        help='target sparsity of model, default value is 0.999 for maximum effort')
    parser.add_argument(
        '--end_sparsity',
        type=int,
        default=prune_config.sparsity_end,
        help='prune until epoch number end_sparsity, this effects the pruning sensitivity')
    parser.add_argument(
        '--prune_freq',
        type=int,
        default=prune_config.pruning_freq,
        help='depicts the frequency of pruning happening every prune_freq episodes')

    FLAGS, unparsed = parser.parse_known_args()
    main()