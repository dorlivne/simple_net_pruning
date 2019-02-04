from Models import  SimpleNet
from configs import ConfigSimpleNetDense as simple_config
from utils.plot_utils import plot_conv_weights, plot_several_graphs
from argparse import ArgumentParser


def main():
    dense_model = SimpleNet(input_size=simple_config.input_size, output_size=simple_config.output_size,
                            model_path=FLAGS.model_path, dropout=FLAGS.dropout)
    sparsity_vs_accuracy, val_vs_episodes, train_vs_episodes = dense_model.fit(n_epochs=FLAGS.n_epochs, learning_rate_schedule=simple_config.learning_rate_schedule,
                                                                               batch_size=FLAGS.batch_size, config=simple_config)
    fig = plot_several_graphs(data=val_vs_episodes, name='val_acc')
    _ = plot_several_graphs(data=train_vs_episodes, name='train_acc', last=1)
    plot_conv_weights(dense_model)
    dense_model.sess.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=simple_config.model_path,
        help=' Directory where to save model checkpoint.')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=simple_config.n_epochs,
        help='number of epoches')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=simple_config.batch_size,
        help='batch_size')
    parser.add_argument(
        '--dropout',
        type=int,
        default=simple_config.dropout,
        help='dropout')
    FLAGS, unparsed = parser.parse_known_args()
    main()