from Models import SimpleNet
from configs import ConfigSimpleNetDense as simple_config
from utils.tensorflow_utils import load_cifar_10
from argparse import ArgumentParser

def main():
    dense_model = SimpleNet(input_size=simple_config.input_size, output_size=simple_config.output_size,
                            model_path=FLAGS.model_path)
    dense_model.load_model()
    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar_10()
    test_acc, test_loss = dense_model.evaluate(set_x=x_test, set_y=y_test, batch_size=128)
    print('Accuracy on test: {accuracy}, loss on test: {loss}'.format(
                accuracy=test_acc, loss=test_loss))
    if test_acc > 0.8:
        dense_model.save_model(simple_config.ready_path)
    dense_model.sess.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=simple_config.model_path,
        help=' Directory where to save model checkpoint.')

    FLAGS, unparsed = parser.parse_known_args()
    main()