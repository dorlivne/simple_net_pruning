from Models import SimpleNet
from configs import ConfigSimpleNetPruned as prune_config
from argparse import ArgumentParser
from model_prune_example.strip_prune_vars import strip_pruning_vars
from utils.tensorflow_utils import  load_cifar_10

def  main():
    pruned_model = SimpleNet(input_size=prune_config.input_size, output_size=prune_config.output_size,
                            model_path=FLAGS.best_path)
    pruned_model.load_model()
    _, _, _, _, x_test, y_test = load_cifar_10()
    test_acc, test_loss = pruned_model.evaluate(x_test, y_test, FLAGS.batch_size)
    sparsity = pruned_model.get_model_sparsity()
    print("sparsity : {} accuracy : {}".format(sparsity, test_acc))
    # this functions needs the name of the output nodes, in this case there is only one named output
    strip_pruning_vars(checkpoint_dir=FLAGS.best_path, output_node_names='output/add',
                       output_dir=FLAGS.output_dir, filename='simplenet_pruned.pb')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        default=prune_config.output_dir,
        help=' Directory where to save model checkpoint.')
    parser.add_argument(
        '--best_path',
        type=str,
        default=prune_config.best_path,
        help=' location of best_pruned model.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=prune_config.batch_size,
        help='batch_size')
    FLAGS, unparsed = parser.parse_known_args()
    main()