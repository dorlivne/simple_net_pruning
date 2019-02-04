from matplotlib import pyplot as plt

def plot_weights(agent, title: str):
    weights_matrices = agent.sess.run(agent.weights_matrices)
    plot_histogram(weights_matrices, title, include_zeros=False)

def plot_histogram(weights_list: list,
                   image_name: str,
                   include_zeros=True,
                   range=(-0.15, 0.15)):

    """A function to plot weights distribution"""

    weights = []
    for w in weights_list:
        weights.extend(list(w.ravel()))

    if not include_zeros:
        weights = [w for w in weights if w != 0]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.hist(weights,
            bins=100,
            facecolor='green',
            edgecolor='black',
            alpha=0.7,
            range=range)

    ax.set_title('Weights distribution')
    ax.set_xlabel('Weights values')
    ax.set_ylabel('Number of weights')

    fig.savefig(image_name + '.png')


def plot_graph(data, name: str):
  fig = plt.figure()
  x = data[0]
  y = data[1]
  plt.plot(x[:], y[:], 'ro')
  plt.xlabel('sparsity')
  plt.ylabel('accuracy')
  plt.title(name)
  plt.grid()
  fig.savefig(name + '.png')

def plot_several_graphs(data, name:str, last = 0):
    """
    a function to plot multiple plots on the same figure
    :param data: data to plot on graph
    :param name: name of data to write on legend
    :param last: if this is the last plot session then we save this figure
    :return:
    """
    fig = plt.figure(0)
    x = data[0]
    y = data[1]
    plt.plot(x[:], y[:], label=name)
    if last:
        plt.legend()
        plt.xlabel('episodes')
        plt.ylabel('accuracy')
        plt.title(name)
        plt.grid()
        fig.savefig('accuracy.png')

def plot_conv_weights(model, title='weights'):
    weights = model.get_flat_weights()
    plot_histogram(weights_list=weights, image_name=title, include_zeros=False)

