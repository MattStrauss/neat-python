"""
# Cards classification example using a feed-forward network #

How to run:
1) Download dataset from https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification
2) Save in Data folder
3) For Windows system: Open the "cards.csv" file in Excel and replace the forward slash / with \ in
   the filepath.
4) Modify the following fields to vary the size of the loaded images: max_images, img_height, labels
"""

import sys
import os
import neat
import visualize
import numpy as np

from load_images import load_images, shuffle_images

# limit training set to 25 images per class = 53 * 25 = 1325 images
max_images = 50     # max number of images per label
img_height = 32      # Original size is 224x224x3, will convert to 8x8x1 grayscale
labels = 5         # number of labels.
inputs, outputs = load_images(max_images, img_height, labels)    # Load images, returns 25 images per class

inputs, outputs = shuffle_images(inputs, outputs)   # shuffle the images

def select_one(array):
    max_index = np.argmax(array)
    array = np.zeros(len(array))
    array[max_index] = 1

    return array


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi, multi_classification=True)
            output = np.argmax(output)
            if output == xo:
                genome.fitness += 1


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 500)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(inputs, outputs):
        output = select_one(winner_net.activate(xi))
        output = np.argmax(output)
        print("expected output {!r}, got {!r}".format(xo, output))

    ####################################################################################
    # # Code for restoring a checkpoint and running from there.
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-181')
    # p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # winner = p.run(eval_genomes, 1)
    ####################################################################################

    # visualize.draw_net(config, winner, True)
    visualize.draw_net(config, winner, True, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)

