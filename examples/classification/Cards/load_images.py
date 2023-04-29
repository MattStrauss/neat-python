
def load_images(max_images, img_height, labels):
    import numpy as np
    from PIL import Image, ImageChops, ImageOps
    import pandas as pd
    import os

    # read metadata for images
    root = os.getcwd()
    filepath = root + "\\Data\\"
    cards = pd.read_csv(filepath + "cards.csv")
    # select test data set
    cards_train = cards.loc[cards['data set'] == 'train']

    # # limit training set to 25 images per class = 53 * 25 = 1325 images
    # max_images = 25     # max number of images per label
    # img_height = 8      # Original size is 224x224x3, will convert to 8x8x1 grayscale
    # labels = 53         # number of labels.

    N = max_images*labels
    inputs = np.empty([N, img_height*img_height])
    outputs = np.empty(N, dtype=int)
    labels_index = range(0,labels+1)
    img_index = 0
    for index in labels_index:
        # Add labels to outputs array
        start_index = index * max_images
        end_index = (index + 1) * max_images

        if start_index < N:
            outputs[start_index : end_index] = index
            temp_view = cards_train.loc[cards['class index'] == index]
            counter = 0
            for index_cards, row in temp_view.iterrows():
                if counter < max_images:
                    img_path = root + "\\Data\\" + row['filepaths']
                    temp = Image.open(img_path).convert('L')     # convert to grayscale
                    temp = temp.resize((img_height,img_height), Image.LANCZOS)  # resize image
                    pix = np.array(temp)
                    pix = pix.flatten() # Flatten 2D image into 1D
                    inputs[img_index, :] = pix  # add to inputs np array
                    counter += 1
                    img_index += 1
                else:
                    break

    return inputs, outputs

def shuffle_images(images, labels):
    import random
    import numpy as np

    indeces = list(range(0, len(labels)))
    random.shuffle(indeces)   # shuffle the indeces

    input = np.empty_like(images)
    output = np.empty_like(labels)
    index = 0
    for i in indeces:
        input[index, :] = images[i, :]
        output[index] = labels[i]
        index += 1

    return input, output


