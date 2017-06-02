import cv2
import numpy as np
from image_functions import Affine, Motionblur
from scipy.ndimage.measurements import label

import random
import copy

import keras.backend as K

from keras.layers import Activation, BatchNormalization, Conv2D, Dropout, concatenate
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import UpSampling2D, Conv2DTranspose, Reshape
from keras.regularizers import l2


"""
Code for the DenseNet implementation based on:
https://github.com/titu1994/DenseNet/

[1] [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v3.pdf)
[2] [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326v2.pdf)

"""


def preprocess_trn_img(img, img_mask, tgt_sz=(240, 384), crop_sz=(224, 224),
                       aug=False, crop=False, p_blur=0.2):

    # Resizing
    img = cv2.resize(img, tgt_sz[::-1])
    img_mask = cv2.resize(img_mask, tgt_sz[::-1])
    img_mask = np.reshape(img_mask, (np.shape(img_mask)[0], np.shape(img_mask)[1], 1))

    # Augmentation
    if aug:
        img, img_mask = Affine(img, img_mask)
        img_mask = img_mask.astype(np.uint8)

        p = random.uniform(0., 1.)
        if p_blur >= p:
            img = Motionblur(img)

        img = (img * 255).astype(np.uint8)
        img_mask = (img_mask * 255).astype(np.uint8)

    # Random crop
    if crop:
        img_h = img.shape[0]
        img_w = img.shape[1]

        dx = np.random.randint(0, img_w - crop_sz[1])
        dy = np.random.randint(0, img_h - crop_sz[0])

        img = img[dy:dy + crop_sz[0], dx:dx + crop_sz[1], :]
        img_mask = img_mask[dy:dy + crop_sz[0], dx:dx + crop_sz[1], :]

    return img, img_mask


def preprocess_val_img(img, img_mask, crop_sz=(224, 224)):

    img_h = float(img.shape[0])
    img_w = float(img.shape[1])

    rescaling_factor = np.min((img_w / crop_sz[1], img_h / crop_sz[0]))
    new_sz = (int(img_h / rescaling_factor), int(img_w / rescaling_factor))

    # Resizing
    img = cv2.resize(img, new_sz[::-1])
    img_mask = cv2.resize(img_mask, new_sz[::-1])
    img_mask = np.reshape(img_mask, (np.shape(img_mask)[0], np.shape(img_mask)[1], 1))

    # Center crop
    img_h = img.shape[0]
    img_w = img.shape[1]

    if img_h == crop_sz[0]:
        dy = 0
    else:
        dy = int((img_h - crop_sz[0]) / 2)

    if img_w == crop_sz[1]:
        dx = 0
    else:
        dx = int((img_w - crop_sz[1]) / 2)

    img = img[dy:dy + crop_sz[0], dx:dx + crop_sz[1], :]
    img_mask = img_mask[dy:dy + crop_sz[0], dx:dx + crop_sz[1], :]

    return img, img_mask


def generator(filepath_list, batch_size=32, tgt_sz=(240, 384), crop_sz=(224, 224),
              shuffle_data=True, aug=True, crop=True, valid=False):
    """
    This generator receives a lisit of filenames from our dataset and returns a preprocessed
    image and mask

    Input:
    filepath_list - list with path to SVHN images.

    Arguments:
    batch_size - size of the mini batch
    tgt_sz - size of the first image rescaling
    crop_sz - size of the final image size after cropping
    aug - boolean to determine whether augmentation will be performed
    crop - boolean to determine whether random cropping will be performed
    valid - boolean to use a different pipeline when processing images
            from the validation set

    Outputs:
    X and y for training or validation

    """

    num_samples = len(filepath_list)
    filelist = copy.copy(filepath_list)

    if shuffle_data and not valid:
        random.shuffle(filelist)

    while True:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = filelist[offset:offset + batch_size]

            if shuffle_data and not valid:
                random.shuffle(batch_samples)

            images = []
            masks = []

            for batch_sample in batch_samples:
                img = cv2.imread('./datasets/deeplearning/udacity/' + batch_sample, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_mask = cv2.imread('./datasets/deeplearning/masks/mask_' + batch_sample, cv2.IMREAD_GRAYSCALE)
                img_mask = np.reshape(img_mask, (np.shape(img_mask)[0], np.shape(img_mask)[1], 1))

                if valid:
                    img, img_mask = preprocess_val_img(img, img_mask, crop_sz=crop_sz)
                else:
                    img, img_mask = preprocess_trn_img(img, img_mask, tgt_sz=tgt_sz,
                                                       crop_sz=crop_sz, aug=aug, crop=crop)

                images.append(img)
                masks.append(img_mask)

            X = np.array(images) / 255
            y = np.array(masks) / 255

            yield X, y


def ConvBlock(x, n_filters, bottleneck=False, p=0, decay=1e-4):

    if bottleneck:
        x = BatchNormalization(gamma_regularizer=l2(decay), beta_regularizer=l2(decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(n_filters * 4, 1, padding='same', use_bias=False,
                   kernel_initializer='he_uniform', kernel_regularizer=l2(decay))(x)

    x = BatchNormalization(gamma_regularizer=l2(decay), beta_regularizer=l2(decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, 3, padding='same', use_bias=False,
               kernel_initializer='he_uniform', kernel_regularizer=l2(decay))(x)

    if p > 0:
        x = Dropout(p)(x)

    return x


def DenseBlock(x, n_layers, n_filters, growth_rate, p=0, decay=1e-4,
               bottleneck=False, grow_nb_filters=True, return_concat_list=False):

    x_list = [x]

    for i in range(n_layers):
        cb = ConvBlock(x, growth_rate, bottleneck, p, decay)
        x_list.append(cb)

        x = concatenate([x, cb])

        if grow_nb_filters:
            n_filters += growth_rate

    if return_concat_list:
        return x, n_filters, x_list
    else:
        return x, n_filters


def TransitionBlock(x, n_filters, compression=1, p=0, decay=1e-4, kernel_sz=3):

    x = BatchNormalization(gamma_regularizer=l2(decay), beta_regularizer=l2(decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(int(n_filters * compression), kernel_sz, padding='same', use_bias=False,
               kernel_initializer='he_uniform', kernel_regularizer=l2(decay))(x)

    if p > 0:
        x = Dropout(p)(x)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    return x


def TransitionUpBlock(x, n_filters, decay=1e-4, up_type='upsampling'):

    if up_type == 'upsampling':
        x = UpSampling2D()(x)
    else:
        x = Conv2DTranspose(n_filters, 3, padding='same', activation='relu',
                            strides=(2, 2), kernel_initializer='he_uniform')(x)
    return x


def get_DenseNet(n_classes, img_input, depth=40, n_dense_blocks=3, growth_rate=12,
                 n_filters=-1, n_layers_per_block=-1, bottleneck=False, compression=1,
                 p=0, decay=1e-4, large_input=False, include_top=True, activation='softmax'):

    assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4'

    if compression != 0.0:
        assert compression <= 1.0 and compression > 0.0, 'compression range needs to be between 0.0 and 1.0'

    # layers in each dense block
    if type(n_layers_per_block) is list or type(n_layers_per_block) is tuple:
        nb_layers = list(n_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (n_dense_blocks + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if n_layers_per_block == -1:
            count = int((depth - 4) / 3)
            nb_layers = [count for _ in range(n_dense_blocks)]
            final_nb_layer = count
        else:
            final_nb_layer = n_layers_per_block
            nb_layers = [n_layers_per_block] * n_dense_blocks

    if bottleneck:
        nb_layers = [int(layer // 2) for layer in nb_layers]

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if n_filters <= 0:
        n_filters = 2 * growth_rate

    # Initial convolution
    if large_input:
        x = Conv2D(n_filters, 7, strides=(2, 2), padding='same', use_bias=False, name='initial_conv2D',
                   kernel_initializer='he_uniform', kernel_regularizer=l2(decay))(img_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    else:
        x = Conv2D(n_filters, 3, padding='same', use_bias=False, name='initial_conv2D',
                   kernel_initializer='he_uniform', kernel_regularizer=l2(decay))(img_input)

    # Add dense blocks
    for block_idx in range(n_dense_blocks - 1):
        x, n_filters = DenseBlock(x, nb_layers[block_idx], n_filters, growth_rate,
                                  p=p, decay=decay, bottleneck=bottleneck)
        # add transition_block
        x = TransitionBlock(x, n_filters, compression=compression, p=p, decay=decay)

        n_filters = int(n_filters * compression)

    # The last dense_block does not have a transition_block
    x, n_filters = DenseBlock(x, final_nb_layer, n_filters, growth_rate,
                              p=p, decay=decay, bottleneck=bottleneck)

    # Final BN-RELU-POOL
    x = BatchNormalization(gamma_regularizer=l2(decay), beta_regularizer=l2(decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dense(n_classes, activation=activation, W_regularizer=l2(decay), b_regularizer=l2(decay))(x)

    return x


def get_FCN_DenseNet(img_input, n_classes=1, n_dense_blocks=5, growth_rate=16,
                     n_layers_per_block=4, compression=1, p=0, decay=1e-4,
                     input_shape=(224, 224, 3), init_conv_filters=48, include_top=True,
                     activation='softmax', upsampling_conv=128, upsampling_type='upsampling',
                     verbose=0):

    upsampling_type = upsampling_type.lower()

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be either "softmax" or "sigmoid"')

    if activation == 'sigmoid' and n_classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    if compression != 0.0:
        assert compression <= 1.0 and compression > 0.0, 'compression range needs to be between 0.0 and 1.0'

    # check if upsampling_conv has minimum number of filters
    # minimum is set to 12, as at least 3 color channels are needed for correct upsampling
    assert upsampling_conv > 12 and upsampling_conv % 4 == 0, 'Parameter `upsampling_conv` number of channels must ' \
                                                              'be a positive number divisible by 4 and greater ' \
                                                              'than 12'

    # layers in each dense block
    if type(n_layers_per_block) is list or type(n_layers_per_block) is tuple:
        nb_layers = list(n_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (n_dense_blocks + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'

        bottleneck_nb_layers = nb_layers[-1]
        rev_layers = nb_layers[::-1]
        nb_layers.extend(rev_layers[1:])
    else:
        bottleneck_nb_layers = n_layers_per_block
        nb_layers = [n_layers_per_block] * (2 * n_dense_blocks + 1)

    # Initial convolution
    x = Conv2D(init_conv_filters, 3, padding='same', use_bias=False, name='initial_conv2D',
               kernel_initializer='he_uniform', kernel_regularizer=l2(decay))(img_input)

    n_filters = init_conv_filters

    skip_list = []

    # Add dense blocks and transition down blocks
    for block_idx in range(n_dense_blocks):
        if verbose > 0:
            print ("\nAdding Dense Block {}".format(block_idx))
            print ("Layers: {}".format(nb_layers[block_idx]))
            print ("Filters: {}".format(n_filters))
            print ("Growth Rate: {}".format(growth_rate))
        x, n_filters = DenseBlock(x, nb_layers[block_idx], n_filters, growth_rate,
                                  p=p, decay=decay, bottleneck=False)

        # store skip connections
        skip_list.append(x)

        # add transition_block
        if verbose > 0:
            print ("\nAdding TD Block {}".format(block_idx))
            print ("Filters: {}".format(n_filters))
            print ("Compression Rate: {}".format(compression))
        x = TransitionBlock(x, n_filters, compression=compression, p=p, decay=decay, kernel_sz=1)

        n_filters = int(n_filters * compression)

    # The last dense block does not have a transition down block
    # the block below returns the concatenated feature map without the concatenation of the input
    if verbose > 0:
        print ("\nAdding Bottleneck Block")
        print ("Layers: {}".format(bottleneck_nb_layers))
        print ("Filters: {}".format(n_filters))
        print ("Growth Rate: {}".format(growth_rate))

    _, n_filters, concat_list = DenseBlock(x, bottleneck_nb_layers, n_filters, growth_rate,
                                           p=p, decay=decay, bottleneck=False,
                                           return_concat_list=True)
    # Reverse the skip list
    skip_list = skip_list[::-1]

    # Add dense blocks and transition up blocks
    for block_idx in range(n_dense_blocks):
        n_filters_keep = growth_rate * nb_layers[n_dense_blocks + block_idx]

        # Upsamplig block upsamples only the feature maps
        l = concatenate(concat_list[1:])

        if verbose > 0:
            print ("\nAdding TU Block {}".format(block_idx))
            print ("Filters: {}".format(n_filters_keep))
        t = TransitionUpBlock(l, n_filters=n_filters_keep, up_type=upsampling_type)

        # concatenate skip connection with transition block
        x = concatenate([t, skip_list[block_idx]])

        # To prevent feature map size to grow in upsampling dense blocks
        # we set grow_nb_filters to False.
        if verbose > 0:
            print ("\nAdding Dense Block {}".format(block_idx))
            print ("Layers: {}".format(nb_layers[n_dense_blocks + block_idx + 1]))
            print ("Filters: {}".format(growth_rate))
            print ("Growth Rate: {}".format(growth_rate))
        DB_up, n_filters, concat_list = DenseBlock(x, nb_layers[n_dense_blocks + block_idx + 1],
                                                   n_filters=growth_rate, growth_rate=growth_rate,
                                                   p=p, decay=decay, bottleneck=False,
                                                   return_concat_list=True, grow_nb_filters=False)

    if include_top:
        x = Conv2D(n_classes, 1, padding='same', activation='linear', use_bias=False,
                   kernel_regularizer=l2(decay))(DB_up)
        rows, cols, channels = input_shape

        x = Reshape((rows * cols, n_classes))(x)
        x = Activation(activation)(x)
        x = Reshape((rows, cols, n_classes))(x)

    return x


def IOU_calc(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IOU_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)


def get_bboxes(img, mask, min_sz=(50, 50)):

    heatmap = mask[:, :, 0]

    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(img), labels, min_sz)

    return draw_img


def draw_labeled_bboxes(img, labels, min_sz=(50, 50), verbose=0):

    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):

        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if verbose > 0:
            print ("\nMin_y: ", np.min(nonzeroy))
            print ("Max_y: ", np.max(nonzeroy))
            print ("\nMin_x: ", np.min(nonzerox))
            print ("Max_y: ", np.max(nonzerox))
            print ("\nHeight: ", np.max(nonzerox) - np.min(nonzerox))
            print ("\nWidth: ", np.max(nonzeroy) - np.min(nonzeroy))

        # Define a bounding box based on min/max x and y
        if ((np.max(nonzeroy) - np.min(nonzeroy) > min_sz[1]) & (np.max(nonzerox) - np.min(nonzerox) > min_sz[0])):
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # Draw the box on the image
            if verbose > 0:
                print ("drawing box for label: ", car_number)
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 1)

    return img


def vehicle_detection(image, model, model_input_sz=(352, 224), bbox_min_sz=(50, 50),
                      lanes=None, heatmap=False):

    img_h = image.shape[0]
    img_w = image.shape[1]

    # We resize our image and add a new dimension (for batch size)
    rsz = cv2.resize(image, (352, 224))
    rsz = np.expand_dims(rsz, 0) / 255

    # Predict vehicles
    car_mask = model.predict(rsz)

    # Convert it to 8 bit integer and scale back to 0-255
    car_mask = np.array(255 * car_mask[0], dtype=np.uint8)

    # Scale the mask up to the original image dimensions
    car_mask = cv2.resize(car_mask, (img_w, img_h))

    if heatmap:
        # Create a 3 channel version, with values only on the Blue channel
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[:, :, 2] = car_mask

        # Add both images together
        if lanes is not None:
            result = cv2.addWeighted(mask, 0.4, lanes, 1, 0)
        else:
            result = cv2.addWeighted(mask, 0.4, image, 1, 0)
    else:
        if lanes is not None:
            result = get_bboxes(lanes, np.expand_dims(car_mask, -1), min_sz=bbox_min_sz)
        else:
            result = get_bboxes(image, np.expand_dims(car_mask, -1), min_sz=bbox_min_sz)

    return result
