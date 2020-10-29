from uwnet import *

def conv_net():
        l = [   make_convolutional_layer(32, 32, 3, 4, 2, 1),
                make_activation_layer(RELU),
                make_maxpool_layer(32, 32, 4, 3, 2),
                make_connected_layer(1024, 10),
                make_activation_layer(SOFTMAX)]
        return make_net(l)

    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# When using a similar number of operations, the fully connected network had an accuracy of 35% while the
# convnet had an accuracy of 51%. These results make sense because convolutional neural networks reduce the image
# input using convolutions, which allows them to be good at extracting features from images based on spatial relationships.
# On the other hand, fully connected networks are densely connected but don't always consider features from
# an image that are most useful for identification (for example, not considering relations by region within an image).
