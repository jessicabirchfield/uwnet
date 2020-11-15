from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            # make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            # make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            # make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            # make_batchnorm_layer(10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = 0.09 # .01
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

# 7.6 Question: What do you notice about training the convnet with/without batch normalization?
# How does it affect convergence? How does it affect what magnitude of learning rate you can use?
# Write down any observations from your experiments:
#
# Training the convnet with batch normalization led to a higher test accuracy of 54.9%,
# compared with a test accuracy of 40.1% when training without batch normalization.
# Without batch normalization, the loss didn't really start converging until around the 30th iteration.
# However, with batch normalization the model converged a lot faster. The loss flucuated a lot during
# the first 3-4 iterations, but then started to converge rapidly by the 5th iteration.
# Regarding the learning rate, we found that generally with batch normalization we could use a higher
# learning rate and achieve higher accuracy than without batch normalization.  Using batch
# normalization and a higher learning rate of 0.09, we achieved a 56.9% test accuracy. Using
# this same learning rate without batch normalization gave a lower test accuracy of 39.4%.
