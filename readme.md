A simple project using handwritten digits from the MNIST database to train and test a simple, three-layer autoencoder. 

TO DO:
-Rewrite data.py to only load the datasets once. (Obvious)
-Set the train_loader data to shuffle.
-Figure out what the hell is going on with validation- maybe generate some images at some point in time to see how we could possibly be getting high validation.
-Investigate the output of BCEloss using the validation image that I loaded in mnist_manip.ipynb- if this is low, something is not working, obviously.

References:
Umberto Michelucci- An Introduction to Autoencoders (2022)
