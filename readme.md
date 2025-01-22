A simple project using handwritten digits from the MNIST database to train and test a simple, three-layer autoencoder. 

Attempted fixes:
-Changed training DataLoader to  shuffle = True
-Changed reduction in BCEloss to reduction="sum" --> After this point, we at least produced things that looked like digits

References:
Umberto Michelucci- An Introduction to Autoencoders (2022)
