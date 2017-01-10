Test = 0
neuron = 8
Lr = 0.01
momentum = 0.01
titleString = 'Test : ' + str(Test) + ', Neurons : ' + str(neuron) + ', Learning Rate : ' + str(Lr) + ', Momentum : ' + str(momentum)
print(titleString)
                              
import neurolab as nl
import numpy as np
import pandas as pd
#import matplotlib.pyplot as pl

trainInput = pd.read_csv('TrainInput.csv')
trainTarget = pd.read_csv('TrainTarget.csv')
testInput = pd.read_csv('TestInput.csv')
testTarget = pd.read_csv('TestTarget.csv')



trainTarget['1'] = trainTarget['1'].map({1: -1, 2: 0, 3:1})
testTarget['1'] = testTarget['1'].map({1: -1, 2: 0, 3:1})

trainInput = (trainInput - trainInput.mean()) / (trainInput.max() - trainInput.min())
testInput = (testInput - testInput.mean()) / (testInput.max() - testInput.min())

# Create network with 2 inputs, 10 neurons in input layer and 1 in output layer
net = nl.net.newff([[-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0]], [neuron,1])


net.errorf = nl.error.SSE() 
net.trainf = nl.train.train_gdx

# Train network
error = net.train(trainInput, trainTarget, epochs=100000, show=1, lr=Lr, goal=0.001, mc=momentum)

# Simulate network
out = net.sim(trainInput)
out2 = net.sim(testInput)

# Plot result
import pylab as pl

pl.subplot(311)
pl.title(titleString)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')


pl.subplot(312)

pl.plot(out, label='Prediction')
pl.plot(trainTarget, label='trainTarget')
pl.xlabel('sample')
pl.ylabel('class')
pl.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


pl.subplot(313)
pl.plot(out2, label='Prediction')
pl.plot(testTarget, label='testTarget')
pl.xlabel('sample')
pl.ylabel('class')
pl.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pl.show()









