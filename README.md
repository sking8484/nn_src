# Description  

Quantcore's implementation of a deep neural network. Used for classification and regression.

#Usage

The class takes six requirements, # input nodes,  # hidden nodes, # layers, # output nodes, #learning rate, output type (specificed with r for regression, c for classification)

The call to train the function is class_.train(input,answer)

The call to test is class_.feed_forward(input)

#Installation

```
pip install QCNN
```

#Sample Code
```

from QCNN import NeuralNetwork
import pandas as pd
import numpy as np

data = []
x = np.linspace(-1,1,401)
y = np.sin(4*x)
for point in range(len(x)):
    data.append([x[point],y[point]])


def shuffle_data():
    train_data = data[:-50]
    test_data = data
    np.random.shuffle(train_data)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_inputs = train_data[train_data.columns[0]].values
    train_answers = train_data[train_data.columns[1]].values
    test_inputs = test_data[test_data.columns[0]].values
    test_answers = test_data[test_data.columns[1]].values

    return train_inputs,train_answers,test_inputs,test_answers






nn = NeuralNetwork(2,2,2,3,.001,'r')



for epoch in range(1000):
    if epoch %10 ==0:
        print(epoch/100)
    normalized_data,inputs,test_data,testing_inputs = shuffle_data()


    for i in range(int(len(normalized_data)/10)):

        nn.train(list(normalized_data.iloc[i]),inputs[i],epoch)

num_correct = 0
points = []
correct = []
for test in range(len(test_data)):
    #inputs = test_data[test]
    nn_output = nn.feed_forward(list(test_data.iloc[test]))
    points.append(nn_output[0])
    output = np.argmax(nn_output)
    ti = testing_inputs[test]

    correct.append(ti)


    answer = np.argmax(ti)

    if output == answer:
        num_correct+= 1

print(num_correct/len(testing_inputs))
plt.plot(points,label='Prediction')
plt.plot(correct,label='Actual')
plt.legend()

plt.show()
```
