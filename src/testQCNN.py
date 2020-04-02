from QCNN import NeuralNetwork
import pandas as pd
import matplotlib.pyplot as plt

nn= NeuralNetwork(2,2,3,1,.1,100000,'c')

data_dict = {0:{'input1':0,'input2':0,'output':0},
            1:{'input1':1,'input2':0,'output':1},
            2:{'input1':0,'input2':1,'output':1},
            3:{'input1':1,'input2':1,'output':0}
            }

df = pd.DataFrame.from_dict(data_dict, orient = 'index')


nn.train(df)
