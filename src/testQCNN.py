from QCNN import NeuralNetwork
import pandas as pd


nn= NeuralNetwork(2,2,3,1,.1,100000, train_test_split =1, split=False)

data_dict = {0:{'input1':0,'input2':0,'output':0},
            1:{'input1':1,'input2':0,'output':1},
            2:{'input1':0,'input2':1,'output':1},
            3:{'input1':1,'input2':1,'output':0}
            }

df = pd.DataFrame.from_dict(data_dict, orient = 'index')



nn.train(df)
