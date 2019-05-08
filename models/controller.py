#%%
import torch
from torch import nn
import torch.nn.functional as F

#%%
class Controller(nn.Module):
    #TODO: Create batched version 
    def __init__(self, num_inputs, controller_size, num_outputs, read_data_size):
        super(Controller,self).__init__()

        self.num_inputs      = num_inputs
        self.controller_size = controller_size
        self.num_outputs     = num_outputs
        self.read_data_size  = read_data_size

        # The LSTM controller memory
        self.lstm = nn.LSTMCell(self.num_inputs, self.controller_size)

        # The final Fully Connected layer that will tranform reads to outputs
        #  [controller_output; previous_reads ] -> output
        self.output_fc = nn.Linear(self.read_data_size, self.num_outputs)
        
        # initialize the fully connected network 
        nn.init.kaiming_uniform_(self.output_fc.weight)
        nn.init.normal_(self.output_fc.bias , std=0.01)

        # The inital bias values for the hidden and cell state of the LSTM
        # these are meant to be reset after every input sequence
        # also they are mean to be learnt by the NTM
        self.lstm_h = torch.zeros([1,self.controller_size])
        self.lstm_c = torch.zeros([1,self.controller_size])

        # Networks that help learn the bias for the LSTM
        self.lstm_h_fc = nn.Linear(1,self.controller_size)
        self.lstm_c_fc = nn.Linear(1,self.controller_size)

        # Call the function that will reset the controller to bias values
        self.reset() 
    
    def forward(self, input_data, prev_reads):
        # concatenate the input and the previous read data
        x = torch.cat([input_data] +prev_reads,dim=1)
        self.lstm_h, self.lstm_c = self.lstm(x, (self.lstm_h,self.lstm_c))

        return self.lstm_h, self.lstm_c 

    def output(self, read_data):
        state = torch.cat([self.lstm_h]+read_data,dim=1)
        output = self.output_fc(state)
        output = F.sigmoid(output)
        return output
    
    def reset(self):
        in_data = torch.tensor([[0.]])  # dummy input
        self.lstm_h= self.lstm_h_fc(in_data)
        self.lstm_c  = self.lstm_c_fc(in_data)

#%%
        
contr = Controller(10+20,100,8,120)





