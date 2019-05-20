#%%
import torch
from torch import nn
import torch.nn.functional as F

#%%
class Controller(nn.Module):
    """The LSTM controller for the NTM. It has an lstm cell that acts as working memory and also a fully connected layer to display output from the conroller.
    
    
    """
    def __init__(self, input_size, controller_size, output_size, read_data_size, device):
        super().__init__()

        self.device = device
        
        self.num_inputs      = input_size
        self.controller_size = controller_size
        self.num_outputs     = output_size
        self.read_data_size  = read_data_size

        # The LSTM controller memory
        self.lstm = nn.LSTMCell(self.num_inputs, self.controller_size).to(device)

        # The final Fully Connected layer that will tranform reads to outputs
        #  [controller_output; previous_reads ] -> output
        self.output_fc = nn.Linear(self.read_data_size, self.num_outputs).to(device)
        
        # initialize the fully connected network 
        nn.init.kaiming_uniform_(self.output_fc.weight)
        nn.init.normal_(self.output_fc.bias , std=0.01)

        # The inital bias values for the hidden and cell state of the LSTM
        # these are meant to be reset after every input sequence
        # also they are mean to be learnt by the NTM
        self.lstm_h = torch.zeros([1,self.controller_size],device = self.device)
        self.lstm_c = torch.zeros([1,self.controller_size],device = self.device)

        # Networks that help learn the bias for the LSTM
        self.lstm_h_fc = nn.Linear(1,self.controller_size).to(device)
        self.lstm_c_fc = nn.Linear(1,self.controller_size).to(device)

        # Call the function that will reset the controller to bias values
        self.reset() 
    
    def forward(self, input_data, prev_reads):
        """Performs a forward propagation of the LSTM controller 
        
        Args:
            input_data (tensor): shape is (batch_size, input_size)
            prev_reads (list of tensors): each list element is a read head's previous data ,shape (batch_size, M) 
        
        Returns:
            hiiden_state,cell_state: The LSTM states
        """ 
        # concatenate the input and the previous read data
        # (batch_size, input_size) + (batch_size, M) -> (batch_size, input_size +M)
#         print(f"input_data :{input_data}")
#         print(prev_reads)
        x = torch.cat([input_data]+prev_reads,dim=1)
        self.lstm_h, self.lstm_c = self.lstm(x, (self.lstm_h,self.lstm_c))

        return self.lstm_h, self.lstm_c 

    def output(self, read_data):
        """A function that produces the output of the controller given read data and current hidden state of the lstm 
        
        Args:
            read_data (list of tensors): each read head's data stored in a list, shape (batch_size,M)
        
        Returns:
            tensor : shape (batch_size, output_size)
        """
        state = torch.cat([self.lstm_h]+read_data,dim=1)
        output = self.output_fc(state)
        output = torch.sigmoid(output)
        return output
    
    # Will need to call this function from the NTM class
    def reset(self, batch_size=1):
        in_data = torch.tensor([[0.]],device=self.device)  # dummy input
        lstm_h= self.lstm_h_fc(in_data)
        self.lstm_h = lstm_h.repeat(batch_size,1)
        lstm_c= self.lstm_c_fc(in_data)
        self.lstm_c = lstm_c.repeat(batch_size,1)


#%%
if __name__ == "__main__":
    input_size = 10
    M = 20
    device = torch.device("cpu")
    output_size = 8
    controller_size = 100

    contr = Controller(input_size+M,controller_size,output_size,M+controller_size)





