#%%
import torch
from torch import nn

from models.controller import Controller
from models.head import Head
from models.memory import Memory
import numpy as np

#%%
class NTM(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 controller_size,
                 memory_units,
                 memory_unit_size,
                 num_heads):
        super().__init__()
        self.controller_size = controller_size
        self.controller = Controller(
            input_size + num_heads * memory_unit_size, controller_size, output_size,
            read_data_size=controller_size + num_heads * memory_unit_size)

        self.memory = Memory(memory_units, memory_unit_size)
        self.memory_unit_size = memory_unit_size
        self.memory_units = memory_units
        self.num_heads = num_heads
        self.heads = nn.ModuleList([])
        for head in range(num_heads):
            self.heads += [
                Head('r', controller_size, key_size=memory_unit_size),
                Head('w', controller_size, key_size=memory_unit_size)
            ]

        self.prev_head_weights = []
        self.prev_reads = []
        self.reset()

    def reset(self, batch_size=1):
        self.memory.reset(batch_size)
        self.controller.reset(batch_size)
        self.prev_head_weights = []
        for i in range(len(self.heads)):
            prev_weight = torch.zeros([batch_size, self.memory.n])
            self.prev_head_weights.append(prev_weight)
        self.prev_reads = []
        for i in range(self.num_heads):
            prev_read = torch.zeros([batch_size, self.memory.m])
            # using random initialization for previous reads
            nn.init.kaiming_uniform_(prev_read)
            self.prev_reads.append(prev_read)

    def forward(self, in_data):
        controller_h_state, controller_c_state = self.controller(
            in_data, self.prev_reads)
        # print(f"controller state:{controller_h_state.shape}")
        read_data = []
        head_weights = []
        for head, prev_head_weight in zip(self.heads, self.prev_head_weights):
            if head.mode == 'r':
                head_weight, r = head(
                    controller_c_state, prev_head_weight, self.memory)
                read_data.append(r)
            else:
                head_weight, _ = head(
                    controller_c_state, prev_head_weight, self.memory)
            head_weights.append(head_weight)

        output = self.controller.output(read_data)

        self.prev_head_weights = head_weights
        self.prev_reads = read_data

        return output#, [h.clone().detach().numpy() for h in head_weights]
    
#%%

if __name__ == "__main__":
    from dataloader import copy_dataloader
    import matplotlib.pyplot as plt

    def print_head(h):
        h = np.array(h)
        r = h[:,0,:,:]
        w = h[:,1,:,:]
        # plt.figure(figsize=(10,20))
        plt.imshow(r.T.squeeze())
        plt.title("Read Head locations")
        plt.show()
        # plt.figure(figsize=(10,20))
        plt.imshow(w.T.squeeze())
        plt.title("Write Head locations")
        plt.show()
    

    device = torch.device("cpu")
    batch_size= 1
    d = copy_dataloader(2,batch_size,8,20,20,device)
    # ntm = NTM(10,8,100,128,20,1)
    ntm = torch.load("checkpoints\i-9900_error_0.000e+00.pt")
    ntm.reset(batch_size)
    x,y = next(d)
    h =[]
    for i in range(x.size(0)):
        input = x[i]
        _,head = ntm(input)
        h.append(head)
    
    outputs = torch.zeros(y.size())

    print_head(h)

    h=[]
    zero_input = torch.zeros([batch_size,10])
    for i in range(y.size(0)):
        outputs[i],head = ntm(zero_input)
        h.append(head)
    
    print_head(h)


#%%
