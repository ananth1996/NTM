#%%
from ntm import NTM
from dataloader import copy_dataloader
from torch import nn, optim
import torch
#%%
input_size=8+2
output_size=8
controller_size=100
memory_units=128
memory_unit_size=20
batch_size=2
num_heads=1
num_batches =1

device = torch.device("cpu")
data = copy_dataloader(100,batch_size,8,1,20,device)

ntm = NTM(input_size,output_size,controller_size,memory_units,memory_unit_size,num_heads)

criterion = nn.BCELoss()
optimizer = optim.RMSprop(ntm.parameters(),
                          lr=1e-4,
                          alpha=0.95,
                          momentum=0.9)

#%%
print("Starting training")
losses =[]
for x,y in data:
    optimizer.zero_grad()
    ntm.reset(batch_size)

    for i in range(x.size(0)):
        input = x[i]
        ntm(input)

    outputs = torch.zeros(y.size())

    zero_input = torch.zeros([batch_size,input_size])
    for i in range(y.size(0)):
        outputs[i] = ntm(zero_input)

    loss = criterion(outputs, y)
    losses.append(loss.item())
    loss.backward()

    nn.utils.clip_grad_value_(ntm.parameters(), 10)
    optimizer.step()
    print(f"Finished batch, loss:{losses[-1]}")

