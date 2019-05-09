#%%
import torch
import random
import numpy as np
#%%
def copy_dataloader(num_batches,batch_size,seq_width, min_seq_len,max_seq_len, device):
    """A function that returns data for the copy task. Each sequence has a random length between `min_seq_len` and `max_seq_len`. Each vector in the sequence has a fixed `seq_width`. They have a start delimite at the beginning and an end delimiter at the end of the sequence.

    Arguments:
        num_batches {int} -- [description]
        batch_size {int} -- [description]
        seq_width {int} -- [description]
        min_seq_len {int} -- [description]
        max_seq_len {int} -- [description]
        device {int} -- [description]
    """
    for _ in range(num_batches):
        # The sequence length is random between min_seq_len and max_seq_len
        seq_len = random.randint(min_seq_len,max_seq_len)

        # create random bits of (seq_len,batch_size,seq_width)
        seq = np.random.binomial(1,0.5, size=(seq_len,batch_size, seq_width))
        seq = torch.from_numpy(seq).to(device)
        # input is 2 bits larger (one start delimiter and one end delimiter)
        # and 2 longer as well 
        inp_seq = torch.zeros([seq_len+2,batch_size,seq_width+2],device=device)

        
        inp_seq[0,:,seq_width]= 1.0 #start delimiter
        inp_seq[1:seq_len+1, : , :seq_width] = seq # copy sequence in middle 
        inp_seq[seq_len+1, : ,seq_width+1 ] = 1.0 # end delimiter 
        
        target_seq = torch.zeros([seq_len,batch_size,seq_width],device=device)
        target_seq[:seq_len,:,:seq_width] = seq

        yield inp_seq,target_seq

#%%
if __name__ == "__main__":
    device = torch.device("cuda")
    b = torch.zeros([1],device=device)
    print(b.device)
#%%
    d = copy_dataloader(num_batches=2,batch_size=2,seq_width=8,min_seq_len=1,max_seq_len=20,device=device)
    a = next(d)
    print(a[0].shape,a[1].shape,a[0].device)
    
    a= next(d)
    print(a[0].shape,a[1].shape,a[0].device)
#%% 