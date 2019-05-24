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


def repeat_copy_dataloader(num_batches, batch_size,seq_width, min_seq_len, max_seq_len, min_repeat, max_repeat ,device):
        
        for _ in range(num_batches):

            # function and values for normalization 
            reps_mean = (max_repeat + min_repeat) / 2
            reps_var = (((max_repeat - min_repeat + 1) ** 2) - 1) / 12
            reps_std = np.sqrt(reps_var)
            def normalize(rep):
                return (rep - reps_mean) / reps_std

            # Sequence length is random between min and max 
            seq_len = random.randint(min_seq_len,max_seq_len)
            #Similar for number of repetitions
            rep = random.randint(min_repeat,max_repeat)

            seq = np.random.binomial(1,0.5, size=(seq_len,batch_size, seq_width))
            seq = torch.from_numpy(seq).to(device)
            # fill in input sequence, two bit longer and wider than target

            input_seq = torch.zeros([seq_len + 2,batch_size, seq_width + 2]).to(device)
            input_seq[0, :, seq_width] = 1.0  # input delimiter
            input_seq[1:seq_len + 1,:, :seq_width] = seq
            # Add the number of repetions required after normalizing 
            input_seq[seq_len + 1, :, seq_width + 1] = normalize(rep)

            target_seq = torch.zeros(
                [seq_len * rep + 1,batch_size, seq_width + 1]).to(device)
            target_seq[:seq_len * rep,:, :seq_width] = seq.repeat(rep, 1,1)
            target_seq[seq_len * rep,:, seq_width] = 1.0  # output delimiter

            yield input_seq,target_seq

def associative_dataloader(num_batches, batch_size,item_width,item_length, min_item_count, max_item_count,device):
    
    for _ in range(num_batches):
        num_item = random.randint(min_item_count,max_item_count)
        
        
        # fill in input two bit wider than target to account for delimiter
        # flags.
        input_items = torch.zeros(
            [(item_length + 1) * (num_item + 1) + 1, batch_size, item_width + 2])
        for i in range(num_item):
            item = np.random.binomial(1,0.5,size=(item_length,batch_size,item_width))
            item = torch.from_numpy(item).to(device)
            
            input_items[(item_length + 1) * i, :,  item_width] = 1.0 # item delimiter 
            input_items[(item_length + 1) * i + 1 :(item_length + 1)
                        * (i + 1), : , :item_width] = item

        # generate query item randomly
        # in case of only one item, torch.randint throws error as num_item-1=0
        query_item = 0
        if num_item != 1:
            query_item = random.randint(0,num_item-1)
        query_seq = input_items[(item_length + 1) * query_item +
                                1:(item_length + 1) * (query_item + 1), : , :item_width]
        
        input_items[(item_length + 1) * num_item, 
                    : ,
                    item_width + 1] = 1.0  # query delimiter
        input_items[(item_length + 1) * num_item + 1:(item_length + 1)* (num_item + 1), 
                    :,
                    :item_width] = query_seq
        input_items[(item_length + 1) * (num_item + 1), 
                    :,
                    item_width + 1] = 1.0  # query delimiter

        # generate target sequences(item next to query in the input list)
        target_item = torch.zeros([item_length,batch_size, item_width])
        # in case of last item, target sequence is zero
        if query_item != num_item - 1:
            target_item[:item_length, :, :item_width] = input_items[
                (item_length + 1) * (query_item + 1) + 1:(item_length + 1) * (query_item + 2),
                :,
                :item_width]

        yield input_items, target_item
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
    d = repeat_copy_dataloader(num_batches=2,batch_size=2,
                               seq_width=8,min_seq_len=1,max_seq_len=10,
                               min_repeat=1,max_repeat=10,device=device)
    a = next(d)
    print(a[0].shape,a[1].shape,a[0].device)
    
    a= next(d)
    print(a[0].shape,a[1].shape,a[0].device)
    
    d = associative_dataloader(num_batches=2,batch_size=2,
                               item_width=6,item_length=3,
                               min_item_count=2, max_item_count=6,
                               device=device)
    a = next(d)
    print(a[0].shape,a[1].shape,a[0].device)
    
    a= next(d)
    print(a[0].shape,a[1].shape,a[0].device)