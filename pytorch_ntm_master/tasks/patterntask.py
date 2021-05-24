"""Copy Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from ntm.aio import EncapsulatedNTM


# Generator of randomized test sequences and test sequences with patterns
def dataloader(num_batches,
               batch_size,
               seq_width,
               seq_min_len,
               seq_max_len):
    """Generator of random and pattern sequences for the repeat copy task.

    Creates random batches of "bits" sequences or in a repating pattern.

    All the sequences within each batch have the same length.
    The length is between `min_len` to `max_len`

    :param num_batches: Total number of batches to generate.
    :param batch_size: Batch size.
    :param seq_width: The width of each item in the sequence.
    :param seq_min_len: Sequence minimum length.
    :param seq_max_len: Sequence maximum length.
    

    NOTE: The output width is `seq_width` + 1, the additional input is used
    by the network to label if it is a pattern or a random sequence.
    """


    for batch_num in range(num_batches):


        seq_len = 5# couldnt come up with more than 10 patterns so I will split them in two to shuffle 
        # them in an attempt to prevent overfitting.. might not make sense


        # Generate the sequence
        seq1 = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width)) # random sequence
        # repeating patterns trying to get all teh bases for 8 integers.. most likely missed a few 
        seq2 = np.array([[[1, 1, 1, 1, 1, 1, 1, 1]], [[1, 1, 1, 1, 0, 0, 0, 0]], [[1, 0, 1, 0, 1, 0, 1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0]], [[0, 1, 0, 1, 0, 1, 0, 1]]]) 
        seq3 = np.array([[[1, 1, 1, 0, 1, 1, 1, 0]], [[0, 0, 1, 1, 0, 0, 1, 1]], [[1, 0, 0, 0, 1, 0, 0, 0]], [[1, 1, 0, 0, 1, 1, 0, 0]], [[0, 0, 0, 0, 1, 1, 1, 1]]])
        
        #shuffling the four pattern sequences to ensure that the patterns are randomly assembled jsut like the random binomial will be. 
        np.random.shuffle(seq2)
        np.random.shuffle(seq3)
        
        choice=[seq1, seq2, seq3] # list of all possible sequences
        rand=random.randint(0,2) # randomly chosing the sequence and saving which type 0=random, 1,2= repeating patterns
        seq=choice[rand] # random sequence chosen 
         # randomly choise between one of the three patern options 
        seq = torch.from_numpy(seq) # creates the torch form the randomly selected data 

        # The input will just be the patterns
        inp = seq
        #however teh output is dependent on if they were random, or had a pattern within the binary data
        #this has to be split as to wether they are patterns or random
        
        if rand == 0: #this is the randomly generated pattern sequence
            outp = torch.zeros(seq_len, batch_size, seq_width +1)
            outp[:seq_len, :, :seq_width] = seq
            outp[:seq_len, :, seq_width] = 0.0 #zero because they are randomly generated patterns. 
        else: # this will be for 1 or 2. the pattern sequences
            outp = torch.zeros(seq_len, batch_size, seq_width +1)
            outp[:seq_len, :, :seq_width] = seq
            outp[:seq_len, :, seq_width] = 1.0     #labeled as 1 for there being a pattern  
            

        yield batch_num+1, inp.float(), outp.float()


@attrs
class PatternTaskParams(object):
    name = attrib(default="patterntask")
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)
    num_heads = attrib(default=1, converter=int)
    sequence_width = attrib(default=8, converter=int)
    sequence_min_len = attrib(default=1, converter=int)
    sequence_max_len = attrib(default=10, converter=int)
    memory_n = attrib(default=128, converter=int)
    memory_m = attrib(default=20, converter=int)
    num_batches = attrib(default=60000, converter=int) # cange depending on what you want to test
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=1e-4, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)


@attrs

class patternModelTraining(object):
    params = attrib(default=Factory(PatternTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # See dataloader documentation
        net = EncapsulatedNTM(self.params.sequence_width, self.params.sequence_width + 1,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m)
        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(self.params.num_batches, self.params.batch_size,
                          self.params.sequence_width,
                          self.params.sequence_min_len, self.params.sequence_max_len)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)

