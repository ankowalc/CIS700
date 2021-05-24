
# CIS700
Shared CIS700 NPL Final Project 
Andrew Kowalczewski
netID: ankowalc

Base code from gh repo clone loudinthecloud/pytorch-ntm https://github.com/loudinthecloud/pytorch-ntm
by Guy Zana, Jules Gagnon-Marchand, and Mark Goldstein 

Code used for this project was a new task called patterndetect.py in the task folder
and notebook pattern_plots.ipynb in the notebooks folder.

## to run the code 
you only need to execute
'python3 train.py --task patterntask' in command line once you have the pytorch_ntm_master folder installed and using the master folder as directory. 

If you already have these dependencies you can add my task patterndetect.py to it adn run it like this. 
I did edit a few of the .py's to update a few errors and to try to fix a zero tensor problem with data.loss[0]

The notebook pattern_plots.ipynb in the notebooks folder to see the plot for training loss or failure to understand implimenting the model. 


### dependencies incude:  
from attr import attrs, attrib, Factory  
import torch  
from torch import nn  
from torch import optim  
import torch.nn.functional as F  
  
import argparse  
import logging  
import time  
import random  
mport matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
  
from glob import glob  
import json  
import os  
import sys  
import re  
import sys  
import attr  
import argcomplete  
import pytest  
 
### Below is the original author's copywrite notice allowing the code's use with alteration provided this is included. 
this notice is also kept with their original readme in the pytorch_ntm_master folder

Their original code worked to complete the copy task and copy-repeat tasks. After testing both tasks I added the patterndetect task using their architecture.


BSD 3-Clause License

Copyright (c) 2017, Guy Zana <guyzana@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
