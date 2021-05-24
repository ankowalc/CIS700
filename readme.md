
# CIS700
Shared CIS700 NPL Final Project 
Base code from gh repo clone loudinthecloud/pytorch-ntm https://github.com/loudinthecloud/pytorch-ntm
by Guy Zana, Jules Gagnon-Marchand, and Mark Goldstein 

Code used for this homework was a new task called patterndetect.py in the task folder
and notebook pattern_plots.ipynb in the notebooks folder.

to run the code you only need to execture
'python3 train.py --task patterntask' in command line once you have the pytorch_ntm_master folder installed from that directory. 

or the notebook pattern_plots.ipynb in the notebooks folder to see the plot for training loss or failure to understand implimenting the model. 


dependencies incude:
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

