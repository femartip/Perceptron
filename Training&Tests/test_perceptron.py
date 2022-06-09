#!/usr/bin/python

import sys 
import numpy as np
from perceptron import perceptron
if len(sys.argv)!=2:
    print('Usage: %s <data>' % sys.argv[0]);
    sys.exit(1);
data=np.loadtxt(sys.argv[1]);
N,L=data.shape;
NTr=int(round(.7*N));
train=data[:NTr,:]; 
w,E,k=perceptron(train);
print(w);
print('E=%d k=%d' % (E,k));

