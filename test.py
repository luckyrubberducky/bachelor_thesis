# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:48:37 2019

@author: zeno
"""

import numpy as np


a = np.array([[1,1],[2,2],[3,3]])
b = np.array([1, 2, 3])
s = np.arange(a.shape[0])
np.random.shuffle(s)
s
print(a[s])
print(b[s])