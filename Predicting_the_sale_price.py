import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#feature scaling
#Feature scaling
#We will do a little preprocessing to our data using the following formula (standardization):

#x′=x−μ/σ 

#where  μ  is the population mean and  σ  is the standard deviation.

x = df_train['GrLivArea']
y = df_train['SalePrice']

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x] 
