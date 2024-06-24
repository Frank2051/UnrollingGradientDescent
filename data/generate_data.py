# generate data for the FNN to train on using cvx py

import cvxpy as cp
import torch
import numpy as np
from utils.gap_function import gap_function as gap_function
from utils.tools import middle_percent as middle_percent
from utils.tools import middle_exclude as middle_exclude
from torch.utils.data import Dataset

#import matplotlib.pyplot as plt

class FloatDataset(Dataset):
    def __init__(self, x_list, y_list):
        
        self.x_list = [np.array(x, dtype=np.float64) for x in x_list]
        self.y_list = np.array(y_list, dtype=np.float64)

        self.x_list = [torch.tensor(x, dtype=torch.float64) for x in self.x_list]
        self.y_list = torch.tensor(self.y_list, dtype=torch.float64)
        

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        x = self.x_list[idx]
        y = self.y_list[idx]
        return x, y

def generate_data(A=np.random.randn(75, 100),sample_size=1200,epsilon = 10e-12):

    # generate y sample on the unit circle (normalization of y imply x*=x/||y||)
    y_dimension=A.shape[0]
    raw_y_sample= np.random.normal(0, 1, size=(sample_size, y_dimension))
    y_norms = np.linalg.norm(raw_y_sample, axis=1, keepdims=True)
    y_sample = raw_y_sample / y_norms

    # searching x using cvxpy then support restrained inversion
    x_sample =[]
    gap_cvx=[]
    gap_restrained_support=[]
    for y in y_sample:

        # Define the problem using cvxpy
        x = cp.Variable(A.shape[1])
        objective = cp.Minimize(cp.norm(y - A @ x, 2)**2)
        constraints = [x >= 0]
        problem = cp.Problem(objective, constraints)

        # Solve the problem
        problem.solve()
        # Truncaturation of negative values
        x.value[x.value<0]=0
        gap_cvx.append(gap_function(A,x.value,y))

        # Calculate x* using the submatrix 
        indices = np.where(x.value >= epsilon)[0]
        sub_A=A[:,indices]
        sub_AtA = np.dot(sub_A.T, sub_A)
        sub_AtA_inv = np.linalg.inv(sub_AtA)
        sub_Aty = np.dot(sub_A.T, y)

        # Compute (A^T A)^-1 A^T y
        x_opti = np.dot(sub_AtA_inv, sub_Aty)
        x_sample.append(x_opti)
        gap_restrained_support.append(gap_function(sub_A,x_opti,y))

    print("mean of the gap with cvx : ", np.mean(gap_cvx))
    print("without 10 percents extrems on both sides: ", np.mean(middle_percent(gap_cvx,20)))
    #plt.hist(gap_cvx)
    #plt.title("distribution of cvx gaps")
    #plt.show()
    print("mean of the gap with restrained support : ", np.mean(gap_restrained_support))
    print("without 10 percent extrems on both sides: ", np.mean(middle_percent(gap_restrained_support,20)))
    #plt.hist(gap_restrained_support)
    #plt.title("distribution of restrained support gaps")
    #plt.show()
    #plt.hist(middle_percent(gap_restrained_support,20))
    #plt.title("distribution of restrained support gaps without 10 percent extrems on both sidess")
    #plt.show()

    #list_gap_mean=[]
    #for i in range(1,101):
    #    list_gap_mean.append(np.mean(middle_percent(gap_restrained_support,i)))
    #plt.plot([k for k in range(1,101)],list_gap_mean)
    #plt.title("gap mean by pourcent of extrems gaps ignored")
    #plt.show()

    triplets = list(zip(gap_restrained_support, x_sample, y_sample))

    # Sort triplets by the first element of each triplet
    sorted_triplets = sorted(triplets, key=lambda x: x[0])

    # Separate sorted triplets back into three lists
    sorted_list1, sorted_list2, sorted_list3 = zip(*sorted_triplets)

    # Convert tuples back to lists
    x_filtered = middle_exclude(list(sorted_list2),20)
    y_filtered = middle_exclude(list(sorted_list3),20)

    dataset = FloatDataset(x_filtered, y_filtered)
    return dataset


