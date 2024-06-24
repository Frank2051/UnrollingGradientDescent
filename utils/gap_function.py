import numpy as np
def gap_function(A,x,y,tolerance=0):

    test= np.dot(A.T,y-np.dot(A,x))
    if np.all(test >= -tolerance):
        raise Exception("gap infini")
    else:
        Ax = np.dot(A, x)
        return np.dot(Ax, Ax) - np.dot(y, Ax)