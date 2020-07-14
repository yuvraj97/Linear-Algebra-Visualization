# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:23:55 2020

@author: yuvraj97
"""


import numpy as np
import matplotlib.pyplot as plt
from Plot2D import Plot2DVectors as vt

class Transform:
    def __init__(self, transform: np.ndarray, vector_label: bool = True):
        self.transform = transform
        self._plot_orig_         = vt("Original vectors", vector_label)
        self._plot_tf_           = vt("Transformed vectors", vector_label)
        self._plot_combine_      = vt("Original[blue] /Transformed[red] vectors", vector_label)
        self._fig_               = plt.figure()
        self.transformed_vectors = None
        self.allowed_opr         = set(np.__all__)
        #plt.close()
    
    def __exec_equation__(self,s: str):
        s = s.lower()
        s = s.replace("^","**")
        r = []
        i=0
        while(i<len(s)):
            c = s[i]
            if(c=='x'):
                r.append(c)
            elif(c.isalpha()):
                cmd = []
                r.append("np.")
                while(i<len(s) and s[i].isalpha()):
                    c = s[i]
                    cmd.append(c)
                    i += 1
                i -= 1
                cmd = "".join(cmd)
                if(cmd in self.allowed_opr):
                    r.append(cmd)
                else:
                    return None, False
            else:
                r.append(c)
            i += 1
        
        return "".join(r), True
    
    def add_equation(self, eq: str, 
                     x_range = (), 
                     count: int   = 10, 
                     color: tuple = ("b","r")
                     ):
        if(x_range==tuple()):
            x_range = (self._plot_orig_._x_range_[0]/2, self._plot_orig_._x_range_[1]/2)
        
        x = np.linspace(x_range[0], x_range[1], count)
        eq, status = self.__exec_equation__(eq)
        if(status==False): # equation is not correct, No need to check for range
            return False, None
        
        try:
            y = eval(eq)
        except:            # equation seems to be correct, but range might not correct
                           # (Range can be wrong like if, equation encounters sqrt(-1) or divide by 0, etc in that range)
            return True, False
        
        ax = self._plot_orig_._fig_.gca()
        ax.scatter(x, y, c=color[0])
        
        ax = self._plot_combine_._fig_.gca()
        ax.scatter(x, y, c=color[0])
        """
        x = np.linspace(x_range_tf[0], x_range_tf[1], count)
        eq, status = self.__exec_equation__(eq, x)
        if(status==False): # equation is not correct, No need to check for range
            return False, None
        
        try:
            y = eval(eq)
        except:            # equation is correct, but range is not correct
            return True, False
        """
        orig_eq = np.vstack((x, y))
        transformed_eq = np.matmul(self.transform, orig_eq)
        
        ax = self._plot_tf_._fig_.gca()
        ax.scatter(transformed_eq[0], transformed_eq [1], c=color[1])
        
        ax = self._plot_combine_._fig_.gca()
        ax.scatter(transformed_eq[0], transformed_eq [1], c=color[1])
        
        (x_min_orig, x_max_orig), (y_min_orig, y_max_orig) = self.__set_axes_limit__(orig_eq, self._plot_orig_)
        self._plot_orig_.set_axes_limit()
        
        (x_min_tf, x_max_tf), (y_min_tf, y_max_tf)         = self.__set_axes_limit__(transformed_eq, self._plot_tf_)
        self._plot_tf_.set_axes_limit()
        
        if(x_min_orig < x_min_tf):
            x_min = x_min_orig
        else:
            x_min = x_min_tf
        
        if(x_max_orig > x_max_tf):
            x_max = x_max_orig
        else:
            x_max = x_max_tf
        
        
        if(y_min_orig < y_min_tf):
            y_min = y_min_orig
        else:
            y_min = y_min_tf
        
        if(y_max_orig > y_max_tf):
            y_max = y_max_orig
        else:
            y_max = y_max_tf
        
        self._plot_combine_._x_range_[0] = x_min - 0.5
        self._plot_combine_._x_range_[1] = x_max + 0.5
        self._plot_combine_._y_range_[0] = y_min - 0.5
        self._plot_combine_._y_range_[1] = y_max + 0.5
        self._plot_combine_.set_axes_limit()
        
        return True, True
        
    def __set_axes_limit__(self, matrix, plot):
        """
        matrix is 2 x n, with two rows, row[0]: x-coordinates and row[1]: y-coordinates
        """
        x_min, x_max = np.nanmin(matrix[0]) - 0.5, np.nanmax(matrix[0]) + 0.5
        y_min, y_max = np.nanmin(matrix[1]) - 0.5, np.nanmax(matrix[1]) + 0.5
        
        if(x_min < plot._x_range_[0]):
            plot._x_range_[0] = x_min
        if(x_max > plot._x_range_[1]):
            plot._x_range_[1] = x_max
        if(y_min < plot._y_range_[0]):
            plot._y_range_[0] = y_min
        if(y_max > plot._y_range_[1]):
            plot._y_range_[1] = y_max
        print(matrix)
        print("QQ",(x_min, x_max), (y_min, y_max))   
        return plot._x_range_, plot._y_range_
    
    def add_vectors(self, vectors: np.ndarray, origin = np.array([0,0])):
        self.transformed_vectors = np.matmul(self.transform, vectors.T).T
        
        self._plot_orig_.add_vectors(vectors, color='b')
        
        self._plot_combine_.add_vectors(self.transformed_vectors, color='r')
        self._plot_combine_.add_vectors(vectors, color='b')
        
        self._plot_tf_.add_vectors(self.transformed_vectors, color='r')
        #plt.close()
    
    def show(self):
        self._plot_combine_._fig_.show()
        self._plot_orig_._fig_.show()
        self._plot_tf_._fig_.show()
    
    def fig(self):
        return (self._plot_orig_._fig_, self._plot_tf_._fig_, self._plot_combine_._fig_)

"""
# Example
transform = np.array([
                    [2.0, -0.5],
                    [1.0,  0.5],
                   ])
vectors = np.array([
                    [1.0, 0.0],
                    [0.0, 1.0],
                   ])


tf = Transform(transform)
tf.add_vectors(vectors)

'''
tf.add_equation("sin(x)*cos(x)",
                x_range=(-3,3),
                count = 100, )
'''


# Adding a full circle
# First half of circle
tf.add_equation("sqrt(9-x^2)",
                x_range=(-3,3),
                count = 100, )
# Second half of circle
tf.add_equation("-sqrt(9-x^2)",
                x_range=(-3,3),
                count = 100, )

tf.show()
"""