# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:23:55 2020

@author: yuvraj97
"""


import numpy as np
import matplotlib.pyplot as plt
from Plot2D import Plot2DVectors as vt

class Transform:
    def __init__(self, transform, vector_label: bool = True):
        self.transform = transform
        self._plot_orig_         = vt("Original vectors", vector_label)
        self._plot_tf_           = vt("Transformed vectors", vector_label)
        self._plot_combine_      = vt("Original[blue] /Transformed[red] vectors", vector_label)
        self._fig_               = plt.figure()
        self.transformed_vectors = None
        #plt.close()
    
    
    def add_vectors(self, vectors, origin=np.array([0,0])):
        self.transformed_vectors = np.matmul(self.transform, vectors.T).T
        
        self._plot_orig_.add_vectors(vectors, color='b')
        
        self._plot_combine_.add_vectors(self.transformed_vectors, color='r')
        self._plot_combine_.add_vectors(vectors, color='b')
        
        self._plot_tf_.add_vectors(self.transformed_vectors, color='r')
        
    def show(self):
        self._plot_combine_._fig_.show()
        self._plot_orig_._fig_.show()
        self._plot_tf_._fig_.show()
    
    def fig(self):
        return (self._plot_orig_._fig_, self._plot_tf_._fig_, self._plot_combine_._fig_)


# Example
transform = np.array([
                    [1,1],
                    [1,2],
                   ])
vectors = np.array([
                    [2,1],
                    [1,0],
                    [3,2],
                    [1,3],
                   ])
tf = Transform(transform)
tf.add_vectors(vectors)
tf.show()