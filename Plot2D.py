# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 07:21:15 2020

@author: yuvraj97
"""

import numpy as np
import matplotlib.pyplot as plt
import os
SIZE = 14
plt.rc('font', size=SIZE)   

class Plot2DVectors:
    def __init__(self, title="", head_width=0.3, head_length=0.2):
        self._x_range_   = None
        self._y_range_   = None
        self._fig_       = plt.figure()
        #plt.close()
        self.vectors     = None
        self.head_width  = head_width
        self.head_length = head_length
        plt.title(title)
        
        
    def add_vectors(self, vectors, origin=np.array([0,0]), color="b"):
        self.vectors   = vectors
        self._x_range_   = [min(vectors[:,0].min(), 0) - 2, max(vectors[:,0].max() + 2, 0)]
        self._y_range_   = [min(vectors[:,1].min(), 0) - 2, max(vectors[:,1].max() + 2, 0)]
        
        ax = self._fig_.gca()
        for v in self.vectors:
            #label = "$\\begin{bmatrix}" + str(v[0]) + "\\\\" + str(v[1]) + "\\end{bmatrix}$"
            ax.arrow(origin[0], origin[1],v[0] - origin[0], v[1] - origin[1], head_width=self.head_width, head_length=self.head_length, fc='k', ec='k')
            ax.text(v[0],v[1], str(v), style='italic', bbox={'facecolor':'red', 'alpha':0.3, 'pad':0.5})
        ax.scatter(origin[0],origin[1])
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        self.set_axes_limit()
        #self._fig_ = plt.gcf()
        #plt.close()
        
    def add_vector(self,vector, origin=np.array([0,0])):
        ax = self._fig_.gca()
        self.vectors = np.vstack((self.vectors, vector))
        if(vector[0] < self._x_range_[0]):
            self._x_range_[0] = vector[0] - 2
        elif(vector[0] > self._x_range_[1]):
            self._x_range_[1] = vector[0] + 2
        
        if(vector[1] < self._y_range_[0]):
            self._y_range_[0] = vector[1] - 2
        elif(vector[1] > self._y_range_[1]):
            self._y_range_[1] = vector[1] + 2
        self.set_axes_limit()
        
        ax.arrow(origin[0], origin[1],vector[0] - origin[0], vector[1] - origin[1], head_width=self.head_width, head_length=self.head_length, fc='k', ec='k')
        ax.text(vector[0],vector[1], str(vector), style='italic', bbox={'facecolor':'red', 'alpha':0.3, 'pad':0.5})
        ax.scatter(origin[0],origin[1])
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        self.set_axes_limit()
        #self._fig_ = plt.gcf()
        #plt.close()
    
    def setX_limit(self):
        ax = self._fig_.gca()
        ax.set_xlim(self._x_range_[0], self._x_range_[1])
        
    def setY_limit(self):
        ax = self._fig_.gca()
        ax.set_ylim(self._y_range_[0], self._y_range_[1])

    def set_axes_limit(self):
        self.setX_limit()
        self.setY_limit()
    
    def savefig(self, name=None, path=None):
        if path==None: path=os.getcwd()
        if name==None: name="fig.png"
        #figure = plt.gcf()
        self._fig_.set_size_inches(8,6)
        self._fig_.savefig(name, dpi = 100)
        #self._fig_.savefig(name, bbox_inches='tight')
        #plt.close()
        
    def fig(self):
        return self._fig_
    
    def show(self):
        self._fig_.show()
        
    
"""
# Example:
vectors = 5*np.array([
                    [1,0],
                    [0,1],
                    [1,1],
                    [1,2],
                    [2,1],    
                   ])
origin = np.array([0,0])

plt2D = Plot2DVectors("Vectors")
plt2D.add_vectors(vectors, origin)
#plt2D.savefig()

vector = np.array([3,3])
plt2D.add_vector(vector)
plt2D.show()
"""