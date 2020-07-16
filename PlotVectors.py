# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 07:21:15 2020

@author: yuvraj97
"""

import numpy as np
import matplotlib.pyplot as plt
import os
SIZE = 8
plt.rc('font', size=SIZE)   

class Plot2DVectors:
    def __init__(self, title="", vector_label = True, head_width=0.15, head_length=0.2):
        self.vector_label = vector_label
        self._x_range_   = [0, 0]
        self._y_range_   = [0, 0]
        self._fig_       = plt.figure()
        self.vectors     = None
        self.head_width  = head_width
        self.head_length = head_length
        plt.title(title)
        plt.grid()
        
    def add_vectors(self, vectors, origin=np.array([0,0]), color="k"):
        self.vectors   = vectors
        self._x_range_   = [min(vectors[:,0].min(), self._x_range_[0]) - 0.5, max(vectors[:,0].max() + 0.5, self._x_range_[1])]
        self._y_range_   = [min(vectors[:,1].min(), self._y_range_[0]) - 0.5, max(vectors[:,1].max() + 0.5, self._y_range_[1])]
        
        ax = self._fig_.gca()
        for v in self.vectors:
            #label = "$\\begin{bmatrix}" + str(v[0]) + "\\\\" + str(v[1]) + "\\end{bmatrix}$"
            ax.arrow(origin[0], origin[1],v[0] - origin[0], v[1] - origin[1], head_width=self.head_width, head_length=self.head_length, fc=color, ec=color, alpha=0.6)
            if(self.vector_label):
                ax.text(v[0],v[1], r"""$[""" + str(v[0]) + """, """ + str(v[1]) + """]$""", style='italic', bbox={'facecolor':color, 'alpha':0.2, 'pad':0.5})
        ax.scatter(origin[0],origin[1])
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        self.set_axes_limit()
        
    def add_vector(self,vector, origin=np.array([0,0])):
        ax = self._fig_.gca()
        self.vectors = np.vstack((self.vectors, vector))
        if(vector[0] < self._x_range_[0]):
            self._x_range_[0] = vector[0] - 0.5
        elif(vector[0] > self._x_range_[1]):
            self._x_range_[1] = vector[0] + 0.5
        
        if(vector[1] < self._y_range_[0]):
            self._y_range_[0] = vector[1] - 0.5
        elif(vector[1] > self._y_range_[1]):
            self._y_range_[1] = vector[1] + 0.5
            
        ax.arrow(origin[0], origin[1],vector[0] - origin[0], vector[1] - origin[1], head_width=self.head_width, head_length=self.head_length, fc='k', ec='k')
        if(self.vector_label):
            ax.text(vector[0],vector[1], str(vector), style='italic', bbox={'facecolor':'red', 'alpha':0.3, 'pad':0.5})
        ax.scatter(origin[0],origin[1])
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        self.set_axes_limit()
        
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


import plotly.graph_objs as go
from plotly.offline import plot

class Plot3DVectors:
    def __init__(self, title=""):
        self._x_range_   = [0, -np.inf]
        self._y_range_   = [0, -np.inf]
        self._z_range_   = [0, -np.inf]
        self._fig_       = None
        self._fig_list_  = []
        self._layout_    = None
        self.vectors     = None
        self._title_     = title
        
    def add_vectors(self, vectors, color="blue", legand="", showlegend=True):
        self.vectors     = vectors.T
        self._x_range_   = [min(self.vectors[0].min(), self._x_range_[0]), max(self.vectors[0].max(), self._x_range_[1])]
        self._y_range_   = [min(self.vectors[1].min(), self._y_range_[0]), max(self.vectors[1].max(), self._y_range_[1])]
        self._z_range_   = [min(self.vectors[2].min(), self._z_range_[0]), max(self.vectors[2].max(), self._z_range_[1])]
        self.__add_data__(vectors,color,legand,showlegend)
    
    def __add_data__(self, vectors, color_, legand, showlegend):
        line_vectors = np.empty(shape=(3,len(vectors)*3))
        i = 0
        for x in self.vectors[0]:
            for c in [0,x, None]:
                line_vectors[0,i] = c
                i += 1
        
        i = 0
        for y in self.vectors[1]:
            for c in [0,y, None]:
                line_vectors[1,i] = c
                i += 1
        
        i = 0
        for z in self.vectors[2]:
            for c in [0, z, None]:
                line_vectors[2,i] = c
                i += 1
        
        vectors_fig = go.Scatter3d(
                                    x=line_vectors[0],
                                    y=line_vectors[1],
                                    z=line_vectors[2],
                                    mode='lines',
                                    name=legand,
                                    marker=dict(color=color_),
                                    showlegend=showlegend
                                )
        scatter = go.Scatter3d(
                                x=self.vectors[0],
                                y=self.vectors[1],
                                z=self.vectors[2],
                                mode='markers',
                                marker=dict(color=color_),
                                name='',
                                showlegend=False
                              )
        txt=[]
        for v in vectors:
            txt.append(r'''[''' + str(v[0]) + ''',''' + str(v[1]) + ''',''' + str(v[2]) + ''']''' )
        txt_plt = go.Scatter3d(
                                x = self.vectors[0],
                                y = self.vectors[1],
                                z = self.vectors[2],
                                mode='text',
                                text=txt,
                                marker={'opacity': 0.3},
                                textfont={'size': 10, 'color': color_},
                                name="Co-ordinates",
                                showlegend=showlegend
                              )
        
        self._fig_list_.extend([vectors_fig,scatter, txt_plt])
        
    def fig(self):
        layout = go.Layout(
                            scene = dict(
                                            xaxis=dict(range=[self._x_range_[0] - 0.5, self._x_range_[1] + 0.5]),
                                            yaxis=dict(range=[self._y_range_[0] - 0.5, self._y_range_[1] + 0.5]),
                                            zaxis=dict(range=[self._z_range_[0] - 0.5, self._z_range_[1] + 0.5])
                                        ),
                            title=dict(text= self._title_)
                          )
        self._layout_ = layout
        fig = go.Figure(data = self._fig_list_, layout = self._layout_)
        fig.update_layout(autosize=False,
                          width=800, height=800,)
        return fig

    def set_axis(self,x_range, y_range, z_range):    
        self._x_range_[0] = x_range[0]
        self._x_range_[1] = x_range[1]
        
        self._y_range_[0] = y_range[0]
        self._y_range_[1] = y_range[1]
        
        self._z_range_[0] = z_range[0]
        self._z_range_[1] = z_range[1]
    
    def get_axis(self):
        return self._x_range_, self._y_range_, self._z_range_, 
    
    def show(self):
        plot(self.fig())

"""
# Example:
vectors = np.array([
                    [1,0,0],
                    [0,1,0],
                    [0,0,1],
                    [1,1,1],
                   ])
plt3D = Plot3DVectors("Vectors")
plt3D.add_vectors(vectors)
plt3D.add_vectors(vectors)
plt3D.show()
"""
