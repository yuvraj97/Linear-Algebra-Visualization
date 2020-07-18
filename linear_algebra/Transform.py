# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:23:55 2020

@author: yuvraj97
"""


import numpy as np
import matplotlib.pyplot as plt
from linear_algebra import Plot2DVectors as vt2D
from linear_algebra import Plot3DVectors as vt3D

class Transform2D:
    def __init__(self, transform: np.ndarray, vector_label: bool = True):
        self.transform = transform
        self._plot_orig_         = vt2D("Original vectors", vector_label)
        self._plot_tf_           = vt2D("Transformed vectors", vector_label)
        self._plot_combine_      = vt2D("Original[blue] /Transformed[red] vectors", vector_label)
        self._fig_               = plt.figure()
        self.transformed_vectors = None
        self.allowed_opr         = set(np.__all__)
        #plt.close()
    
    def __exec_equation__(self,s: str):
        s = s.lower()
        s = s.replace("^","**")
        s = s.replace("[","(")
        s = s.replace("]",")")
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


tf = Transform2D(transform)
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


import plotly.graph_objs as go
from plotly.subplots import make_subplots

class Transform3D:
    def __init__(self, transform: np.ndarray):
        self.transform           = transform
        self._plot_orig_         = vt3D("Original vectors")
        self._plot_tf_           = vt3D("Transformed vectors")
        self._plot_combine_      = vt3D("Original[blue] /Transformed[red] vectors")
        self.transformed_vectors = None
        self.allowed_opr         = set(np.__all__)
        
    def __exec_equation__(self,s: str):
        s = s.lower()
        s = s.replace("^","**")
        s = s.replace("[","(")
        s = s.replace("]",")")
        r = []
        i=0
        while(i<len(s)):
            c = s[i]
            if(c=='x' or c=='y'):
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
    
    @staticmethod
    def __flatten__(x,y,z):
        vectors = np.empty(shape=(3,np.product(x.shape)))
        vectors[0] = x.reshape((np.product(x.shape)))
        vectors[1] = y.reshape((np.product(y.shape)))
        vectors[2] = z.reshape((np.product(z.shape)))
        return vectors
    
    def add_equation(self, eq: str, 
                     range_ = (), 
                     count: int   = 10,
                     opacity = 0.5,
                     ):
        if(range_==tuple()):
            range_ = (self._plot_combine_._x_range_[0], self._plot_combine_._x_range_[1])
        
        x = np.linspace(range_[0], range_[1], count)
        y = np.linspace(range_[0], range_[1], count)
        x, y = np.meshgrid(x, y)

        eq, status = self.__exec_equation__(eq)
        if(status==False): # equation is not correct, No need to check for range
            return False, None
        
        try:
            z = eval(eq)
        except:            # equation seems to be correct, but range might not correct
                           # (Range can be wrong like if, equation encounters sqrt(-1) or divide by 0, etc in that range)
            return True, False

        trace1 = go.Surface(x=x, y=y, z=z, opacity=opacity, colorscale='Viridis', showscale=False)#, col)
        
        vectors = self.__flatten__(x,y,z)
        tf = np.matmul(self.transform, vectors)
        x, y, z = tf[0].reshape(x.shape), tf[1].reshape(y.shape), tf[2].reshape(z.shape)
        trace2 = go.Surface(x=x, y=y, z=z, opacity=opacity, colorscale='YlOrRd', showscale=False)#, col)
        
        self._plot_combine_._fig_list_.extend([trace1,trace2])
        self._plot_orig_._fig_list_.extend([trace1])
        self._plot_tf_._fig_list_.extend([trace2])
        
        
        (x_min_orig, x_max_orig), (y_min_orig, y_max_orig), (z_min_orig, z_max_orig) = self.__get_axes_limit__(vectors, self._plot_orig_)
        self._plot_orig_.set_axis((x_min_orig, x_max_orig), (y_min_orig, y_max_orig), (z_min_orig, z_max_orig))
        
        (x_min_tf, x_max_tf),     (y_min_tf, y_max_tf)    , (z_min_tf, z_max_tf)     = self.__get_axes_limit__(tf, self._plot_tf_)
        self._plot_tf_.set_axis((x_min_tf, x_max_tf),     (y_min_tf, y_max_tf)    , (z_min_tf, z_max_tf))
        
        x_min = min(x_min_orig, x_min_tf)
        x_max = max(x_max_orig, x_max_tf)
        
        y_min = min(y_min_orig, y_min_tf)
        y_max = max(y_max_orig, y_max_tf)
        
        z_min = min(z_min_orig, z_min_tf)
        z_max = max(z_max_orig, z_max_tf)
        self._plot_combine_.set_axis((x_min, x_max),     (y_min, y_max)    , (z_min, z_max))
        
        #print((x_min, x_max),     (y_min, y_max)    , (z_min, z_max))
        #print((x_min_orig, x_max_orig), (y_min_orig, y_max_orig), (z_min_orig, z_max_orig))
        #print((x_min_tf, x_max_tf),     (y_min_tf, y_max_tf)    , (z_min_tf, z_max_tf))
   
        return True, True
        
    def __get_axes_limit__(self, matrix, plot):
        x_min, x_max = min(np.nanmin(matrix[0]), plot._x_range_[0]), max(np.nanmax(matrix[0]), plot._x_range_[1])
        y_min, y_max = min(np.nanmin(matrix[1]), plot._x_range_[0]), max(np.nanmax(matrix[1]), plot._x_range_[1])
        z_min, z_max = min(np.nanmin(matrix[2]), plot._x_range_[0]), max(np.nanmax(matrix[2]), plot._x_range_[1])
        
        return (x_min, x_max), (y_min, y_max), (z_min, z_max)
        
    def add_vectors(self, vectors: np.ndarray):
        self.transformed_vectors = np.matmul(self.transform, vectors.T).T
        self._plot_orig_.add_vectors(
                                         vectors,
                                         color='blue',
                                         legand="Original vectors",
                                         showlegend=True
                                    )
        
        self._plot_combine_.add_vectors(
                                            self.transformed_vectors, 
                                            color='red', 
                                            legand="Transformed vectors", 
                                            showlegend=True
                                       )
        
        self._plot_combine_.add_vectors(
                                            vectors, 
                                            color='blue', 
                                            legand="Original vectors", 
                                            showlegend=True
                                       )
        
        self._plot_tf_.add_vectors(
                                        self.transformed_vectors, 
                                        color='red', 
                                        legand="Transformed vectors", 
                                        showlegend=True
                                   )
    
    def show(self):
        self._plot_orig_.show()
        self._plot_tf_.show()
        self._plot_combine_.show()
    
    def fig_side_by_side(self):
        fig = make_subplots(rows=1, cols=2,
                            specs=[[{'type': 'surface'}, {'type': 'surface'}]],)
        for trace in self._plot_orig_._fig_list_:
            fig.add_trace(trace=trace, row=1, col=1)
        
        for trace in self._plot_tf_._fig_list_:
            fig.add_trace(trace=trace, row=1, col=2)
        fig.update_layout(width=800, height=500,showlegend = False,)
        return fig
    
    def fig_combine(self):
        return self._plot_combine_.fig()
    
    def fig_orig(self):
        return self._plot_orig_.fig()
    
    def fig_tf(self):
        return self._plot_tf_.fig()
    
    def fig(self):
        return (self._plot_orig_.fig(), self._plot_tf_.fig(), self._plot_combine_.fig())
    
    
"""
# Example
transform3D = np.array([
                    [-1.0,  0.0,  0.0],
                    [ 0.0, -1.0,  0.0],
                    [ 0.0,  0.0, -1.0]
                   ])
vectors3D = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1],
                   ])


tf3D = Transform3D(transform3D)
tf3D.add_vectors(vectors3D)
tf3D.add_equation(
                    "sqrt[9 - (x-0)^2 - (y-0)^2] + 0",
                    range_  = [-3, 3],
                    count   = 30, 
                    opacity = 0.5,
                 )
tf3D.show()
"""