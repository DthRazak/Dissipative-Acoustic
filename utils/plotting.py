import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.dolfinx import eval_pointvalues

class Mpl2DPlotter():
    
    def __init__(self, layout=[["1"]], projection='2d', fontsize=16):
        """
        Creates figure with mosaic layout
        """
        
        self.projection = projection
        self.fontsize = fontsize
        
        self.fig = plt.figure(
            constrained_layout=True, 
            figsize=(16, 8)
        )
        
        if projection == '3d':
            self.axd = self.fig.subplot_mosaic(
                layout,
                empty_sentinel="",
                subplot_kw={"projection": "3d"}
            )
        else:
            self.axd = self.fig.subplot_mosaic(
                layout,
                empty_sentinel=""
            )
        
    def update_figure(self, **updates):
        update_fun = {
            'fontsize': lambda fontsize: setattr(self, 'fontsize', fontsize),
            'figsize' : lambda figsize: self.fig.set_size_inches(*figsize),
            'suptitle': lambda suptitle: self.fig.suptitle(suptitle, fontsize=self.fontsize)
        }
        
        for update, params in updates.items():
            update_fun[update](params)

                
    def plot(self, fun, type, points, N, axes_id, title, cmap=cm.coolwarm, **kwargs):
        x = np.linspace(points[0][0], points[0][1], N[0])
        y = np.linspace(points[1][0], points[1][1], N[1])
        
        xx, yy = np.meshgrid(x, y)
        lin_shape = xx.shape[0] * xx.shape[1]

        compute_points = np.hstack((xx.ravel(),
                                    yy.ravel(),
                                    np.zeros(lin_shape))
                                  ).reshape((3, lin_shape)).T
        values = eval_pointvalues(fun, compute_points)
        
        if type == 'real':
            mag = np.linalg.norm(np.real(values), axis=1).reshape(xx.shape)
        elif type == 'imag':
            mag = np.linalg.norm(np.imag(values), axis=1).reshape(xx.shape)
        else:
            raise TypeError(f"Unknown type of function: {type}. Should be `real` or `imag`")
        
        if self.projection == '2d':
            surf = self.axd[axes_id].contourf(xx, yy, mag, 25, cmap=cmap)
            self.fig.colorbar(surf, ax=self.axd[axes_id])
        elif self.projection == '3d':            
            surf = self.axd[axes_id].plot_surface(xx, yy, mag, cmap=cmap)
            self.fig.colorbar(surf, ax=self.axd[axes_id], shrink=0.75)
        
        self.axd[axes_id].set_title(title, fontsize=self.fontsize)
        self.axd[axes_id].set_xlabel('x', fontsize=self.fontsize) 
        self.axd[axes_id].set_ylabel('y', fontsize=self.fontsize)
            
    def save(self, filename):
        self.fig.savefig(filename, bbox_inches='tight')
