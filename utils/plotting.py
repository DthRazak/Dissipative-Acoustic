import numpy as np

from matplotlib import animation, cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path

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

                
    def plot(self, fun, type, project, points, N, axes_id, title, cmap=cm.coolwarm, **kwargs):
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
            values = np.real(values)
        elif type == 'imag':
            values = np.imag(values)
        else:
            raise TypeError(f"Unknown type of values: {type[0]}. Should be `real` or `imag`")
        
        if project == 'x' or project == 'scalar':
            zz = values[:, 0].reshape(xx.shape)
        elif project == 'y':
            zz = values[:, 1].reshape(xx.shape)
        elif project == 'mag':
            zz = np.linalg.norm(values, axis=1).reshape(xx.shape)
        else:
            raise TypeError(f"Unknown type of projection: {project}. Should be `x`, `y` or `mag`")
        
        if self.projection == '2d':
            surf = self.axd[axes_id].contourf(xx, yy, zz, 25, cmap=cmap)
            self.fig.colorbar(surf, ax=self.axd[axes_id])
        elif self.projection == '3d':            
            surf = self.axd[axes_id].plot_surface(xx, yy, zz, cmap=cmap)
            self.fig.colorbar(surf, ax=self.axd[axes_id], shrink=0.75)
        
        self.axd[axes_id].set_title(title, fontsize=self.fontsize)
        self.axd[axes_id].set_xlabel('x', fontsize=self.fontsize) 
        self.axd[axes_id].set_ylabel('y', fontsize=self.fontsize)
            
    def save(self, filename):
        self.fig.savefig(filename, bbox_inches='tight')


class Mpl2DAnimator():
    
    def __init__(self, layout=[["1"]], time=[0], projection='2d', figsize=(16, 8), fontsize=16):
        self.layout = layout
        self.projection = projection
        self.figsize = figsize
        self.fontsize = fontsize
        self.suptitle = ''
        self.time = time
        
        self.function_data = list()

    def update_figure(self, **updates):
        if 'figsize' in updates.keys():
            self.figsize = updates['figsize']
        if 'suptitle' in updates.keys():
            self.suptitle = updates['suptitle']
        if 'fontsize' in updates.keys():
            self.fontsize = updates['fontsize']
        
    def add_fun(self, fun, time_data, type, project, points, N, axes_id, title, cmap=cm.coolwarm, **kwargs):
        self.function_data.append((fun, time_data, type, project, points, N, axes_id, title, cmap))
        
    def write(self, path, filename='animation.mp4', fps=10, interval=500, frame_ext='png'):
        # Plot frames and save them as temorary files
        for idx, t in enumerate(self.time):
            mpl = Mpl2DPlotter(layout=self.layout, projection=self.projection)
            updates = {
                'figsize': self.figsize,
                'fontsize': self.fontsize,
                'suptitle': f'{self.suptitle} t = {t:.4}'
            }
            mpl.update_figure(**updates)
            
            for fun, time_data, type, project, points, N, axes_id, title, cmap in self.function_data:
                fun.x.array[:] = time_data[idx]
                _ = mpl.plot(fun=fun, type=type, project=project, points=points, N=N, axes_id=axes_id, title=title)
                
                mpl.save(f'{path}/frame_{idx}.{frame_ext}')
                plt.close()
        
        # Animator cofiguration
        fig = plt.figure()
        fig.set_size_inches(*self.figsize)
        ax = plt.gca()
        plt.axis('off')

        def init():
            im.set_data(im0)

            return im,

        def animate(i):
            fname = f"{path}/frame_{i}.{frame_ext}"

            img = plt.imread(fname)
            im.set_data(img)

            return im,

        im0 = plt.imread(f"{path}/frame_{0}.{frame_ext}")
        im = ax.imshow(im0)

        anim = animation.FuncAnimation(fig, animate, init_func=init, repeat=True,
                                       frames=range(1, len(self.time)), interval=interval, 
                                       blit=True, repeat_delay=1000)

        # Write animation to file
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'))
        anim.save(f'{path}/{filename}', writer=writer)
        
        plt.close()
        
        # Remove temporary frames
        for i in range(len(self.time)):
            Path(f"{path}/frame_{i}.{frame_ext}").unlink()

