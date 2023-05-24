import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def createFigure(title, N_sides, data_to_plot, legends_label, rgrid=[0.2, 0.4, 0.6, 0.8], colors=['b', 'r', 'g', 'm', 'y']):
    """
    title:          A string for figure title.
    N_sides:        number of dimension for radar chart
    data_to_plot:   data structure needs to be a tuple with 2 elements:
                    (  ['dim1', 'dim2', 'dim3', ....],   # The corresponding label of dimensions
                       [[num, num, num, .....],          # The actual data to plot:
                        [num, num, num, .....],          # One array is one instance
                        [num, num, num, .....],
                        ....     
                    ) 
                    Note: The len of both elements need to equal to N_sides
    legends_label:  A list of strings, each of which is the name/ label of each instance.
    rgrid:          A list that contains the values to set up the inner radial gridlines.
    colors:         ['r', 'b', 'y', .....]                # color codes for each dimension
                    Choose your colors here:
                    https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    """

    theta = radar_factory(N_sides, frame='circle')
    spoke_labels = data_to_plot[0]
    fig, ax = plt.subplots(subplot_kw=dict(projection='radar'))

    N_instance = len(data_to_plot[1])
    if len(colors) < N_instance:
        colors = colors + colors[: N_instance-len(colors)]
    if len(colors) > N_instance:
        colors = colors[:N_instance]
    
    ax.set_rigids = ax.set_rgrids(rgrid)
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                    horizontalalignment='center', verticalalignment='center')
    
    for d, color in zip(data_to_plot[1], colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25) 
    
    ax.set_varlabels(spoke_labels)

    # Legend
    legend = ax.legend(legends_label, loc=(0.9, .95),
                       labelspacing=0.1, fontsize='small')

    return fig, ax
