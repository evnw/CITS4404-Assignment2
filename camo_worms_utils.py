# adds/changes
# modified colour_at_t
# added get_mean_colour_under
# added get_dots

# Imports
import numpy as np
import imageio.v3 as iio
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.bezier as mbezier
from sklearn.metrics.pairwise import euclidean_distances

rng = np.random.default_rng()
Path = mpath.Path
mpl.rcParams['figure.dpi']= 72 #size of images

# Global variables

IMAGE_DIR = 'images'
IMAGE_NAME='original'
MASK = [320, 560, 160, 880] # ymin ymax xmin xmax


# Read, crop and display image and stats

def crop (image, mask):
    h, w = np.shape(image)
    return image[max(mask[0],0):min(mask[1],h), max(mask[2],0):min(mask[3],w)]

def prep_image (imdir, imname, mask):
    print("Image name (shape) (intensity max, min, mean, std)\n")
    image = np.flipud(crop(iio.imread(imdir+'/'+imname+".png"), mask))
    print("{} {} ({}, {}, {}, {})".format(imname, np.shape(image), np.max(image), np.min(image), round(np.mean(image),1), round(np.std(image),1)))
    plt.imshow(image, vmin=0, vmax=255, cmap='gray', origin='lower') # use vmin and vmax to stop imshow from scaling
    plt.show()
    return image

def points_in_circle_np(radius, x0=0, y0=0):
    x_ = np.arange(int(x0 - radius - 1), int(x0 + radius + 1))
    y_ = np.arange(int(y0 - radius - 1), int(y0 + radius + 1))
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
    # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
    for x, y in zip(x_[x], y_[y]):
        yield x, y

class Camo_Worm:
    def __init__(self, x, y, r, theta, deviation_r, deviation_gamma, width, colour):
        self.x = x
        self.y = y
        self.r = r
        self.theta = theta
        self.dr = deviation_r
        self.dgamma = deviation_gamma
        self.width = width
        self.colour = colour
        p0 = [self.x - self.r * np.cos(self.theta), self.y - self.r * np.sin(self.theta)]
        p2 = [self.x + self.r * np.cos(self.theta), self.y + self.r * np.sin(self.theta)]
        p1 = [self.x + self.dr * np.cos(self.theta+self.dgamma), self.y + self.dr * np.sin(self.theta+self.dgamma)]
        self.bezier = mbezier.BezierSegment(np.array([p0, p1,p2]))

    def control_points (self):
        return self.bezier.control_points

    def path (self):
        return mpath.Path(self.control_points(), [Path.MOVETO, Path.CURVE3, Path.CURVE3])

    def patch (self):
        return mpatches.PathPatch(self.path(), fc='None', ec=str(self.colour), lw=self.width, capstyle='round')

    def intermediate_points(self, intervals=None):
        if intervals is None:
            intervals = max(3, int(np.ceil(self.r/8)))
        return self.bezier.point_at_t(np.linspace(0,1,intervals))

    def approx_length(self):
        intermediates = self.intermediate_points()
        eds = euclidean_distances(intermediates,intermediates)
        return np.sum(np.diag(eds,1))

    # def colour_at_t(self, t, image):
    #     intermediates = np.int64(np.round(np.array(self.bezier.point_at_t(t)).reshape(-1,2)))
    #     colours = [image[point[1],point[0]] for point in intermediates]
    #     return(np.array(colours)/255)

    def colour_at_t(self, t, image):
        point = np.int64(np.round(np.array(self.bezier.point_at_t(t)).reshape(-1,2)))[0]
        # ignore points outside image
        xmin, xmax = [0, image.shape[0]]
        ymin, ymax = [0, image.shape[1]]
        if(    point[1] > xmin
           and point[1] < xmax
           and point[0] > ymin
           and point[0] < ymax ):
            colours = image[point[1],point[0]] # reversed as [y, x]
            colour = np.array(colours)/255
            return colour
        else:
            return -1
    
    # def get_mean_colour_under(self, num_intervals, image):
    #     t_vals = np.linspace(0,1,num_intervals)
    #     avg_colour = np.mean([self.colour_at_t(t, image) for t in t_vals])
    #     return avg_colour

    def points_in_worm(self, n_points_mod: float=1.0):
        n_points = math.ceil(self.approx_length() * n_points_mod / self.width)
        points = []
        for point in self.intermediate_points(n_points):
            points += points_in_circle_np(self.width, point[0], point[1])
        return list(set(points))
    
    def get_colour_under(self, image):
        colours = []
        xmin, xmax = [0, image.shape[0]]
        ymin, ymax = [0, image.shape[1]]
        for point in self.points_in_worm():
            if(    point[1] > xmin
                and point[1] < xmax
                and point[0] > ymin
                and point[0] < ymax ):
                colours += [image[point[1], point[0]]]
        # if len 0 then entirely off screen
        if len(colours) == 0:
            return None
        return np.array(colours)/255
    
    def get_colour_around(self, image):
        # increase width and use uncommon points to get colour around worm
        bigger_worm = Camo_Worm(self.x, self.y, self.r, self.theta, self.dr, self.dgamma, self.width*1.3, self.colour)
        points_bigger = bigger_worm.points_in_worm(n_points_mod=2.0)
        points_smaller = self.points_in_worm()
        points_around = list(set(points_bigger) - set(points_smaller))

        colours = []
        xmin, xmax = [0, image.shape[0]]
        ymin, ymax = [0, image.shape[1]]
        for point in points_around:
            if(    point[1] > xmin
                and point[1] < xmax
                and point[0] > ymin
                and point[0] < ymax ):
                colours += [image[point[1], point[0]]]
        # if len 0 then entirely off screen
        if len(colours) == 0:
            print("x", self.x,"y", self.y,"r", self.r,"theta", self.theta,"dr", self.dr,"dg", self.dgamma,"width", self.width,"colour", self.colour)
            print(points_smaller)
            print(points_around)
            return None
        return np.array(colours)/255


class Drawing:
    def __init__ (self, image):
        self.fig, self.ax = plt.subplots()
        self.image = image
        self.im = self.ax.imshow(self.image, cmap='gray', origin='lower')

    def add_patches(self, patches):
        try:
            for patch in patches:
                self.ax.add_patch(patch)
        except TypeError:
            self.ax.add_patch(patches)

    def add_dots(self, points, radius=4, **kwargs):
        try:
            for point in points:
                self.ax.add_patch(mpatches.Circle((point[0],point[1]), radius, **kwargs))
        except TypeError:
            self.ax.add_patch(mpatches.Circle((points[0],points[1]), radius, **kwargs))

    def add_worms(self, worms):
        try:
            self.add_patches([w.patch() for w in worms])
        except TypeError:
            self.add_patches([worms.patch()])

    def show(self, save=None):
        if save is not None:
            plt.savefig(save)
        plt.show()

# Example of a random worm. You may do this differently.

    # centre points, angles and colour chosen from uniform distributions
    # lengths chosen from normal distributions with two std parameters passed
    # width chosen from gamma distribution with shape parameter 3 and scale passed

def random_worm (imshape, init_params):
    (radius_std, deviation_std, width_theta) = init_params
    (ylim, xlim) = imshape
    midx = xlim * rng.random()
    midy = ylim * rng.random()
    r = radius_std * np.abs(rng.standard_normal())
    theta = rng.random() * np.pi
    dr = deviation_std * np.abs(rng.standard_normal())
    dgamma = rng.random() * np.pi
    colour = rng.random()
    width = width_theta * rng.standard_gamma(3)
    return Camo_Worm(midx, midy, r, theta, dr, dgamma, width, colour)

# Initialise a random clew

def initialise_clew (size, imshape, init_params):
    clew = []
    for i in range(size):
        clew.append(random_worm(imshape, init_params))
    return clew

def observe_clew(clew, image):
    dots=[]
    for worm in clew:
        (p1,p2,p3) = worm.intermediate_points(3)
        dots.append(p1)
        dots.append(p2)
        dots.append(p3)

    drawing = Drawing(image)
    drawing.add_worms(clew)
    drawing.add_dots(dots, 1, color="yellow")
    drawing.show()
    return