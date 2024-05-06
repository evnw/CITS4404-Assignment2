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

def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
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

    def points_in_worm(self):
        n_points = math.ceil(self.approx_length() / self.width)
        points = []
        for point in self.intermediate_points(n_points):
            points += points_in_circle_np(self.width, point[0], point[1])
        return list(set(points))
    
    def get_mean_colour_under(self, image):
        colours = []
        xmin, xmax = [0, image.shape[0]]
        ymin, ymax = [0, image.shape[1]]
        for point in self.points_in_worm():
            if(    point[1] > xmin
                and point[1] < xmax
                and point[0] > ymin
                and point[0] < ymax ):
                colours += [image[point[1], point[0]]]
        # if len 0 then entirely off screen so penalise
        if len(colours) == 0:
            colours += [100*255]
        return np.mean(np.array(colours)/255)
    
    def edge_points (self, intervals, print_results=False):
        """
        Genenerates edge points around a worm, based on width
        """
        edge_points = []

        # Get Parallels returns two parallel lines separated by a width. 
        # This is not quite the true edge points as the end caps will be incorrect
        edge_points = np.array(
            mbezier.get_parallels(self.control_points(), self.width)
        )

        # The points returned are control points, not intermediate points.
        line1 = mbezier.BezierSegment(edge_points[0])
        line2 = mbezier.BezierSegment(edge_points[1])

        # Borrowed from the intermediate_points() method
        line1points = line1.point_at_t(np.linspace(0,1,intervals))
        line2points = line2.point_at_t(np.linspace(0,1,intervals))

        # Merge the two parallel lines into one continuous set of points:
        merged = np.concatenate((line1points, line2points), axis=0)

        if(print_results):
            print("control points: \n{}".format(self.control_points()))
            print("Edge Points: \n{}".format(merged))

        return merged
    

    def mate(self, partner, image):
        
        
        # 45% chance to pick one parent
        # 45% chance to pick other
        # 10% chance to mutate

        (radius_std, deviation_std, width_theta) = (40, 30, 1)
        (ylim, xlim) = image.shape
        
        #centerpoint coord x
        rand = rng.random() # 0 to 1
        if(rand < 0.45):       ch_x = self.x
        elif(rand < 0.9):      ch_x = partner.x
        else:                  ch_x = xlim * rng.random()
        #else:                  ch_x = (self.x + partner.x)/2

        #centerpoint coord y
        rand = rng.random()
        if(rand < 0.45):       ch_y = self.y
        elif(rand < 0.9):      ch_y = partner.y
        else:                  ch_y = ylim * rng.random()
        #else:                  ch_y = (self.y + partner.y)/2

        #radius
        rand = rng.random()       
        if(rand < 0.45):       ch_r = self.r
        elif(rand < 0.9):      ch_r = partner.r
        else:                  ch_r = radius_std * np.abs(rng.standard_normal())
        #else:                  ch_r = (self.r + partner.r)/2

        #angle from x axis
        rand = rng.random()        
        if(rand < 0.45):       ch_theta = self.theta
        elif(rand < 0.9):      ch_theta = partner.theta
        else:                  ch_theta = rng.random() * np.pi
        #else:                  ch_theta = (self.theta + partner.theta)/2

        # radius of deviation from midpoint
        rand = rng.random()        
        if(rand < 0.45):       ch_dr = self.dr
        elif(rand < 0.9):      ch_dr = partner.dr
        else:                  ch_dr = deviation_std * np.abs(rng.standard_normal())
        #else:                  ch_dr = (self.dr + partner.dr)/2

        #angle of line joining midpoint and curve
        rand = rng.random()        
        if(rand < 0.45):       ch_dgamma = self.dgamma
        elif(rand < 0.9):      ch_dgamma = partner.dgamma
        else:                  ch_dgamma = rng.random() * np.pi
        #else:                  ch_dgamma = (self.dgamma + partner.dgamma)/2

        #thickness
        rand = rng.random()        
        if(rand < 0.45):       ch_width = self.width
        elif(rand < 0.9):      ch_width = partner.width
        else:                  ch_width = width_theta * rng.standard_gamma(3)
        #else:                  ch_width = (self.width + partner.width)/2

        # worm colour - 0 to 255
        rand = rng.random()
        if(rand < 0.45):       ch_colour = self.colour
        elif(rand < 0.9):      ch_colour = partner.colour
        else:                  ch_colour = rng.random()
        #else:                  ch_colour = (self.colour + partner.colour)/2
        
        child = Camo_Worm(
            ch_x,
            ch_y,
            ch_r,
            ch_theta,
            ch_dr,
            ch_dgamma,
            ch_width,
            ch_colour
        )
        
        """
        print("Self:")
        self.print_variables()
        print("Partner:")
        partner.print_variables()
        print("Child:")
        child.print_variables()
        """
        
        return child


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





class Double_Drawing:
    """
    An extra utility class that has two images side by side.
    One with image, one with black background
    """
    def __init__ (self, image):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2)
        self.image = image
        self.im1 = self.ax1.imshow(self.image, cmap='gray', origin='lower')
        self.im2 = self.ax2.imshow(self.image, cmap='gray', vmin=255, vmax=255, origin='lower') # White background

    def add_patches1(self, patches):
        try:
            for patch in patches:
                self.ax1.add_patch(patch)
        except TypeError:
            self.ax1.add_patch(patches)

    def add_patches2(self, patches):
        try:
            for patch in patches:
                self.ax2.add_patch(patch)
        except TypeError:
            self.ax2.add_patch(patches)

    def add_dots(self, points, radius=4, **kwargs):
        try:
            for point in points:
                self.ax1.add_patch(mpatches.Circle((point[0],point[1]), radius, **kwargs))
                self.ax2.add_patch(mpatches.Circle((point[0],point[1]), radius, **kwargs))
        except TypeError:
            self.ax1.add_patch(mpatches.Circle((points[0],points[1]), radius, **kwargs))
            self.ax2.add_patch(mpatches.Circle((points[0],points[1]), radius, **kwargs))

    def add_worms(self, worms):
        try:
            self.add_patches1([w.patch() for w in worms])
            self.add_patches2([w.patch() for w in worms])
        except TypeError:
            self.add_patches1([worms.patch()])
            self.add_patches2([worms.patch()])

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