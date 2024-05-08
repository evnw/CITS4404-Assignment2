# Imports

import cv2
import numpy as np
import imageio.v3 as iio

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.bezier as mbezier
from sklearn.metrics.pairwise import euclidean_distances

from skimage import metrics

rng = np.random.default_rng()
Path = mpath.Path
mpl.rcParams['figure.dpi']= 72 #size of images

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
        self.mask = None

    def create_mask(self, image):
        self.mask = WormMask(self, image)

    def control_points (self):
        return self.bezier.control_points

    def path (self):
        return mpath.Path(self.control_points(), [Path.MOVETO, Path.CURVE3, Path.CURVE3])

    def patch (self):
        return mpatches.PathPatch(self.path(), fc='None', ec=str(self.colour), lw=self.width, capstyle='round')

    def intermediate_points (self, intervals=None):
        if intervals is None:
            intervals = max(3, int(np.ceil(int(self.r / 8))))
        return self.bezier.point_at_t(np.linspace(0,1,intervals))

    def approx_length (self):
        intermediates = self.intermediate_points()
        eds = euclidean_distances(intermediates,intermediates)
        return np.sum(np.diag(eds,1))

    def colour_at_t(self, t, image):
        intermediates = np.int64(np.round(np.array(self.bezier.point_at_t(t)).reshape(-1,2)))
        colours = [image[point[1],point[0]] for point in intermediates]
        return(np.array(colours)/255)
    
    def environment_fitness(self, image, points=3):
        arr = []
        steps = np.pi / 16
        intermediates = self.intermediate_points(points)
        for point in intermediates:
            if (point[0] < 720 and point[1] < 240): 
                arr.append(255 - abs(self.colour * 255 - image[int(point[1]), int(point[0])]))
                rads = 0 
                while (rads < 2 * np.pi):
                    x = int(point[0] + 2 * np.cos(rads))
                    y = int(point[1] + 2 * np.sin(rads))
                    if (x < 720 and y < 240):
                        arr.append(255 - abs(self.colour * 255 - image[y, x]))
                        rads += steps
                    else:
                        arr.append(0)
                        rads += steps
            else:
                arr.append(0)
        return np.average(arr)
    
    def environment_fitness2(self):
        return self.mask.colour_difference()

    def distance(self, x, y):
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)

# The worm mask class
class WormMask:
    def __init__(self, worm: Camo_Worm, image):
        self.worm = worm
        self.points = math.ceil(worm.approx_length() * 5 / worm.width)
        self.image = image
        if max(self.image[0]) > 1:
            self.image = self.image / 255
        self.mask = self.create_mask()

    # Create a rectangle mask around the worm    
    def create_mask(self):
        new_img = np.full(self.image.shape, 0, dtype=np.float64)
        self.max_x = 0
        self.min_x = new_img.shape[1]
        self.max_y = 0
        self.min_y = new_img.shape[0]
        pts = self.worm.intermediate_points(self.points)
        for pt in pts:
            if pt[0] < new_img.shape[1] and pt[1] < new_img.shape[0]:
                cv2.circle(new_img, (int(pt[0]), int(pt[1])), int(self.worm.width), self.worm.colour, -1) 
                self.max_x = int(max(self.max_x, pt[0] + self.worm.width))
                self.max_y = int(max(self.max_y, pt[1] + self.worm.width))
                self.min_x = int(min(self.min_x, pt[0] - self.worm.width))
                self.min_y = int(min(self.min_y, pt[1] - self.worm.width))
        if self.min_x < 0: self.min_x = 0
        if self.min_y < 0: self.min_y = 0
        return new_img[self.min_y:self.max_y, self.min_x:self.max_x]
    
    # Crop the image under the mask
    def crop_xy(self, x, y, width_x, width_y):
        return self.image[y : y + width_y, x : x + width_x].copy()
    
    # Calculate the colour difference between the mask and the image
    def colour_difference(self):
        image_crop = self.crop_xy(self.min_x, self.min_y, self.max_x - self.min_x, self.max_y - self.min_y)
        arr = []
        for i in range(0, image_crop.shape[0]):
            for j in range(0, image_crop.shape[1]):
                if self.mask[i, j] != 0:
                    arr.append(1.0 - abs(image_crop[i, j] - self.mask[i, j]))
        return np.average(arr)

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

    def show_image(self, save=None):
        plt.axis('off')
        if save is not None:
            plt.savefig(save)
        plt.show()

    