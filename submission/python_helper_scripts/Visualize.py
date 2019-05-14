#!/usr/bin/env python
# coding: utf-8

# In[259]:


from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
import os
from scipy.spatial.distance import cdist
import copy
from scipy.signal import convolve2d
from scipy.ndimage import rank_filter
from scipy.stats import norm
from scipy.misc import imsave
from PIL import Image


# In[ ]:


data = [(1, 49), (2, 53), (3, 62), (4, 72), (5, 69),(6, 65), (7,70), (8, 72), (9, 73), (10, 72), (11, 71), (12, 73), (13, 72), (14, 72), (16, 72), (18, 73)]


# In[ ]:


x = [k[0] for k in data]


# In[ ]:


y = [k[1] for k in data]


# In[ ]:


plt.plot(x, y, color = 'orange')
# plt.plot(range(1, len(accuracy_0_1 )+1), accuracy_0_1, color = 'green')
plt.xlabel("patch dimensions")
plt.ylabel("Total correct out of 100")
plt.title("Tinyimage KNN(3)")
plt.savefig('./images/Tinyimage knn(3).png')
plt.show()


# In[ ]:





# In[2]:


class CatsvDogsDataset(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.labels_file = os.path.join(dataset_dir, 'labels.txt')
        assert os.path.exists(self.labels_file)
        self.__read_labels_file__()
        self.train_indices = np.where(self.image_sets == 1)[0]
        self.val_indices = np.where(self.image_sets == 2)[0]
        self.test_indices = np.where(self.image_sets == 3)[0]

    def __read_labels_file__(self):
        self.image_names = []
        self.class_ids = []
        self.image_sets = []
        with open(self.labels_file, 'r') as f:
            for line in f.readlines():
                n, i, s = line.strip().split()
                self.image_names.append(n)
                self.class_ids.append(int(i))
                self.image_sets.append(int(s))
        self.image_names = np.array(self.image_names)
        self.class_ids = np.array(self.class_ids)
        self.image_sets = np.array(self.image_sets)


def read_dataset(dataset_dir):
    return CatsvDogsDataset(dataset_dir)


# In[231]:


def extract_local_features(im, r, stride):
    features = []
    start = (0 + r, 0 + r)
    
    while(start[0] + r < im.shape[0]):
        while(start[1] + r < im.shape[1]):
            patch = []
            for i in range(int(start[0] - r), int(start[0] + r + 1)):
                for j in range(int(start[1] - r), int(start[1] + r + 1)):
                    patch.append(im[i, j])
            features.append(patch)
            start = (start[0], start[1] + stride)
        start = (start[0] + stride, 0 + r)    
    return features
    
def extract_sift_features(im, r, stride):
    features = []
    start = [[0 + r, 0 + r]]
    while(start[0][0] + r < im.shape[0]):
        while(start[0][1] + r < im.shape[1]):
            features.append(find_sift(I=im, circles = np.asarray(start), radius = r)[0])
            start = [[start[0][0], start[0][1] + stride]]
        start = [[start[0][0] + stride, 0 + r]]
    return np.asarray(features)
    
def createDictionary(img_list, r, stride, clusterCount, imdb):
    features = []
    for im in img_list:
        npImage = imread(os.path.join(imdb.image_dir, im))
        npImage = rgb2gray(npImage)
        features.extend(extract_local_features(npImage, r, stride))
    
    kmeans = KMeans(n_clusters=clusterCount, random_state=0).fit(features)
    return kmeans.cluster_centers_

def createSiftDictionary(img_list, r, stride, clusterCount, imdb):
    features = []
    for im in img_list:
        npImage = imread(os.path.join(imdb.image_dir, im))
        npImage = rgb2gray(npImage)
        features.extend(extract_sift_features(npImage, r, stride))
    
    kmeans = KMeans(n_clusters=clusterCount, random_state=0).fit(features)
    return kmeans.cluster_centers_


# In[249]:


imdb = read_dataset("./data")


# In[194]:





# In[240]:


def createSiftDictionary(img_list, r, stride, clusterCount, imdb):
    features = []
    for im in img_list[:2]:
        npImage = imread(os.path.join(imdb.image_dir, im))
        npImage = rgb2gray(npImage)
        features.extend(extract_sift_features(npImage, r, stride))
        print("asdf")
    print("*******************")
    kmeans = KMeans(n_clusters=clusterCount, random_state=0).fit(features)
    return kmeans.cluster_centers_


# In[242]:


train_val = list(imdb.train_indices) + list(imdb.val_indices)
train_val = imdb.image_names[train_val]
clusterCount = 128
centers = createSiftDictionary(train_val, 8, 20, clusterCount, imdb)


# In[245]:


def bow_patch_features(imdb, dictionarysize, radius, stride):
    '''
     STEP 1: Write a function that extracts dense grayscale patches from an image
     STEP 2: Learn a dictionary
               -- sample many desriptors (~10k) from train+val images
               -- learn a dictionary using k-means
     STEP 3: Loop over all the images  and extract
             features (same as step 1). Build global histograms over these.
    '''
    train_val = list(imdb.train_indices) + list(imdb.val_indices)
    train_val = imdb.image_names[train_val]
    clusterCount = dictionarysize
    
    centers = createDictionary(train_val, radius, stride, clusterCount, imdb)
    return centers
#     features = np.zeros((len(imdb.image_names), clusterCount) )

#     for index, image in enumerate(imdb.image_names):
#         hist = np.zeros((clusterCount))
#         npImage = rgb2gray(imread(os.path.join(imdb.image_dir, image)))
#         f1 = extract_local_features(npImage, radius, stride)
#         distMatrix = cdist(f1, centers, 'sqeuclidean')
#         min_index = np.argmin(distMatrix, axis=1)
#         for i in min_index:
#             hist[i] += 1
#         features[index, :] = hist
#     return features


# In[266]:


dic = bow_patch_features(imdb, 128, 9, 20)


# In[269]:


image_dict = np.zeros((128, 19, 19))
for i in range(dic.shape[0]):
    image_dict[i] = dic[i].reshape((19,19))


# In[270]:


def save_img_c(array, filename):
    """
    Saves given numpy array to ./images folder with given filename
    """
    imsave("./images/"+ filename, array)


# In[271]:


def save_img(array, filename):
    """
    Saves given numpy array to ./images folder with given filename
    """
    img_new = Image.fromarray(array)
    img_new = img_new.convert("RGB")
    fp = open("./images/" + filename, "wb")
    img_new.save(fp)
    fp.close()
    return None


# In[272]:


for i in range(image_dict.shape[0]):
    save_img(image_dict[i] * 255, "test" + str(i) + ".png")


# In[273]:


image_dict[0]


# In[ ]:





# In[160]:


test = features.copy()


# In[162]:


for i in range(test.shape[0]):
    test[i] = test[i] / test[i].sum()


# In[168]:


test[133].sum()


# In[186]:


def gen_dgauss(sigma):
    """
    Generates the horizontally and vertically differentiated Gaussian filter

    Parameters
    ----------
    sigma: float
        Standard deviation of the Gaussian distribution

    Returns
    -------
    Gx: numpy.ndarray
        First degree derivative of the Gaussian filter across rows
    Gy: numpy.ndarray
        First degree derivative of the Gaussian filter across columns
    """
    f_wid = 4 * np.floor(sigma)
    G = norm.pdf(np.arange(-f_wid, f_wid + 1),
                 loc=0, scale=sigma).reshape(-1, 1)
    G = G.T * G
    Gx, Gy = np.gradient(G)

    Gx = Gx * 2 / np.abs(Gx).sum()
    Gy = Gy * 2 / np.abs(Gy).sum()

    return Gx, Gy

def find_sift(I, circles, radius= 8):
    """
    Compute non-rotation-invariant SITF descriptors of a set of circles

    Parameters
    ----------
    I: numpy.ndarray
        Image
    circles: numpy.ndarray
        An array of shape `(ncircles, 3)` where ncircles is the number of
        circles, and each circle is defined by (x, y, r), where r is the radius
        of the cirlce
    enlarge_factor: float
        Factor which indicates by how much to enlarge the radius of the circle
        before computing the descriptor (a factor of 1.5 or large is usually
        necessary for best performance)

    Returns
    -------
    sift_arr: numpy.ndarray
        Array of SIFT descriptors of shape `(ncircles, 128)`
    """
    assert circles.ndim == 2 and circles.shape[1] == 2,         'Use circles array (keypoints array) of correct shape'
    I = I.astype(np.float64)
    if I.ndim == 3:
        I = rgb2gray(I)

    NUM_ANGLES = 8
    NUM_BINS = 4
    NUM_SAMPLES = NUM_BINS * NUM_BINS
    ALPHA = 9
    SIGMA_EDGE = 1

    ANGLE_STEP = 2 * np.pi / NUM_ANGLES
    angles = np.arange(0, 2 * np.pi, ANGLE_STEP)

    height, width = I.shape[:2]
    num_pts = circles.shape[0]

    sift_arr = np.zeros((num_pts, NUM_SAMPLES * NUM_ANGLES))

    Gx, Gy = gen_dgauss(SIGMA_EDGE)

    Ix = convolve2d(I, Gx, 'same')
    Iy = convolve2d(I, Gy, 'same')
    I_mag = np.sqrt(Ix ** 2 + Iy ** 2)
    I_theta = np.arctan2(Ix, Iy + 1e-12)

    interval = np.arange(-1 + 1/NUM_BINS, 1 + 1/NUM_BINS, 2/NUM_BINS)
    gridx, gridy = np.meshgrid(interval, interval)
    gridx = gridx.reshape((1, -1))
    gridy = gridy.reshape((1, -1))

    I_orientation = np.zeros((height, width, NUM_ANGLES))

    for i in range(NUM_ANGLES):
        tmp = np.cos(I_theta - angles[i]) ** ALPHA
        tmp = tmp * (tmp > 0)

        I_orientation[:, :, i] = tmp * I_mag

    for i in range(num_pts):
        cy, cx = circles[i, :2]
#         r = circles[i, 2]
        r = radius

        gridx_t = gridx * r + cx
        gridy_t = gridy * r + cy
        grid_res = 2.0 / NUM_BINS * r

        x_lo = np.floor(np.max([cx - r - grid_res / 2, 0])).astype(np.int32)
        x_hi = np.ceil(np.min([cx + r + grid_res / 2, width])).astype(np.int32)
        y_lo = np.floor(np.max([cy - r - grid_res / 2, 0])).astype(np.int32)
        y_hi = np.ceil(
            np.min([cy + r + grid_res / 2, height])).astype(np.int32)

        grid_px, grid_py = np.meshgrid(
            np.arange(x_lo, x_hi, 1),
            np.arange(y_lo, y_hi, 1))
        grid_px = grid_px.reshape((-1, 1))
        grid_py = grid_py.reshape((-1, 1))

        dist_px = np.abs(grid_px - gridx_t)
        dist_py = np.abs(grid_py - gridy_t)

        weight_x = dist_px / (grid_res + 1e-12)
        weight_x = (1 - weight_x) * (weight_x <= 1)
        weight_y = dist_py / (grid_res + 1e-12)
        weight_y = (1 - weight_y) * (weight_y <= 1)
        weights = weight_x * weight_y

        curr_sift = np.zeros((NUM_ANGLES, NUM_SAMPLES))
        for j in range(NUM_ANGLES):
            tmp = I_orientation[y_lo:y_hi, x_lo:x_hi, j].reshape((-1, 1))
            curr_sift[j, :] = (tmp * weights).sum(axis=0)
        sift_arr[i, :] = curr_sift.flatten()

    tmp = np.sqrt(np.sum(sift_arr ** 2, axis=-1))
    if np.sum(tmp > 1) > 0:
        sift_arr_norm = sift_arr[tmp > 1, :]
        sift_arr_norm /= tmp[tmp > 1].reshape(-1, 1)

        sift_arr_norm = np.clip(sift_arr_norm, sift_arr_norm.min(), 0.2)

        sift_arr_norm /= np.sqrt(
            np.sum(sift_arr_norm ** 2, axis=-1, keepdims=True))

        sift_arr[tmp > 1, :] = sift_arr_norm

    return sift_arr


# In[244]:


print(imdb.image_names[105])


# In[8]:


import numpy as np


# In[9]:


t = np.array([1, 4 , 54, 2323])


# In[14]:


np.nonzero(t == 44)[0][0]


# In[16]:


type(sorted(t))


# In[22]:


t = [23,4, 2,345, 34]


# In[23]:


t.index(4)


# In[ ]:




