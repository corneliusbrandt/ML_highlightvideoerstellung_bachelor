import numpy as np
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation


'''
Script used for data augmentation. Main function is called in dataset_builder and creates augmented
dataset from the original one. It is iportant that the labels are not changed during augmentation.
The augmentation techniques used are based on the ones used in the paper "Data Augmentation for 
Wearable Sensor Data in Human Activity Recognition Tasks" by Terry Um et al. (https://arxiv.org/abs/1901.05948).
Even tough Permutation is stated as one of the most effective techniques in the paper, it is not implemented here
as it changes the temporal structure of the time series. It is importatn to choose augmentation techniques
that are physically meaningful.
Augmentation techniques and code inspored by: https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/tree/master

'''

def load_data(dataset_path):
    data = np.load(dataset_path)

    X = data['X']
    y = data['y']

    return X, y

def add_gaussian_noise(X, mean=0, std=0.05):
    noise = np.random.normal(loc=mean, scale=std, size=X.shape)
    return X + noise

def add_rotation(X, angle_range=(-15, 15)):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    # Implement rotation logic here (e.g., using OpenCV or scipy)
    return X  # Return the rotated data

# from git repo menationed above
# wie muss ich das zitieren?
def add_time_warping(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum

def add_rot_plus_time_warping(X, angle_range=(-15, 15), sigma=0.2):
    pass

#form git repo menationed above
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

def augment_data(X, y):
    pass




if __name__ == "__main__":
    X, y = load_data("datasets_output/train_dataset.npz")