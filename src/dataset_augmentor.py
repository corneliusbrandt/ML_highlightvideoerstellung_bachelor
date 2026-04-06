import numpy as np
from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat


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

# from git repo menationed above
# wie muss ich das zitieren?
def add_rotation(X, angle_range=(-15, 15)):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])  # Random rotation axis
    angle = np.random.uniform(low=angle_range[0], high=angle_range[1])  # Random rotation angle
    return np.matmul(X , axangle2mat(axis,angle))

# from git repo menationed above
# wie muss ich das zitieren?
# changed to have the same time warping for all channels (assisted by chatgpt)
def add_time_warping(X, sigma=0.2):
    n_steps, n_channels = X.shape
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to create warped timeline
    # Make the last value to have X.shape[0] to ensure the same length after warping
    t_scale = (X.shape[0]-1)/tt_cum[-1]
    tt_cum = tt_cum * t_scale
    
    t_original = np.arange(n_steps)

    X_warped = np.zeros_like(X)

    #interpolate on new timeline and evaluate at original time steps for each channel
    for ch in range(n_channels):
        X_warped[:, ch] = np.interp(
            t_original,
            tt_cum,
            X[:, ch]
        )

    return X_warped
    
def add_rot_plus_time_warping(X, angle_range=(-15, 15), sigma=0.2):
    X_rot = add_rotation(X, angle_range)
    X_warped = add_time_warping(X_rot, sigma)
    return X_warped

#form git repo menationed above
# changed to have the same time warping for all channels (assisted by chatgpt)
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    n_steps, n_channels = X.shape
    # generate evenly spaced points between 0 and length of the array -> knots are "anqors" for splines
    xx = np.linspace(0, n_steps-1, num=knot+2)
    #generate random values around 1 for each knot 
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2,))
    x_range = np.arange(n_steps)
    #fit spline to the created points and evaluate them at the original time steps
    c = CubicSpline(xx, yy)(x_range)
    return c

def augment_data(X, y):
    for window, label in zip(X, y):
        pass
pass
    

def test_visualization(X, y):
    pass







if __name__ == "__main__":
    X, y = load_data("datasets_output/train_dataset.npz")