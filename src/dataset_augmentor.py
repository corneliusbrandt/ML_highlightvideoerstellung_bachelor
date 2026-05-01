import numpy as np
from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat
import matplotlib.pyplot as plt
from helper import load_data

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

def add_gaussian_noise(X, mean=0, std=0.05):
    noise = np.random.normal(loc=mean, scale=std, size=X.shape)
    return X + noise

# from git repo menationed above
# wie muss ich das zitieren?
# rewritten to rotate 3 seonsors with 9 channels each -> otherwise would have to rotate in 27D -> not very physically meaningful (assisted by chatgpt)
def add_rotation(X, angle_range=(-np.deg2rad(20), np.deg2rad(20))):

    X_rot = X.copy()
    axis = np.random.uniform(low=-1, high=1, size=3)  # Random rotation axis 3D vector
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    angle = np.random.uniform(low=angle_range[0], high=angle_range[1])  # Random rotation angle

    R = axangle2mat(axis, angle)  # Rotation matrix

    # rotate sensors piecewise (each sensor has 9 channels: 3 for Euler, 3 for Acceleration, 3 for Gyroscope)
    for sensor in range(3):
        base_idx = sensor * 9

        #Euler (Euler is not rotatet -> makes no sense physically)
        X_rot[:, base_idx:base_idx+3] = X[:, base_idx:base_idx+3]

        #Acceleration
        X_rot[:, base_idx+3:base_idx+6] = np.matmul(X[:, base_idx+3:base_idx+6], R)

        #Gyroscope
        X_rot[:, base_idx+6:base_idx+9] = np.matmul(X[:, base_idx+6:base_idx+9], R)

    return X_rot

# from git repo menationed above
# wie muss ich das zitieren?
# changed to have the same time warping for all channels (assisted by chatgpt)
def add_time_warping(X, sigma=0.5):
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
    
def add_rot_plus_time_warping(X, angle_range=(-np.deg2rad(20), np.deg2rad(20)), sigma=0.2):
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


# main functio to augment dataset. Only returns augmented windows that have to be appended to the main dataset.

'''
Here it may be good to implement a logic which matches the count of the different classes. For example if 3 is underrepresented it will augment more windos
with the label 4.
'''
def augment_data(X, y):
    augmented_X = []
    augmented_y = []
    for window, label in zip(X, y):
        if label != 5:
            augmented_X.append(add_gaussian_noise(window))
            augmented_y.append(label)
            augmented_X.append(add_rotation(window))
            augmented_y.append(label)
            augmented_X.append(add_time_warping(window))
            augmented_y.append(label)
            augmented_X.append(add_rot_plus_time_warping(window))
            augmented_y.append(label)
    
    return np.array(augmented_X), np.array(augmented_y)


#plot written by chatgpt 5.3 on 16.04.26
def test_visualization(window_original, window_augmented):
    n_samples, n_channels = window_original.shape
    time_steps = np.arange(n_samples)

    group_size = 3
    n_groups = n_channels // group_size  # here: 27 → 9

    n_cols = 3
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for g in range(n_groups):
        ax = axes[g]
        start = g * group_size
        end = start + group_size

        # colors for the 3 channels
        colors = plt.cm.tab10.colors

        for i, ch in enumerate(range(start, end)):
            # Original
            l1, = ax.plot(time_steps,
                          window_original[:, ch],
                          color=colors[i],
                          linestyle='-',
                          linewidth=1)

            # Augmented
            l2, = ax.plot(time_steps,
                          window_augmented[:, ch],
                          color=colors[i],
                          linestyle='--',
                          linewidth=1)

        ax.set_title(f'Channels {start+1}-{end}', fontsize=10)
        ax.grid(alpha=0.3)

        if g // n_cols == n_rows - 1:
            ax.set_xlabel('Time')
        if g % n_cols == 0:
            ax.set_ylabel('Value')

    #remove empty subplots
    for j in range(n_groups, len(axes)):
        fig.delaxes(axes[j])

    # global legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', label='Original'),
        Line2D([0], [0], color='black', linestyle='--', label='Augmented')
    ]

    fig.legend(handles=legend_elements,
               loc='upper center',
               ncol=2,
               frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    X, y = load_data("datasets_output/train_dataset.npz")
    augmented_X, augmented_y = augment_data(X, y)
    print("Original dataset shape:", X.shape, y.shape)
    print("Augmented dataset shape:", augmented_X.shape, augmented_y.shape)
    
    # visualize one example
    original_windows_augmented = []
    for window, label in zip(X, y):
        if label != 5:
            original_windows_augmented.append(window)
    original_windows_augmented = np.array(original_windows_augmented)
    print("Original windows that were augmented shape:", original_windows_augmented.shape)
    

    #for each original window there are 4 augmented windows (gaussian noise, rotation, time warping, rotation + time warping) 
    #index for augmented = index for original * 4 + augmentation type (0-3)
    test_visualization(original_windows_augmented[0], augmented_X[3])
