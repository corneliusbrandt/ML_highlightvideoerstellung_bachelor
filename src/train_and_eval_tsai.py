import numpy as np
from tsai.all import *
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from helper import load_data
from focal_loss import FocalLoss
from helper import calculate_class_weights


#Datapreparation
batch_size = 32

#Training Loop
n_epochs = 20
learning_rate = 0.00001
weight_scaling_factor = 2
min_weight = 0.01
gamma = 3

#threshold for converting probabilities to binary predictions
threshold = 0.57

#---------------------------------------------------------------------------
# load data
#---------------------------------------------------------------------------

X_train, y_train = load_data("datasets_output/train_dataset_augmented.npz")
X_val, y_val = load_data("datasets_output/val_dataset_augmented.npz")

#---------------------------------------------------------------------------
# Adjust shape
# from: (samples, time_steps, channels)
# to:  (samples, channels, time_steps)
#---------------------------------------------------------------------------
X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))

#---------------------------------------------------------------------------
# Print out data shapes and class distribution
#---------------------------------------------------------------------------
print("X_train:", X_train.shape, X_train.dtype, X_train.flags["C_CONTIGUOUS"])
print("y_train:", y_train.shape, y_train.dtype)
print("Klassen Train:", np.unique(y_train, return_counts=True))
print("X_val:", X_val.shape, X_val.dtype, X_val.flags["C_CONTIGUOUS"])
print("y_val:", y_val.shape, y_val.dtype)
print("Klassen Val:", np.unique(y_val, return_counts=True))

#---------------------------------------------------------------------------
# Train/Val 
# tsai works with combined datasets and uses indices to split them -> Train and Val are NOT beeing combined
#---------------------------------------------------------------------------
X = np.concatenate([X_train, X_val], axis=0)
y = np.concatenate([y_train, y_val], axis=0)

train_idxs = np.arange(len(X_train))
val_idxs = np.arange(len(X_train), len(X_train) + len(X_val))

splits = (train_idxs, val_idxs)

#---------------------------------------------------------------------------
# create dataloaders
#---------------------------------------------------------------------------
dls = get_ts_dls(
    X,
    y,
    splits=splits,
    bs=batch_size,
    tfms=[None, TSCategorize()],
    shuffle_train=True
)

#---------------------------------------------------------------------------
# create models
#---------------------------------------------------------------------------
n_channels = X_train.shape[1]
n_classes = len(np.unique(y_train))


model_resnet = ResNet(
    c_in=n_channels,
    c_out=n_classes,
)

model_inceptiontime = InceptionTime(
    c_in=n_channels,
    c_out=n_classes,
)

model_FCN = FCN(
    c_in=n_channels,
    c_out=n_classes,
)

model_omniscale = OmniScaleCNN(
    c_in=n_channels,
    c_out=n_classes,
    seq_len=X_train.shape[2]
)


models = [model_resnet, model_inceptiontime, model_FCN, model_omniscale]
print("n_channels:", n_channels)
print("n_classes:", n_classes)

#---------------------------------------------------------------------------
# Initialize class weights and loss function
#---------------------------------------------------------------------------
class_weights = calculate_class_weights(y_train, num_classes=n_classes, scaling_factor=weight_scaling_factor, min_weight=min_weight)
loss_function = FocalLoss(gamma=gamma, alpha=class_weights, task_type='multi-class', num_classes=n_classes)


#---------------------------------------------------------------------------
# Training and Evaluation Loop
#---------------------------------------------------------------------------
for model in models:
    print("\n--- Training model:", model.__class__.__name__, "---")

    learn = Learner(
        dls,
        model,
        loss_func=loss_function,
        opt_func=Adam,
        metrics=[accuracy, F1Score(average='macro'), Precision(average='macro'), Recall(average='macro')],
        path=Path("src/Models"),
        cbs = [SaveModel(monitor='f1_score',fname=f"{model.__class__.__name__}_multiclass_tsai", verbose=True)]
    )


    # Train
    learn.fit_one_cycle(
        n_epoch=n_epochs,
        lr_max=learning_rate
    )


    # Validation
    preds, targets = learn.get_preds(ds_idx=1)
    probs = preds.numpy()
    y_true = targets.numpy()
    y_pred = np.argmax(probs, axis=1)


    # Prediction wih argmax
    print("\n--- Prediction mit argmax ---")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0, average='macro'))


    # prediction with threshold
    # only for binary classification
    event_probs = probs[:, 1]
    y_pred_threshold = (event_probs > threshold).astype(int)

    print("\n--- Prediction with Threshold ---")
    print("Threshold:", threshold)
    print(confusion_matrix(y_true, y_pred_threshold))
    print(classification_report(y_true, y_pred_threshold, zero_division=0))
    print("F1:", f1_score(y_true, y_pred_threshold, zero_division=0, average='macro'))