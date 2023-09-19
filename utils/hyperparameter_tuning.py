from utils.mlp_sparse_model_tf2 import MLPSparseModel
from utils.mlp_plain_model_tf2 import MLPPlainModel
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

def nn_l1_val(X_train1, Y_train1, X_train2, Y_train2, n_layer, lambd, lr_initial):
    config = dict()
    config['num_input'] = X_train1.shape[1]
    config['num_layer'] = n_layer
    config['num_neuron'] = 128
    config['lambda'] = lambd
    config['verbose'] = 0

    dir_output = 'C:/Users/Downloads/'

    # Build and train model
    # print(config, dir_output)
    model = MLPSparseModel(config)
    model.build_train()
    # print(Y_train1)
    model.train(X_train1, Y_train1, lr_initial)

    # Evaluate trained model on validation data
    Y_pred_val = model.predict(X_train2)
    # print(Y_pred_val)
    abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
    rel_error = np.mean(np.abs(np.divide(Y_train2 - Y_pred_val, Y_train2)))

    return abs_error, rel_error

def hyperparameter_tuning(args=[[],[],[],[],[]]):
    '''
    tune the hyperparameter of neural network and return the optimal hyperparameters
    :param N_features: the number of input dimension
    :param X_train1: the training x
    :param Y_train1: the training y
    :param X_train2: the validation x
    :param Y_train2: the validation y
    :return: the optimal hyperparameters
    '''
    # print(args)
    N_features = args[0]
    X_train1 = args[1]
    Y_train1 = args[2]
    X_train2 = args[3]
    Y_train2 = args[4]
    # print(N_features, X_train1, Y_train1, X_train2, Y_train2)
    print('Tuning hyperparameters for neural network ...')
    print('Step 1: Tuning the number of layers and the learning rate ...')
    temp_config = dict()
    temp_config['num_input'] = N_features
    temp_config['num_neuron'] = 128
    temp_config['lambda'] = 'NA'
    temp_config['decay'] = 'NA'
    temp_config['verbose'] = 0

    abs_error_all = np.zeros((15, 4))
    abs_error_all_train = np.zeros((15, 4))
    abs_error_layer_lr = np.zeros((15, 2))
    abs_err_layer_lr_min = 100
    count = 0
    layer_range = range(2, 5)
    lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 4)
    for n_layer in layer_range:
        temp_config['num_layer'] = n_layer
        for lr_index, lr_initial in enumerate(lr_range):
            model = MLPPlainModel(temp_config)
            model.build_train()
            model.train(X_train1, Y_train1, lr_initial)

            Y_pred_train = model.predict(X_train1)
            abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
            abs_error_all_train[int(n_layer), lr_index] = abs_error_train

            Y_pred_val = model.predict(X_train2)
            abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
            abs_error_all[int(n_layer), lr_index] = abs_error
            # print(abs_error)

        # Pick the learning rate that has the smallest train cost
        # Save testing abs_error correspond to the chosen learning_rate
        temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
        temp_idx = np.where(abs(temp) < 0.0001)[0]
        if len(temp_idx) > 0:
            lr_best = lr_range[np.max(temp_idx)]
            err_val_best = abs_error_all[int(n_layer), np.max(temp_idx)]
        else:
            lr_best = lr_range[np.argmin(temp)]
            err_val_best = abs_error_all[int(n_layer), np.argmin(temp)]

        abs_error_layer_lr[int(n_layer), 0] = err_val_best
        abs_error_layer_lr[int(n_layer), 1] = lr_best

        if abs_err_layer_lr_min >= abs_error_all[int(n_layer), np.argmin(temp)]:
            abs_err_layer_lr_min = abs_error_all[int(n_layer),
                                                 np.argmin(temp)]
            count = 0
        else:
            count += 1

        if count >= 2:
            break
    abs_error_layer_lr = abs_error_layer_lr[abs_error_layer_lr[:, 1] != 0]

    # Get the optimal number of layers
    n_layer_opt = layer_range[np.argmin(abs_error_layer_lr[:, 0])] + 5

    # Find the optimal learning rate of the specific layer
    temp_config['num_layer'] = n_layer_opt
    for lr_index, lr_initial in enumerate(lr_range):
        model = MLPPlainModel(temp_config)
        model.build_train()
        model.train(X_train1, Y_train1, lr_initial)

        Y_pred_train = model.predict(X_train1)
        abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
        abs_error_all_train[int(n_layer), lr_index] = abs_error_train

        Y_pred_val = model.predict(X_train2)
        abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
        abs_error_all[int(n_layer), lr_index] = abs_error

    temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
    temp_idx = np.where(abs(temp) < 0.0001)[0]
    if len(temp_idx) > 0:
        lr_best = lr_range[np.max(temp_idx)]
    else:
        lr_best = lr_range[np.argmin(temp)]

    temp_lr_opt = lr_best
    print('The optimal number of layers: {}'.format(n_layer_opt))
    print('The optimal learning rate: {:.4f}'.format(temp_lr_opt))

    print('Step 2: Tuning the l1 regularized hyperparameter ...')
    # Use grid search to find the right value of lambda
    lambda_range = np.logspace(-2, np.log10(100), 20)
    error_min = np.zeros((1, len(lambda_range)))
    rel_error_min = np.zeros((1, len(lambda_range)))
    decay = 'NA'
    for idx, lambd in enumerate(lambda_range):
        val_abserror, val_relerror = nn_l1_val(X_train1, Y_train1,
                                               X_train2, Y_train2,
                                               n_layer_opt, lambd, temp_lr_opt)
        # print(val_abserror, val_relerror)
        error_min[0, idx] = val_abserror
        rel_error_min[0, idx] = val_relerror

    # Find the value of lambda that minimize error_min
    lambda_f = lambda_range[np.argmin(error_min)]
    print('The optimal l1 regularizer: {:.4f}'.format(lambda_f))

    return n_layer_opt, lambda_f, temp_lr_opt
