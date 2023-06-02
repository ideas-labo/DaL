import os
import numpy as np
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

def get_non_zero_indexes(whole_data, total_tasks):
    (N, n) = whole_data.shape
    n = n - 1
    delete_index = set()
    temp_index = list(range(N))
    for i in range(total_tasks):
        temp_Y = whole_data[:, n - i]
        for j in range(len(temp_Y)):
            if temp_Y[j] == 0:
                delete_index.add(j)
    non_zero_indexes = np.setdiff1d(temp_index, list(delete_index))
    return non_zero_indexes


def process_training_data(whole_data, training_index, N_features, n, main_task):
    temp_X = whole_data[training_index, 0:N_features]
    # scale x
    temp_max_X = np.amax(temp_X, axis=0)
    if 0 in temp_max_X:
        temp_max_X[temp_max_X == 0] = 1
    temp_X = np.divide(temp_X, temp_max_X)
    X_train = np.array(temp_X)

    # Split train data into 2 parts (67-33)
    N_cross = int(np.ceil(len(temp_X) * 2 / 3))
    X_train1 = (temp_X[0:N_cross, :])
    X_train2 = (temp_X[N_cross:len(temp_X), :])

    ### process y
    temp_Y = whole_data[training_index, n - main_task][:, np.newaxis]
    # scale y
    temp_max_Y = np.max(temp_Y) / 100
    if temp_max_Y == 0:
        temp_max_Y = 1
    temp_Y = np.divide(temp_Y, temp_max_Y)
    Y_train = np.array(temp_Y)

    # Split train data into 2 parts (67-33)
    Y_train1 = (temp_Y[0:N_cross, :])
    Y_train2 = (temp_Y[N_cross:len(temp_Y), :])

    return temp_max_X, X_train, X_train1, X_train2, temp_max_Y, Y_train, Y_train1, Y_train2


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def build_model(regression_mod='RF', test_mode=True, training_X=[], training_Y=[]):
    """
    to build the specified regression model, given the training data
    :param regression_mod: the regression model to build
    :param test_mode: won't tune the hyper-parameters if test_mode == False
    :param training_X: the array of training features
    :param training_Y: the array of training label
    :return: the trained model
    """
    model = None
    if regression_mod == 'RF':
        model = RandomForestRegressor(random_state=0)
        max = 3
        if len(training_X)>30: # enlarge the hyperparameter range if samples are enough
            max = 6
        param = {'n_estimators': np.arange(10, 100, 20),
                 'criterion': ('mse', 'mae'),
                 'max_features': ('auto', 'sqrt', 'log2'),
                 'min_samples_split': np.arange(2, max, 1)
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = RandomForestRegressor(**gridS.best_params_, random_state=0)

    elif regression_mod == 'KNN':
        min = 2
        max = 3
        if len(training_X)>30:
            max = 16
            min = 5
        model = KNeighborsRegressor(n_neighbors=min)
        param = {'n_neighbors': np.arange(2, max, 2),
                 'weights': ('uniform', 'distance'),
                 'algorithm': ['auto'],  # 'ball_tree','kd_tree'),
                 'leaf_size': [10, 30, 50, 70, 90],
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = KNeighborsRegressor(**gridS.best_params_)

    elif regression_mod == 'SVR':
        model = SVR()
        param = {'kernel': ('linear', 'rbf'),
                 'degree': [2, 3, 4, 5],
                 'gamma': ('scale', 'auto'),
                 'coef0': [0, 2, 4, 6, 8, 10],
                 'epsilon': [0.01, 1]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = SVR(**gridS.best_params_)

    elif regression_mod == 'DT':
        model = DecisionTreeRegressor(random_state=0)
        max = 3
        if len(training_X)>30:
            max = 6
        param = {'criterion': ('mse', 'friedman_mse', 'mae'),
                 'splitter': ('best', 'random'),
                 'min_samples_split': np.arange(2, max, 1)
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = DecisionTreeRegressor(**gridS.best_params_, random_state=0)

    elif regression_mod == 'LR':
        model = LinearRegression()
        param = {'fit_intercept': ('True', 'False'),
                 # 'normalize': ('True', 'False'),
                 'n_jobs': [1, -1]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = LinearRegression(**gridS.best_params_)

    elif regression_mod == 'KR':
        x1 = np.arange(0.1, 5, 0.5)
        model = KernelRidge()
        param = {'alpha': x1,
                 'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                 'coef0': [1, 2, 3, 4, 5]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = KernelRidge(**gridS.best_params_)

    return model


def init_dir(dir_name):
    """Creates directory if it does not exists"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
