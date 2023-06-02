import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import genfromtxt
from imblearn.over_sampling import SMOTE
from utils.mlp_sparse_model_tf2 import MLPSparseModel
from utils.general import build_model
from utils.hyperparameter_tuning import nn_l1_val, hyperparameter_tuning
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # set the parameters
    test_mode = False  ### to tune the hyper-pearameters, set to False
    save_file = False  ### to save the results, set to True
    enable_DaL = True ### to enable the training of DaL, set to True
    enable_baseline_models = False  ### to compare DaL with other ML models (RQ2), set to True
    enable_deepperf = False  ### to compare DaL with Deepperf (RQ1), set to True
    seed = 3  # the base random seed, for replicating the results
    depths = [1]  ### to run experiments comparing different depths (RQ3), add depths here, starting from 1
    total_experiments = 30  ### the total number of repeated runs, for controlling the start point of runs
    N_experiments = [0, 0, 0, 0, 30]  ### to control the number of experiment, each element corresponds to a sample size
    total_tasks = 1  # number of performance metrics in the csv file, starting from 1
    task_index = 1  # the index of the performance metric to learn, starting from 1

    ### ---change the lines below to switch subject system--- ###
    subject_system = 'Apache_AllNumeric'
    # subject_system = 'BDBC_AllNumeric'
    # subject_system = 'BDBJ_AllNumeric'
    # subject_system = 'x264_AllNumeric'
    # subject_system = 'hsmgp_AllNumeric'
    # subject_system = 'hipacc_AllNumeric'
    # subject_system = 'VP8'
    # subject_system = 'Lrzip'
    ### ---change the lines above to switch subject system--- ###

    dir_data = 'Data/{}.csv'.format(subject_system)  ### change this line to locate the dataset files
    print('Dataset: ' + subject_system)

    whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)  # read the csv file into array
    (N, n) = whole_data.shape  # get the number of rows and columns
    print('Total sample size: ', N)
    N_features = n - total_tasks  # calculate the number of features
    print('N_features: ', N_features)

    ### ---change the lines below to specify the training sample sizes--- ###
    sample_sizes = []
    if subject_system == 'Apache_AllNumeric':
        sample_sizes = [9, 18, 27, 36, 45]
    elif subject_system == 'BDBC_AllNumeric':
        sample_sizes = [18, 36, 54, 72, 90]
    elif subject_system == 'BDBJ_AllNumeric':
        sample_sizes = [26, 52, 78, 104, 130]
    elif subject_system == 'x264_AllNumeric':
        sample_sizes = [16, 32, 48, 64, 80]
    elif subject_system == 'hsmgp_AllNumeric':
        sample_sizes = [77,173,384,480,864]
    elif subject_system == 'hipacc_AllNumeric':
        sample_sizes = [261, 528, 736, 1281,2631]
    elif subject_system == 'VP8':
        sample_sizes = [121, 273, 356, 467, 830]
    elif subject_system == 'Lrzip':
        sample_sizes = [127, 295, 386, 485, 907]
    else:
        sample_sizes = np.multiply(N_features, [1, 2, 3, 4, 5]) # in the default case, the sizes are [n, 2n, ... 5n], n = N_features
    ### ---change the lines above to specify the training sample sizes--- ###

    # run experiemnts for each sample size
    for i_train, N_train in enumerate(sample_sizes):
        # specify the number of testing samples
        N_test = 0
        if (N > N_train > 2):
            N_test = N - N_train  ### the default testing size is all the samples except the training ones
        else:
            N_test = 1

        # min_samples = int(0.15 * N_train)  # minimum samples required in each division
        min_samples = 2

        # specify the file name to save
        saving_file_name = '{}_{}_{}-{}_{}.txt'.format(dir_data.split('/')[1].split('.')[0], N_train,
                                                       total_experiments - N_experiments[i_train], total_experiments,
                                                       time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time())))
        # save the results when needed
        if save_file and N_experiments[i_train] != 0:
            with open(saving_file_name, 'w') as f:
                f.write('N_train={} N_test={} min_samples={}'.format(N_train, N_test,
                                                                     min_samples))  # the first line is the sizes

        # for the specified number of repeated runs
        for ne in range(total_experiments - N_experiments[i_train], total_experiments):
            print('\nRun {}: '.format(ne + 1))
            print('N_train: ', N_train)
            print('N_test: ', N_test)
            print('min_division_samples: ', min_samples)

            if save_file:
                with open(saving_file_name, 'a') as f:  # write the number of the run
                    f.write('\nRun {}'.format(ne + 1))

            # set the seed of randomness in order to replicate the results
            random.seed(ne * seed)  # the seed in each run is different

            # delete the zero-performance samples
            non_zero_indexes = []
            delete_index = set()
            temp_index = list(range(N))
            for i in range(total_tasks):
                temp_Y = whole_data[:, n - i - 1]
                for j in range(len(temp_Y)):
                    if temp_Y[j] == 0:
                        delete_index.add(j)
                # temp_Y = np.delete(temp_Y,[delete_index],axis=0)
            non_zero_indexes = np.setdiff1d(temp_index, list(delete_index))  # get all the remaining samples
            # total_index = non_zero_indexes  # set the total training indexes

            # randomly generate the testing samples
            testing_index = random.sample(list(non_zero_indexes), N_test)
            non_zero_indexes = np.setdiff1d(non_zero_indexes, testing_index)

            # randomly generate the training samples
            non_zero_indexes = random.sample(list(non_zero_indexes), N_train)
            # print('training sample indexes: {}'.format(non_zero_indexes))
            # print('testing samples: {}'.format(testing_index))

            # compute the weights of each feature using Mutual Information, for eliminating insignificant features
            weights = []
            feature_weights = mutual_info_regression(whole_data[non_zero_indexes, 0:N_features],
                                                     whole_data[non_zero_indexes, n - task_index], random_state=0)
            # print('Computing weights of {} samples'.format(len(non_zero_indexes)))
            for i in range(N_features):
                weight = feature_weights[i]
                # print('Feature {} weight: {}'.format(i, weight))
                weights.append(weight)

            # for each depths (to analyze the sensitivity of DaL to depth (RQ3))
            for max_depth in depths:
                print('\n---DNN_DaL depth {}---'.format(max_depth))
                # initialize variables
                start = time.time()  # to measure the training time
                max_X = []
                max_Y = []
                config = []
                lr_opt = []
                models = []
                clusters = []
                X_train = []
                Y_train = []
                X_train1 = []
                Y_train1 = []
                X_train2 = []
                Y_train2 = []
                X_test = []
                Y_test = []
                cluster_indexes_all = []

                # generate clustering labels based on the dividing conditions of DT
                print('Dividing...')
                # get the training X and Y for clustering
                Y = whole_data[non_zero_indexes, n - task_index][:, np.newaxis]
                X = whole_data[non_zero_indexes, 0:N_features]

                # build and train a CART to extract the dividing conditions
                DT = build_model('DT', test_mode, X, Y)
                DT.fit(X, Y)
                tree_ = DT.tree_  # get the tree structure

                # the function to extract the dividing conditions recursively,
                # and divide the training data into clusters (divisions)
                from sklearn.tree import _tree


                def recurse(node, depth, samples=[]):
                    indent = "  " * depth
                    if depth <= max_depth:
                        if tree_.feature[node] != _tree.TREE_UNDEFINED:  # if it's not the leaf node
                            left_samples = []
                            right_samples = []
                            # get the node and the dividing threshold
                            name = tree_.feature[node]
                            threshold = tree_.threshold[node]
                            # split the samples according to the threshold
                            for i_sample in range(0, len(samples)):
                                if X[i_sample, name] <= threshold:
                                    left_samples.append(samples[i_sample])
                                else:
                                    right_samples.append(samples[i_sample])
                            # check if the minimum number of samples is statisfied
                            if (len(left_samples) <= min_samples or len(right_samples) <= min_samples):
                                print('{}Not enough samples to cluster with {} and {} samples'.format(indent,
                                                                                                      len(left_samples),
                                                                                                      len(right_samples)))
                                cluster_indexes_all.append(samples)
                            else:
                                print("{}{} samples with feature {} <= {}:".format(indent, len(left_samples), name,
                                                                                   threshold))
                                recurse(tree_.children_left[node], depth + 1, left_samples)
                                print("{}{} samples with feature {} > {}:".format(indent, len(right_samples), name,
                                                                                  threshold))
                                recurse(tree_.children_right[node], depth + 1, right_samples)
                    # the base case: add the samples to the cluster
                    elif depth == max_depth + 1:
                        cluster_indexes_all.append(samples)


                # run the defined recursive function above
                recurse(0, 1, non_zero_indexes)

                k = len(cluster_indexes_all)  # the number of divided subsets
                if save_file:
                    with open(saving_file_name, 'a') as f:  # write the number of the samples in each cluster
                        f.write('\ndepth{} N_samples of {} divisions:'.format(max_depth, k))
                        for i in range(k):
                            f.write(' {}'.format(len(cluster_indexes_all[i])))

                # if there is only one cluster, DaL can not be used
                if k <= 1:
                    print(
                        'Error: samples are less than the minimum number (min_samples={}), please add more samples'.format(
                            min_samples))
                    continue  # end this run

                # extract training samples from each cluster
                N_trains = []  # N_train for each cluster
                cluster_indexes = []
                for i in range(k):
                    if int(N_train) > len(cluster_indexes_all[i]):  # if N_train is too big
                        N_trains.append(int(len(cluster_indexes_all[i])))
                    else:
                        N_trains.append(int(N_train))
                    # sample N_train samples from the cluster
                    cluster_indexes.append(random.sample(cluster_indexes_all[i], N_trains[i]))

                # generate the samples and labels for classification
                total_index = cluster_indexes[0]  # samples in the first cluster
                clusters = np.zeros(int(len(cluster_indexes[0])))  # labels for the first cluster
                for i in range(k):
                    if i > 0:  # the samples and labels for each cluster
                        total_index = total_index + cluster_indexes[i]
                        clusters = np.hstack((clusters, np.ones(int(len(cluster_indexes[i]))) * i))

                # get max_X and max_Y for scaling
                max_X = np.amax(whole_data[total_index, 0:N_features], axis=0)  # scale X to 0-1
                if 0 in max_X:
                    max_X[max_X == 0] = 1
                max_Y = np.max(whole_data[total_index, n - task_index]) / 100  # scale Y to 0-100
                if max_Y == 0:
                    max_Y = 1

                # get the training data for each cluster
                for i in range(k):  # for each cluster
                    temp_X = whole_data[cluster_indexes[i], 0:N_features]
                    temp_Y = whole_data[cluster_indexes[i], n - task_index][:, np.newaxis]
                    # Scale X and Y
                    X_train.append(np.divide(temp_X, max_X))
                    Y_train.append(np.divide(temp_Y, max_Y))
                X_train = np.array(X_train)
                Y_train = np.array(Y_train)

                # get the testing data
                X_test = whole_data[testing_index, 0:N_features]
                X_test = np.divide(X_test, max_X)  # scale X
                Y_test = whole_data[testing_index, n - task_index][:, np.newaxis]

                # split train data into 2 parts for hyperparameter tuning
                for i in range(0, k):
                    N_cross = int(np.ceil(X_train[i].shape[0] * 2 / 3))
                    X_train1.append(X_train[i][0:N_cross, :])
                    Y_train1.append(Y_train[i][0:N_cross, :])
                    X_train2.append(X_train[i][N_cross:N_trains[i], :])
                    Y_train2.append(Y_train[i][N_cross:N_trains[i], :])

                # process the sample to train a classification model
                X_smo = whole_data[total_index, 0:N_features]
                y_smo = clusters
                for j in range(N_features):
                    X_smo[:, j] = X_smo[:, j] * weights[j]  # assign the weight for each feature

                # SMOTE is an oversampling algorithm when the sample size is too small
                enough_data = True
                for i in range(0, k):
                    if len(X_train[i]) < 5:
                        enough_data = False
                if enough_data:
                    smo = SMOTE(random_state=1, k_neighbors=3)
                    X_smo, y_smo = smo.fit_resample(X_smo, y_smo)

                # build a random forest classifier to classify testing samples
                forest = RandomForestClassifier(random_state=0)
                # tune the hyperparameters if not in test mode
                if not test_mode:
                    param = {"min_samples_split": [2, 10, 20],
                             "max_depth": [None, 2, 5, 10],
                             "min_samples_leaf": [1, 5, 10],
                             "max_leaf_nodes": [None, 5, 10, 20],
                             }
                    print('Hyperparameter Tuning...')
                    gridS = GridSearchCV(forest, param)
                    gridS.fit(X_smo, y_smo)
                    forest = RandomForestClassifier(**gridS.best_params_, random_state=0)
                forest.fit(X_smo, y_smo)  # training

                # classify the testing samples
                testing_clusters = []  # classification labels for the testing samples
                X = whole_data[testing_index, 0:N_features]
                for j in range(N_features):
                    X[:, j] = X[:, j] * weights[j]  # assign the weight for each feature
                for temp_X in X:
                    temp_cluster = forest.predict(temp_X.reshape(1, -1))  # predict the dedicated local DNN using RF
                    testing_clusters.append(int(temp_cluster))
                # print('Testing size: ', len(testing_clusters))
                # print('Testing sample clusters: {}'.format((testing_clusters)))

                if enable_DaL:
                    ### Train DNN_DaL
                    print('Training DaL...')
                    if test_mode == True:  # default hyperparameters, just for testing
                        for i in range(0, k):
                            # define the configuration for constructing the NN
                            temp_lr_opt = 0.123
                            n_layer_opt = 3
                            lambda_f = 0.123
                            temp_config = dict()
                            temp_config['num_neuron'] = 128
                            temp_config['num_input'] = N_features
                            temp_config['num_layer'] = n_layer_opt
                            temp_config['lambda'] = lambda_f
                            temp_config['verbose'] = 0
                            config.append(temp_config)
                            lr_opt.append(temp_lr_opt)

                    ## tune DNN for each cluster (division) with multi-thread
                    elif test_mode == False:  # only tune the hyperparameters when not test_mode
                        from concurrent.futures import ThreadPoolExecutor

                        # create a multi-thread pool
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            args = []  # prepare arguments for hyperparameter tuning
                            for i in range(k):  # for each division
                                args.append([N_features, X_train1[i], Y_train1[i], X_train2[i], Y_train2[i]])
                            # optimal_params contains the results from the function 'hyperparameter_tuning'
                            for optimal_params in pool.map(hyperparameter_tuning, args):
                                print('Learning division {}... ({} samples)'.format(i + 1, len(X_train[i])))
                                n_layer_opt, lambda_f, temp_lr_opt = optimal_params  # unzip the optimal parameters
                                # define the configuration for constructing the DNN
                                temp_config = dict()
                                temp_config['num_neuron'] = 128
                                temp_config['num_input'] = N_features
                                temp_config['num_layer'] = n_layer_opt
                                temp_config['lambda'] = lambda_f
                                temp_config['verbose'] = 0
                                config.append(temp_config)
                                lr_opt.append(temp_lr_opt)

                    for i in range(k):
                        # train a local DNN model using the optimal hyperparameters
                        model = MLPSparseModel(config[i])
                        model.build_train()
                        model.train(X_train[i], Y_train[i], lr_opt[i])
                        models.append(model)  # save the trained models for each cluster

                    # compute the MRE (MAPE) using the testing samples
                    print('Testing...')
                    rel_errors = []
                    # rel_error3 = []
                    count = 0  # count the correctly classified clusters
                    for i in range(len(testing_clusters)):  # for each testing sample
                        best_cluster = 0
                        # print('Sample {} performance {} configuration {}:'.format(testing_index[i], Y_test[i], X_test[i]))
                        temp_RE = []
                        # test the sample using every model
                        for d in range(k):
                            temp_RE.append(np.abs(
                                np.divide(max_Y * models[d].predict(X_test[i][np.newaxis, :]) - Y_test[i],
                                          Y_test[i]))[0][0] * 100)
                            # print('cluster {} RE: {}'.format(d, temp_RE[-1]))
                        best_cluster = list(np.where(temp_RE == np.min(temp_RE))[0])  # get the most accurate cluster
                        rel_errors.append(temp_RE[testing_clusters[i]])  # save the RE of each sample
                        # count how many samples are classified to the best clusters
                        for temp in best_cluster:
                            if temp == testing_clusters[i]:
                                count += 1
                    rel_errors = np.mean(rel_errors)  # average the REs to get the MRE
                    print('Best division rate: {}/{} = {:.2f}'.format(count, len(testing_clusters),
                                                                  count / len(testing_clusters)))
                    print('> DNN_DaL MRE: {}'.format(round(rel_errors, 2)))

                    # End measuring time
                    end = time.time()
                    time_dal = ((end - start) / 60)
                    print('DNN_DaL total time cost (minutes): {:.2f}'.format(time_dal))

                    # save the MRE and time
                    if save_file:
                        with open(saving_file_name, 'a') as f:  # save the results
                            # f.write('\nRun {}'.format(ne+1))
                            f.write('\ndepth{} DNN_DaL RE: {}'.format(max_depth, np.mean(rel_errors)))
                            f.write('\ndepth{} DNN_DaL_time (minutes): {}'.format(max_depth, time_dal))
                            f.write('\nBest shot rate: {:.2f}'.format(count / len(testing_clusters)))

                ### Train different ML models with DaL framework
                if enable_baseline_models:
                    for regression_mod in ['RF', 'KNN', 'SVR', 'DT', 'LR', 'KR']:
                        # check if there's enough data
                        not_enough_data = False
                        for i in range(0, k):
                            if len(X_train[i]) < 5:  # at least 5 samples are needed for tuning the local models
                                print(
                                    '\n---At least 5 samples are required for tuning {}-DaL---'.format(regression_mod))
                                not_enough_data = True

                        # initialize variables
                        start = time.time()
                        models = []
                        random.seed(ne * seed)
                        print('\n---{}-DaL depth {}---'.format(regression_mod, max_depth))

                        # train a local model for each cluster
                        for i in range(0, k):
                            print('Training division {}... ({} samples)'.format(i + 1, len(X_train[i])))

                            # build regression models
                            if not_enough_data:
                                model = build_model(regression_mod, True, X_train[i], Y_train[i][:, 0])
                            else:
                                model = build_model(regression_mod, test_mode, X_train[i], Y_train[i][:, 0])
                            model.fit(X_train[i], Y_train[i][:, 0])
                            models.append(model)

                        rel_errors = []
                        count = 0
                        for i in range(len(testing_clusters)):  # for each testing sample
                            best_cluster = 0

                            temp_RE = []
                            # test the sample using every model
                            for d in range(k):
                                temp_RE.append(np.abs(
                                    np.divide(max_Y * models[d].predict(X_test[i][np.newaxis, :]) - Y_test[i],
                                              Y_test[i]))[0] * 100)
                                # print('cluster {} RE: {}'.format(d, temp_RE[-1]))
                            best_cluster = np.argmin(temp_RE)  # the best cluster
                            rel_errors.append(temp_RE[testing_clusters[i]])  # the predicted cluster
                            # count how many clusters are predicted correctly
                            if best_cluster == testing_clusters[i]:
                                count += 1

                        rel_errors = np.mean(rel_errors)  # average the REs to get the MRE
                        print('> {}_DaL MRE: {}'.format(regression_mod, round(rel_errors, 2)))

                        print('Best shot rate: {}/{} = {}'.format(count, len(testing_clusters),
                                                                  count / len(testing_clusters)))

                        # End measuring time
                        end = time.time()
                        temp_time = (end - start) / 60
                        print('{}_DaL Time cost (minutes): {:.2f}'.format(regression_mod, temp_time))
                        # save the results
                        if save_file:
                            with open(saving_file_name, 'a') as f:  # save the results
                                f.write(
                                    '\ndepth{} {}_DaL RE: {}'.format(max_depth, regression_mod, np.mean(rel_errors)))
                                f.write(
                                    '\ndepth{} {}_DaL_time (minutes): {}'.format(max_depth, regression_mod, temp_time))

            ### training DeepPerf
            if enable_deepperf:
                print('\n---Training DeepPerf---')
                # initialize time
                start = time.time()

                # get max X and Y for scaling
                max_X = np.amax(whole_data[total_index, 0:N_features], axis=0)
                if 0 in max_X:
                    max_X[max_X == 0] = 1
                max_Y = np.max(whole_data[total_index, n - task_index]) / 100
                if max_Y == 0:
                    max_Y = 1

                # get the training and testing data
                X_train_deepperf = whole_data[total_index, 0:N_features]
                Y_train_deepperf = whole_data[total_index, n - task_index][:, np.newaxis]
                X_test_deepperf = whole_data[testing_index, 0:N_features]
                Y_test_deepperf = whole_data[testing_index, n - task_index][:, np.newaxis]
                print('Deepperf training size: ', len(X_train_deepperf))
                print('Deepperf testing size: ', len(X_test_deepperf))

                # Scale X and Y
                X_train_deepperf = np.divide(X_train_deepperf, max_X)
                Y_train_deepperf = np.divide(Y_train_deepperf, max_Y)
                X_test_deepperf = np.divide(X_test_deepperf, max_X)

                # Split train data into 2 parts (67-33)
                N_cross = int(np.ceil(len(X_train_deepperf) * 2 / 3))
                X_train1 = X_train_deepperf[0:N_cross, :]
                Y_train1 = Y_train_deepperf[0:N_cross]
                X_train2 = X_train_deepperf[N_cross:len(X_train_deepperf), :]
                Y_train2 = Y_train_deepperf[N_cross:len(X_train_deepperf)]

                # default hyperparameters, just for testing
                if test_mode == True:
                    lr_opt = 0.123
                    n_layer_opt = 3
                    lambda_f = 0.123
                    config = dict()
                    config['num_neuron'] = 128
                    config['num_input'] = N_features
                    config['num_layer'] = n_layer_opt
                    config['lambda'] = lambda_f
                    config['verbose'] = 0
                # if not test_mode, tune the hyperparameters
                else:
                    n_layer_opt, lambda_f, lr_opt = hyperparameter_tuning(
                        [N_features, X_train1, Y_train1, X_train2, Y_train2])

                    # save the hyperparameters
                    config = dict()
                    config['num_neuron'] = 128
                    config['num_input'] = N_features
                    config['num_layer'] = n_layer_opt
                    config['lambda'] = lambda_f
                    config['verbose'] = 0

                # train the DeepPerf model
                deepperf_model = MLPSparseModel(config)
                deepperf_model.build_train()
                deepperf_model.train(X_train_deepperf, Y_train_deepperf, lr_opt)

                # compute MRE
                rel_error = []
                print('Testing...')
                Y_pred_test = deepperf_model.predict(X_test_deepperf)
                Y1_pred_test = max_Y * Y_pred_test[:, 0:1]
                rel_error = np.mean(
                    np.abs(np.divide(Y_test_deepperf.ravel() - Y1_pred_test.ravel(), Y_test_deepperf.ravel()))) * 100
                print('> Deepperf MRE: {}'.format(rel_error))

                # End measuring time
                end = time.time()
                time_deepperf = ((end - start) / 60)
                print('Deepperf Time cost (minutes): {:.2f}'.format(time_deepperf))

                # save the results
                if save_file:
                    with open(saving_file_name, 'a') as f:  # save the results
                        f.write('\nDeepperf RE: {}'.format(rel_error))
                        f.write('\nDeepperf_time (minutes): {}'.format(time_deepperf))

            ### train baseline models if specified
            if enable_baseline_models:
                for regression_mod in ['RF', 'KNN', 'SVR', 'DT', 'LR', 'KR']:
                    # initialization
                    models = []
                    random.seed(ne * seed)
                    print('\n---Training {}---'.format(regression_mod))
                    start = time.time()
                    # generate training and testing samples
                    X_train_deepperf = whole_data[total_index, 0:N_features]
                    Y_train_deepperf = whole_data[total_index, n - task_index][:, np.newaxis]
                    X_test_deepperf = whole_data[testing_index, 0:N_features]
                    Y_test_deepperf = whole_data[testing_index, n - task_index][:, np.newaxis]
                    print('{} training size: {}'.format(regression_mod, len(X_train_deepperf)))
                    print('{} testing size: {}'.format(regression_mod, len(X_test_deepperf)))

                    # there must be at least 4 samples to train the baseline models
                    if len(X_train_deepperf) < 5:
                        print('\n---Not enough samples for {}---\n'.format(regression_mod))
                        continue

                    # Scale X and Y
                    X_train_deepperf = np.divide(X_train_deepperf, max_X)
                    Y_train_deepperf = np.divide(Y_train_deepperf, max_Y)
                    X_test_deepperf = np.divide(X_test_deepperf, max_X)

                    # Split train data into 2 parts (67-33)
                    N_cross = int(np.ceil(len(X_train_deepperf) * 2 / 3))
                    X_train1 = X_train_deepperf[0:N_cross, :]
                    Y_train1 = Y_train_deepperf[0:N_cross]
                    X_train2 = X_train_deepperf[N_cross:len(X_train_deepperf), :]
                    Y_train2 = Y_train_deepperf[N_cross:len(X_train_deepperf)]

                    # train baseline model
                    baseline_model = build_model(regression_mod, test_mode, X_train_deepperf, Y_train_deepperf[:, 0])
                    baseline_model.fit(X_train_deepperf, Y_train_deepperf[:, 0])

                    # compute MRE
                    rel_error = []
                    print('Testing...')
                    Y_pred_test = baseline_model.predict(X_test_deepperf)
                    # print(Y_pred_test)
                    Y1_pred_test = max_Y * Y_pred_test
                    rel_error = np.mean(
                        np.abs(
                            np.divide(Y_test_deepperf.ravel() - Y1_pred_test.ravel(), Y_test_deepperf.ravel()))) * 100
                    print('> {} MRE: {}'.format(regression_mod, rel_error))

                    # End measuring time
                    end = time.time()
                    training_time = ((end - start) / 60)
                    print('{} Time cost (minutes): {:.2f}'.format(regression_mod, training_time))

                    # save the results
                    if save_file:
                        with open(saving_file_name, 'a') as f:  # save the results
                            f.write('\n{} RE: {}'.format(regression_mod, rel_error))
                            f.write('\n{}_time (minutes): {}'.format(regression_mod, training_time))