# Predicting Software Performance with Divide-and-Learn
> Predicting the performance of highly configurable software systems is the foundation for performance testing and quality assurance. To that end, recent work has been relying on machine/deep learning to model software performance. However, a **crucial yet unaddressed challenge** is how to cater for the **sparsity** inherited from the configuration landscape: the influence of configuration options (features) and the distribution of data samples are highly sparse.
> 
> In this paper, we propose an approach based on the concept of **'divide-and-learn'**, dubbed *DaL*. The basic idea is that, to handle sample sparsity, we divide the samples from the configuration landscape into **distant divisions**, for each of which we build a **regularized Deep Neural Network** as the local model to deal with the feature sparsity. A newly given configuration would then be assigned to the right model of division for the final prediction. 
> 
> Experiment results from eight real-world systems and five sets of training data reveal that, compared with the state-of-the-art approaches, *DaL* performs **no worse than the best counterpart on 33 out of 40 cases** (within which 26 cases are significantly better) with up to **1.94×** improvement on accuracy; requires fewer samples to reach the same/better accuracy; and producing acceptable training overhead. Practically, *DaL* also considerably improves different global models when using them as the underlying local models, which further strengthens its flexibility. 
> 
This repository contains the **key codes**, **full data used**, **raw experiment results** and **the supplementary tables** for the paper.

# Citation

>Jingzhi Gong and Tao Chen. Predicting Software Performance with Divide-and-Learn. *In Proceedings of the ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE)*, December 3–9, 2023, San Francisco, CA, USA., 13 pages.

# Documents

- **DaL_main.py**: 
the *main program* for using DaL, which automatically reads data from csv files, trains and evaluates, and save the results.

- **utils**

    └─ **general.py**:
    contains utility functions to build DNN and other ML models.
    
    └─ **hyperparameter_tuning.py**:
    contains the function that efficiently tunes hyperparameters of DNN.
    
    └─ **mlp_plain_model_tf2.py**:
    contains functions to construct and train plain DNN. 
    
    └─ **mlp_sparse_model_tf2.py**:
    contains functions to construct and build DNN with L1 regularization. 
    
- **Raw_results**:
contains the raw experiment results for all the research questions.

- **Data**:
performance datasets of 8 subject systems as specified in the paper.

- **Table4_full.pdf**:
supplementary document for Table4 in the paper.

# Prerequisites and Installation
1. Download all the files into the same folder/clone the repository.

2. Install the specified version of Python and Tensorflow:
the codes have been tested with **Python 3.6 - 3.9** and **Tensorflow 2.x**, other versions might cause errors.

3. Install all missing packages according to **requirements.txt** and runtime messages.


# Run *DaL*

- **Command line**: cd to the folder with the codes, input the command below, and the rest of the processes will be fully automated.

        python DaL_main.py
        
- **Python IDE (e.g. Pycharm)**: Open the *DaL_main.py* file on the IDE, and simply click 'Run'.


# Demo Experiment
The main program *DaL_main.py* defaultly runs a demo experiment that evaluates *DaL* with 5 sample sizes of *Lrzip*, 
each repeated 30 times, without hyperparameter tuning (to save demonstration time).

A **successful run** would produce similar messages as below: 

        Run 1
        N_train:  127
        N_test:  5057
        ---DNN_DaL depth 1---
        Dividing...
          106 samples with feature 5 <= 0.5:
          21 samples with feature 5 > 0.5:
        Training...
        Testing...
        Best shot rate: 4885/5057 = 0.9659877397666601
        > DNN_DaL MRE: 30.88
        DNN_DaL total time cost (minutes): 0.91

The results will be saved in a file in the same directory with a name in the format *'System_Nsamples_Nexperiments_Date'*, for example, *'Lrzip_127_01-30_05-05'*.

# Change Experiment Settings
To run more complicated experiments, alter the codes following the instructions below and comments in *DaL_main.py*.

#### To switch between subject systems
    Comment and Uncomment lines 34-41 following the comments in DaL_main.py.

    E.g., to run DaL with Apache, uncomment line 34 'subject_system = 'Apache_AllNumeric'' and comment out the other lines.


#### To save the experiment results
    Set 'save_file = True' at line 22.
    
    
#### To tune the hyperparameters (takes longer time)
    Set line 21 with 'test_mode = False'.


#### To change the number of experiments for specified sample size(s)
    Change 'N_experiments' at line 29, where each element corresponds to a sample size. 

    For example, to simply run the first sample size with 30 repeated runs, set 'N_experiments = [30, 0, 0, 0, 0]'.

#### To change the sample sizes of a particular system
    Edit lines 55-71.

    For example, to run Apache with sample sizes 10, 20, 30, 40 and 50: set line 56 with 'sample_sizes = [10, 20, 30, 40, 50]'.


#### To compare DaL with DeepPerf
    1. Set line 21 with 'test_mode = False'.

    2. Set line 25 with 'enable_deepperf = True'.


#### To compare DaL with other ML models (RF, DT, LR, SVR, KRR, kNN) and DaL framework with these models (DaL_RF, DaL_DT, DaL_LR, DaL_SVR, DaL_KRR, DaL_kNN)
    1. Set line 21 with 'test_mode = False'.

    2. Set line 24 with 'enable_baseline_models = True'.


#### To run DaL with different depth d
    Add the dedicated d into the list 'depths' at line 27.
    
    E.g, run DaL with d=2: set 'depths = [2]'.

    E.g, run DaL with d=3 and d=4, respectively: set 'depths = [3, 4]'.


# State-of-the-art Performance Prediction Models
Below are the repositories of the SOTA performance prediction models, which are evaluated and compared with *DaL* in the paper. 

- [DeepPerf](https://github.com/DeepPerf/DeepPerf)

    A deep neural network performance model with L1 regularization and efficient hyperparameter tuning.

- [DECART](https://github.com/jmguo/DECART)

    CART with data-efficient sampling method.

- [SPLConqueror](https://github.com/se-sic/SPLConqueror)

    Linear regression with optimal binary and numerical sampling method and stepwise feature selection.

- [Perf-AL](https://github.com/GANPerf/GANPerf)

    Novel GAN-based performance model with a generator to predict performance and a discriminator to distinguish the actual and predicted labels.
    


Note that *DaL_main.py* only compares *DeepPerf* because it is formulated in the most similar way to *DaL*, while the others are developed under different programming languages or have different ways of usage. 

Therefore, to compare *DaL* other SOTA models, please refer to their original pages (you might have to modify or reproduce their codes to ensure the compared models share the same set of training and testing samples).
