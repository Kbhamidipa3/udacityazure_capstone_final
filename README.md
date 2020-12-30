# Capstone Project - Las Vegas TripAdvisor Reviews

## Overview
This Capstone project is done as part of the Udacity Azure ML Nanodegree program. In this project, Tripadvisor dataset for 21 Las Vegas hotels between January and August of 2015 is accessed from the UCI Machine Learning Repository. A "Classification" type Machine Learning model was built and studied using an Azure ML pipeline. Two different methods were used for tuning the models - Hyperdrive Tuning and Azure AutoML run. The best model from each method were compared and the one with the highest accuracy was deployed and tested. 

## Summary
**The Las Vegas TripAdvisor dataset contained various metrics about the features available at the hotel, the type and location of customer, TripAdvisor membership duration, period of stay, and a Score between 1 and 5 from each of the reviewers. The objective of the project is to make a recommendation to a potential customer on whether or not the hotel should be chosen. The Classification model would return "Yes" if the Score is equal to and greater than 3 and "No" otherwise.**

**Both Hyperparameter Tuning with HyperDrive and Automated ML methods were evaluated. While the difference was relatively smaller, the Automated ML method showed the best performance between the two with a “Best Accuracy” of 0.91859.**

## Scikit-learn Pipeline
**The high-level Scikit-learn Pipeline used to setup the pipeline is shown in the image below:**

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig1.jpg)

In this project, two different models were used to train and compared as shown in the image below:
![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig2.jpg)

### Data cleanup - train.py module:
The data was accessed from the "https://archive.ics.uci.edu/ml/datasets/Las+Vegas+Strip" link. Moro et al., 2017 (Moro, S., Rita, P., & Coelho, J. (2017). Stripping customers' feedback on hotels through data mining: The case of Las Vegas Strip. Tourism Management Perspectives, 23, 41-52.) performed data mining on the reviews available on the Trip Advisor site between January and August of 2015 and generated 504 rows of data. 














There are several steps involved in setting up and running the trained model, which is done in train.py for the HyperDrive model.
* Tabular data is first accessed from the following link and dataset is created:
https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

* Then the data is converted to dataframe using Pandas and missing values are dropped.

* Subsequently the data is cleaned to get it ready for training the model. This involves translating categorical data into numerical data before fitting a machine learning model. 
* One hot encoding is one such method used where each category under a given parameter is split into it’s own column and assigned values ‘0’ or ‘1’ depending on whether an item doesn’t belong or belong under the given category respectively. Then all the individual columns are added back to the dataframe.

* Another method used to generate numerical data from categorical values is using key-value pairs from Dictionaries.


* Other columns that have binary values like “married” or “not married” are assigned values 1 and 0 respectively.

* A separate column is created for label values that we want to predict by popping it out from the dataframe.


* Then the data is split into train and test values using a certain split such that more data is assigned under “train” data to get accurate model fits. In the following code, 67% of the data is used for training and remaining 33% is used for testing. Setting random_state (42 in the current model) to a specified value can ensure that the data will return same results after each execution. 

* Logistic Regression model is used to fit the train data. And two parameters were chosen – Inverse of Regularization Strength (denoted by C) and Maximum Iterations (denoted by max_iter). These are tuned as explained in the next subsection.

 
* The accuracy of the model is then determined using the score method.

* The trained model is then saved to the output folder using joblib.dump().



 ### Hyperparameter Tuning parameters
**Hyperparameter tuning method is used to tune the two parameters defined in train.py to achieve the best prediction accuracy- Inverse of Regularization Strength (denoted by C) and Maximum Iterations (denoted by max_iter). The former parameter, “C”, helps to avoid overfitting. The parameter max_iter dictates the maximum number of iterations for the regression model so that the model doesn't run for too long resulting in diminishing returns. These parameters are randomly sampled using the RandomParameterSampling method to understand the impact of the parameters on the output prediction accuracy.**

**Specifying an early stopping policy improves computational efficiency by terminating poorly performing runs. BanditPolicy was chosen as the early stopping policy in this project with slack factor and evaluation interval as the parameters. Every run is compared to the Best performing run at the end of the specified evaluation interval and in combination with the slack factor (allowed slack value compared to the best performing model) determines whether the run should be continued or terminated.**

## AutoML
**Unlike the Hypertuning method used earlier, Automated ML method automates the iterative tasks associated with the machine learning models thereby improving efficiency and productivity. Accuracy is used as the primary metric similar to the first method.**

### AutoML parameters
**For train/test splitting, no specific test size is entered for the AutoML case, so a default of 25% test size will be used. As the final objective is to predict each potential customer as "y" or "n", which is a classification problem, task is set as "Classification". The other parameter used is "experiment_timeout_minutes", which is set to 30 minutes per project specifications. This metric is important to ensure the model terminates within a reasonable time. Accuracy is used as the primary metric similar to the first method and the objective is to maximize the accuracy. Number of cross-validations is set to 5, meaning each training uses 4/5th of the data, while the remaining 1/5th of the data is used for validation. The final metric reported out is the average of the five individual metrics. The cleaned x_train and y_train data from the train.py file is concatenated to make a single trained dataset, uploaded to the cloud and fed to the training_data parameter. Column "y" (the predicted column) is assigned to the label_name parameter. Parameters "enable_voting_ensemble" and "enable_stack_ensemble" have not been specified, so the default values are set to "True".**


### Pipeline comparison
#### Pipeline and accuracy differences between Hyperparameter tuning and Automated ML:
##### Hyperparameter Tuning:
**As specified in Figure 1, in the Hyperparameter tuning method, the tabular data is split into test/train data using the train.py model and Scikit-learn is used to perform Logistic Regression. This is subsequently called in the Hyperparameter tuning code and the parameters are randomly sampled. The parameters seemed to however have no impact on the final accuracy as all the runs performed using different combinations of parameters yielded the same accuracy of 0.9105265. The best model identified by the hypertuning method (hp_trained_model.pkl) is uploaded to the main Github folder.**


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%201.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%202.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%203.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%204.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%205.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%206.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%207.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%208.jpg)

#### Automated ML:
**On the other hand, in the case of Automated ML, cleaned data obtained from the train.py module was split into train and test data within the Automated ML code itself and the data was uploaded to a datastore. The Automated ML method evaluated 45 different runs and chose "VotingEnsemble" as the best performing model with an accuracy of 0.91753. While the difference in accuracy is relatively smaller, Automated ML method definitely showed a higher accuracy than the Hyperparameter tuning method. This could be attributed to the superiority of the Automated ML method in sweeping through a more optimum space to find the best fit. The best model identified by the Automated ML method (automl_best_model.pkl) is uploaded to the main Github folder.**

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%201.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%202.jpg)

### Best Automated ML Method:
**As shown below, "VotingEnsemble" method is chosen as the best Automated ML method as it had the highest accuracy (0.91753) of all the models. VotingEnsemble method implements soft voting on an ensemble of previous Auto ML runs and "Stacking" is the ensemble learning method used. Soft vote uses average of predicted probabilities.**


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%203.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%204.jpg)


**The algorithms used in the ensemble operation and the corresponding weights are listed below.**


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%205.jpg)


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%206.JPG)




**The image below shows "Precision - Recall" plot. It shows how closely the model tracks the ideal behavior. The closer the curve to the ideal line, the better is the model.** 


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%2010.JPG)


**Another way to interpret the accuracy of the model is using "Calibration Curve". The datapoints are tracking the ideal line well.** 


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%2011.JPG)


**The following confusion matrix shows that the model has the most "True Positives" predictions (21138), which confirms that the trained model is performing well. "True Negatives" are also higher than "False Negatives".**


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%2012.JPG)



## Tabulated Summary of the Models:

**In summary comparing the two models, the accuracies are as follows:**

Hyperparameter Tuning Accuracy | Automated ML Accuracy
------------ | -------------
0.91052|0.91753



## Future work
**In the future, the following improvements can be made to the models to potentially improve accuracy:**
* Try different train/test split ratios
* Explore other sampling methods
* Sample other parameters that haven't been tested in this project

## Proof of cluster clean up
**Cluster cleanup is included in the code.**
