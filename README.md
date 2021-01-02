# Capstone Project - Las Vegas Tripadvisor Reviews

## Overview
This Capstone project is done as part of the Udacity Azure ML Nanodegree program. In this project, Tripadvisor dataset for 21 Las Vegas hotels between January and August of 2015 is accessed from the UCI Machine Learning Repository. A "Classification" type Machine Learning model was built and studied using an Azure ML pipeline. Two different methods were used for tuning the models - Hyperdrive Tuning and Azure AutoML run. The best model from each method were compared and the one with the highest accuracy was deployed and tested. 

## Summary
**The Las Vegas Tripadvisor dataset contained various metrics about the features available at the hotel, the type and location of customer, Tripadvisor membership duration, period of stay, and a Score between 1 and 5 from each of the reviewers. The objective of the project is to make a recommendation to a potential customer on whether or not a particular hotel should be chosen based on these attributes. The Classification model would return "Yes" if the Score is equal to and greater than 3 and "No" otherwise.**

**Both Hyperparameter Tuning with HyperDrive and Automated ML methods were evaluated. There was barely any difference in the accuracy values from the two models, both showed an accuracy around 0.9206. So one of the models, Automated ML method was chosen for deployment.**

## High-Level Pipeline
**The high-level pipeline is shown in the image below:**
Data was accessed from an external site, uploaded to Github and registered using "Registered Datasets" functionality available in Azure. Then the data is cleaned using train.py module and prepared separately as desired for the AutoML method and Hyperdrive method. The best models from each of the two methods are compared and the model with the overall highest accuracy is chosen for deployment and tested. Finally, the entire procedure is described in this README documentation.

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig1.jpg)

In this project, two different models were used to train and compared as shown in the image below:
![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig2.JPG)

### Data cleanup - train.py module:
The data was accessed from the "https://archive.ics.uci.edu/ml/datasets/Las+Vegas+Strip" link. Moro et al., 2017 (Moro, S., Rita, P., & Coelho, J. (2017). Stripping customers' feedback on hotels through data mining: The case of Las Vegas Strip. Tourism Management Perspectives, 23, 41-52.) performed data mining on the reviews available from the Trip Advisor site between January and August of 2015 and generated 504 rows of data. This dataset was uploaded to the following Github link:
https://raw.githubusercontent.com/Kbhamidipa3/udacityazure_capstone_final/main/LasVegasTripAdvisorReviews-Dataset.csv
The data was registered using "Registered Datasets" feature in Azure. Loading of data was done using the above github url link (copy the link from Raw) and as shown in the image below.

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig3.JPG)

Once the data is loaded, the train.py accesses the data and converts it into a pandas dataframe. Then all spaces and periods in the column names are replaced with underscores. 

The next very important step is to modify the data under label "Score" to represent binary classification. For this purpose, any scores with values 3 and above are assigned a value of '1', while rest are assigned a value of '0'. 

The next step is to fill the missing data as explained in the following sub-section.

#### Missing Data
As can be seen in the following image, there was data missing in some rows in the Nr. rooms, User. continent, Member years, Review month and Review weekday columns. 

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig4.JPG)

Each of this missing data is filled using different methods as can be noticed from the train.py code and explained below: 
1. Nr_rooms column was updated by performing a simple Google search of the specified hotel. 
2. User_continent fields were updated based on the reviewer's country details. 
3. Review_month is updated using the middle month from the "Period_of_stay" column. For example, if the traveler's "Period_of_stay" is listed as Jun-Aug, then the row with missing data under "Review_month" is set to July and so-on. 
4. Missing rows under "Review_weekday" were randomly selected to be "Sunday"
5. All the missing rows under these coulmns are updated using key-value pairs from Dictionaries. 
6. Missing rows under column "Member_years" were updated randomly using values between 1 and 10.



#### Data preparation for AutoML method:
The dataset available at this point suffices for running AutoML models. However, the data needs to be split into training, validation and testing data. For this purpose, the overall data is split into train and test data in 75%/25% ratio first and the available train data is eventually split into 75%/25% for training and validation. All three datasets are exported as csv files as shown in image below and uploaded to Github:

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig5.JPG)

AutoML train data: https://raw.githubusercontent.com/Kbhamidipa3/udacityazure_capstone_final/main/LV_github_automl_train.csv
AutoML validation data: https://raw.githubusercontent.com/Kbhamidipa3/udacityazure_capstone_final/main/LV_github_automl_validation.csv
AutoML test data: https://raw.githubusercontent.com/Kbhamidipa3/udacityazure_capstone_final/main/LV_github_automl_test.csv

#### Data preparation for Hyperdrive method:
There are more steps remaining to get the model available for the Hyperdrive method and these are done in the train.py model as well. 
1. Categorical data should be converted to numerical data before fitting a Hyperdrive model. 
2. One-hot encoding is one such method used where each category under a given parameter is split into it’s own column and assigned values ‘0’ or ‘1’ depending on whether an item doesn’t belong or belong under the given category respectively. Then all the individual columns are added back to the dataframe.
3. The final dataset generated is uploaded to the following Github link:
https://raw.githubusercontent.com/Kbhamidipa3/udacityazure_capstone_final/main/LV_github_hypertune.csv
4. The data is split into train and test values using a certain split such that more data is assigned under “train” data to get accurate model fits. In the following code, 75% of the data is used for training and remaining 33% is used for testing. Setting random_state (42 in the current model) to a specified value can ensure that the data will return same results after each execution. 
5. Logistic Regression model is used to fit the train data. And two parameters were chosen – Inverse of Regularization Strength (denoted by C) and Maximum Iterations (denoted by max_iter) as shown in the following image. These are tuned as explained in the next subsection.

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig6.JPG)

6. The accuracy of the model is then determined using the score method.
7. The trained model is then saved to the output folder using joblib.dump().



### Hyperparameter Tuning parameters
Hyperparameter tuning method is used to tune the two parameters defined in train.py to achieve the best prediction accuracy- Inverse of Regularization Strength (denoted by C) and Maximum Iterations (denoted by max_iter). The former parameter, “C”, helps to avoid overfitting. The parameter max_iter dictates the maximum number of iterations for the regression model so that the model doesn't run for too long resulting in diminishing returns. These parameters are randomly sampled using the RandomParameterSampling method to understand the impact of the parameters on the output prediction accuracy. Four sets of "C" values and five sets of "max_iter" values were chosen for this particular study as shown in the image below:

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig7.JPG)

Specifying an early stopping policy improves computational efficiency by terminating poorly performing runs. BanditPolicy was chosen as the early stopping policy in this project with slack factor and evaluation interval as the parameters. Every run is compared to the Best performing run at the end of the specified evaluation interval and in combination with the slack factor (allowed slack value compared to the best performing model) determines whether the run should be continued or terminated.**


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig8.JPG)

## AutoML
Unlike the Hypertuning method used earlier, Automated ML method automates the iterative tasks associated with the machine learning models thereby improving efficiency and productivity. Accuracy is used as the primary metric similar to the first method.

### AutoML parameters
As the final objective is to predict each potential hotel recommendation as "y" or "n", which is a binary classification problem, task is set as "Classification". The other parameter used is "experiment_timeout_hours", which is set to 0.3 per project specifications. This metric is important to ensure the model terminates within a reasonable time. Accuracy is used as the primary metric similar to the first method and the objective is to maximize the accuracy. "iteration_timeout_minutes" is set to 5 and "max_concurrent_iterations" is set to 4 meaning 4 iterations can be concurrently done. Column "Score" (the predicted column) is assigned to the label_name parameter. Parameters "enable_voting_ensemble" and "enable_stack_ensemble" have not been specified, so the default values are set to "True".**

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig9.JPG)

### Pipeline comparison
#### Pipeline and accuracy differences between Hyperparameter tuning and Automated ML:
##### Hyperparameter Tuning:

In the Hyperparameter tuning method, the tabular data is split into test/train data using the train.py model and Scikit-learn is used to perform Logistic Regression. This is subsequently called in the Hyperparameter tuning code and the parameters are randomly sampled. 


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig10.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig11.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig11b.jpg)

The parameters seemed to however have little impact on the final accuracy as all the runs performed using different combinations of "C" and "max_iter" yielded an accuracy between 0.9126 and 0.9206.

The RunDetails widget images are shown below:


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig11c.jpg)


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig11d.jpg)


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig11e.jpg)


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig11f.jpg)

All the relevant plots and other details are shown below:

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig12.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig13.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig14.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig15.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig16.jpg)


The best model identified by the hypertuning method (hp_trained_model.pkl) is uploaded to the main Github folder.


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig17.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig18.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig18a.jpg)

#### Automated ML:
On the other hand, in the case of Automated ML, cleaned data obtained from the train.py module was split into train, validation and test data and accessed from Github links. The Automated ML method evaluated around 40 different runs and chose "StandardScaleWrapper, XGBoostClassifier" as the best performing model with an accuracy of 0.9206. "Voting Ensemble" model showed similar accuracy. So, similar to the Hyperdrive models, Automated ML models also showed little difference in accuracy.


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig19.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig20.jpg)

The best model identified by the Automated ML method (model.pkl) is uploaded to the main Github folder.


### Best Automated ML Method:
As shown below, "StandardScaleWrapper, XGBoostClassifier" method is chosen as the best Automated ML method as it had the highest accuracy (0.92857) of all the models. VotingEnsemble method implements soft voting on an ensemble of previous Auto ML runs and "Stacking" is the ensemble learning method used. Soft vote uses average of predicted probabilities.

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig21.jpg)

The RunDetails widget images are shown below:

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig21a.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig21b.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig21c.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig21d.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig21e.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig21f.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig21g.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig21h.jpg)


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig22.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig23.jpg)

The image below shows "Precision - Recall" plot. It shows how closely the model tracks the ideal behavior. The closer the curve to the ideal line, the better is the model.

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig24.jpg)

Another way to interpret the accuracy of the model is using "Calibration Curve". The datapoints are not tracking as well as expected.

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig25.jpg)

The following confusion matrix shows that the model has the most "True Positives" predictions (115), which confirms that the trained model is performing well. However, the higher "False Positives" and fewer "True Negatives" is concerning. This indicates Classification imbalance in the original classification dataset, which is also the reason for not so great looking "Calibration Curve".  This could be improved by chosing data with more balanced positives and negatives (future recommendation).


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig27.jpg)

The overall summary is provided in the following image.

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig26.jpg)





## Tabulated Summary of the Models:

The two methods, Hyperdrive and Automated ML showed same accuracy of 0.9206. 


**In summary comparing the two models, the accuracies are as follows:**

Hyperparameter Tuning Accuracy | Automated ML Accuracy
------------ | -------------
0.9206|0.9206

## Deploy and Test the Model:

The next step is to deploy the best model for consumption by the end user. For this, the best model is selected from the AutoML run and is deployed using Python code. As shown in the image below, the deployment was successful as indicated by the "Healthy" status.

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig28.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig29.jpg)

Key-based autherntication and Application Insights are enabled. Swagger json file, autherntication keys and REST Endpoints are created.

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig30.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig31.jpg)


Then the deployed model is successfully tested as shown below and a confusion matrix is generated based on the test data:

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig32.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig33.jpg)

## Screencast Link:

The Screencast can be found at the following link:

https://www.youtube.com/watch?v=gn9YNdmaMZo

## Files Uploaded:

All files are uploaded to the "Zipped_Folder.rar" file, which is uploaded to Github.

## Future work:
**In the future, the following improvements can be made to the models to potentially improve accuracy:**
* Even though the accuracy of the model was high, Precision-Recall curves showed poor "Macro" behavior. This is caused because of classification-imbalance, which is due to the quality of the classification dataset. In the future, there is a possibility to chose a more balanced dataset.
* Explore other sampling methods for Hyperdrive tuning
* Sample other parameters that haven't been tested in this project
* Evaluate other classification models

## Proof of cluster clean up
The aci service created for deployment and the compute cluster originally created were both deleted towards the end of the project.


![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig36.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig37.jpg)

Similarly the compute cluster created for Hyperdrive is deleted as well.

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig38.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig39.jpg)

## Standout Suggestions
As part of Standout suggestion, implemented converting the model into ONNX format as shown in the image below:

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_capstone_final/blob/main/images/Fig34.jpg)

