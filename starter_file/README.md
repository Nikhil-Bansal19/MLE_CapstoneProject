# Diabetes Predictions Using Microsoft Azure

This project is part of the Udacity Azure ML Nanodegree. The dataset used in this project is originally from NIDDK. The objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. We will use Azure to configure a cloud-based machine learning production model and deploy it. We use Hyper Drive and Auto ML methods to develop the model. Then the model with higest accuary is retrieved(voting ensemble in this case) and deployed in cloud with Azure Container Instances(ACI) as a webservice, also by enabling the authentication. Once the model is deployed, the behaviour of the endpoint is analysed by getting a response from the service and logs are retrived at the end.

# Dataset

## Overview

This Dataset is available publicy in Kaggle

The datasets consists of several medical predictor variables and one target variable (Outcome). Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and more.
<img width="1727" alt="Screenshot 2023-07-29 at 8 22 38 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/c3cfebf8-5e80-4aac-ad7d-1f355705a348">


##Variables Description
Pregnancies Number of times pregnant
Glucose Plasma glucose concentration in an oral glucose tolerance test
BloodPressure Diastolic blood pressure (mm Hg)
SkinThickness Triceps skinfold thickness (mm)
Insulin Two hour serum insulin
BMI Body Mass Index
DiabetesPedigreeFunction Diabetes pedigree function
Age Age in years
Outcome Class variable (either 0 or 1). 268 of 768 values are 1, and the others are 0

## Task

The objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset

## Access

In order to access the dataset, I used 2 different methods for 2 different models.

## HyperDrive

For the moodel,trained with HyperDrive functionalities,the dataset is saved in one of the public respositories (my git repo) and loaded with the help of TabularDataset.

## AutoML

For the model trained with AutoML functionalities, the dataset is registered with the help of "from local files" option and loaded from Azure workspace.
<img width="1224" alt="Screenshot 2023-07-29 at 6 48 58 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/818907bc-e093-4cf0-b6b7-cca51be06a39">



# Automated ML

AutoML also referred as Automated machine learning or automated ML, is the process of automating the time consuming, iterative tasks of machine learning model development. Automated ML is applied when you want Azure Machine Learning to train and tune a model for you using the target metric you specify.

Initially the dataset(health care diabetes.csv) is registered with the help of "from local files" option and loaded from Azure workspace. It is then converted to pandas dataframe,before that the Workspace,experiment and the cluster are created.

AutoMLConfig is then defined for successfully executing the AutoML run with automl_settings and compute target , the automl_settings is defined with experiment timeout as ‘30’ minutes,task type as ‘Classification’(Classification is a type of supervised learning in which models learn using training data, and apply those learnings to new data. Classification models is to predict which categories new data will fall into based on learnings from its training data) primary metric as ‘accuracy’, label column as ‘Outcome’ 
,n cross validations as ‘5’,training_dataset to “registered dataset”,max concurrent iterations as “4” and featurization as “auto”.

The models used here are LightGBM,SVM, XGBoostClassifier, RandomForest, VotingEnsemble, StackEnsemble etc and It also uses different pre-processing techniques like Standard Scaling, Min Max Scaling, Sparse Normalizer, MaxAbsScaler,MinAbsScaler etc.

Finally the experiment is being submitted and using the Notebook widget(RunDetails(remote_run).show()), it is visualized. Once the run is successfully completed or executed, the best model is retrieved and saved in output folder in source directory.

<img width="635" alt="Screenshot 2023-07-29 at 6 57 48 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/3d5f759a-a45d-4e8b-b7dc-be0d4ec425a8">
<img width="1200" alt="Screenshot 2023-07-29 at 6 58 14 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/3e43ad59-9981-4bbf-a550-4847b83181d7">
<img width="841" alt="Screenshot 2023-07-29 at 6 58 26 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/27df570f-5774-41d1-a6df-ea172993371a">
<img width="1077" alt="Screenshot 2023-07-29 at 7 21 04 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/81376e84-9155-4364-81d7-cb1642697fd4">

<img width="1219" alt="Screenshot 2023-07-29 at 7 22 51 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/9c417f1a-a1f8-4699-a230-beae7682fb0c">


<img width="890" alt="Screenshot 2023-07-29 at 7 24 59 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/51c4c16b-04fa-4494-8cb1-cdfb87f742e5">
<img width="1204" alt="Screenshot 2023-07-29 at 7 23 05 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/c997849c-d40b-42f1-b919-b9c897f012b1">
<img width="1194" alt="Screenshot 2023-07-29 at 7 23 20 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/06ae22ce-1bd2-43fd-a396-cdb49e6f6d8c">

<img width="838" alt="Screenshot 2023-07-29 at 7 28 41 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/3335a66e-c8bc-490a-a9ce-bdf03bc85574">


## Results
The first model we built is with the help of Azure AutoML for training many types of models such as LightGBM, XGBoostClassifier, RandomForest, VotingEnsemble, StackEnsemble etc.The best accuracy obtained from this model is 78.39%

<img width="1193" alt="Screenshot 2023-07-29 at 7 23 42 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/44b2d13d-72e5-44a9-b6f4-338649b6153b">
<img width="1128" alt="Screenshot 2023-07-29 at 7 23 55 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/1f229c4e-af08-4944-988a-e33faf63d745">


# Hyperparameter Tuning

Initially in the training script (train.py),the dataset (health care diabetes.csv) is retrieved from the URL (https://raw.githubusercontent.com/Nikhil-Bansal19/MLE_CapstoneProject/master/health%20care%20diabetes.csv) provided using TabularDatasetFactory Class (Contains methods to create a tabular dataset for Azure Machine Learning).Then the data is being split as train and test with the ratio of 70:30.

The classification algorithm used here is Logistic Regression.Logistic regression is a well-known method in statistics that is used to predict the probability of an outcome, and is especially popular for classification tasks. The algorithm predicts the probability of occurrence of an event by fitting data to a logistic function. Then the training(train.py) script is passed to estimator and HyperDrive configurations to predict the best model and accuracy.The HyperDrive run is executed successfully with the help of parameter sampler, policy, estimator and HyperDrive Config, before that Azure workspace,experiment and cluster is created successfully.

The policy is defined.Bandit policy is used here with slack_factor=0.1,evaluation_internal=1, delay_evaluation=5.Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.Then,the Hyperparameters are then tuned (Hyperparameters are adjustable parameters that let you control the model training process and Hyperparameter tuning is the process of finding the configuration of hyperparameters that results in the best performance) with the help of parameter sampler.Random Sampling is used here.It is used on ‘--C’ ( Inverse of regularization parameter ) which is a control variable that retains strength modification of Regularization by being inversely positioned to the Lambda regulator and ‘--max_iter’ (Maximum number of iterations to converge) which defines the number of times we want the learning to happen and this helps in solving high complex problems with large training hours as per the instructions.For ‘--C’, the parameter is set as (0.1,1) and for ‘—max_iter’ the parameter is set as (50,100,150,200) which returns a value uniformly distributed between low and high.

The ScriptRunConfig is created with the training script(train.py), directory path and cluster name, and created an environment using conda_dependencies.yml file and then the configurations for runs is created using hyperdrive_run_config with using run_config(created using ScriptRunConfig),hyperparameter sampling(Random Sampling), policy(Bandit policy) and also included primary metric(Accuracy),it’s goal (Maximize) , maximum total runs(10) and maximum concurrent runs(4).

Finally the experiment is being submitted and using the Notebook widget(RunDetails(hyperdrive_run).show()), it is visualized. Once the run is successfully completed or executed, the best model is retrieved and saved in output folder in source directory.The saved model then registered.

<img width="880" alt="Screenshot 2023-07-29 at 9 17 25 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/52628fa4-7dce-49e0-b232-665c0689c130">
<img width="1190" alt="Screenshot 2023-07-29 at 9 20 30 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/aa48b24e-cee1-4f5b-9d15-e0e4d5d63879">
<img width="777" alt="Screenshot 2023-07-29 at 9 21 36 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/cdafdb58-ce29-4e0c-adbd-2afac3c3f9ff">
<img width="836" alt="Screenshot 2023-07-29 at 9 21 44 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/602c4df2-8e8f-4365-97c0-7c0003de26eb">
<img width="1205" alt="Screenshot 2023-07-29 at 9 31 19 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/37e8135f-ec18-4bb4-a5bd-ecb261840659">


## Results
The second model we built is trained with Logistic regression and HyperDrive parameters which are tuned with the help of Azure ML python SDK and HyperDrive tools(Azure HyperDrive functionalities).The best accuracy obtained from this model is 75.19%
<img width="782" alt="Screenshot 2023-07-29 at 9 33 02 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/a4d43b42-d621-4047-b6e9-0ff641d6a6ad">

<img width="1210" alt="Screenshot 2023-07-29 at 9 31 48 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/e34d887f-c320-4d1a-b6ee-e25b73414158">
<img width="1154" alt="Screenshot 2023-07-29 at 9 31 56 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/263d7432-dcc7-4322-9ade-95034734cb1b">

<img width="833" alt="Screenshot 2023-07-29 at 9 33 41 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/fd13f25a-ea47-4b84-9006-1dd99e3e4b49">
<img width="887" alt="Screenshot 2023-07-29 at 9 33 55 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/da88a73c-07b8-4698-96bc-6b73c4da971c">

The model can be futher improved with the help of tuning other parameters such as the criterion used to define the optimal split and the min/max samples leaf number of samples in the leaf node of each tree.

# Model Deployment
HyperDrive’s best run accuracy = 75.19%

AutoML’s best run accuracy = 78.39%

<img width="1224" alt="Screenshot 2023-07-29 at 9 34 05 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/14108814-6528-4719-a977-e5b2b39337f9">

Thus,Automl's model has the highest accuracy.The model with the best accuracy is deployped as per the instructions,so the AutoML's best model is deployed.

Initially, the best model is registered and it's necessary files are downloaded.Then the Environment and inference is created with the help of required conda dependencies and score.py script file which has the intialization and exit function defined for the best model and the model is deployed with ACI(Azure Container Instance) and configurations such as cpu_cores=1, memory_gb=1.Once the deployment is sucessful, applications insights is enabled and the state of the service is verified.Then the behaviour of the endpoint is analyzed and the service is deleted

<img width="869" alt="Screenshot 2023-07-29 at 8 47 19 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/725c54fb-4dfc-4ef5-8093-8b1a3d42661e">
<img width="886" alt="Screenshot 2023-07-29 at 8 47 38 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/228c4c4c-dae5-4fc7-afe7-81f10841e11e">
<img width="1147" alt="Screenshot 2023-07-29 at 8 50 09 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/a221932b-70a6-4ab7-ad58-383f40ed5985">
<img width="616" alt="Screenshot 2023-07-29 at 8 50 23 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/4cda52fc-da61-42fd-9b31-f1588b53c2ad">
<img width="800" alt="Screenshot 2023-07-29 at 8 50 36 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/6d087c97-e738-471c-93c1-21d0b782094e">
<img width="808" alt="Screenshot 2023-07-29 at 8 53 58 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/1fb9602f-bd43-4c2c-8886-10dab0c6d519">
<img width="1232" alt="Screenshot 2023-07-29 at 8 57 44 PM" src="https://github.com/Nikhil-Bansal19/MLE_CapstoneProject/assets/47290347/0c89a4de-6bb9-4593-9180-3198cfcd33d7">


