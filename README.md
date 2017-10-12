# KTH_bigdata_project
Twitter sentiment analysis

## Naive Bayes
Simply run bayes_classifiers.py

## SVM
Run svm_testing.py 

## LSTM
dir : KTH_bigdata_project/sentiment_analysis_tensorflow-master/

for train, run: python train.py

for predict, run: python predict.py --checkpoints_dir <checkpoints directory>  
  for example: `python predict.py --checkpoints_dir checkpoints/1481294288`

for more information: see the readme file inside the project

## CNN
1. Run Train_CNN.py for training process 

2. Checkpoints will be saved in runs folder

3. Run Test_CNN_v2_final.py for test process 

Hints:
You should change the directory of input data in the test and train process based on your own data directory  
text_cnn.py contain the model for CNN. data_helpers.py and data_input_v3.py doing some process related to input data of CNN

