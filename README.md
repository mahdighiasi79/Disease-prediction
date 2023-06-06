# Disease-prediction
In this project, we aim to predict whether a person has some disease or not.
The prediction is based on features like age, sex, fbs, etc.
Number 1 in the "disease" column indicates that the corresponding person has a disease. And number 0 indicates the opposite.
So, we have got two classes of data.

There is no preprocessing as the data are already clear. 
The model we use for this classification task is Naive Bayes.
For categorical features, we calculate the probabilities of occurring each class for each possible value.
For numerical features, we assume that the values are distributed normally. Therefore, we calculate the needed probabilities by the normal distribution formula.

For evaluation, we split the data into two separate sets: train set, and test set.
We use the train set to calculate the needed probabilities for Naive Bayes. And we use the test set to report the predictions' accuracy.
Based on testing, the proposed model gives an accuracy of about  87.5%.
