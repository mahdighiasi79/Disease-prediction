import helper_functions as hf
import pandas as pd
import numpy as np
import math

df = pd.read_csv("Dataset3.csv")
probabilities = {}
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def CalculateProbabilities(train_df):
    labels = np.array(train_df["disease"])
    class0 = (labels == 0)
    class1 = (labels == 1)
    num_class0 = np.sum(class0, axis=0, keepdims=False)
    num_class1 = np.sum(class1, axis=0, keepdims=False)
    p_class0 = num_class0 / train_size
    p_class1 = num_class1 / train_size
    probabilities["disease"] = [p_class0, p_class1]

    for column in df.columns:
        feature = np.array(train_df[column])

        if column in categorical_features:
            values = hf.ExtractValues(feature)
            probability = {}

            for value in values.keys():
                matches = (feature == value)
                p0 = matches * class0
                p1 = matches * class1
                p0 = np.sum(p0, axis=0, keepdims=False) / num_class0
                p1 = np.sum(p1, axis=0, keepdims=False) / num_class1
                probability[value] = [p0, p1]

            probabilities[column] = probability

        elif column in numerical_features:
            matches0 = []
            matches1 = []
            for i in range(train_size):
                if class0[i]:
                    matches0.append(feature[i])
                if class1[i]:
                    matches1.append(feature[i])
            matches0 = np.array(matches0)
            matches1 = np.array(matches1)
            probabilities[column] = [hf.MeanVariance(matches0), hf.MeanVariance(matches1)]


def Predict(record):
    p0 = 1
    p1 = 1

    for column in df.columns:
        value = record[column]

        if column in numerical_features:
            mean0 = probabilities[column][0][0]
            mean1 = probabilities[column][1][0]
            variance0 = probabilities[column][0][1]
            variance1 = probabilities[column][1][1]
            p0 *= hf.NormalDistribution(value, mean0, variance0)
            p1 *= hf.NormalDistribution(value, mean1, variance1)

        elif column in categorical_features:
            p0 *= probabilities[column][value][0]
            p1 *= probabilities[column][value][1]

    p0 *= probabilities["disease"][0]
    p1 *= probabilities["disease"][1]

    if p0 > p1:
        return 0
    else:
        return 1


if __name__ == "__main__":
    num_records = len(df)
    test_size = math.floor(0.2 * len(df))
    train_size = num_records - test_size
    test_ids = np.arange(math.floor(train_size / 2), math.floor(train_size / 2) + test_size)
    train_set = df.drop(test_ids, axis=0)
    CalculateProbabilities(train_set)

    true_predictions = 0
    for record_id in test_ids:
        row = df.iloc(0)[record_id]
        label = row["disease"]
        predicted_label = Predict(row)
        if label == predicted_label:
            true_predictions += 1
    accuracy = (true_predictions / test_size) * 100
    print("accuracy: ", accuracy, "%")
