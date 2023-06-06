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

    for column in df.columns:
        feature = np.array(train_df[column])

        if column in categorical_features:
            values = hf.ExtractValues(feature)
            probability = {}

            for value in values.keys():
                matches = (feature == value)
                num_matches = np.sum(matches, axis=0, keepdims=False)
                class0 = (labels == 0)
                class1 = (labels == 1)
                p0 = matches * class0
                p1 = matches * class1
                p0 = np.sum(p0, axis=0, keepdims=False) / num_matches
                p1 = np.sum(p1, axis=0, keepdims=False) / num_matches
                probability[value] = [p0, p1]

            probabilities[column] = probability

        elif column in numerical_features:
            class0 = (labels == 0)
            class1 = (labels == 1)
            matches0 = feature * class0
            matches1 = feature * class1
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
