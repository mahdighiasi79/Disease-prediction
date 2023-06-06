import math
import numpy as np

outlier_threshold = 0.0001


def ExtractValues(feature):
    values = {}
    for value in feature:
        if values.get(value) is None:
            values[value] = 1
        else:
            values[value] += 1
    return values


def HasMissingValue(feature):
    for element in feature:
        if element == '?':
            return True
    return False


def MeanVariance(feature):
    records = len(feature)
    mean = np.sum(feature, axis=0, keepdims=False) / records
    variance = np.power(feature - mean, 2)
    variance = np.sum(variance, axis=0, keepdims=False) / (records - 1)
    return [mean, variance]


def NormalDistribution(x, mean, variance):
    coefficient = 1 / pow(2 * math.pi * variance, 0.5)
    exponent = (pow(x - mean, 2) / variance) / -2
    result = coefficient * math.exp(exponent)
    return result


def Entropy(feature):
    records = len(feature)
    values = {}

    for value in feature:
        if values.get(value) is None:
            values[value] = 1
        else:
            values[value] += 1

    entropy = 0
    for key in values.keys():
        probability = values[key] / records
        entropy += probability * math.log2(probability)
    entropy *= -1
    return entropy


def MutualInformation(feature1, feature2):
    feature1_np = np.array(feature1)
    feature2_np = np.array(feature2)
    records = len(feature1)
    outcomes1 = np.array([])
    outcomes2 = np.array([])

    for value in feature1:
        if value not in outcomes1:
            outcomes1 = np.append(outcomes1, value)

    for value in feature2:
        if value not in outcomes2:
            outcomes2 = np.append(outcomes2, value)

    n1 = len(outcomes1)
    n2 = len(outcomes2)
    probability_matrix = np.zeros((n1, n2))

    for i in range(n1):
        f1 = (feature1_np == outcomes1[i])
        for j in range(n2):
            f2 = (feature2_np == outcomes2[j])
            f3 = f1 * f2
            count = np.sum(f3, axis=0)
            if count == 0:
                count = records
            probability_matrix[i][j] = count
    probability_matrix /= records

    joint_entropy = probability_matrix * np.log2(probability_matrix)
    joint_entropy = np.sum(joint_entropy, axis=0)
    joint_entropy = np.sum(joint_entropy, axis=0)
    joint_entropy *= -1

    mutual_information = Entropy(feature1) + Entropy(feature2) - joint_entropy
    return mutual_information


def DetectOutliers(feature):
    records = len(feature)

    categories = {}
    for category in feature:
        if categories.get(category) is None:
            categories[category] = 1
        else:
            categories[category] += 1

    noise_categories = []
    for key in categories.keys():
        if categories[key] < (records * outlier_threshold):
            noise_categories.append(key)

    result = []
    for i in range(records):
        if feature[i] in noise_categories:
            result.append(i)
    return result
