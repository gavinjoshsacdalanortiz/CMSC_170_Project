#Name: John Michael V. Pabiz
#Student Number: 2023-07564
#Section: W-3L
#Program Description: This program simulates a k nearest neighbor algorithm.
#AI Declaration: Generative AI was used to debug and clean up codes given by the user. Generative AI was also used to provide suggestions for code structure and functions to use.
#               The final code was reviewed and modified by the author to ensure accuracy and functionality.

import csv
import os
import random
import numpy as np #install numpy

TRAIN_RATIO = 0.80
SEED = 1


def minkowski(A, B, p):
    return np.power(np.sum(np.abs(np.array(A) - np.array(B)) ** p), 1/p)
    #credit: https://www.geeksforgeeks.org/maths/minskowski-distance/


def load_data(path):
    # each row becomes (features, label, row_id); row_id is the line number in the file
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if len(row) == 0:      # skip blank lines
                continue
            features = [float(val) for val in row[0:4]]
            label = row[4]
            data.append((features, label, line_no))
    return data


def stratified_split(data, train_ratio, seed):
    # the file is sorted by class, so we shuffle each class separately
    # and take the same percentage from each one (keeps the class ratio in both sets)
    random.seed(seed)
    by_class = {}
    for item in data:
        by_class.setdefault(item[1], []).append(item)

    train, test = [], []
    for label in sorted(by_class):
        items = by_class[label]
        random.shuffle(items)
        cut = round(len(items) * train_ratio)
        train.extend(items[:cut])
        test.extend(items[cut:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test


def fit_min_max(train):
    # min and max of each feature, computed from the TRAINING set only
    cols = np.array([item[0] for item in train])
    return cols.min(axis=0), cols.max(axis=0)


def scale(data, mins, maxs):
    # min-max normalization: (x - min) / (max - min), result is between 0 and 1
    ranges = np.where(maxs - mins == 0, 1, maxs - mins)
    scaled = []
    for features, label, row_id in data:
        new_features = ((np.array(features) - mins) / ranges).tolist()
        scaled.append((new_features, label, row_id))
    return scaled


def knn(k, p, test_features, train):
    distances = []
    zero = 0
    one = 0

    for train_features, train_label, _ in train:
        distance = minkowski(test_features, train_features, p)
        distances.append((distance, train_label))

    distances.sort()

    closest_k = distances[:k]

    for dist, label in closest_k:
        if label == '0':
            zero += 1
        else:
            one += 1

    if zero > one:
        return '0'
    else:
        return '1'


def evaluate(actual, predicted):
    # class '1' is treated as the positive class
    tp = tn = fp = fn = 0
    for a, pr in zip(actual, predicted):
        if a == '1' and pr == '1':
            tp += 1
        elif a == '0' and pr == '0':
            tn += 1
        elif a == '0' and pr == '1':
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return tp, tn, fp, fn, accuracy, precision, recall, f1


base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "dataset", "data_banknote_authentication.txt")

all_data = load_data(data_path)
training_set, test_set = stratified_split(all_data, TRAIN_RATIO, SEED)

mins, maxs = fit_min_max(training_set)
training_set = scale(training_set, mins, maxs)
test_set = scale(test_set, mins, maxs)

k = int(input("Enter k: "))
p = int(input("Enter p: "))

print("K-Nearest Neighbors for Banknote Authentication Data")

print(f"Dataset size: {len(all_data)}")
print(f"Train size: {len(training_set)}")
print(f"Test size: {len(test_set)}")
print(f"Split: stratified {int(TRAIN_RATIO * 100)}/{int((1 - TRAIN_RATIO) * 100)}, seed = {SEED}")
print("Preprocessing: min-max normalization (fitted on training set)")
print(f"k = {k}, p = {p}")

actual = []
predictions = []
final_test = []

for features, label, row_id in test_set:
    prediction = knn(k, p, features, training_set)
    actual.append(label)
    predictions.append(prediction)
    final_test.append((row_id, features, label, prediction))

# Every cell is padded to width 10, then a tab follows, so all columns line up.
headers = ["Row No.", "Variance", "Skewness", "Curtosis", "Entropy", "Actual", "Predicted"]
print("\t".join(f"{h:<10}" for h in headers))
print("-" * 120)

for row_id, features, label, prediction in final_test:
    cells = [str(row_id)] + [f"{val:.5f}" for val in features] + [label, prediction]
    print("\t".join(f"{c:<10}" for c in cells))

tp, tn, fp, fn, accuracy, precision, recall, f1 = evaluate(actual, predictions)

print()
print("Confusion Matrix")
print("\t".join(f"{h:<14}" for h in ["", "Predicted 0", "Predicted 1"]))
print("\t".join(f"{c:<14}" for c in ["Actual 0", str(tn), str(fp)]))
print("\t".join(f"{c:<14}" for c in ["Actual 1", str(fn), str(tp)]))
print()
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")