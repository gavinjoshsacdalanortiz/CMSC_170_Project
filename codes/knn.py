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

all_data = []
training_set = []
test_set = []


def minkowski(A, B, p):
    return np.power(np.sum(np.abs(np.array(A) - np.array(B)) ** p), 1/p)
    #credit: https://www.geeksforgeeks.org/maths/minskowski-distance/

def knn(k, p, test_features, train):
    distances = []
    zero = 0
    one = 0

    for train_features, train_label in train:
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


base_dir = os.path.dirname(os.path.abspath('_file_'))
data_path = os.path.join(base_dir, "..", "dataset", "data_banknote_authentication.txt")

# the banknote file has no header row, 4 features, and the label in column 5
with open(data_path, "r") as i:
    data = csv.reader(i)

    for j in data:
        if len(j) == 0:      # skip blank lines
            continue
        features = [float(val) for val in j[0:4]]
        label = j[4]

        all_data.append((features, label))


# the file is sorted by class (all 0s first, then all 1s), so shuffle before splitting
random.seed(1)
random.shuffle(all_data)

split = int(len(all_data) * 0.70)

for i in range(split):
    training_set.append(all_data[i])

for i in range(split, len(all_data)):
    test_set.append(all_data[i])


k = int(input("Enter k: "))
p = int(input("Enter p: "))

final_test = []
predictions = []

print("K-Nearest Neighbors for Banknote Authentication Data")
print(f"Dataset size: {len(all_data)}")
print(f"Train size: {len(training_set)}")
print(f"Test size: {len(test_set)}")
print(f"k = {k}, p = {p}")

for test_item in test_set:

    test_set_features = test_item[0]
    actual_label = test_item[1]

    prediction = knn(k, p, test_set_features, training_set)
    predictions.append(prediction)

    final_test.append((test_set_features, actual_label, prediction))

headers = ["Test Index", "Variance", "Skewness", "Curtosis", "Entropy", "Actual", "Predicted"]
print("\t".join(f"{h:<10}" for h in headers))
print("-" * 120)

index = 1
for features, actual_label, prediction in final_test:
    cells = [str(index)] + [f"{val:.5f}" for val in features] + [actual_label, prediction]
    print("\t".join(f"{c:<10}" for c in cells))
    index = index + 1

correct = 0
for i in range(len(test_set)):
    if test_set[i][1] == predictions[i]:
        correct = correct + 1

percent = (correct / len(test_set)) * 100

print(f"Accuracy: {percent:.2f}%")