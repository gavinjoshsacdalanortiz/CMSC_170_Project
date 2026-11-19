import csv
import os
import random
import numpy as np #install numpy

TRAIN_RATIO = 0.80
SEED = 1
MAX_EPOCHS = 1000


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

learning_rate = float(input("Enter learning rate: "))
bias = float(input("Enter bias: "))

#weights: w0 = bias, all other weights start at 0
n = len(training_set[0][0]) + 1
w = [bias] + [0.0] * (n - 1)

#training: repeat over the training set until every row is classified correctly
epoch = 0
converged = False

while not converged and epoch < MAX_EPOCHS:
    epoch = epoch + 1
    converged = True

    for features, label, row_id in training_set:
        x = [1.0] + features            #x0 = 1
        if label == '1':                #labels '0' and '1' become -1 and +1
            d = 1
        else:
            d = -1

        #linear combiner output and predicted output
        v = 0
        for i in range(n):
            v = v + w[i] * x[i]

        if v >= 0:
            y = 1
        else:
            y = -1

        #update the weights if wrong: w = w + learning_rate * (d - y) * x
        if y != d:
            converged = False
            for i in range(n):
                w[i] = w[i] + learning_rate * (d - y) * x[i]

print("Perceptron for Banknote Authentication Data")
print(f"Dataset size: {len(all_data)}")
print(f"Train size: {len(training_set)}")
print(f"Test size: {len(test_set)}")
print(f"Split: stratified {int(TRAIN_RATIO * 100)}/{int((1 - TRAIN_RATIO) * 100)}, seed = {SEED}")
print("Preprocessing: min-max normalization (fitted on training set)")
print(f"Learning rate = {learning_rate}, Bias, w0 = {bias}")

if converged:
    print(f"Training: converged after {epoch} epochs")
else:
    print(f"Training: stopped after {epoch} epochs (data is not perfectly linearly separable)")
print("Final Adjusted Weight: " + ", ".join(f"{val:.4f}" for val in w))

#testing: classify every test row using the final weights
actual = []
predictions = []
final_test = []

for features, label, row_id in test_set:
    x = [1.0] + features

    v = 0
    for i in range(n):
        v = v + w[i] * x[i]

    if v >= 0:
        prediction = '1'
    else:
        prediction = '0'

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