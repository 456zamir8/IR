'''A) Calculate precision, recall, and F-measure for a given set of retrieval results. 
   B) Use an evaluation toolkit to measure average precision and other evaluation metrics.'''

# A)
def calculate_metrics(retrieved_set, relevant_set):
    true_positive = len(retrieved_set.intersection(relevant_set))
    false_positive = len(retrieved_set.difference(relevant_set))
    false_negative = len(relevant_set.difference(retrieved_set))

    print("True Positive: ", true_positive,
          "\nFalse Positive: ", false_positive,
          "\nFalse Negative: ", false_negative, "\n")

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# Use sets here:
retrieved_set = set(["doc1", "doc2", "doc3"])
relevant_set = set(["doc1", "doc4"])

precision, recall, f1 = calculate_metrics(retrieved_set, relevant_set)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# B)
from sklearn.metrics import average_precision_score

y_true = [0, 1, 1, 0, 1, 1] #Binary Prediction 
y_scores = [0.1, 0.4, 0.35, 0.8, 0.65, 0.9] #Model's estimation score 

average_precision = average_precision_score(y_true, y_scores)

print(f'Average Precision: {average_precision:.2f}')