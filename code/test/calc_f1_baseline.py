from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
from utils_0 import read_task_in_jsonl
import re

def parse_input(x):
    pred, label = x
    
    key = label.split("\n")[1]
    if key.find(":")!=-1:
        key = key.split(":")[1]
    else:
        key = key.split("smell of ")[1].split(".")[0]
        
    if label.split("\n")[1].find(" no ") == -1:
        label = 1
    else:
        label = 0
    
    if pred.find(key) != -1 and pred.find(" no ") == -1:
        pred = 1
    else:
        if  key == "Multifaceted Abstraction" and label == 1:
            print("-"*30)
            print(pred)
            print("\n")
            pass
        pred = 0
    return pred, label, key

    # for preds in pred.split("\n"):
    #     if preds.find(key) != -1 and preds.find(" no ") == -1:
    #         return 1, label, key
    #     # else:
    #     #     if pred.find(key)!=-1:
    #     #         print(pred)
    #     #         print("\n")
    #     #     pred = 0
    # return 0, label, key
  

def calculate_metrics(datas):
    """Calculate F1, recall, and precision for each label."""
    # Parse the input texts
    
    all_labels = {}
    
    preds = []
    labels = []
    for data in datas:
        pred = data['prediction']
        label = data['label_output']
        
        pred,label,key = parse_input([pred,label])
        
        # print(key)
        # print(pred)
        # print("\n")
        # print(label)
        # print("\n")
        # print(pred_samples)
        # print(label_samples)

        # Get a list of all unique labels
        
        all_labels[key] = 1  # Use only labels in the ground truth
    print(all_labels)
    y_true = {}
    y_pred = {}
    for key in all_labels:
       y_true[key] = []
       y_pred[key] = []
    for data in datas:
        pred = data['prediction']
        label = data['label_output']
        
        pred,label,key = parse_input([pred,label])
        # Prepare ground truth and predictions
        y_true[key].append(label)
        y_pred[key].append(pred)

       
    # Calculate metrics for each label
    metrics = {}
    for i, label in enumerate(all_labels):
        y_true_label = y_true[label]
        y_pred_label = y_pred[label]

        precision = precision_score(y_true_label, y_pred_label)
        recall = recall_score(y_true_label, y_pred_label)
        f1 = f1_score(y_true_label, y_pred_label)

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return metrics

# data = read_task_in_jsonl("/data/liangwj/codellama/design_implementation_smell_no_smell/design0w-100wunsmell_200wimplementation0w-300wunsmell_200w/smell_llama_train_on_baseline/merged_inference.jsonl")
# data = read_task_in_jsonl("/data/liangwj/codellama/design_implementation_smell_no_smell/design0w-100wunsmell_200wimplementation0w-300wunsmell_200w/newbase_merged_tuned_on_baseline/merged_inference.jsonl")
data = read_task_in_jsonl("/data/liangwj/codellama/design_implementation_smell_no_smell/design0w-100wunsmell_200wimplementation0w-300wunsmell_200w/newbase_copy_for_testbaseline/merged_inference.jsonl")


metrics = calculate_metrics(data)

# Print metrics
for label, metric in metrics.items():
    print(f"Label: {label}")
    print(f"  Precision: {metric['precision']:.3f}")
    print(f"  Recall: {metric['recall']:.3f}")
    print(f"  F1 Score: {metric['f1']:.3f}\n")
