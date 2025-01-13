from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
from utils_0 import read_task_in_jsonl
import re

def parse_input(text, sample = None, all_labels = None):
    """Parse the input text list into a dictionary of samples and their corresponding labels."""
    samples = defaultdict(list)
    
    # Regular expression to match "In the Class level of ..." or "In the Method level of ..."

    lines = text.strip().splitlines()
    current_sample = sample
    
    for line in lines:
        if line.find(" level of ")!=-1 or line.find(" level of ")!=-1:
            current_sample = line.split("level of ")[1].split(" ")[0]
            current_sample = current_sample.split(":")[0]
            # print(current_sample)
        elif re.match(r'^\d+:', line) and current_sample:
            label = " ".join(line.split(':', 1)[1].split(" ")[0:2])
            label = label.split(":")[0]
            if label == "ere is" or label == "There is":
                continue
            samples[current_sample].append(label)
        elif all_labels is not None and current_sample:
            for label in all_labels:
                if line.find(label.lower())!=-1:
                    samples[current_sample].append(label)
        elif line.find(" because ")!=-1 and current_sample:
            label = line.split('because ')[0]
            if label.find(" is ")!=-1:
                label = label.split(" is ")[1]
            label = label.split(" ")
            if len(label) > 4:
                continue
            label = " ".join(label[:2])
            if label.find(".")!=-1:
                label = label.split(".")[1]
            if label == "ere is" or label == "There is":
                continue
            samples[current_sample].append(label)
            # print(label)
       
    return samples

def calculate_metrics(datas):
    """Calculate F1, recall, and precision for each label."""
    # Parse the input texts
    
    all_labels = set()
    
    preds = []
    labels = []
    for data in datas[:]:
        pred = data['prediction']
        label = data['label_output']
        
       
        label_samples = parse_input(label)
        if len(label_samples) == 1:
            for k in label_samples:
                x = k
        else:
            x = None
        pred_samples = parse_input(pred,x)
        
        # for x in label_samples:
        #     true_labels = set(label_samples.get(x, []))
        #     if "Long Method" in true_labels:
                        
        #         print("-"*30)
        #         print(pred)
        #         print()
        #         print(label_samples)
        #         print(pred_samples)
        
        # print(pred)
        # print("\n")
        # print(label)
        # print("\n")
        # print(pred_samples)
        # print(label_samples)

        # Get a list of all unique labels
        for smells in label_samples.values():
            all_labels.update(smells)  # Use only labels in the ground truth
    
    y_true = []
    y_pred = []
    for data in datas[:]:
        pred = data['prediction']
        label = data['label_output']
        
        label_samples = parse_input(label)
        if len(label_samples) == 1:
            for k in label_samples:
                x = k
        else:
            x = None
        pred_samples = parse_input(pred,x, all_labels)
        for x in label_samples:
            true_labels = set(label_samples.get(x, []))
            if "Complex Conditional" in true_labels:
                        
                print("-"*30)
                print(pred)
                print()
                print(label_samples)
                print(pred_samples)
        # Prepare ground truth and predictions
       

        for sample in label_samples.keys():  # Iterate only over samples in label
            true_labels = set(label_samples.get(sample, []))
            if sample not in pred_samples:
                pred_labels = ([0] * len(all_labels))
            else:
                pred_labels = set(pred_samples.get(sample, []))

            y_true.append([1 if label in true_labels else 0 for label in all_labels])
            y_pred.append([1 if label in pred_labels else 0 for label in all_labels])

        # For samples in prediction but not in label, treat as all-zero ground truth
    print(all_labels)
    # Calculate metrics for each label
    metrics = {}
    for i, label in enumerate(all_labels):
        y_true_label = [y[i] for y in y_true]
        y_pred_label = [y[i] for y in y_pred]

        precision = precision_score(y_true_label, y_pred_label)
        recall = recall_score(y_true_label, y_pred_label)
        f1 = f1_score(y_true_label, y_pred_label)

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return metrics


data = read_task_in_jsonl("../../output/newbase/merged_inference.jsonl")

metrics = calculate_metrics(data)

with open("all_test_f1.txt",'r') as f:
    lines = f.readlines()
    for line in lines:
        x = line.split("\n")[0].split("&")
        a = x[0]
        st = ""
        if len(x) > 5:
            b = x[4]
        else:
            b = None
        for label, metric in metrics.items():
            if a.find(label) > -1:
                st+=label+"&"+f"{metric['precision']:.3f}"+"&"+f"{metric['recall']:.3f}"+"&"+f"{metric['f1']:.3f}"
                break
        if b is not None:
            for label, metric in metrics.items():
                if b.find(label) > -1:
                    st+="&"+label+"&"+f"{metric['precision']:.3f}"+"&"+f"{metric['recall']:.3f}"+"&"+f"{metric['f1']:.3f}"
                    break
        print(st+"\cr")
  

# Print metrics
for label, metric in metrics.items():
    print(f"Label: {label}")
    print(f"  Precision: {metric['precision']:.3f}")
    print(f"  Recall: {metric['recall']:.3f}")
    print(f"  F1 Score: {metric['f1']:.3f}\n")
