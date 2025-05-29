#!/usr/bin/env python3
"""
Advanced LLM Classification Metrics Generator

This script analyzes the output from LLM classification tasks,
specifically for algal vs bacterial sequence classification.
It handles various tag formats and provides detailed metrics.
"""

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc

def parse_file(file_path):
    """
    Parse the LLM inference results file and extract true and predicted labels
    
    Arguments:
        file_path (str): Path to the file containing inference results
        
    Returns:
        tuple: Lists of true labels and predicted labels
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split content into sections
    sections = re.split(r'==>\s+(.+?)\s+<==', content)
    
    # Initialize result lists
    true_labels = []
    predicted_labels = []
    sequence_ids = []
    
    # Track current section type
    current_section_type = None
    
    for i, section in enumerate(sections):
        if i == 0:  # Skip initial empty section if present
            continue
            
        # Check if this is a section header
        if 'Algal-example' in section:
            current_section_type = 'algal'
            continue
        elif 'Bacterial-example' in section:
            current_section_type = 'bacterial'
            continue
            
        # Process lines in the section
        if current_section_type:
            lines = section.strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                    
                # Extract sequence ID
                seq_id_match = re.match(r'^([^>]+?)\s', line)
                if seq_id_match:
                    seq_id = seq_id_match.group(1)
                else:
                    seq_id = "unknown_id"
                
                # Add true label based on section
                true_labels.append(current_section_type)
                sequence_ids.append(seq_id)
                
                # Check for predicted tag
                if re.search(r'<@+>', line):
                    predicted_labels.append('algal')
                elif re.search(r'<!+>', line):
                    predicted_labels.append('bacterial')
                else:
                    predicted_labels.append('unknown')
    
    return true_labels, predicted_labels, sequence_ids

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate comprehensive classification metrics
    
    Arguments:
        true_labels (list): List of true class labels
        predicted_labels (list): List of predicted class labels
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Convert labels for sklearn functions
    classes = ['algal', 'bacterial']
    label_map = {label: i for i, label in enumerate(classes)}
    
    # Convert to numeric form
    true_numeric = np.array([label_map.get(label, 2) for label in true_labels])
    pred_numeric = np.array([label_map.get(label, 2) for label in predicted_labels])
    
    # Filter out unknowns for main metrics
    known_indices = [i for i, pred in enumerate(predicted_labels) if pred != 'unknown']
    true_known = [true_labels[i] for i in known_indices]
    pred_known = [predicted_labels[i] for i in known_indices]
    
    # Overall accuracy (including unknowns as wrong predictions)
    accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
    
    if true_known and pred_known:
        # Convert to numeric
        true_known_numeric = np.array([label_map[label] for label in true_known])
        pred_known_numeric = np.array([label_map[label] for label in pred_known])
        
        # Calculate precision, recall, and F1 (excluding unknowns)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_known_numeric, 
            pred_known_numeric,
            labels=[0, 1],  # algal, bacterial
            zero_division=0
        )
        
        # Create confusion matrix
        cm = confusion_matrix(
            true_known_numeric, 
            pred_known_numeric,
            labels=[0, 1]
        )
        
        # Full classification report
        report = classification_report(
            true_known_numeric,
            pred_known_numeric,
            labels=[0, 1],
            target_names=classes,
            output_dict=True
        )
    else:
        precision = recall = f1 = support = [[0, 0]]
        cm = np.zeros((2, 2))
        report = {}
    
    # Count occurrences and calculate per-class metrics
    class_metrics = {}
    for class_name in classes:
        class_indices = [i for i, label in enumerate(true_labels) if label == class_name]
        total = len(class_indices)
        
        if total == 0:
            class_metrics[class_name] = {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "unknown": 0,
                "accuracy": 0,
                "error_rate": 0
            }
            continue
            
        correct = sum(1 for i in class_indices if predicted_labels[i] == class_name)
        unknown = sum(1 for i in class_indices if predicted_labels[i] == "unknown")
        incorrect = total - correct - unknown
        
        class_metrics[class_name] = {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "unknown": unknown,
            "accuracy": correct / total if total > 0 else 0,
            "error_rate": (incorrect + unknown) / total if total > 0 else 0
        }
    
    # Compile all metrics
    metrics = {
        "accuracy": accuracy,
        "class_metrics": class_metrics,
        "confusion_matrix": cm,
        "precision": {classes[i]: precision[i] for i in range(len(classes))},
        "recall": {classes[i]: recall[i] for i in range(len(classes))},
        "f1": {classes[i]: f1[i] for i in range(len(classes))},
        "support": {classes[i]: support[i] for i in range(len(classes))},
        "classification_report": report,
        "macro_f1": np.mean(f1),
        "weighted_f1": np.sum(f1 * support) / np.sum(support) if np.sum(support) > 0 else 0
    }
    
    return metrics

def display_results(metrics, output_file=None):
    """
    Display comprehensive results and optionally save to file
    
    Arguments:
        metrics (dict): Dictionary containing all calculated metrics
        output_file (str, optional): Path to save results to
    """
    # Start capturing output if needed
    if output_file:
        import io
        output_capture = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_capture
    
    # Print header
    print("\n" + "="*60)
    print("          LLM CLASSIFICATION METRICS REPORT")
    print("="*60)
    
    # Overall metrics
    print("\n=== OVERALL METRICS ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    
    # Confusion matrix
    cm = metrics["confusion_matrix"]
    class_labels = ["Algal", "Bacterial"]
    
    print("\n=== CONFUSION MATRIX ===")
    print(f"{'':15} | {'Predicted Algal':15} | {'Predicted Bacterial':20}")
    print("-" * 55)
    for i, label in enumerate(class_labels):
        print(f"{label:15} | {int(cm[i][0]):15} | {int(cm[i][1]):20}")
    
    # Per-class metrics
    print("\n=== PER-CLASS METRICS ===")
    print(f"{'Class':10} | {'Precision':10} | {'Recall':10} | {'F1 Score':10} | {'Support':10}")
    print("-" * 60)
    for class_name in ['algal', 'bacterial']:
        precision = metrics['precision'][class_name]
        recall = metrics['recall'][class_name]
        f1 = metrics['f1'][class_name]
        support = metrics['support'][class_name]
        print(f"{class_name.capitalize():10} | {precision:.4f}     | {recall:.4f}     | {f1:.4f}     | {int(support):10}")
    
    # Detailed class counts
    print("\n=== DETAILED CLASS COUNTS ===")
    for class_name, class_data in metrics["class_metrics"].items():
        print(f"{class_name.capitalize()} class:")
        print(f"  Total samples: {class_data['total']}")
        if class_data['total'] > 0:
            print(f"  Correctly classified: {class_data['correct']} ({class_data['correct']/class_data['total']*100:.2f}%)")
            print(f"  Incorrectly classified: {class_data['incorrect']} ({class_data['incorrect']/class_data['total']*100:.2f}%)")
            print(f"  Unknown: {class_data['unknown']} ({class_data['unknown']/class_data['total']*100:.2f}%)")
        print()
    
    # If saving to file
    if output_file:
        # Restore stdout
        sys.stdout = original_stdout
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(output_capture.getvalue())
        
        print(f"Results saved to {output_file}")

def generate_visualizations(metrics, true_labels, predicted_labels, output_prefix=None):
    """
    Generate visualizations of the metrics
    
    Arguments:
        metrics (dict): Dictionary containing all calculated metrics
        true_labels (list): List of true class labels
        predicted_labels (list): List of predicted class labels
        output_prefix (str, optional): Prefix for output image files
    """
    # Create confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    cm = metrics["confusion_matrix"]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ["Algal", "Bacterial"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j]), 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    
    # Create per-class metrics bar chart
    plt.figure(figsize=(10, 6))
    
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics_names))
    width = 0.35
    
    algal_values = [metrics['precision']['algal'], metrics['recall']['algal'], metrics['f1']['algal']]
    bacterial_values = [metrics['precision']['bacterial'], metrics['recall']['bacterial'], metrics['f1']['bacterial']]
    
    plt.bar(x - width/2, algal_values, width, label='Algal')
    plt.bar(x + width/2, bacterial_values, width, label='Bacterial')
    
    plt.ylabel('Score')
    plt.title('Performance Metrics by Class')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_metrics_by_class.png", dpi=300, bbox_inches='tight')

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='LLM Classification Metrics Generator')
    parser.add_argument('file', help='Path to the file containing LLM inference results')
    parser.add_argument('-o', '--output', help='Path to save the metrics report')
    parser.add_argument('-v', '--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('-p', '--prefix', default='llm_metrics', help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Parse file and calculate metrics
    true_labels, predicted_labels, sequence_ids = parse_file(args.file)
    metrics = calculate_metrics(true_labels, predicted_labels)
    
    # Display results
    output_file = f"{args.prefix}_report.txt" if args.output else None
    display_results(metrics, output_file)
    
    # Generate visualizations if requested
    if args.visualize:
        generate_visualizations(metrics, true_labels, predicted_labels, args.prefix)
    
    # Return number of misclassifications (for automated testing)
    misclassifications = sum(t != p for t, p in zip(true_labels, predicted_labels))
    return misclassifications

if __name__ == "__main__":
    sys.exit(main())
