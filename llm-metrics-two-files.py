#!/usr/bin/env python3
"""
LLM Classification Metrics Generator for Two-File Analysis

This script analyzes the LLM classification results from two separate files:
- One containing algal sequences (true algal samples)
- One containing contaminant sequences (true contaminant samples)

It extracts the predicted tags and calculates comprehensive metrics.
"""

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

def parse_files(algal_file, contaminant_file):
    """
    Parse the algal and contaminant files to extract true and predicted labels
    
    Arguments:
        algal_file (str): Path to the file containing algal sequences
        contaminant_file (str): Path to the file containing contaminant sequences
        
    Returns:
        tuple: Lists of true labels and predicted labels
    """
    true_labels = []
    predicted_labels = []
    sequence_ids = []
    
    # Process algal file (all true labels are 'algal')
    with open(algal_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Skip header or non-data lines
            if line.startswith('==>') or line.startswith('('): ## or not re.search(r'-|_', line):
                continue
                
            # Extract sequence ID
            seq_id_match = re.match(r'^([^\s]+)', line)
            if seq_id_match:
                seq_id = seq_id_match.group(1)
            else:
                seq_id = "unknown_id"
            
            # Add to tracking lists
            true_labels.append('algal')
            sequence_ids.append(seq_id)
            
            # Determine predicted label based on tags
            if '@' in line:
                predicted_labels.append('algal')
            elif '!' in line:
                predicted_labels.append('contaminant')
            else:
                predicted_labels.append('unknown')
            #if re.search(r'<@+>', line):
             #   predicted_labels.append('algal')
            #elif re.search(r'<!+>', line):
             #   predicted_labels.append('contaminant')
            #else:
             #   predicted_labels.append('unknown')
    
    # Process contaminant file (all true labels are 'contaminant')
    with open(contaminant_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Skip header or non-data lines
            if line.startswith('==>') or line.startswith('('): ## or not re.search(r'\.|_', line):
                continue
                
            # Extract sequence ID
            seq_id_match = re.match(r'^([^\s]+)', line)
            if seq_id_match:
                seq_id = seq_id_match.group(1)
            else:
                seq_id = "unknown_id"
            
            # Add to tracking lists
            true_labels.append('contaminant')
            sequence_ids.append(seq_id)
            
            # Determine predicted label based on tags
           # if re.search(r'<@+>', line):
            #    predicted_labels.append('algal')
            #elif re.search(r'<!+>', line):
             #   predicted_labels.append('contaminant')
            #e#lse:
             #   predicted_labels.append('unknown')
            # Determine predicted label based on symbols (@ for algal, ! for contaminant)
            if '@' in line:
                predicted_labels.append('algal')
            elif '!' in line:
                predicted_labels.append('contaminant')
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
    classes = ['algal', 'contaminant']
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
            labels=[0, 1],  # algal, contaminant
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
        precision = recall = f1 = support = [0, 0]
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
        "weighted_f1": np.sum(f1 * support) / np.sum(support) if np.sum(support) > 0 else 0,
        "total_samples": len(true_labels),
        "total_correct": sum(t == p for t, p in zip(true_labels, predicted_labels)),
        "total_unknown": predicted_labels.count("unknown")
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
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correctly classified: {metrics['total_correct']} ({metrics['total_correct']/metrics['total_samples']*100:.2f}%)")
    print(f"Unknown predictions: {metrics['total_unknown']} ({metrics['total_unknown']/metrics['total_samples']*100:.2f}%)")
    print(f"Overall accuracy: {metrics['accuracy']:.4f}")
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
    for class_name in ['algal', 'contaminant']:
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

def generate_visualizations(metrics, output_prefix=None):
    """
    Generate visualizations of the metrics
    
    Arguments:
        metrics (dict): Dictionary containing all calculated metrics
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
    else:
        plt.show()
    
    # Create per-class metrics bar chart
    plt.figure(figsize=(10, 6))
    
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics_names))
    width = 0.35
    
    algal_values = [metrics['precision']['algal'], metrics['recall']['algal'], metrics['f1']['algal']]
    contaminant_values = [metrics['precision']['contaminant'], metrics['recall']['contaminant'], metrics['f1']['contaminant']]
    
    plt.bar(x - width/2, algal_values, width, label='Algal')
    plt.bar(x + width/2, contaminant_values, width, label='Bacterial')
    
    plt.ylabel('Score')
    plt.title('Performance Metrics by Class')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_metrics_by_class.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Create class distribution pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Algal class distribution
    algal_data = metrics['class_metrics']['algal']
    algal_labels = ['Correct', 'Incorrect', 'Unknown']
    algal_values = [algal_data['correct'], algal_data['incorrect'], algal_data['unknown']]
    ax1.pie(algal_values, labels=algal_labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Algal Class Predictions')
    
    # Bacterial class distribution
    contaminant_data = metrics['class_metrics']['contaminant']
    contaminant_labels = ['Correct', 'Incorrect', 'Unknown']
    contaminant_values = [contaminant_data['correct'], contaminant_data['incorrect'], contaminant_data['unknown']]
    ax2.pie(contaminant_values, labels=contaminant_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Bacterial Class Predictions')
    
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_class_distribution.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()

def create_misclassified_report(true_labels, predicted_labels, sequence_ids, output_file=None):
    """
    Create a report of misclassified sequences
    
    Arguments:
        true_labels (list): List of true class labels
        predicted_labels (list): List of predicted class labels
        sequence_ids (list): List of sequence IDs
        output_file (str, optional): Path to save the report to
    """
    misclassified = []
    for i, (true, pred, seq_id) in enumerate(zip(true_labels, predicted_labels, sequence_ids)):
        if true != pred:
            misclassified.append({
                'id': seq_id,
                'true': true,
                'predicted': pred
            })
    
    # Start capturing output
    if output_file:
        import io
        output_capture = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_capture
    
    # Print header
    print("\n" + "="*60)
    print("          MISCLASSIFIED SEQUENCES REPORT")
    print("="*60)
    print(f"\nTotal misclassified: {len(misclassified)} out of {len(true_labels)} ({len(misclassified)/len(true_labels)*100:.2f}%)\n")
    
    # Print algal sequences misclassified as contaminant
    print("\n--- ALGAL SEQUENCES MISCLASSIFIED AS BACTERIAL ---")
    algal_as_contaminant = [m for m in misclassified if m['true'] == 'algal' and m['predicted'] == 'contaminant']
    for item in algal_as_contaminant:
        print(f"ID: {item['id']}")
    print(f"Total: {len(algal_as_contaminant)}")
    
    # Print contaminant sequences misclassified as algal
    print("\n--- BACTERIAL SEQUENCES MISCLASSIFIED AS ALGAL ---")
    contaminant_as_algal = [m for m in misclassified if m['true'] == 'contaminant' and m['predicted'] == 'algal']
    for item in contaminant_as_algal:
        print(f"ID: {item['id']}")
    print(f"Total: {len(contaminant_as_algal)}")
    
    # Print unknown classifications
    print("\n--- SEQUENCES WITH UNKNOWN CLASSIFICATION ---")
    unknown = [m for m in misclassified if m['predicted'] == 'unknown']
    for item in unknown:
        print(f"ID: {item['id']} (True: {item['true']})")
    print(f"Total: {len(unknown)}")
    
    # If saving to file
    if output_file:
        # Restore stdout
        sys.stdout = original_stdout
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(output_capture.getvalue())
        
        print(f"Misclassified report saved to {output_file}")

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='LLM Classification Metrics Generator for Two-File Analysis')
    parser.add_argument('algal_file', help='Path to the file containing algal sequences')
    parser.add_argument('contaminant_file', help='Path to the file containing contaminant sequences')
    parser.add_argument('-o', '--output', help='Path to save the metrics report')
    parser.add_argument('-m', '--misclassified', help='Path to save the misclassified sequences report')
    parser.add_argument('-v', '--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('-p', '--prefix', default='llm_metrics', help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Parse files and calculate metrics
    true_labels, predicted_labels, sequence_ids = parse_files(args.algal_file, args.contaminant_file)
    metrics = calculate_metrics(true_labels, predicted_labels)
    
    # Display results
    output_file = f"{args.prefix}_report.txt" if args.output else None
    display_results(metrics, output_file)
    
    # Generate visualizations if requested
    if args.visualize:
        generate_visualizations(metrics, args.prefix)
    
    # Create misclassified report if requested
    if args.misclassified:
        misclassified_file = f"{args.prefix}_misclassified.txt" if args.misclassified is True else args.misclassified
        create_misclassified_report(true_labels, predicted_labels, sequence_ids, misclassified_file)
    
    # Return number of misclassifications (for automated testing)
    misclassifications = sum(t != p for t, p in zip(true_labels, predicted_labels))
    return misclassifications

if __name__ == "__main__":
    sys.exit(main())
