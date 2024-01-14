# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# ML
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)

# Interactive visualization
import ipywidgets as widgets
from IPython.display import display, clear_output

# ==================================================================================================
# Calculating evaluation metrics
# ==================================================================================================
def roc_auc_score(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return roc_auc, fpr, tpr

def precision_recall_auc_score(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    return pr_auc, precision, recall

# ==================================================================================================
# Interactive visualizations
# ==================================================================================================
# Plotting functions for interactive version
def plot_precision_recall_curve(ax, true_labels, scores, threshold):
    """
    Plots the Precision-Recall curve on the given axes.
    """
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    pr_auc = auc(recall, precision)

    closest_threshold_index = np.argmin(np.abs(thresholds - threshold))
    ax.clear()
    ax.plot(
        recall, precision, color="green", lw=2, label=f"PR curve (area = {pr_auc:.2f})"
    )
    ax.scatter(
        recall[closest_threshold_index], precision[closest_threshold_index], color="red"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")


def plot_roc_curve(ax, true_labels, scores, threshold, zoom_in):
    """
    Plots the ROC curve on the given axes.
    """
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)

    closest_threshold_index = np.argmin(np.abs(thresholds - threshold))
    ax.clear()
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    ax.scatter(fpr[closest_threshold_index], tpr[closest_threshold_index], color="red")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")

    if zoom_in:
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0.7, 1.1)


def plot_anomaly_score_histogram(ax, scores, threshold, bins=30):
    ax.clear()
    ax.hist(scores, bins=bins, color="blue", alpha=0.7)
    ax.axvline(
        x=threshold, color="red", linestyle="--", label=f"Threshold: {threshold:.2f}"
    )
    ax.set_title("Anomaly Scores Distribution")
    ax.set_xlabel("Anomaly Scores")
    ax.set_ylabel("Frequency")
    ax.legend()


def plot_confusion_matrix(ax, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ax.clear()
    ax.imshow(np.zeros_like(cm), cmap="binary", aspect="auto")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    classes = ["Normal", "Anomaly"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(classes)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center", color="black")


# Define the interactive plot function
def interactive_plot(scores, true_labels):
    threshold_slider = widgets.FloatSlider(
        value=np.min(scores),
        min=np.min(scores),
        max=np.max(scores),
        step=0.01,
        description="Threshold:",
        continuous_update=False,
    )
    zoom_checkbox = widgets.Checkbox(value=False, description="Zoom in ROC")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    def update(change):
        threshold = threshold_slider.value
        zoom_in = zoom_checkbox.value
        predicted_labels = scores >= threshold

        # Clear all axes
        for ax in axes.flatten():
            ax.clear()

        # Update plots
        plot_roc_curve(axes[0, 0], true_labels, scores, threshold, zoom_in)
        plot_anomaly_score_histogram(axes[0, 1], scores, threshold)
        plot_precision_recall_curve(axes[1, 0], true_labels, scores, threshold)
        plot_confusion_matrix(axes[1, 1], true_labels, predicted_labels)

        clear_output(wait=True)
        display(threshold_slider)
        display(zoom_checkbox)
        display(fig)

    threshold_slider.observe(update, names="value")
    zoom_checkbox.observe(update, names="value")
    update({"new": threshold_slider.value})  # Initial plot


# ==================================================================================================
# Other plotting functions
# ==================================================================================================


def plot_performance_ds_t1_t2(
    y_scores_ds, y_scores_t1, y_scores_t2, true_labels, zoom_in=False
):
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot ROC Curve
    for scores, color, label in zip(
        [y_scores_ds, y_scores_t1, y_scores_t2],
        ["green", "blue", "red"],
        ["Data space", "FA t1", "FA t2"],
    ):
        fpr, tpr, _ = roc_curve(
            true_labels, scores
        )  # Remove unused variable "thresholds"
        roc_auc = auc(fpr, tpr)
        axes[0].plot(
            fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.2f})"
        )
    axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Receiver Operating Characteristic")
    if zoom_in:
        axes[0].set_xlim(0, 0.3)
        axes[0].set_ylim(0.7, 1.1)

    # Plot Precision-Recall Curve
    for scores, color, label in zip(
        [y_scores_ds, y_scores_t1, y_scores_t2],
        ["green", "blue", "red"],
        ["Data space", "FA t1", "FA t2"],
    ):
        precision, recall, _ = precision_recall_curve(
            true_labels, scores
        )  # Remove unused variable "thresholds"
        pr_auc = auc(recall, precision)
        axes[1].plot(
            recall, precision, color=color, lw=2, label=f"{label} (AUC = {pr_auc:.2f})"
        )
    axes[1].set_xlabel("Recall")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")

    # Plot Anomaly Score Histogram
    for scores, color, label in zip(
        [y_scores_ds, y_scores_t1, y_scores_t2],
        ["green", "blue", "red"],
        ["Data space", "FA t1", "FA t2"],
    ):
        # Normalizing arrays
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        axes[2].hist(
            scores,
            bins=30,
            color=color,
            alpha=0.7,
            label=f"{label} anomaly scores",
        )
    axes[2].set_title("Anomaly Scores Distribution")
    axes[2].set_xlabel("Anomaly Scores")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def plot_train_val_loss(model_history):
    # Get train and validation losses
    train_loss = model_history['loss']
    val_loss = model_history['val_loss']

    # Plot train and validation losses
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')

    # Add title and labels
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
   
    # Hide the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False) 
    

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()