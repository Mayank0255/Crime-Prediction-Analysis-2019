import seaborn
import matplotlib.pyplot as plt


def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.title("Confusion Matrix")

    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()

# define data
data = [[4, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 1, 0, 4, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 0, 0],
        [0, 0, 1, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 25, 0, 0, 0],
        [0, 0, 0, 0, 0, 30, 1, 0],
        [0, 0, 0, 0, 0, 0, 21, 0],
        [0, 0, 0, 0, 0, 0, 0, 21]]
# define labels
labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

# create confusion matrix
plot_confusion_matrix(data, labels, "../Visuals/v2/confusion_matrix.png")
