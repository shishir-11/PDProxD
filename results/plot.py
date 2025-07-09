import matplotlib.pyplot as plt
import numpy as np

class Plotting:
    def __init__(self,precision=None,f1=None,accuracy=None,recall=None):
        self.precision_dict = precision
        self.f1_dict = f1
        self.accuracy_dict = accuracy
        self.recall_dict = recall

    def plot(self):
        model_names = list(self.precision_dict.keys())

        # Gather metric values in order
        accuracy = [self.accuracy_dict[name] for name in model_names]
        precision = [self.precision_dict[name] for name in model_names]
        recall = [self.recall_dict[name] for name in model_names]
        f1 = [self.f1_dict[name] for name in model_names]

        x = np.arange(len(model_names))  # label locations
        width = 0.2  # width of each bar

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
        rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision')
        rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall')
        rects4 = ax.bar(x + 1.5*width, f1, width, label='F1 Score')

        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylim(0, 1)
        ax.legend()

        for rects in [rects1, rects2, rects3, rects4]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()
        
    def acit(self,res_dict):
        plt.figure(figsize=(10, 6))

        # Plot the accuracy for each model
        for model_name, accuracies in res_dict.items():
            iterations = list(range(10, 51))  # Iterations from 10 to 50
            plt.plot(iterations, accuracies, label=f'Accuracy for {model_name}')
        
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Iterations for Multiple Models (Iterations 10-50)')
        plt.legend()
        plt.grid(True)
        plt.show()