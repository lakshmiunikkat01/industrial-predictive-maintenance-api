import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10,6))
    sns.barplot(x=sorted_importances[:10], y=sorted_features[:10])
    plt.title("Top 10 Feature Importances")
    plt.show()
