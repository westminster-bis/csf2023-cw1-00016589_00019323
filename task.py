import numpy as np
from matplotlib.figure import Figure
from scipy.stats import chi2_contingency
from tickcounter.util import plot_trend
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
maindata = 'C:/Users/TEMA/OneDrive/Desktop/data.csv'
dataset = pd.read_csv(maindata)
questionsdata = 'C:/Users/TEMA/OneDrive/Desktop/Data2.csv'
df2 = pd.read_csv(questionsdata)
print(dataset.head())
# second data sheet
print(df2.head(100))
# Display the first few rows


question_columns = ['Order', 'Questions', 'Option & scores']
print(df2[question_columns].head())

# description of the dataset
print("Dataset Description:")
print("Number of Rows:", dataset.shape[0])
print("Number of Columns:", dataset.shape[1])
print("\nAvailable Features/Columns:")
print(dataset.columns.tolist())


# Displaying a sample of the dataset
print("\nSample Data:")
print(dataset.head(1000))
def plot_columns(data, col_list, plot_type, n_col=2, **kwargs):
    n_row = len(col_list) // n_col + 1
    plt.figure(figsize=(n_col * 5, n_row * 4))  # Adjust the size as needed

    for i, col in enumerate(col_list):
        ax = plt.subplot(n_row, n_col, i + 1)

        if plot_type == "hist":
            sns.histplot(data=data, x=col, multiple="stack", **kwargs)

        elif plot_type == "bar":
            sns.barplot(data=data, x=col, **kwargs)

        elif plot_type == "count":
            sns.countplot(data=data, x=col, **kwargs)

        elif plot_type == "box":
            sns.boxplot(data=data, x=col, **kwargs)

        elif plot_type == "line":
            x = kwargs.get('x', None)
            sns.lineplot(data=data, x=x, y=col, ax=ax, **kwargs)

        elif plot_type == "trend":
            # Assuming plot_trend is a defined function
            x = kwargs.get('x', None)
            plot_trend(data=data, x=x, y=col, ax=ax, **kwargs)

        elif plot_type == "top":
            # Assuming 'top' is defined
            top = kwargs.get('top', 5)
            temp = data[col].value_counts()
            if top > 0:
                sns.barplot(x=temp.index[0:top], y=temp[0:top])
            else:
                sns.barplot(x=temp.index[-1:top:-1], y=temp[-1:top:-1])

        else:
            raise ValueError(f"Invalid plot_type argument: {plot_type}")

        ax.set_title(f"Distribution of {col}")

    plt.tight_layout()
    #plt.show()