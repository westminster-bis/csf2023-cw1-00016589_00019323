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

