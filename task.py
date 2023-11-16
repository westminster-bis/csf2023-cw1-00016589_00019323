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


# Function to identify missing values
def identify_missing_values(dataframe):
    return dataframe.isnull().sum()

# Function to handle missing values
def handle_missing_values(dataframe, strategy='drop', fill_value=None):
    if strategy == 'drop':
        # Drop rows with missing values
        return dataframe.dropna()
    elif strategy == 'fill':
        # Fill missing values with a specified value
        return dataframe.fillna(fill_value)
    else:
        raise ValueError("Invalid strategy. Use 'drop' or 'fill'.")

# Function to convert data types
def convert_data_types(dataframe, column, new_type):
    dataframe[column] = dataframe[column].astype(new_type)
    return dataframe

# Path to the CSV file
maindata = 'C:/Users/TEMA/OneDrive/Desktop/data.csv'
# Reading the dataset
dataset = pd.read_csv(maindata)

# Identifying missing values
missing_values = identify_missing_values(dataset)
print("Missing Values per Column:")
print(missing_values)

# Handling missing values
# Example: Drop rows with missing values
dataset_cleaned = handle_missing_values(dataset)


# Display cleaned dataset
print("\nCleaned Dataset:")
print(dataset_cleaned.head())


class Encoder:
    def init(self, mapping, default=None, neutral=None, name="Encoder", inverse=False):
        self.mapping = mapping
        self.default = default
        self.neutral = neutral
        if inverse:
            self.mapping = {k: max(mapping.values()) - v + min(mapping.values()) for k, v in mapping.items()}

    def encode(self, response):
        return self.mapping.get(response, self.default)
# Instantiate the Encoder
response_encoder = Encoder({
    "Strong Agree": 5,
    "Agree": 4,
    "Neither": 3,
    "Disagree": 2,
    "Strong Disagree": 1
}, default=3, neutral=3, name="Response Encoding")

# Function to transform the DataFrame
def transform_responses(df, encoder, columns):
    transformed_df = df.copy()
    for col in columns:
        if col in transformed_df.columns:
            transformed_df[col] = transformed_df[col].apply(encoder.encode)
    return transformed_df

df = pd.read_csv('C:/Users/TEMA/OneDrive/Desktop/data.csv')

columns_to_transform = ['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
df_transformed = transform_responses(df, response_encoder, columns_to_transform)

class Scoring:
    from tickcounter.questionnaire import Encoder
    e1 = Encoder({
        "Strong Agree": 5,
        "Agree": 4,
        "Neither": 3,
        "Disagree": 2,
        "Strong Disagree": 1
    }, default=3, neutral=3, name="Agree-ness")
    e2 = Encoder(template=e1, name="Disagree-ness")

    from tickcounter.questionnaire import QuartileLabel
    l1 = QuartileLabel(q=2, labels=['Below average', 'Above average'], name='Above/Below average')
    from tickcounter.questionnaire import Scoring
    s1 = Scoring(labeling=l1, encoding={
        e1: ['7', '8', '10', '14', '17'],
        e2: ['6', '9', '11', '12', '13', '15', '16']
    }, name="time_management")
    scores = s1.score(df)
    df['time_management_score'] = scores


    print(df)

    # Add scores to the DataFrame
    df['time_management'] = scores

    # Calculate and validate labels
    labels = s1.label(df_transformed)
    df_transformed = pd.concat([df_transformed, labels], axis=1)

    # Display the updated DataFrame
    print(df_transformed)
class Encoder:
    def init(self, mapping, default=None, neutral=None, name="Encoder"):
        self.mapping = mapping
        self.default = default
        self.neutral = neutral
        self.name = name

    def count_neutral(self, df, return_rule=False):
        # Assuming 'neutral' is the neutral response value
        # Count neutral responses in each column
        neutral_counts = df.apply(lambda col: sum(col== self.neutral))
        if return_rule:
            # Return rule information
            rule = {col: self.name if col in self.mapping else np.nan for col in df.columns}
            return neutral_counts, pd.Series(rule)
        return neutral_counts
# Instantiate Encoder
e1 = Encoder({
    "Strong Agree": 5,
    "Agree": 4,
    "Neither": 3,  # Assuming this is the neutral response
    "Disagree": 2,
    "Strong Disagree": 1
}, default=3, neutral=3, name="Agree-ness")

# Count neutral responses
total, rule = e1.count_neutral(df, return_rule=True)

# Print the rule information
print(rule)
# Define a threshold for removal
threshold = len(df.columns) / 2
# Grouping by Gender and counting

data = {
    'Gender': ['Male', 'Female', 'Non-binary', 'Male', 'Female', 'Non-binary', 'Male'],
    'Count': [10, 20, 5, 10, 20, 5, 10]  # Example counts
}
dataset = pd.DataFrame(data)

maindata = 'C:/Users/TEMA/OneDrive/Desktop/data.csv'
dataset = pd.read_csv(maindata)
root = tk.Tk()
root.title("Data Analysis")
root.geometry("800x600")

# Frame for plotting
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)


def show_plot(plot_function):
    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)

    # Call the plotting function
    plot_function(dataset, ax)

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
