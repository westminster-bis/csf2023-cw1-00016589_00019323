
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
maindata = 'data.csv'
dataset = pd.read_csv(maindata)
questionsdata = 'Data2.csv'
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
    'Count': [10, 20, 5, 10, 20, 5, 10]
}
dataset = pd.DataFrame(data)
#code for tkinter
maindata = 'data.csv'
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

def dynamic_plot():
    x_column = x_column_var.get()
    y_column = y_column_var.get()

    # Create a new figure and axis
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)

    # Generate the plot
    sns.barplot(x=x_column, y=y_column, data=dataset, ax=ax)

# Clear previous plot from the plot_frame and display the new plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def plot_age_distribution(data, ax):
    sns.countplot(data=data, x='Age', ax=ax)


    def filter_data(plot_function=None):
        global dataset
        # Example: Filter based on a condition, such as 'Age' > 20
        filtered_data = dataset[dataset['Age'] > 20]
        show_plot(plot_function, filtered_data)

    filter_button = ttk.Button(root, text="Filter Data", command=filter_data)
    filter_button.pack(side=tk.TOP, fill=tk.X)
def apply_filter():
    global dataset
    try:
        min_age = int(min_age_entry.get())
        max_age = int(max_age_entry.get())
        filtered_data = dataset[(dataset['Age'] >= min_age) & (dataset['Age'] <= max_age)]
        show_plot(plot_age_distribution, filtered_data)
    except ValueError:
        messagebox.showerror("Error", "Invalid age entered")
def dynamic_plot():
    def dynamic_plot():
        x_column = x_column_var.get()
        y_column = y_column_var.get()

        # Create a new figure and axis for the plot
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Check if y_column is numeric, if not, use a count plot
        if dataset[y_column].dtype.kind in 'bifc':  # Numeric types
            sns.barplot(x=x_column, y=y_column, data=dataset, ax=ax)
        else:
            sns.countplot(x=x_column, data=dataset, ax=ax)  # Default to count plot

        # Clear previous plot from plot_frame and display the new plot
        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


x_column_var = tk.StringVar()
x_column_dropdown = ttk.Combobox(root, textvariable=x_column_var, values=dataset.columns.tolist())
x_column_dropdown.pack(side=tk.TOP, fill=tk.X)

y_column_var = tk.StringVar()
y_column_dropdown = ttk.Combobox(root, textvariable=y_column_var, values=dataset.columns.tolist())
y_column_dropdown.pack(side=tk.TOP, fill=tk.X)

plot_button = ttk.Button(root, text="Plot Columns", command=dynamic_plot)
plot_button.pack(side=tk.TOP, fill=tk.X)

# Entry fields for age filter
min_age_entry = tk.Entry(root)
min_age_entry.pack(side=tk.TOP, fill=tk.X)
max_age_entry = tk.Entry(root)
max_age_entry.pack(side=tk.TOP, fill=tk.X)

# Button to apply filter
filter_button = ttk.Button(root, text="Apply Age Filter", command=apply_filter)
filter_button.pack(side=tk.TOP, fill=tk.X)
def show_summary():
    summary_window = tk.Toplevel(root)
    summary_text = tk.Text(summary_window)
    summary_text.insert(tk.END, str(dataset.describe()))
    summary_text.pack()

summary_button = ttk.Button(root, text="Show Data Summary", command=show_summary)
summary_button.pack(side=tk.TOP, fill=tk.X)
def export_data():
    try:
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filename:
            filtered_data.to_csv(filename, index=False)
            messagebox.showinfo("Information", "Data exported successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

export_button = ttk.Button(root, text="Export Data", command=export_data)
export_button.pack(side=tk.TOP, fill=tk.X)

def plot_nationality_distrubution(data, ax):
    sns.countplot(data=data, x='Nationality', ax=ax)
def plot_Program_distrubution(data, ax):
        sns.countplot(data=data, x='Program', ax=ax)
def plot_English_distrubution(data, ax):
    sns.countplot(data=data, x='English', ax=ax)
def plot_Attendance_distrubution(data, ax):
    sns.countplot(data=data, x='Attendance', ax=ax)
def plot_Academic_distrubution(data, ax):
    sns.countplot(data=data, x='Academic', ax=ax)
def plot_Course_distrubution(data, ax):
            sns.countplot(data=data, x='Course', ax=ax)
def plot_gender_distribution(data, ax):
    sns.countplot(data=data, x='Gender', ax=ax)  # Replace 'Gender' with your column name
# Buttons to trigger different plots
age_button = ttk.Button(root, text="Plot Age Distribution", command=lambda: show_plot(plot_age_distribution))
age_button.pack(side=tk.TOP, fill=tk.X)

gender_button = ttk.Button(root, text="Plot Gender Distribution", command=lambda: show_plot(plot_gender_distribution))
gender_button.pack(side=tk.TOP, fill=tk.X)
nationality_button = ttk.Button(root,text="Plot Nationality Distrubution",command=lambda: show_plot(plot_nationality_distrubution))
nationality_button.pack(side=tk.TOP,fill=tk.X)
program_button = ttk.Button(root, text="Plot Program Distribution", command=lambda: show_plot(plot_Program_distrubution))
program_button.pack(side=tk.TOP, fill=tk.X)
english_button = ttk.Button(root, text="Plot English Distribution", command=lambda: show_plot(plot_English_distrubution))
english_button.pack(side=tk.TOP, fill=tk.X)
attendance_button = ttk.Button(root, text="Plot Attendance Distribution", command=lambda: show_plot(plot_Attendance_distrubution))
attendance_button.pack(side=tk.TOP, fill=tk.X)
academic_button = ttk.Button(root, text="Plot Academic Distribution", command=lambda: show_plot(plot_Academic_distrubution))
academic_button.pack(side=tk.TOP, fill=tk.X)
course_button = ttk.Button(root, text="Plot Course Distribution", command=lambda: show_plot(plot_Course_distrubution))
course_button.pack(side=tk.TOP, fill=tk.X)

root.mainloop()