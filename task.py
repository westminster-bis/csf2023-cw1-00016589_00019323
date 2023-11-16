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
def plot_age_distribution(data, ax):
    sns.countplot(data=data, x='Age', ax=ax)  # Replace 'Age' with your column name
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
# Frame for the selector and button
frame_controls = tk.Frame(root)
frame_controls.pack(fill='x')

# Label for the selector
label = tk.Label(frame_controls, text="Choose Gender Column:")
label.pack(side=tk.LEFT, padx=5, pady=5)

# Combobox to select the gender column
selector = ttk.Combobox(frame_controls, values=list(dataset.columns), state="readonly")
selector.set('Gender')  # default value
selector.pack(side=tk.LEFT, padx=5, pady=5)

# Button to plot gender distribution
button = ttk.Button(frame_controls, text='Plot Distribution', command=plot_gender_distribution)
button.pack(side=tk.LEFT, padx=5, pady=5)

# Run the application
root.mainloop()
#plt.show()
# Count the occurrences of each nationality
nationality_counts = df['Nationality'].value_counts().reset_index()
nationality_counts.columns = ['Nationality', 'Count']

# Set the size of the plot
plt.figure(figsize=(100, 8))

# Create a bar plot

# Assuming nationality_counts is a DataFrame with columns 'Nationality' and 'Count'
sns.barplot(x='Count', y='Nationality', hue='Nationality', data=nationality_counts, palette='Spectral', legend=False)
# Add other customization as needed

# Add labels and title
plt.xlabel('Count', fontsize=14)
plt.ylabel('Nationality', fontsize=14)
plt.title('Nationality Counts', fontsize=16)

# Add value labels
for index, value in enumerate(nationality_counts['Count']):
    plt.text(value, index, str(value))

#plt.show()
# Count the occurrences of each program
program_counts = df['Program'].value_counts().reset_index()
program_counts.columns = ['Program', 'Count']
# Set the size of the plot
plt.figure(figsize=(14, 10))

# Create a bar plot
sns.barplot(x='Program', y='Count', data=program_counts, palette='Spectral')

# Add labels and title
plt.xlabel('Program', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Counts in Each Program', fontsize=16)
# Rotate x-labels for better readability
plt.xticks(rotation=45)

# Add value labels
for index, value in enumerate(program_counts['Count']):
    plt.text(index, value, str(value), ha='center')

#plt.show()
# Count the occurrences of each course
course_counts = df['Course'].value_counts().reset_index()
course_counts.columns = ['Course', 'Count']
# Set the size of the plot
plt.figure(figsize=(12, 8))

# Create a horizontal bar plot
sns.barplot(x='Count', y='Course', data=course_counts, palette='Spectral')

# Add labels and title
plt.xlabel('Count', fontsize=14)
plt.ylabel('Course', fontsize=14)
plt.title('Counts in Each Course', fontsize=16)

# Add value labels on the bars
for index, value in enumerate(course_counts['Count']):
    plt.text(value, index, str(value), va='center')

#plt.show()
# Count the occurrences of each English score
english_counts = df['English'].value_counts().reset_index()
english_counts.columns = ['English Course Mark', 'Count']

# Set the size of the plot
plt.figure(figsize=(12, 8))

# Create a bar plot
sns.barplot(x='English Course Mark', y='Count', data=english_counts, palette='Spectral')

# Add labels and title
plt.xlabel('English Course Mark', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Counts in Each English Course Mark', fontsize=16)

# Rotate x-labels for better readability
plt.xticks(rotation=45)

# Add value labels on the bars
for index, value in enumerate(english_counts['Count']):
    plt.text(index, value, str(value), ha='center', va='bottom')

#plt.show()
# Count the occurrences of each academic score
academic_counts = df['Academic'].value_counts().reset_index()
academic_counts.columns = ['Academic Course Mark', 'Count']
# Set the size of the plot
plt.figure(figsize=(12, 8))

# Create a bar plot
sns.barplot(x='Academic Course Mark', y='Count', data=academic_counts, palette='Spectral')

# Add labels and title
plt.xlabel('Academic Course Mark', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Counts in Each Academic Course Mark', fontsize=16)

# Rotate x-labels for better readability
plt.xticks(rotation=45)

# Add value labels on the bars
for index, value in enumerate(academic_counts['Count']):
    plt.text(index, value, str(value), ha='center', va='bottom')

#plt.show()

# Count the occurrences of each attendance group
attendance_counts = df['Attendance'].value_counts().reset_index()
attendance_counts.columns = ['Attendance Group', 'Count']
# Set the size of the plot
plt.figure(figsize=(12, 8))

# Create a bar plot
sns.barplot(x='Attendance Group', y='Count', data=attendance_counts, palette='Spectral')

# Add labels and title
plt.xlabel('Attendance Group', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Counts in Each Attendance Group', fontsize=16)

# Rotate x-labels for better readability
plt.xticks(rotation=45)

# Add value labels on the bars
for index, value in enumerate(attendance_counts['Count']):
    plt.text(index, value, str(value), ha='center', va='bottom')


# Example of fixing the FutureWarning in barplot
sns.barplot(x='Program', y='Count', data=program_counts, hue='Program', palette='Spectral', legend=False)



# Initialize global variables
dataset = None

# Load Data Function
def load_data():
    global dataset
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        dataset = pd.read_csv(file_path)
        messagebox.showinfo("Information", "Data loaded successfully!")
        print(dataset.head())  # You can change this to display in the GUI

# Handle Missing Values Function
def handle_missing_values():
    global dataset
    if dataset is not None:
        dataset.dropna(inplace=True)
        messagebox.showinfo("Information", "Missing values handled.")
    else:
        messagebox.showerror("Error", "Data not loaded.")
# Plotting Function
def plot_graphage():
    global dataset
    if dataset is not None:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=dataset, x='Age')  # Replace with your column
        plt.show()
    else:
        messagebox.showerror("Error", "Data not loaded.")
def plot_graphgender():
    global dataset
    if dataset is not None:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=dataset, x='Gender')  # Replace with your column
        plt.show()
    else:
        messagebox.showerror("Error", "Data not loaded.")
# Main Application Window
root = tk.Tk()
root.title("Data Analysis Project")
root.geometry("800x600")

# Load Data Button
load_button = tk.Button(root, text="Load Data", command=load_data)
load_button.pack()

# Handle Missing Values Button
handle_missing_button = tk.Button(root, text="Handle Missing Values", command=handle_missing_values)
handle_missing_button.pack()

# Plot Graph age Button
plot_button = tk.Button(root, text="Plot graph Age", command=plot_graphage())
plot_button.pack()
# Plot Graph  Button
plot_button = tk.Button(root, text="Plot graph Gender", command=plot_graphgender())
plot_button.pack()
plot_button = tk.Button(root, text="Plot graph Age", command=plot_graphage())
plot_button.pack()
#
# Run the application
root.mainloop()


def show_plot(figure):
    # Create a new window to display the plot
    new_window = tk.Toplevel(root)
    canvas = FigureCanvasTkAgg(figure, new_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Function to plot data
def plot_data(column):
    # Prepare the data
    count_data = df[column].value_counts().reset_index()
    count_data.columns = [column, 'Count']

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=column, y='Count', data=count_data, palette='Spectral', ax=ax)
    ax.set_title(f'Counts for {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')

    # Show the plot in a new window
    show_plot(fig)


# Function to perform and show chi-square test results
def chi_square_test(column1, column2):
    contingency_table = pd.crosstab(df[column1], df[column2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    result_text = f"Chi-Square Test between {column1} and {column2}:\nChi2 value: {chi2}\np-value: {p}\nDegrees of Freedom: {dof}"
    # Show the results in a popup window
    result_popup = tk.Toplevel(root)
    ttk.Label(result_popup, text=result_text, justify=tk.LEFT).pack()


# Main application window
root = tk.Tk()
root.title("Data Analysis")
#   Dropdown to select the column for plotting
column_var = tk.StringVar(root)
column_label = ttk.Label(root, text="Select Column:")
column_label.pack(side=tk.TOP, fill=tk.X)
column_dropdown = ttk.Combobox(root, textvariable=column_var, values=df.columns.tolist())
column_dropdown.pack(side=tk.TOP, fill=tk.X)

# Button to perform the plot
plot_button = ttk.Button(root, text="Plot Data", command=lambda: plot_data(column_var.get()))
plot_button.pack(side=tk.TOP, fill=tk.X)

# Entry fields to specify columns for chi-square test
entry1_var = tk.StringVar(root)
entry2_var = tk.StringVar(root)
entry1_label = ttk.Label(root, text=" colum :")
entry1_label.pack(side=tk.TOP, fill=tk.X)
entry1_entry = ttk.Entry(root, textvariable=entry1_var)
entry1_entry.pack(side=tk.TOP, fill=tk.X)
entry2_label = ttk.Label(root, text="C Test:")
entry2_label.pack(side=tk.TOP, fill=tk.X)
entry2_entry = ttk.Entry(root, textvariable=entry2_var)
entry2_entry.pack(side=tk.TOP, fill=tk.X)

# Button to perform chi-square test
chi_square_button = ttk.Button(root, text="Perform Test",
                               command=lambda: chi_square_test(entry1_var.get(), entry2_var.get()))
chi_square_button.pack(side=tk.TOP, fill=tk.X)

root.mainloop()
# Create a figure with grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Load your data
maindata = 'C:/Users/TEMA/OneDrive/Desktop/data.csv'
dataset = pd.read_csv(maindata)

# Initialize the main application window
root = tk.Tk()
root.title("Data Analysis")
root.geometry("800x600")
# Frame for plotting
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)

def show_plot(plot_function, *args):
    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)

    # Call the plotting function with the figure and axis
    plot_function(fig, ax, *args)

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def plot_gender_distribution(fig, ax):
    # Your logic to plot gender distribution
    sns.countplot(data=dataset, x='Gender', ax=ax)  # Replace 'Gender' with your column name

def plot_other_chart(fig, ax):
    # Example plotting logic for another chart
    sns.countplot(data=dataset, x='OtherColumn', ax=ax)  # Replace 'OtherColumn' with actual column name

# Buttons to trigger different plots
button_gender = ttk.Button(root, text="Plot Gender Distribution", command=lambda: show_plot(plot_gender_distribution))
button_gender.pack(side=tk.TOP, fill=tk.X)

button_other = ttk.Button(root, text="Plot Other Chart", command=lambda: show_plot(plot_other_chart))
button_other.pack(side=tk.TOP, fill=tk.X)

# Additional buttons and functionalities can be added similarly

#root.mainloop()
