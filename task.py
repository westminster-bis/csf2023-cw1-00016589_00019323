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