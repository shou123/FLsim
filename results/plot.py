import matplotlib.pyplot as plt
import re 

# Define the file path
file_path = "/home/shiyue/FLsim/results/distance_values.txt"

# Initialize a dictionary to store client data
client_data = {}

# Read the data from the file
with open(file_path, "r") as file:
    for line in file:
        parts = line.split()
        client_number = int(parts[1][:-2])  # Extract the client number 
        norm_value = float(parts[3])  # Extract the norm value
        if client_number not in client_data:
            client_data[client_number] = []
        client_data[client_number].append(norm_value)
# Set the figure size (adjust the width and height as needed)
plt.figure(figsize=(25, 15))
# Plot the data for all clients
for client_number, data in client_data.items():
    rounds = range(len(data))  # X-axis values
    plt.plot(rounds, data, label=f'Client {client_number}')

# Set plot labels and legend
plt.title("Norm Values for Different Clients")
plt.xlabel("Rounds")
plt.ylabel("Distance")
plt.legend(loc="upper right")

# Define the number of legend columns
num_legend_columns = 2  # Adjust as needed

# Create a legend with multiple columns
plt.legend(loc="upper right", ncol=num_legend_columns)

# Show the plot
plt.show()