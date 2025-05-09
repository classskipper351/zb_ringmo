import re
from time import sleep

# Initialize arrays
iterations = []
train_times = []

# Regular expressions to match iteration and elapsed time
iter_pattern = r"iteration\s+(\d+)/\s+\d+"
time_pattern = r"elapsed time per iteration \(ms\):\s+([\d.]+)"

# Read input from console until EOF or empty line
print("Enter log lines (press Enter twice to finish):")
while True:
    line = input()
    if line == "":
        break
    # Extract iteration
    iter_match = re.search(iter_pattern, line)
    if iter_match:
        iterations.append(int(iter_match.group(1)))
    # Extract elapsed time
    time_match = re.search(time_pattern, line)
    if time_match:
        train_times.append(float(time_match.group(1)))

# Print the arrays
sleep(3)
print("Iterations:", iterations)
print("Train Times (ms):", train_times)
print("average train time",sum(train_times)/len(train_times) )