import os

# Set the input and output directories
input_dir = "./experiments/aruco-x-offset-gaussian-filter-9-3-mean-filter-250-3/data"
output_dir = "./experiments/aruco-x-offset-negative-100-col1gauss-9-3-col23mean-250-3/"

# Iterate over each file in the input directory
for filename in os.listdir(input_dir):
  # Open the input file for reading
  with open(os.path.join(input_dir, filename), "r") as input_file:
    # Open the output file for writing
    with open(os.path.join(output_dir, filename), "w") as output_file:
      # Iterate over each line in the input file
      for line in input_file:
        # Split the line into columns
        col1, col2, col3 = line.strip().split()
        
        # Replace 0 with -100 in the second column
        print(col2)
        if col2 == "0.000000000000000000e+00":
          col2 = "-100"
        if col3 == "0.000000000000000000e+00":
          col3 = "-100"
        
        # Write the modified line to the output file
        output_file.write(f"{col1} {col2} {col3}\n")
