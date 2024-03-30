#!/bin/bash

# Get user input for the Python script file
read -p "Enter the path to the Python script file: " script_file

# Get additional flags from the user
read -p "Enter additional flags for the script (if any): " additional_flags

# Run the Python script with user-specified file and flags, and append the output to output.md
time python "$script_file" $additional_flags >> output.md 2>&1

# Post-process output.md to remove newline characters
# sed -i ':a;N;$!ba;s/\n/ /g' output.md
