#!/bin/bash

# Input file containing the parameters (first 7 columns)
input_file="cosmologies.txt"

# Function to run the Python script with a row's parameters and the row number
run_python_script() {
    local line="$1"
    local row_number="C$2"
    python your_script.py $line $row_number
}

export -f run_python_script

# Read the input file and pass each line with its row number to parallel
awk '{print $0, NR}' "$input_file" | parallel -j 96 --colsep ' ' run_python_script {1..8} {9}

