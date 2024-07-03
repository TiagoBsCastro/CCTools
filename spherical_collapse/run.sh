#!/bin/bash

# Input file containing the parameters (first 8 columns)
input_file="cosmologies.txt"

# Initialize arrays for each parameter
param1=()
param2=()
param3=()
param4=()
param5=()
param6=()
param7=()
param8=()

# Read the input file and populate the arrays
while IFS=' ' read -r p1 p2 p3 p4 p5 p6 p7 p8; do
    param1+=("$p1")
    param2+=("$p2")
    param3+=("$p3")
    param4+=("$p4")
    param5+=("$p5")
    param6+=("$p6")
    param7+=("$p7")
    param8+=("$p8")
done < "$input_file"

# Function to run the Python script with a row's parameters and the row number
run_python_script() {
    local p1="$1"
    local p2="$2"
    local p3="$3"
    local p4="$4"
    local p5="$5"
    local p6="$6"
    local p7="$7"
    local p8="$8"
    local index="$9"
    python deltac.py "$p1" "$p2" "$p3" "$p4" "$p5" "$p6" "$p7" "$p8" "$index"
}

export -f run_python_script

# Use parallel to run the script for each index
parallel -j 96 run_python_script ::: "${param1[@]}" :::+ "${param2[@]}" :::+ "${param3[@]}" :::+ "${param4[@]}" :::+ "${param5[@]}" :::+ "${param6[@]}" :::+ "${param7[@]}" :::+ "${param8[@]}" :::+ $(seq 0 ${#param1[@]})
