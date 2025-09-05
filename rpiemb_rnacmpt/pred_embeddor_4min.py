import subprocess
import re
import pandas as pd

rpi_embeddor = ["python3", "src/inference.py"]

# Step 1: Load the CSV file into a pandas DataFrame
# df = pd.read_csv('maxprobe_seqs_extracted.csv')

# Step 2: Function that processes the RNA and Protein sequences
# def process_sequences(rna_seq, protein_seq):

    # return prediction


# Step 3: Iterate over the rows and apply the process to each RNA and Protein sequence
# df['RPIEmbeddor_pred'] = df.apply(lambda row: process_sequences(row['rna_seq'], row['protein_seq']), axis=1)

# Step 4: Save the new dataframe with the additional column to a CSV
# df.to_csv(, index=False)































import csv

# Step 1: Read the CSV file
input_file = 'minprobe_seqs_extracted.csv'
output_file = 'rpiembeddor_minprobes_preds.csv'

def process_sequences(rna_seq, protein_seq):
    # Setting up call for RPIEmbeddor
    process = subprocess.Popen(rpi_embeddor, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, cwd=r'/home/fr/fr_fr/fr_jg590/rpi-main', text=True)
    # Specify first stdin
    process.stdin.write(protein_seq + "\n")
    process.stdin.flush()
    # Specify second stdin
    process.stdin.write(rna_seq + "\n")
    process.stdin.flush()
    # Get stdout
    stdout, stderr = process.communicate()
    print(stdout)

    # Adapt prediction text to add it to result_table file
    if stdout != None:
        # Handle the case when for any reason there was stdout but with no prediction in it
        positive_pred = stdout.find("POSITIVE")
        negative_pred = stdout.find("NEGATIVE")
        if positive_pred != -1:
            prediction = 1
        elif negative_pred != -1:
            prediction = 0
        # Case both variables are -1, neither positive nor negative found in any of the variables
        else:
            prediction = "N/A Pred"
        # prediction = stdout
    elif stderr != None:
        prediction = stderr
    return prediction

# Read the input CSV
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile, delimiter=',')  # Adjust delimiter if necessary
    headers = next(reader)  # Read header row
    data = list(reader)  # Read the remaining rows

# Step 2: Add new column header
headers.append("RPIEmbeddor_pred")

# Step 3: Process each row
for row in data:
    rna_seq = row[1] # Assuming 'rna_seq' is the second column
    protein_seq = row[2]  # Assuming 'protein_seq' is the third column
    row.append(process_sequences(rna_seq, protein_seq))

# Step 4: Write the updated data to a new CSV file
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(headers)  # Write header row
    writer.writerows(data)  # Write processed rows

print("CSV with RPIEmbeddor_pred column has been saved.")
