import csv

def txt_to_csv(input_txt_file, output_csv_file, delimiter="\t"):
    """
    Converts a text file to a CSV file.

    Args:
        input_txt_file (str): Path to the input .txt file.
        output_csv_file (str): Path to the output .csv file.
        delimiter (str): Delimiter used in the .txt file (default is tab).
    """
    try:
        with open(input_txt_file, 'r') as txt_file, open(output_csv_file, 'w', newline='') as csv_file:
            reader = csv.reader(txt_file, delimiter=delimiter)
            writer = csv.writer(csv_file)
            for row in reader:
                writer.writerow(row)
        print(f"Conversion successful! CSV saved to {output_csv_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    input_txt = "input.txt"  # Replace with your .txt file path
    output_csv = "output.csv"  # Replace with your desired .csv file path
    txt_to_csv(input_txt, output_csv)