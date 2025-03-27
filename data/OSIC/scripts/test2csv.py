"""
Module for converting text files to CSV files.
"""

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
        with open(input_txt_file, 'r', encoding='utf-8') as txt_file, \
             open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(txt_file, delimiter=delimiter)
            writer = csv.writer(csv_file)
            for row in reader:
                writer.writerow(row)
        print(f"Conversion successful! CSV saved to {output_csv_file}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except IOError as e:
        print(f"I/O error: {e}")
    except csv.Error as e:
        print(f"CSV error: {e}")
    except Exception as e:  # Optional: Keep this for unexpected errors
        print(f"Unexpected error during conversion: {e}")

if __name__ == "__main__":
    INPUT_TXT = "input.txt"  # Replace with your .txt file path
    OUTPUT_CSV = "output.csv"  # Replace with your desired .csv file path
    txt_to_csv(INPUT_TXT, OUTPUT_CSV)
