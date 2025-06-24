#csv_cleaner
#BuiltWithLove by @papitx0
import csv
import os
from collections import defaultdict

def remove_duplicates_from_csv(input_file, column='plate_number', keep='first'):
    """
    Removes duplicate rows from a CSV file based on a specified column.
    Overwrites the original CSV file with the cleaned data.

    Args:
        input_file (str): Path to the input CSV file.
        column (str): Column name to check for duplicates. Defaults to 'plate_number'.
        keep (str/bool): Strategy for keeping duplicates: 'first', 'last', or False (keep only unique rows).

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Read the CSV file
        with open(input_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames
            rows = list(reader)

        # Validate column existence
        if column not in headers:
            print(f"Error: Column '{column}' not found in the CSV file.")
            print(f"Available columns: {headers}")
            return False

        # Process rows to remove duplicates
        cleaned_rows = _process_duplicates(rows, column, keep)

        if cleaned_rows is None:
            print(f"Error: Invalid 'keep' parameter: {keep}")
            print("Valid values are: 'first', 'last', or False")
            return False

        # Overwrite the original file with cleaned data
        with open(input_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(cleaned_rows)

        # Print summary
        print(f"Original rows: {len(rows)}")
        print(f"Cleaned rows: {len(cleaned_rows)}")
        print(f"Removed rows: {len(rows) - len(cleaned_rows)}")
        print(f"Output saved to: {input_file}")

        return True

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False

def _process_duplicates(rows, column, keep):
    """
    Processes rows to remove duplicates based on the specified strategy.

    Args:
        rows (list): List of rows (dictionaries) from the CSV.
        column (str): Column name to check for duplicates.
        keep (str/bool): Strategy for keeping duplicates: 'first', 'last', or False.

    Returns:
        list: List of cleaned rows, or None if 'keep' is invalid.
    """
    if keep == 'first':
        seen = set()
        cleaned_rows = []
        for row in rows:
            value = row[column]
            if value not in seen:
                cleaned_rows.append(row)
                seen.add(value)
    elif keep == 'last':
        seen = set()
        cleaned_rows = []
        for row in reversed(rows):
            value = row[column]
            if value not in seen:
                cleaned_rows.append(row)
                seen.add(value)
        cleaned_rows.reverse()
    elif keep is False:
        value_counts = defaultdict(int)
        for row in rows:
            value_counts[row[column]] += 1
        unique_values = {value for value, count in value_counts.items() if count == 1}
        cleaned_rows = [row for row in rows if row[column] in unique_values]
    else:
        return None

    return cleaned_rows

#remove_duplicates_from_csv('x.csv')
#remove_duplicates_from_csv('x.csv', column='plate_number')
#remove_duplicates_from_csv('x.csv', column='plate_number', keep='last')



remove_duplicates_from_csv('parking_data/anpr_detections.csv', column='plate_number')
remove_duplicates_from_csv('parking_data/parking_data.csv', column='plate_number')
remove_duplicates_from_csv('parking_data/priority_queue.csv', column='plate_number')
