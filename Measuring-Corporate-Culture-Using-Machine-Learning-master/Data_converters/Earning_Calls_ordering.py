import json

def sort_earnings_transcripts(input_file_path, output_file_path):
    """
    Sorts the earnings transcripts in a JSON file by year and quarter within each company (ticker).

    Parameters:
        input_file_path (str): Path to the input JSON file containing earnings transcripts.
        output_file_path (str): Path to save the sorted JSON data.
    """
    # Load the JSON data from the file specified in 'input_file_path'
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    # Organize data by ticker, then sort each group by year and quarter
    sorted_data = []
    tickers = sorted(set(item['ticker'] for item in data))  # Get unique tickers
    for ticker in tickers:
        entries = [item for item in data if item['ticker'] == ticker]  # Filter entries by ticker
        entries.sort(key=lambda x: (x['year'], x['quarter']))  # Sort entries by year and quarter
        sorted_data.extend(entries)  # Add sorted entries back to the list

    # Save the sorted data to a new JSON file specified in 'output_file_path'
    with open(output_file_path, 'w') as file:
        json.dump(sorted_data, file, indent=4)

    print(f"Data sorted and saved to {output_file_path}")

# Example usage:
input_path = '/Users/olivierhagoort/Desktop/University/Year 3/Semester 2/Thesis/Data Collection/Trial/Test_Set.json'  # Specify the path to your input JSON file
output_path = '/Users/olivierhagoort/Desktop/Measuring-Corporate-Culture-Using-Machine-Learning-master/Data_converters/Q&A_Earning_Calls.json'  # Specify the path where you want the sorted data saved

sort_earnings_transcripts(input_path, output_path)
