import json

def create_firm_document_mapping(input_file, output_file):
    # Load JSON data from file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Open the output file
    with open(output_file, 'w') as file:
        # Write headers
        file.write("document_ID,firm_ID,time_of_earning_call\n")

        # Initialize a counter for document IDs
        document_id = 1

        # Process each entry in the JSON data
        for entry in data:
            # Extract the ticker symbol, year, and quarter
            firm_id = entry["ticker"]
            year = entry["year"]
            quarter = entry["quarter"]

            # Construct the time of the earning call string
            time_of_call = f"{year} Q{quarter}"

            # Write the data to the output file
            file.write(f"{document_id}.F,{firm_id},{time_of_call}\n")

            # Increment the document ID
            document_id += 1

# Usage
input_json_path = '/Users/olivierhagoort/Desktop/University/Year 3/Semester 2/Thesis/Data Collection/Trial/labeled_earnings_call_transcripts_2013_2023(Ordered).json'  # Modify with the actual path to your JSON data file
output_csv_path = '/Users/olivierhagoort/Desktop/Measuring-Corporate-Culture-Using-Machine-Learning-master/data/id2firms.csv'  # Modify with the desired output path

create_firm_document_mapping(input_json_path, output_csv_path)
