import json

def convert_json_to_single_paragraph_transcripts(input_file, output_file):
    # Load the data from the JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Open the output text file
    with open(output_file, 'w') as file:
        for item in data:
            # Remove new lines within the transcript and write it as one paragraph
            transcript = item["transcript"].replace('\n', ' ')
            file.write(f'{transcript}\n')  # Only one newline to separate entries

# Full path specification for where your files are stored
input_path = '/Users/olivierhagoort/Desktop/University/Year 3/Semester 2/Thesis/Data Collection/Trial/labeled_earnings_call_transcripts_2013_2023(Ordered).json'  # Adjust with the actual full path
output_path = '/Users/olivierhagoort/Desktop/Measuring-Corporate-Culture-Using-Machine-Learning-master/data/document_test.txt'  # Adjust with the actual full path

convert_json_to_single_paragraph_transcripts(input_path, output_path)
