import json
import re

def extract_qa_sections(input_filepath, output_filepath):
    # Load the JSON data
    with open(input_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    qa_sections = []
    found_sections = 0

    # Define the start pattern more broadly and include multiple cues for Q&A starts
    start_pattern = re.compile(r"\nOperator.*?:|\nModerator.*?:", re.IGNORECASE)
    q_and_a_cues = re.compile(r"question and answer|q&a|first question|we will now begin the question|open the floor", re.IGNORECASE)

    # Pattern to avoid false positives
    false_start_pattern = re.compile(r"Welcome|Introduction", re.IGNORECASE)

    for entry in data:
        transcript = entry.get('transcript', '')

        # Start the search
        start_matches = list(start_pattern.finditer(transcript))
        valid_start_index = None

        for match in start_matches:
            start_index = match.end()
            # Check the next 1500 characters for more context
            subsequent_text = transcript[start_index:start_index + 1500]

            # Confirm it's the right area by finding Q&A cues and avoiding false starts
            if q_and_a_cues.search(subsequent_text) and not false_start_pattern.search(subsequent_text):
                valid_start_index = start_index
                break

        if valid_start_index is None:
            print(f"No valid Q&A start found for {entry['ticker']} Q{entry['quarter']} {entry['year']}")
            continue

        # Search for the end index after the valid start
        end_index = transcript.rfind("\nOperator", valid_start_index)
        if end_index != -1:
            end_index += len("\nOperator")
        else:
            print(f"End phrase not found for {entry['ticker']} Q{entry['quarter']} {entry['year']}")
            continue

        # Extract Q&A section if valid indices are found
        if valid_start_index < end_index:
            qa_text = transcript[valid_start_index:end_index].strip()
            new_entry = {
                'ticker': entry['ticker'],
                'year': entry['year'],
                'quarter': entry['quarter'],
                'transcript': qa_text
            }
            qa_sections.append(new_entry)
            found_sections += 1

    # Save the extracted Q&A sections to a new JSON file
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(qa_sections, outfile, indent=4)

    print(f"Extracted {found_sections} Q&A sections, saved to {output_filepath}")

# Example usage
input_file_path = '/Users/olivierhagoort/Desktop/University/Year 3/Semester 2/Thesis/Data Collection/Trial/labeled_earnings_call_transcripts_2013_2023(Ordered).json'
output_file_path = '/Users/olivierhagoort/Desktop/Measuring-Corporate-Culture-Using-Machine-Learning-master/Data_converters/output_qa_sections.json'
extract_qa_sections(input_file_path, output_file_path)
