def create_document_ids_from_transcripts(input_file, output_file):
    # Read the document
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Filter lines that contain actual transcripts
    transcripts = [line for line in lines if "transcript" in line and line.strip()]

    # Generate document IDs for actual transcripts
    with open(output_file, 'w') as file:
        for i, transcript in enumerate(transcripts, start=1):
            file.write(f'{i}.F\n')

# Usage
input_document_path = '/Users/olivierhagoort/Desktop/University/Year 3/Semester 2/Thesis/Data Collection/Trial/labeled_earnings_call_transcripts_2013_2023(Ordered).json'  # Modify with the actual path to your document
output_document_ids_path = '/Users/olivierhagoort/Desktop/Measuring-Corporate-Culture-Using-Machine-Learning-master/data/document_ids.txt'  # Modify with the desired output path

create_document_ids_from_transcripts(input_document_path, output_document_ids_path)
