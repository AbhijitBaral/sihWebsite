import os
import re
import csv
import PyPDF2
from transformers import pipeline

# Step 1: Extract Text from PDF or Text File
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        text = file.read()
    return text

# Step 2: Sentence Segmentation (Simple approach using regex)
def split_into_sentences(text):
    # This is a basic sentence segmentation. You can improve this with more complex NLP techniques.
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

# Step 3: Generate Multiple Questions from Sentences
def generate_multiple_questions(sentences):
    # Using a question generation model with beam search for generating multiple questions
    question_generator = pipeline('text2text-generation', model='mrm8488/t5-base-finetuned-question-generation-ap')
    
    qna_pairs = []
    for sentence in sentences:
        # Generate multiple questions using beam search
        questions = question_generator(
            f"generate questions: {sentence}",
            num_return_sequences=3,  # Generate 3 questions
            num_beams=10,
            # num_beams=5,               # Beam search with 5 beams
            max_new_tokens=50          # Limit the length of generated questions
        )
        # for question in questions:
        #     qna_pairs.append((question['generated_text'], sentence))

        for question in questions:
            # Remove the "question:" prefix if present
            clean_question = question['generated_text'].replace("question:", "").strip()
            qna_pairs.append((clean_question, sentence))
    
    return qna_pairs

# Step 4: Save Questions and Answers to CSV
def save_to_csv(qna_pairs, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer'])  # CSV header
        for question, answer in qna_pairs:
            writer.writerow([question, answer])

def main(input_path, output_path):
    # Check file type
    file_extension = os.path.splitext(input_path)[1].lower()

    if file_extension == '.pdf':
        text = extract_text_from_pdf(input_path)
    elif file_extension == '.txt':
        text = extract_text_from_txt(input_path)
    else:
        print("Unsupported file type. Please provide a .txt or .pdf file.")
        return

    # Step 2: Split text into sentences
    sentences = split_into_sentences(text)

    # Step 3: Generate multiple questions from sentences
    qna_pairs = generate_multiple_questions(sentences)

    # Step 4: Save to CSV
    save_to_csv(qna_pairs, output_path)
    print(f"CSV file generated at {output_path}")

# Example usage
if __name__ == "__main__":
    input_file = './uni_history.txt'  # or 'input.txt'
    output_csv = 'output.csv'
    main(input_file, output_csv)
