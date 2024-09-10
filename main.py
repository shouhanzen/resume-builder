import json
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

# Load OpenAI API key from environment
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Step 1: Read the resume from a text file
def read_resume_from_file(file_path):
    """Reads resume text from a file."""
    try:
        with open(file_path, 'r') as file:
            resume_text = file.read()
        return resume_text
    except FileNotFoundError:
        print("The file was not found.")
        return None

# Step 2: Request the job details
def request_job_details():
    """Asks user for job description and job link."""
    job_description = input("Please enter the job description: \n")
    job_link = input("Please enter the job description link: \n")
    return job_description, job_link

# Step 3: Generate resume in natural text
def generate_resume_text(resume, job_description):
    """Generates a high-quality resume in natural language."""
    
    # Initialize the chat model
    model = ChatOpenAI()

    # Create a prompt template to generate the resume in text format
    prompt = PromptTemplate(
        template="Rewrite the resume to best match the job description while ensuring it passes ATS. "
                 "Make sure it is a high-quality professional resume.\n\nResume:\n{resume}\nJob Description:\n{job_description}",
        input_variables=["resume", "job_description"]
    )
    
    # Execute the chain and get the result
    result = model(prompt.invoke({"resume": resume, "job_description": job_description}))

    return result.content

# Step 4: Convert the resume into structured JSON format
def format_resume_to_json(resume_text):
    """Formats the resume into a machine-parsable JSON format."""
    
    # Initialize the chat model
    model = ChatOpenAI()

    # Create a prompt template for JSON formatting
    prompt = PromptTemplate(
        template="Convert the following resume into a structured JSON format with an array of jobs (each containing a list of accomplishments) "
                 "and a skills section (mapping skill category to a list of skills/technologies):\n\nResume:\n{resume_text}",
        input_variables=["resume_text"]
    )
    
    # Execute the chain and parse the result into JSON
    result = model(prompt.invoke({"resume_text": resume_text}))

    return json.loads(result.content)

# Step 5: Save the revised resume with the job link to a new file
def save_new_resume_to_file(new_resume, job_link, output_file):
    """Saves the revised resume in JSON format along with the job link to a new file."""
    with open(output_file, 'w') as file:
        file.write("Generated Resume (in JSON format):\n\n")
        file.write(json.dumps(new_resume, indent=4))
        file.write("\n\nJob Description Link: ")
        file.write(job_link)
    print(f"New resume saved to {output_file}")

# Main function to run the script
def main():
    resume_file_path = "resume.txt"
    
    # Step 1: Read the resume from a text file
    resume = read_resume_from_file(resume_file_path)
    if not resume:
        return
    
    # Step 2: Request the job description and job link from the user
    job_description, job_link = request_job_details()
    
    # Step 3: Generate the new resume in natural text
    resume_text = generate_resume_text(resume, job_description)
    
    # Step 4: Format the generated resume into structured JSON format
    formatted_resume = format_resume_to_json(resume_text)
    
    # Step 5: Save the new resume with the job description link to a new file
    output_file = "new_resume_with_link.json"
    save_new_resume_to_file(formatted_resume, job_link, output_file)

if __name__ == "__main__":
    main()
