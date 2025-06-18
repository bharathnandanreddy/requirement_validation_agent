import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid


# Google Cloud Libraries
from google.cloud import storage
from google.cloud.aiplatform_v1beta1.types import content

# LangChain Components
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader # You can add others like CSVLoader, etc.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.schema import Document

# --- Configuration ---
# IMPORTANT: Replace these with your actual GCP Project ID, Region, and GCS Bucket.
PROJECT_ID = "hacker2025-team-97-dev" # e.g., "my-awesome-project-12345"
REGION = "us-central1"              # e.g., "us-central1", "europe-west4"
GCS_BUCKET_NAME = "hacker2025-team-97-dev.appspot.com" # e.g., "my-reåquirements-bucket"


# File paths within your GCS bucket
SOURCE_REQUIREMENT_FOLDER = "source documents/"  # Or .pdf, .docx
RESPONSE_DOCUMENT_FILE = "response documents/A2Airbus_design.docx"      # Or .pdf, .docx

# --- Model Configuration ---
LLM_MODEL_NAME = "gemini-2.0-flash-001"
EMBEDDING_MODEL_NAME = "text-embedding-005" # For Vertex AI Embeddings
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 2048

# --- Text Splitter Configuration ---.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Initialize GCP Clients ---
storage_client = storage.Client(project=PROJECT_ID)

# --- Helper Functions ---

def download_file_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str) -> str:
    """Downloads a blob from the bucket to a local file."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")
    return destination_file_name

def load_document(file_path: str) -> List[Document]:
    """Loads content from a local file based on its extension."""
    file_extension = Path(file_path).suffix.lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path) # Use python-docx or docx2txt
    elif file_extension == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Only .pdf, .docx, .txt are supported.")
    return loader.load()

def initialize_llms_and_embeddings() -> (VertexAI, VertexAI):
    """Initializes Vertex AI LLM and Embeddings models."""
    llm = VertexAI(
        model_name=LLM_MODEL_NAME,
        project=PROJECT_ID,
        location=REGION,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        generation_config={
            "candidate_count": 1 # We usually only need one response for this task
        }
    )
    embeddings = VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        project=PROJECT_ID,
        location=REGION,
    )
    return llm, embeddings

def extract_requirements(llm: VertexAI, document_chunks: List[Document]) -> List[Dict[str, str]]:
    """
    Extracts structured requirements from document chunks using an LLM.
    The LLM is prompted to output a JSON list of requirements.
    """
    print("\n--- Step 2: Extracting Requirements ---")
    extracted_requirements = []
    
    extraction_prompt = PromptTemplate(
        input_variables=["text_chunk"],
        template="""You are an expert aerospace engineer tasked with identifying technical requirements.
        Carefully read the following text and extract all distinct engineering requirements.
        Each requirement must be a complete, concise, and verifiable statement.
        Output your findings as a JSON array, where each element is an object with a 'id' (e.g., 'REQ-001', 'NFR-001') and a 'text' field.
        If no requirements are found, return an empty JSON array [].

        Example of desired output:
        ```json
        [
            {{
                "id": "REQ-001",
                "text": "The system shall autonomously navigate from point A to point B."
            }},
            {{
                "id": "NFR-PERF-001",
                "text": "The system shall complete tasks within 10 minutes."
            }}
        ]
        ```

        Text to analyze:
        {text_chunk}
        """
    )
    extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)

    # Process chunks in batches or sequentially
    for i, chunk in enumerate(document_chunks):
        print(f"Processing chunk {i+1}/{len(document_chunks)} for requirement extraction...")
        try:
            raw_llm_output = extraction_chain.run(text_chunk=chunk.page_content)
            # Attempt to parse JSON. Gemini sometimes adds markdown.
            json_str = raw_llm_output.strip().replace("```json\n", "").replace("\n```", "")
            
            # Simple validation to ensure it's a list before extending
            temp_requirements = json.loads(json_str)
            if isinstance(temp_requirements, list):
                extracted_requirements.extend(temp_requirements)
            else:
                print(f"Warning: LLM output for chunk {i+1} was not a list: {json_str}")

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM output for chunk {i+1}: {e}")
            print(f"Raw LLM output: {raw_llm_output}")
        except Exception as e:
            print(f"An unexpected error occurred during extraction for chunk {i+1}: {e}")

    # Deduplicate based on 'id' if there's overlap in chunks
    unique_requirements = {}
    for req in extracted_requirements:
        if 'id' in req and 'text' in req:
            unique_requirements[req['id']] = req['text']
    
    final_requirements = [{"id": req_id, "text": req_text} for req_id, req_text in unique_requirements.items()]
    print(f"Extracted {len(final_requirements)} unique requirements.")
    return final_requirements


def create_vector_store(embeddings: VertexAIEmbeddings, document_chunks: List[Document]) -> Chroma:
    """Creates a Chroma vector store from document chunks."""
    print("\n--- Step 3: Creating Vector Store for Response Document ---") #.
    vectorstore = Chroma.from_documents(documents=document_chunks, embedding=embeddings)
    print("Vector store created.")
    return vectorstore

def validate_requirement( #.
    llm: VertexAI,
    vectorstore: Chroma,
    requirement: Dict[str, str]
) -> Dict[str, Any]:
    """
    Validates a single requirement against the response document using semantic search
    and an LLM for final judgment.
    """
    req_id = requirement['id']
    req_text = requirement['text']
    print(f"Validating {req_id}: {req_text}")

    # Retrieve semantically similar chunks from the response document
    retrieved_docs = vectorstore.similarity_search(req_text, k=3) # Get top 3 relevant chunks
    response_snippets = "\n\n".join([doc.page_content for doc in retrieved_docs])

    validation_prompt = PromptTemplate(
        input_variables=["requirement_id", "requirement_text", "response_snippets"],
        template="""You are an AI assistant specialized in validating engineering requirements against response documents.
        Given a specific requirement and relevant snippets from a response document, analyze the coverage.

        Requirement ID: {requirement_id}
        Requirement Text: {requirement_text}

        Relevant Response Snippets:
        ---
        {response_snippets}
        ---

        Based on the provided snippets, determine if the requirement is:
        - **COMPLETE**: Fully addressed with clear evidence.
        - **PARTIAL**: Partially addressed, some aspects are covered but others are missing or unclear.
        - **MISSING**: Not addressed at all by the provided snippets.
        - **CONTRADICTORY**: The snippets directly conflict with the requirement.

        Provide your analysis in a JSON format:
        {{
            "id": "{requirement_id}",
            "status": "COMPLETE" | "PARTIAL" | "MISSING" | "CONTRADICTORY",
            "explanation": "Detailed explanation of why this status was assigned, referencing specific parts of the requirement and snippets.",
            "remediation_suggestion": "Suggest specific actions to fully address or clarify the requirement, if applicable. If complete, state 'N/A'."
        }}
        """
    )
    validation_chain = LLMChain(llm=llm, prompt=validation_prompt)

    try:
        raw_llm_output = validation_chain.run(
            requirement_id=req_id,
            requirement_text=req_text,
            response_snippets=response_snippets
        )
        json_str = raw_llm_output.strip().replace("```json\n", "").replace("\n```", "")
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from validation LLM output for {req_id}: {e}")
        print(f"Raw LLM output: {raw_llm_output}")
        return {
            "id": req_id,
            "status": "ERROR_PARSING_LLM_OUTPUT",
            "explanation": f"Failed to parse LLM response: {e}. Raw output: {raw_llm_output}",
            "remediation_suggestion": "Manual review required."
        }
    except Exception as e:
        print(f"An unexpected error occurred during validation for {req_id}: {e}")
        return {
            "id": req_id,
            "status": "ERROR_UNEXPECTED",
            "explanation": f"An unexpected error occurred: {e}",
            "remediation_suggestion": "Manual review required."
        }

def generate_validation_report(validation_results: List[Dict[str, Any]]) -> str:
    """Generates a formatted summary report."""
    print("\n--- Step 5: Generating Validation Report ---")
    report_lines = ["# Requirement Validation Report\n"]
    
    # Summary
    complete_count = sum(1 for r in validation_results if r.get('status') == 'COMPLETE')
    partial_count = sum(1 for r in validation_results if r.get('status') == 'PARTIAL')
    missing_count = sum(1 for r in validation_results if r.get('status') == 'MISSING')
    contradictory_count = sum(1 for r in validation_results if r.get('status') == 'CONTRADICTORY')
    error_count = sum(1 for r in validation_results if 'ERROR' in r.get('status', ''))

    report_lines.append(f"## Summary\n")
    report_lines.append(f"- Total Requirements Analyzed: {len(validation_results)}")
    report_lines.append(f"- Complete: {complete_count}")
    report_lines.append(f"- Partial: {partial_count}")
    report_lines.append(f"- Missing: {missing_count}")
    report_lines.append(f"- Contradictory: {contradictory_count}")
    report_lines.append(f"- Errors: {error_count}\n")

    # Detailed Results
    report_lines.append("## Detailed Results\n")
    for result in validation_results:
        report_lines.append(f"### Requirement ID: {result.get('id', 'N/A')}")
        report_lines.append(f"**Status:** {result.get('status', 'UNKNOWN')}")
        report_lines.append(f"**Explanation:** {result.get('explanation', 'No explanation provided.')}")
        report_lines.append(f"**Remediation Suggestion:** {result.get('remediation_suggestion', 'N/A')}\n")
        report_lines.append("---\n") # Separator
    
    return "\n".join(report_lines)


# --- Main Execution Flow ---
def main():
    print(f"Starting Requirement Validation Agent for Project: {PROJECT_ID}, Region: {REGION}")
    print(f"GCS Bucket: {GCS_BUCKET_NAME}")

    # Create a temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = "downloads"
        os.makedirs(tmp_dir_path, exist_ok=True)

        print("\n--- Step 1: Document Ingestion and Preprocessing ---")
        try:

            print("\n--- Step 1A: Fetching all requirement files ---")
            # Downloading source documents from GCS
            source_docs = []
            blobs = storage_client.list_blobs(GCS_BUCKET_NAME, prefix=SOURCE_REQUIREMENT_FOLDER)

            for blob in blobs:
                if blob.name.endswith("/") or not any(blob.name.endswith(ext) for ext in [".pdf", ".docx", ".txt"]):
                    continue  # Skip folders or unsupported files
                print(f"Processing {blob.name}")
                unique_suffix = uuid.uuid4().hex[:8]
                original_name = Path(blob.name).name
                unique_name = f"{unique_suffix}_{original_name}"
                local_filename = tmp_dir_path+"/"+unique_name
                try:
                    download_file_from_gcs(GCS_BUCKET_NAME, blob.name, str(local_filename))
                    docs = load_document(str(local_filename))
                    source_docs.extend(docs)
                except Exception as e:
                    print(f"Failed to process {blob.name}: {e}")

            print(f"Loaded {len(source_docs)} documents from {SOURCE_REQUIREMENT_FOLDER}")

            # Split all documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )
            source_chunks = text_splitter.split_documents(source_docs)
            print(f"Total chunks from all requirement documents: {len(source_chunks)}")
            # Download response documents from GCS
            response_local_path = "downloads/response.docx"
            download_file_from_gcs(GCS_BUCKET_NAME, RESPONSE_DOCUMENT_FILE, str(response_local_path))

            # Load and split documents
            response_docs = load_document(str(response_local_path))

            response_chunks = text_splitter.split_documents(response_docs)
            print(f"Response document split into {len(response_chunks)} chunks.")

        except Exception as e:
            print(f"Error during document ingestion: {e}")
            return

        # Initialize LLM and Embeddings
        try:
            llm, embeddings = initialize_llms_and_embeddings() #.¸¸
        except Exception as e:
            print(f"Error initializing Vertex AI models. Check your PROJECT_ID, REGION, and API permissions: {e}")
            return

        # Extract requirements from the source document
        extracted_requirements = extract_requirements(llm, source_chunks)
        if not extracted_requirements:
            print("No requirements extracted. Exiting.")
            return

        # Create vector store from the response document
        response_vector_store = create_vector_store(embeddings, response_chunks)

        # Perform validation for each extracted requirement
        print("\n--- Step 4: Comparing and Validating Requirements ---")
        validation_results = []
        for req in extracted_requirements:
            result = validate_requirement(llm, response_vector_store, req)
            validation_results.append(result)
            print(f"  -> {result.get('id')}: {result.get('status')}")

        # Generate and print the final report
        final_report = generate_validation_report(validation_results)
        print("\n" + "="*80)
        print("FINAL VALIDATION REPORT")
        print("="*80)
        print(final_report)

        # Optionally save the report to a file
        report_output_path = "validation_report.md"
        with open(report_output_path, "w") as f:
            f.write(final_report)
        print(f"\nReport saved to {report_output_path}")

if __name__ == "__main__":
    main()