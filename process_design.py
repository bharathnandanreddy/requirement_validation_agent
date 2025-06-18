import google.generativeai as genai
from google.cloud import storage
import base64
import json


model = genai.GenerativeModel("gemini-1.5-pro")

def load_pdf_bytes(bucket, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()

def validate_requirements(pdf_bytes: bytes, requirements: list[str], batch_size: int = 25) -> list[dict]:
    all_results = []

    for i in range(0, len(requirements), batch_size):
        batch = requirements[i:i + batch_size]

        prompt = f"""
You are an aerospace systems engineer.

Analyze the attached PDF design document and assess the coverage status for each of the following requirements. For each requirement, determine whether it is:
- Fully Covered
- Partially Covered
- Not Covered

Respond strictly in JSON array format like:
[
  {{
    "requirement": "<requirement text>",
    "status": "Fully Covered / Partially Covered / Not Covered",
    "evidence_summary": "Mention sections, figures, or content from the document that support your judgment."
  }},
  ...
]

Requirements:
{json.dumps(batch, indent=2)}
"""

        try:
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", "data": pdf_bytes},
                    {"text": prompt}
                ]
            )
            parsed = json.loads(response.text[8:-4])
            all_results.extend(parsed)
        except Exception as e:
            for req in batch:
                all_results.append({
                    "requirement": req,
                    "status": "Parsing Failed",
                    "evidence_summary": str(e)
                })

    return all_results

def process_design(requirements):
    bucket_name = "hacker2025-team-97-dev.appspot.com"
    blob_name = "response documents/sample_design.pdf"
    pdf_bytes = load_pdf_bytes(bucket_name, blob_name)
    results = validate_requirements(pdf_bytes, requirements)

    with open("output_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Done. Saved to output_results.json")
