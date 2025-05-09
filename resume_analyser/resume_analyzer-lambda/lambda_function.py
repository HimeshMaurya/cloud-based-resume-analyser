import os
import json
import base64
from io import BytesIO
import logging
import tempfile
import uuid # For generating unique submission IDs
import datetime # For adding a timestamp

import boto3
from multipart import parse_form_data

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
# Initialize clients outside the handler for potential performance benefits
# (reuses connections across invocations if Lambda container is warm)
try:
    textract = boto3.client('textract')
    comprehend = boto3.client('comprehend')
    # Using the client interface for DynamoDB put_item
    dynamodb_client = boto3.client('dynamodb')
    # If you prefer the resource interface for put_item, you can use this instead:
    # dynamodb_resource = boto3.resource('dynamodb')
    # table = dynamodb_resource.Table('YourApplicantsTable') # Initialize table resource here
    logger.info("AWS clients initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize AWS clients: {e}")
    # Depending on your error handling strategy, you might raise an exception
    # here or handle it within the handler.

# --- Configuration ---
DYNAMODB_TABLE_NAME = 'YourApplicantsTable' # <<<< !!! REPLACE WITH YOUR TABLE NAME !!! >>>>
# ---------------------


def lambda_handler(event, context):
    """
    AWS Lambda handler function to process resume and job description,
    calculate a match score using Textract and Comprehend,
    log Comprehend responses, and store results in DynamoDB.

    1. Decode & parse multipart/form-data (resume PDF, jd_text).
    2. Save PDF to /tmp & read bytes.
    3. Call Textract.detect_document_text on PDF bytes.
    4. Run Comprehend key-phrase detection on extracted text and jd_text.
    5. Log Comprehend responses to CloudWatch.
    6. Compute overlap score based on matching key phrases.
    7. Store submission details, score, and matches in DynamoDB.
    8. Return JSON score and details.
    """
    logger.info("Lambda function started.")

    # --- 1. Decode the (possibly base64) request body ---
    raw_body = event.get('body', '')
    if event.get('isBase64Encoded', False):
        try:
            body_bytes = base64.b64decode(raw_body)
            logger.info("Request body was Base64 encoded, decoded.")
        except Exception as e:
             logger.error(f"Base64 decode error: {e}")
             return {'statusCode': 400, 'body': json.dumps({'message': 'Error decoding request body'})}
    else:
        body_bytes = raw_body.encode('utf-8')
        logger.info("Request body was not Base64 encoded, encoded to utf-8.")

    content_type = event['headers'].get('content-type', '')
    logger.info(f"Content-Type: {content_type}")

    # --- 2. Build minimal WSGI environ for parse_form_data ---
    # The 'multipart' library expects a WSGI-like environment
    environ = {
        'REQUEST_METHOD': 'POST',
        'CONTENT_TYPE': content_type,
        'CONTENT_LENGTH': str(len(body_bytes)),
        'wsgi.input': BytesIO(body_bytes),
    }
    logger.info("Prepared WSGI environment for multipart parsing.")

    # --- 3. Parse form fields & file uploads ---
    resume_part = None
    jd_text = None
    try:
        form, files = parse_form_data(environ)

        # Validate required fields are present
        if 'resume' not in files:
            logger.error("Missing 'resume' file in multipart data.")
            return {'statusCode': 400, 'body': json.dumps({'message': 'Missing "resume" file part.'})}
        if 'jd_text' not in form:
            logger.error("Missing 'jd_text' field in multipart data.")
            return {'statusCode': 400, 'body': json.dumps({'message': 'Missing "jd_text" form field.'})}

        resume_part = files['resume']
        jd_text = form['jd_text']
        logger.info(f"Parsed form data. Resume filename: {resume_part.filename}, JD text length: {len(jd_text) if jd_text else 0} characters.")

    except Exception as e:
        logger.error(f"Error parsing multipart data: {e}")
        return {'statusCode': 400, 'body': json.dumps({'message': f'Error parsing form data: {e}'})}

    # --- 4. Save PDF locally & read its bytes (for Textract) ---
    # Using tempfile is recommended for safer temporary file handling
    pdf_bytes = None
    tmp_path = None # Define tmp_path outside try for finally block
    try:
        # Create a temporary file that will be automatically deleted when closed,
        # but we need delete=False because Textract needs to read bytes after we write them.
        # We will clean up manually in the finally block.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
             tmp_path = tmp.name
             logger.info(f"Saving PDF to temporary path: {tmp_path}")
             tmp.write(resume_part.file.read())
             logger.info(f"Successfully wrote PDF content to {tmp_path}.")

        # Re-open the temp file to read its bytes for the Textract API call
        with open(tmp_path, 'rb') as f:
             pdf_bytes = f.read()
        logger.info(f"Read {len(pdf_bytes)} bytes from PDF.")

    except Exception as e:
         logger.error(f"Error saving/reading PDF file from temporary path: {e}")
         return {'statusCode': 500, 'body': json.dumps({'message': f'Error processing PDF file: {e}'})}
    finally:
         # Clean up the temporary file
         if tmp_path and os.path.exists(tmp_path):
             try:
                 os.remove(tmp_path)
                 logger.info(f"Cleaned up temporary file: {tmp_path}")
             except Exception as cleanup_e:
                 logger.error(f"Error cleaning up temporary file {tmp_path}: {cleanup_e}")


    # --- 5. Synchronous text detection via Textract ---
    extracted_text = ""
    if pdf_bytes: # Only proceed if PDF bytes were successfully read
        try:
            logger.info("Calling Textract detect_document_text...")
            resp = textract.detect_document_text(
                Document={'Bytes': pdf_bytes}
            )
            logger.info("Textract call complete.")

            # Collect only LINE blocks
            lines = [
                blk['Text']
                for blk in resp.get('Blocks', [])
                if blk.get('BlockType') == 'LINE' and 'Text' in blk # Ensure 'Text' key exists
            ]
            extracted_text = "\n".join(lines)
            logger.info(f"Extracted {len(lines)} lines from PDF (total text length: {len(extracted_text)}).")

        except Exception as e:
            logger.error(f"Error during Textract processing: {e}")
            return {'statusCode': 500, 'body': json.dumps({'message': f'Error extracting text from PDF: {e}'})}
    else:
         logger.error("PDF bytes not available after reading from temp file.")
         return {'statusCode': 500, 'body': json.dumps({'message': 'Failed to read PDF bytes for Textract.'})}


    # --- 6. Comprehend key-phrase detection ---
    rpt = {}
    jpt = {}
    try:
        # Check if extracted text or jd_text are empty before calling Comprehend
        if not extracted_text:
             logger.warning("Extracted text from resume is empty.")
        else:
            logger.info("Calling Comprehend detect_key_phrases for Resume...")
            # Max text size for Comprehend sync operations is 5000 bytes.
            # If extracted_text is larger, you might need to split it or use async.
            # For simplicity here, we'll truncate if needed (basic handling).
            text_to_comprehend = extracted_text[:5000]
            if len(extracted_text) > 5000:
                logger.warning(f"Truncated resume text for Comprehend from {len(extracted_text)} to 5000 bytes.")
            rpt = comprehend.detect_key_phrases(Text=text_to_comprehend, LanguageCode='en')
            logger.info("Comprehend call for Resume complete.")

        if not jd_text:
             logger.warning("Job description text is empty.")
        else:
            logger.info("Calling Comprehend detect_key_phrases for Job Description...")
            # Apply similar truncation/check for JD text if needed
            jd_text_to_comprehend = jd_text[:5000]
            if len(jd_text) > 5000:
                 logger.warning(f"Truncated JD text for Comprehend from {len(jd_text)} to 5000 bytes.")
            jpt = comprehend.detect_key_phrases(Text=jd_text_to_comprehend, LanguageCode='en')
            logger.info("Comprehend call for Job Description complete.")

    except Exception as e:
         logger.error(f"Error during Comprehend processing: {e}")
         return {'statusCode': 500, 'body': json.dumps({'message': f'Error processing text with Comprehend: {e}'})}


    # --- 7. Log Comprehend Responses (to CloudWatch) ---
    logger.info("\n--- Raw Comprehend Response (Resume) ---")
    logger.info(json.dumps(rpt, indent=2)) # Use json.dumps for pretty printing
    logger.info("---------------------------------------\n")

    logger.info("\n--- Raw Comprehend Response (Job Description) ---")
    logger.info(json.dumps(jpt, indent=2)) # Use json.dumps for pretty printing
    logger.info("-----------------------------------------------\n")


    # --- 8. Compute key phrase overlap score ---
    score = 0
    matches = set()
    try:
        res_key_phrases = {p['Text'].lower() for p in rpt.get('KeyPhrases', []) if 'Text' in p}
        jd_key_phrases  = {p['Text'].lower() for p in jpt.get('KeyPhrases', []) if 'Text' in p}

        matches = res_key_phrases & jd_key_phrases # Set intersection
        score = len(matches)

        logger.info(f"Resume key phrases count: {len(res_key_phrases)}")
        logger.info(f"JD key phrases count: {len(jd_key_phrases)}")
        logger.info(f"Matching key phrases count: {score}")

    except Exception as e:
        logger.error(f"Error computing score: {e}")
        # Decide how to handle: return error or return score 0?
        # For now, we'll log and continue, returning potentially score 0.
        pass


    # --- 9. Store submission details in DynamoDB ---
    submission_id = str(uuid.uuid4()) # Generate a unique ID for this submission
    timestamp = datetime.datetime.utcnow().isoformat() # Add a timestamp

    # Prepare the item for DynamoDB PutItem API call
    # Need to use DynamoDB AttributeValue types ('S', 'N', 'L' etc.)
    dynamodb_item = {
        'submissionId': {'S': submission_id}, # Partition Key (String type)
        'timestamp': {'S': timestamp},        # Timestamp (String type)
        'score': {'N': str(score)},           # Score (Number type - must be string)
        'matchingPhrases': {'L': [{'S': phrase} for phrase in list(matches)]}, # Matching phrases (List of Strings)
        'extractedResumeText': {'S': extracted_text}, # Extracted resume text (String)
        'jobDescriptionText': {'S': jd_text} # Job description text (String)
        # Add other fields if collected/extracted (e.g., applicantName, email)
    }

    try:
        logger.info(f"Attempting to store item {submission_id} in DynamoDB table {DYNAMODB_TABLE_NAME}...")
        response = dynamodb_client.put_item( # Use dynamodb_client
            TableName=DYNAMODB_TABLE_NAME,
            Item=dynamodb_item
        )
        logger.info(f"Successfully stored item {submission_id} in DynamoDB. Response: {response}")
        # You can optionally add the submission_id to the response returned to the user
        # This could be useful for retrieval later
        # result_payload['submission_id'] = submission_id

    except Exception as e:
        logger.error(f"Critical Error storing item {submission_id} in DynamoDB table {DYNAMODB_TABLE_NAME}: {e}")
        # If saving to DB is critical, you might return a 500 error here.
        # For now, we'll log the critical error but still return the score to the user.
        # Depending on requirements, you might want to send a notification or retry.
        pass # Continue returning the score even if saving fails


    # --- 10. Return the scoring result as JSON ---
    # Prepare the response payload for the API Gateway/Function URL
    result_payload = {
        'score': score,
        'matching_phrases': list(matches), # Convert set back to list for JSON response
        'message': f'Match Score: {score}'
    }
    # Optional: Add the submission ID to the response
    # result_payload['submission_id'] = submission_id


    logger.info(f"Returning final JSON response with score: {score}")

    return {
        'statusCode': 200, # Indicate success
        'headers': {
            # CORS headers should ideally be configured on the Function URL or API Gateway
            # but can be added here if not using console config (less recommended)
            # 'Access-Control-Allow-Origin': '*', # Example: allows requests from any origin
            # 'Access-Control-Allow-Headers': 'Content-Type',
            'Content-Type': 'application/json' # Specify the content type of the response body
        },
        'body': json.dumps(result_payload) # The response body must be a string
    }