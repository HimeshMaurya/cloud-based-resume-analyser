# cloud-based-resume-analyser
Gemini
Resume Screening with AWS Services
New chat
My stuff
Notebooks
New notebook
Gems
Chats
AWS Certification Prep: Final Review
Pinned chat
normal general questions
Pinned chat
Resume Screening with AWS Services
Full-Stack Integration Assessment Guidance
Making a Private GitHub Repo Public
Resume Improvement Suggestions
Skin Redness After Bathing Explained
AI Learning Roadmap: From Basics to Advanced
Hair Loss Science: Causes and Solutions
Internship Presentation Prompt and Script
Laptop Crashed: Possible Causes and Solutions
YouTube Monetization Requirements Explained
Breaking the Cycle of Addiction
OnePlus 11 Moon Photography Pro Mode
Visualizing Four Dimensions in Two
Weekly Task Consolidation and Planning
Personalized Activity Suggestions
S3 to Kinesis Data Streams Solution
Religions' Doomsday Predictions and Texts
Dominance, China, and India's Tech
LOS Light: Troubleshooting Fiber Connection Issues
Network Load Balancer Target Routing Explained
Losing Interest Due to Social Conflict
Indus Waters Treaty Dispute Explained
Developer's "Holy" Database Deletion Joke
Backend Engineer: Why VectorShift?
Resume for DevOps Internship
Set YouTube Music Alarm on Devices
OnePlus Live Alerts App Support
Cloud Computing and Virtualization Overview
IaaS, PaaS, and SaaS Explained
Samal's varied contextual meanings
SAML Usage Explained
NoSQL Performance Considerations Explained
Database Types and Examples
Document vs Key-Value Databases

Settings and help
Conversation with Gemini
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



wht this do

This code is an AWS Lambda function written in Python that acts as an automated "Resume Screener." It compares a job applicant's resume (PDF) against a job description (JD) to see how well they match.

Here is the step-by-step breakdown of what happens when this code runs:

1. Receiving the Data
The function is triggered (likely by an API call) and receives a multipart form containing two things:

A PDF file (the resume).

A text string (the job description).

2. Extracting Text (AWS Textract)
Since computers can't "read" a PDF directly as text, the code sends the file to AWS Textract. Textract uses Optical Character Recognition (OCR) to convert the visual lines of the PDF into actual strings of text that the program can analyze.

3. Finding Key Phrases (AWS Comprehend)
Once it has the text from both the resume and the JD, it sends them to AWS Comprehend. This service uses Natural Language Processing (NLP) to identify important "key phrases" (like "Python," "Project Management," or "Data Analysis") rather than just looking at every single word like "the" or "and."

4. Scoring the Match
The code performs a "Set Intersection" between the resume phrases and the JD phrases.

The Logic: If the JD asks for "Cloud Computing" and the resume has "Cloud Computing," that’s a match.

The Score: The final score is simply the count of matching phrases.

5. Saving to the Database (AWS DynamoDB)
The function saves a record of the entire process into a DynamoDB table. It stores:

A unique ID for the submission.

A timestamp.

The final score.

The list of phrases that matched.

The full text of both the resume and the JD for future reference.

6. Returning the Result
Finally, it sends a JSON response back to the user (or the website that called it) containing the score and the list of matching keywords.

Summary of AWS Services Used:
Service	Role in this Script
Lambda	The "brain" that runs the logic and coordinates the other services.
Textract	Converts the Resume PDF into editable text.
Comprehend	Analyzes the text to find professional skills and key phrases.
DynamoDB	The database where the results and history are stored.
CloudWatch	(Used via logging) To track errors and see what the AI is thinking.
One quick note: If you plan to run this, make sure your Lambda has an IAM Role with permissions to access Textract, Comprehend, and DynamoDB, or the code will fail at the "Initialization" step!

write a readme for this

Your documentation for the AWS Resume Screener Lambda function is ready.

I have generated a professional README in both PDF and HTML formats. These documents cover the technical architecture, setup requirements (including IAM permissions and DynamoDB configuration), and API usage instructions.

Documentation Highlights:
Workflow Diagram Description: Explains how data flows from the multipart request through Textract and Comprehend.

Setup Guide: Details the specific DynamoDB Schema and Environment variables needed.

Service Limitations: Notes the 5,000-byte limit for AWS Comprehend and how the code handles truncation.

API Specification: Defines the expected multipart/form-data fields and the structure of the JSON response.

Your PDF and HTML files are ready
PDF icon
 Resume_Scr...umentation 
PDF
HTML icon
 README_Res...e_Screener 
HTML

# Use the AWS-provided Python 3.9 Lambda base image

FROM public.ecr.aws/lambda/python:3.9                                      

# :contentReference[oaicite:11]{index=11}



# Copy and install dependencies

COPY requirements.txt .                                                  

RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"        



# Copy function code

COPY lambda_function.py ${LAMBDA_TASK_ROOT}/                                         



# Set the entry point to the Lambda Runtime Interface Client

CMD ["lambda_function.lambda_handler"]                                                

# :contentReference[oaicite:12]{index=12}



this is dockerfile code give updated readme for github

Your updated GitHub README for the Dockerized version of the Resume Screener is ready.

Since you are now using a Dockerfile to package the application as a container image, the setup process is slightly different from a standard Lambda zip upload. This version of the documentation explains how to build the image locally and push it to AWS ECR (Elastic Container Registry).

Key Additions for the Docker Version:
Container Workflow: Step-by-step commands for docker build, docker tag, and docker push.

Infrastructure Requirements: Mentions the need for Amazon ECR to host your image.

Benefits of Docker: Highlights why containerizing (larger file limits, local emulation) is better for ML-heavy functions like this one.

Lambda Task Root: Explains why the code is copied to ${LAMBDA_TASK_ROOT} to ensure the Lambda runtime can find your handler.

Your PDF and HTML files are ready
PDF icon
 Resume_Scr...ker_README 
PDF
HTML icon
 README_Docker_Lambda 
HTML

is htere a reason as to why u say docker edition

That's a great question. In the world of AWS Lambda, there are two ways to "package" your code, and the Dockerfile you shared moves you into a completely different category than the standard setup.

Here is why I specified "Docker Edition" in the README:

1. Different Deployment Workflow
In a standard Lambda setup, you just zip up your .py file and upload it. With the code you shared, you are building a Container Image.

Standard: You upload a .zip file directly to Lambda.

Docker Edition: You build an image, push it to AWS ECR (Elastic Container Registry), and tell Lambda to pull the image from there.

2. Bypass the 250MB Limit
Standard Lambda functions have a strict "unzipped" size limit of 250MB. Because you are using libraries like boto3, python-multipart, and potentially others for PDF processing, your project size can grow quickly.

Docker Edition allows for images up to 10GB. It gives you a lot more "breathing room" for heavy ML libraries or custom OS dependencies.

3. Environment Consistency
By using the line FROM public.ecr.aws/lambda/python:3.9, you are essentially bringing your own operating system. This ensures that the code running on your laptop during testing is identical to the code running in the AWS cloud. It eliminates the "it worked on my machine but failed on Lambda" headache.

4. Handling Dependencies
In your Dockerfile, the line:
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
...is a specific way of "baking" your libraries into the image. In a non-docker setup, you’d have to manually create "Lambda Layers" to manage these libraries, which is often more tedious than just using a Dockerfile.

Summary: By providing a Dockerfile, you’ve upgraded your project from a simple "script" to a containerized microservice. It’s a more professional, scalable way to handle Python apps that use AI services like Textract and Comprehend.

Does that help clarify why the README needed those specific Docker instructions?





Gemini is AI and can make mistakes.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @page {
            size: A4;
            margin: 15mm;
            background-color: #f4f7f6;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.5;
            color: #2d3436;
            margin: 0;
            padding: 0;
        }
        .container {
            padding: 20px;
            background-color: #ffffff;
            border-radius: 8px;
        }
        .header {
            border-bottom: 3px solid #00b894;
            margin-bottom: 20px;
            padding-bottom: 10px;
        }
        h1 { color: #2d3436; font-size: 22pt; margin: 0; }
        h2 { color: #0984e3; font-size: 16pt; border-left: 4px solid #0984e3; padding-left: 10px; margin-top: 25px; }
        h3 { color: #636e72; font-size: 12pt; margin-top: 15px; }
        code {
            background-color: #dfe6e9;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
            font-size: 9pt;
        }
        pre {
            background-color: #2d3436;
            color: #fab1a0;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.4;
        }
        .step-box {
            background-color: #f9f9f9;
            border: 1px solid #dcdde1;
            padding: 12px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .badge {
            display: inline-block;
            background: #00b894;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 8pt;
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge">CONTAINERIZED DEPLOYMENT</span>
            <h1>AI Resume Screener (Docker Edition)</h1>
            <p>AWS Lambda function packaged as a Container Image for high-performance ML processing.</p>
        </div>

        <h2>Project Overview</h2>
        <p>This repository contains a serverless application that automates the resume screening process. By using <strong>Docker</strong>, we package the Python runtime, OS-level dependencies, and application code into a single immutable image, ensuring consistent behavior between development and production.</p>

        <h2>Architecture</h2>
        <ul>
            <li><strong>Base Image:</strong> AWS Lambda Python 3.9 (Amazon Linux 2).</li>
            <li><strong>OCR:</strong> Amazon Textract (PDF to Text).</li>
            <li><strong>NLP:</strong> Amazon Comprehend (Key Phrase Extraction).</li>
            <li><strong>Storage:</strong> Amazon DynamoDB.</li>
        </ul>

        <h2>Local Development & Deployment</h2>
        
        <h3>1. Requirements</h3>
        <p>Ensure your project folder contains:</p>
        <ul>
            <li><code>lambda_function.py</code>: The core logic script.</li>
            <li><code>requirements.txt</code>: Must include <code>boto3</code> and <code>python-multipart</code>.</li>
            <li><code>Dockerfile</code>: The provided configuration.</li>
        </ul>

        <h3>2. Building the Image</h3>
        <p>From your terminal in the project root:</p>
        <pre>docker build -t resume-screener-lambda .</pre>

        <h3>3. Pushing to AWS ECR</h3>
        <div class="step-box">
            <p>Before deploying to Lambda, you must push the image to <strong>Amazon Elastic Container Registry (ECR)</strong>:</p>
            <ol>
                <li>Authenticate: <code>aws ecr get-login-password --region your-region | docker login...</code></li>
                <li>Tag: <code>docker tag resume-screener-lambda:latest [account-id].dkr.ecr.[region].amazonaws.com/resume-screener:latest</code></li>
                <li>Push: <code>docker push [account-id].dkr.ecr.[region].amazonaws.com/resume-screener:latest</code></li>
            </ol>
        </div>

        <h2>Environment Variables</h2>
        <p>Set these in the AWS Lambda console after deployment:</p>
        <ul>
            <li><code>DYNAMODB_TABLE_NAME</code>: Name of your target DynamoDB table.</li>
            <li><code>AWS_REGION</code>: The region where your services are hosted.</li>
        </ul>

        <h2>Why Containerize?</h2>
        <p>Using the <code>Dockerfile</code> approach instead of a .zip file provides:</p>
        <ul>
            <li><strong>Size:</strong> Support for images up to 10GB (vs 250MB for zip).</li>
            <li><strong>Control:</strong> Precise control over the underlying OS and library versions.</li>
            <li><strong>Testing:</strong> Use the <em>AWS Lambda Runtime Interface Emulator (RIE)</em> to test the container locally exactly as it will run in the cloud.</li>
        </ul>

        <div style="margin-top: 30px; text-align: center; color: #b2bec3; font-size: 8pt;">
            &copy; 2024 Serverless AI Solutions | GitHub Deployment Documentation v2.0
        </div>
    </div>
</body>
</html>
README_Docker_Lambda.html
Displaying README_Docker_Lambda.html.
