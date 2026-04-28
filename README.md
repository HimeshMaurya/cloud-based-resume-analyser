# AI Resume Screener (Docker Edition)

An automated talent matching backend that uses **Amazon Textract** for OCR and **Amazon Comprehend** for NLP to score candidate resumes against job descriptions.

## 🚀 Why Docker?
This project is packaged as a **Container Image** rather than a standard `.zip` deployment. This allows for:
* **Larger Dependencies:** Up to 10GB image size (bypassing the 250MB Lambda limit).
* **Consistency:** The Python 3.9 environment is identical in local testing and production.
* **Simplified Deployment:** All libraries are "baked" into the image using the `Dockerfile`.

---

## 🛠 Tech Stack
* **Compute:** AWS Lambda (Container Image)
* **OCR:** Amazon Textract (`detect_document_text`)
* **NLP:** Amazon Comprehend (`detect_key_phrases`)
* **Database:** Amazon DynamoDB
* **Runtime:** Python 3.9

---

## 📋 Prerequisites
1.  **AWS CLI** configured with appropriate permissions.
2.  **Docker** installed and running.
3.  **DynamoDB Table:** Created with a Partition Key named `submissionId` (String).
4.  **IAM Role:** The Lambda needs a role with `Textract`, `Comprehend`, and `DynamoDB:PutItem` permissions.

---

## 📦 Local Setup & Deployment

### 1. Build the Image
Navigate to the project root (where the `Dockerfile` is located) and run:
```bash
docker build -t resume-screener .
```

### 2. Push to Amazon ECR
You must host the image in the **Elastic Container Registry** before Lambda can use it.

```bash
# Login to ECR (Replace [region] and [account-id])
aws ecr get-login-password --region [region] | docker login --username AWS --password-stdin [account-id].dkr.ecr.[region].amazonaws.com

# Create a repository (if not already done)
aws ecr create-repository --repository-name resume-screener

# Tag and Push
docker tag resume-screener:latest [account-id].dkr.ecr.[region].amazonaws.com/resume-screener:latest
docker push [account-id].dkr.ecr.[region].amazonaws.com/resume-screener:latest
```

### 3. Create the Lambda Function
1.  Go to the AWS Lambda Console -> **Create function**.
2.  Select **Container image**.
3.  Browse for your image in ECR.
4.  Under **Configuration > Environment variables**, add:
    * `DYNAMODB_TABLE_NAME`: Your table name.

---

## 📑 API Specification

**Endpoint:** `POST` (via Function URL or API Gateway)  
**Content-Type:** `multipart/form-data`

### Request Fields:
| Field | Type | Description |
| :--- | :--- | :--- |
| `resume` | File (.pdf) | The candidate's resume. |
| `jd_text` | String | The full text of the job description. |

### Example Response:
```json
{
  "score": 12,
  "matching_phrases": ["python", "aws", "docker", "sql"],
  "message": "Match Score: 12"
}
```

---

## ⚠️ Known Limitations
* **Text Truncation:** Amazon Comprehend has a 5,000-byte limit for synchronous calls. This script automatically truncates text longer than 5,000 characters.
* **Timeouts:** OCR processing can be slow. Set your Lambda timeout to at least **30 seconds**.
