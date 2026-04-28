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
