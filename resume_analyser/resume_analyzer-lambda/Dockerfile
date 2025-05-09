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
