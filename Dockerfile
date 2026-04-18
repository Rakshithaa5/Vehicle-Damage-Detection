# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the scanner_app folder and models
COPY scanner_app/ ./scanner_app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Expose the port the app runs on (HF Spaces uses 7860)
EXPOSE 7860

# Run the application using gunicorn with a higher timeout for model loading
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "scanner_app.app:app"]
