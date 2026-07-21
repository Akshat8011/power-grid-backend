# Use a lightweight, stable version of Python
FROM python:3.9-slim

# Create a folder inside the container for your app
WORKDIR /app

# Copy your updated requirements file first
COPY requirements.txt .

# Install the dependencies (Docker will cache this step to make future builds faster)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main.py and the rest of your backend files
COPY . .

# CRITICAL: Hugging Face exclusively routes web traffic to port 7860
EXPOSE 7860

# Start the server using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
