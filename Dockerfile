# Use the official Python image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container and install the required dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the Flask application code into the container
COPY . .

# Expose the port your Flask app will run on
EXPOSE 5001

# Define the command to run your Flask application
# CMD ["python", "run.py"]
#use gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "run:app"]