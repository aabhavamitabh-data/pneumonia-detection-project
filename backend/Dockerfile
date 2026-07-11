FROM python:3.11-slim

WORKDIR /app

# Install only the dependencies needed to serve predictions
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy the rest of the project
COPY . .

ENV PORT=8080
EXPOSE 8080

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT