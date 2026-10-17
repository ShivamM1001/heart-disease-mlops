## Docker Deployment

### Prerequisites
- Docker Desktop installed and running.
- A trained model available at `artifacts/best_model.joblib`.

### Build the Docker image

From the project root, run:

```powershell
docker build -t heart-disease-api:1.0 .
```

### Run the container

```powershell
docker run -d --name heart-disease-container -p 8000:8000 heart-disease-api:1.0
```

### Verify the API

- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

### Test a prediction

Open `/docs`, expand `POST /predict`, click **Try it out**, enter the 13 required input features, and click **Execute**.

### Container management

Check running containers:

```powershell
docker ps
```

View container logs:

```powershell
docker logs heart-disease-container
```

Stop the container:

```powershell
docker stop heart-disease-container
```

Remove the stopped container:

```powershell
docker rm heart-disease-container
```

### Screenshots

The `screenshots/` folder contains the Docker container status, API documentation, health check, and prediction response evidence.


## Monitoring and Logging

The API includes lightweight monitoring endpoints and application logging.

- `GET /health` — reports API health and whether the model is loaded.
- `GET /metrics` — reports request totals, response status counts, prediction totals, prediction classes, and process uptime.
- `POST /predict` — generates a prediction and logs the predicted class.

Request logs include the HTTP method, endpoint, response status, and request duration. Prediction logs record the predicted class without logging individual input feature values.

### Limitations

Metrics are stored in memory and reset when the API process restarts. They are intended for this educational project and are not a persistent monitoring system or a Prometheus exporter. The model is for demonstration only and must not be used for clinical diagnosis.


**Disclaimer:** This project is for educational purposes only and must not be used to diagnose heart disease or make clinical decisions.
