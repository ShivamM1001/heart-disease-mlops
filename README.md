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



## Kubernetes Deployment

The Heart Disease Prediction API can be deployed locally using Kubernetes through Docker Desktop.

### Prerequisites

- Docker Desktop with Kubernetes enabled
- `kubectl` installed
- The trained model at `artifacts/best_model.joblib`
- Docker image `heart-disease-api:1.1`

### Deployment Files

- `k8s/deployment.yaml` — defines the API Deployment, replica count, resource limits, and health probes.
- `k8s/service.yaml` — exposes the API through a NodePort Service.

### Deploy the Application

Build the Docker image if it does not already exist:

```powershell
docker build -t heart-disease-api:1.1 .
```

Load the local image into the kind cluster:

```powershell
kind load docker-image heart-disease-api:1.1 --name docker-desktop
```

Apply the Kubernetes manifests:

```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Check deployment status:

```powershell
kubectl get deployments
kubectl get pods
kubectl get services
```

Wait until the API pod reports `1/1 Running`.

### Access the API Locally

Forward local port 8001 to the Kubernetes Service:

```powershell
kubectl port-forward service/heart-disease-api-service 8001:8000
```

Keep the port-forwarding terminal open.

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/health"
```

Monitoring:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/metrics"
```

Interactive API documentation:

http://127.0.0.1:8001/docs

Prediction endpoint:

`POST http://127.0.0.1:8001/predict`

### Kubernetes Evidence

Screenshots of the deployment, Service, health check, prediction, and monitoring output are stored in the `screenshots/` directory.

### Limitations

This is a local Kubernetes deployment, not a publicly hosted service. The API metrics are held in memory and reset when the process restarts. The model is an educational demonstration and must not be used for clinical diagnosis.


**Disclaimer:** This project is for educational purposes only and must not be used to diagnose heart disease or make clinical decisions.
