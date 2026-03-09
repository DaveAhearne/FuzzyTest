# Fuzzy CNN — CIFAR-10 Image Classifier

A convolutional neural network trained on CIFAR-10, exported to ONNX, and served via a FastAPI REST API. Accepts an image upload and returns ranked predictions with confidence scores across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Local Setup

Create a virtual environment at the root:
```
python -m venv .venv
```

Activate the environment:
* WindowsCommand Prompt: `.venv\Scripts\activate.bat`
* WindowsPowerShell: `.venv\Scripts\Activate.ps1`
* macOS / Linuxbash / zshsource: `.venv/bin/activate`


Install all dependency groups for development:

```bash
pip install -e ".[serve,train,test]"
```

Copy the example env file and adjust as needed:

```bash
cp .env.example .env
```

## Commands
* `fuzzy_train`: Train the model. CIFAR-10 data downloads automatically on first run
* `fuzzy_export`: Export the trained checkpoint to ONNX
* `fuzzy_serve`: Start the API server
* `pytest`: Run the test suite

API docs are available at `http://localhost:8000/` when the server is running.

## Docker

### Build

```bash
docker build -f docker/Dockerfile.train -t fuzzy-train .
docker build -f docker/Dockerfile.serve -t fuzzy-serve .
```

### Train and Export

Trains the model and exports it to ONNX in one step. Mount both directories to persist the outputs on the host. The extra `--rm` flag just cleans it up afterwards so you're left with the artifacts

```bash
docker run --name fuzzy-train --rm \
  -v [YOUR_CHECKPOINT_PATH]:/app/src/fuzzy_cnn/train/checkpoints \
  -v [YOUR_MODEL_STORE_PATH]:/app/src/fuzzy_cnn/serve/model_store \
  fuzzy-train
```

### Serve

```bash
docker run --name fuzzy-api-serve \
  -p 8000:8000 \
  -v [YOUR_MODEL_STORE_PATH]:/app/src/fuzzy_cnn/serve/model_store \
  fuzzy-serve
```