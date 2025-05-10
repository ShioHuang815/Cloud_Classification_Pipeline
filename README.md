# Cloud Classification Pipeline

A reproducible pipeline for classifying cloud types based on image features, with automated data cleaning, model training, and artifact storage.

![Cloud Classification](https://img.shields.io/badge/Machine%20Learning-Cloud%20Classification-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Compatible-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview

This project implements an end-to-end pipeline for cloud classification using machine learning. The system processes cloud imagery, extracts relevant features, trains classification models, and stores resulting artifacts for deployment or further analysis.

## 🔧 Prerequisites

- Docker
- Python 3.9+
- AWS CLI (optional, for S3 access)

## 🚀 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/cloud-classification.git
   cd cloud-classification
   ```

2. Build the Docker image:
   ```bash
   docker build -t cloud-classifier -f dockerfiles/Dockerfile .
   ```

## 🏃‍♂️ Running the Pipeline

### Local Execution
```bash
python pipeline.py
```

### Docker Execution
```bash
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/artifacts:/app/artifacts \
           cloud-classifier
```

## 🧪 Testing

### Running Tests Locally
```bash
python tests/unit_tests.py
```

### Running Tests in Docker
```bash
docker run cloud-classifier python tests/unit_tests.py
```

## 📏 Linting (PEP8 Compliance)

```bash
# Install pylint if needed
pip install pylint

# Run linting
docker run cloud-classifier pylint pipeline.py tests modules
```

## ☁️ AWS Configuration (Optional)

To enable S3 uploads for artifact storage:

1. Set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_BUCKET_NAME=your-bucket-name
   ```

2. Or configure in `config.yaml`:
   ```yaml
   aws:
     bucket_name: "your-bucket-name"
     s3_folder: "cloud-classifier"
     region_name: "us-east-1"
   ```

## 📁 Folder Structure

```
.
├── pipeline.py           # Main pipeline execution script
├── README.md             # This documentation
├── requirements.txt      # Python dependencies
├── .pylintrc             # Linting configuration
├── data
│   └── clouds.csv        # Input dataset
├── modules
│   ├── data_cleaning.py  # Data preprocessing module
│   ├── model_training.py # ML training module
│   └── aws_util.py       # AWS integration utilities
├── config
│   └── config.yaml       # Configuration settings
├── dockerfiles
│   └── Dockerfile        # Docker container definition
└── tests
    ├── test_cleaning.py  # Unit tests for data cleaning
    ├── test_training.py  # Unit tests for model training
    └── test_aws.py       # Unit tests for AWS utilities
```

## 📊 Pipeline Workflow

1. **Data Loading**: Imports cloud imagery data from CSV files
2. **Data Cleaning**: Preprocesses images, handles missing values
3. **Feature Extraction**: Identifies key cloud characteristics 
4. **Model Training**: Trains classification models using processed data
5. **Model Evaluation**: Evaluates performance metrics
6. **Artifact Storage**: Saves models and metadata locally or to S3

## 📝 Notes

* All Python code complies with PEP8 standards
* The included `.pylintrc` ensures consistent linting
* Tests can be run both locally and in Docker containers
* AWS uploads are optional and require proper credentials

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
