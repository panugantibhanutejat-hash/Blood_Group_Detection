# Deployment Instructions for Blood Group Detection System

## Repository Status

The code has been successfully pushed to the GitHub repository:
https://github.com/chaitu0-12/Blood_group_detection

## Repository Structure

The repository contains all the necessary files for the application except the large model file, which has been excluded due to GitHub's file size limitations.

### Files Included in Repository:
- Application source code (`app.py`, `calibrated_predictor.py`, `pdf_report.py`)
- Web templates and static assets
- Requirements file
- Documentation (README.md)
- Configuration files (.gitignore)

### Files Excluded from Repository:
- `calibrated_model.h5` (92.55 MB) - The pre-trained model file

## Deployment Steps

### 1. Clone the Repository
```bash
git clone https://github.com/chaitu0-12/Blood_group_detection.git
cd Blood_group_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Obtain the Pre-trained Model

Due to GitHub's file size limitations, you need to obtain the `calibrated_model.h5` file separately:

#### Option A: Download from Releases (Recommended)
1. Visit the [Releases page](https://github.com/chaitu0-12/Blood_group_detection/releases)
2. Download the `calibrated_model.h5` file
3. Place the file in the project root directory

#### Option B: Contact Repository Owner
Contact the repository owner to obtain the model file directly.

#### Option C: Train Your Own Model
Use the provided training scripts to train your own model.

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Application
Open your web browser and go to `http://localhost:5000`

## System Requirements

- Python 3.7 or higher
- At least 2GB RAM (4GB recommended)
- At least 100MB free disk space for dependencies
- Additional 100MB for the model file

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Make sure all dependencies are installed with `pip install -r requirements.txt`

2. **Model Loading Error**: Ensure `calibrated_model.h5` is in the project root directory

3. **Port Already in Use**: If port 5000 is occupied, modify `app.py` to use a different port

4. **Memory Issues**: The model requires significant memory. Close other applications if experiencing issues.

## Additional Notes

- The application uses calibrated confidence scores for more reliable predictions
- All uploaded fingerprint images are stored temporarily in `static/uploads/`
- User accounts and detection history are stored in `app.db` (SQLite database)
- PDF reports are generated on-demand and saved in `static/uploads/`

## Support

For any issues or questions, please contact the repository owner or create an issue on GitHub.