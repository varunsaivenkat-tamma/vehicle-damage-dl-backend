# Car Damage Estimation System

A Flask-based web application that uses AI/ML models to detect car damage types, assess severity, and estimate repair costs.

## Features

✅ **Damage Detection** - YOLOv8 models detect damage types and severity levels
✅ **Cost Estimation** - Machine learning model predicts repair costs based on damage and vehicle info
✅ **Image Annotation** - Visual bounding boxes showing detected damages
✅ **Error Handling** - Comprehensive validation and error management
✅ **Beautiful UI** - Modern, responsive web interface with gradient design
✅ **Professional Results** - Detailed damage breakdown and printable reports

## Project Structure

```
car-damage-app/
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── static/                     # User uploaded images and processed results
├── templates/
│   ├── index.html             # Upload and form page
│   └── result.html            # Results display page
└── Final Damage/              # Pre-trained ML models
    ├── DamageTypebest.pt      # YOLO model for damage type detection
    ├── Severitybest.pt        # YOLO model for severity classification
    ├── cost_model.pkl         # RandomForest/DecisionTree for cost prediction
    ├── label_encoders.pkl     # Categorical encoders
    └── feature_columns.pkl    # Feature column names
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Setup Steps

1. **Navigate to the project directory:**
   ```powershell
   cd c:\Users\ADMIN\Downloads\car-damage-app\car-damage-app
   ```

2. **Install required packages:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```powershell
   python app.py
   ```

## Running the Application

### Development Mode
```powershell
python app.py
```

The application will start at: `http://localhost:5000`

### Production Mode
For production deployment, set environment variables:
```powershell
$env:FLASK_ENV = "production"
python app.py
```

## Usage

1. **Open the Application**
   - Navigate to `http://localhost:5000` in your browser

2. **Upload an Image**
   - Click the upload area or drag-and-drop a car image (JPG, PNG, or GIF)
   - Maximum file size: 10MB

3. **Enter Vehicle Details**
   - **Brand**: Car manufacturer (e.g., Ford, Honda, Toyota)
   - **Model**: Vehicle model (e.g., Focus, Civic)
   - **Year**: Manufacturing year (1900-2025)
   - **Fuel Type**: Petrol or Diesel
   - **Vehicle Type**: SUV, Sedan, or Hatchback
   - **Color**: Car color (e.g., Black, White)

4. **Get Results**
   - View the annotated image with bounding boxes
   - See cropped images of each detected damage
   - Get damage breakdown with estimated costs
   - Print or share the report

## API Endpoints

### GET `/`
Displays the main upload form and vehicle details page.

### POST `/predict`
Processes the uploaded image and vehicle details.

**Request Format:**
- Form data with file upload and vehicle details
- Image formats: JPG, JPEG, PNG, GIF
- Max size: 10MB

**Response:**
- HTML page with annotated images and cost estimates
- Error messages for invalid inputs (400, 500 status codes)

## Error Handling

The application validates:
- ✅ Image file presence and format
- ✅ Image file size
- ✅ All required vehicle details
- ✅ Year is within valid range (1900-2025)
- ✅ Image readability and processing

All errors are logged and returned with descriptive messages.

## Dependencies

- **flask**: Web framework
- **opencv-python**: Image processing
- **numpy**: Numerical computing
- **ultralytics**: YOLO model framework
- **joblib**: Model serialization
- **pandas**: Data manipulation
- **scikit-learn**: ML models and preprocessing
- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities

See `requirements.txt` for complete list with versions.

## Logging

Application logs are output to console at INFO level. Logs include:
- Model loading status
- Image processing events
- Validation messages
- Error details

## Performance Notes

- **First run**: May take time to download YOLO models
- **Inference time**: ~2-5 seconds per image (depends on hardware)
- **Cost calculation**: ~100ms per damage detected

## Troubleshooting

### Models not loading
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Reinstall packages
```powershell
pip install -r requirements.txt
```

### Port already in use
**Solution**: Change port in app.py or kill process on port 5000

### Static directory errors
**Solution**: Ensure write permissions on the `static/` directory

## Future Improvements

- [ ] Downgrade scikit-learn to match training version
- [ ] Add database for storing predictions
- [ ] Implement user authentication
- [ ] Add damage severity recommendations
- [ ] Export reports as PDF
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Real-time webcam detection

## License

This project uses pre-trained YOLO models and scikit-learn models.
See individual package licenses for details.

## Support

For issues or questions:
1. Check the error message carefully
2. Review the logs in console output
3. Verify all required files are present in `Final Damage/` directory
4. Ensure Python 3.9+ is installed

---

**Version**: 1.0.0  
**Last Updated**: November 29, 2025
