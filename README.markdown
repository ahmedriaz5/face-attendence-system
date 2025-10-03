# Face Recognition Attendance System

A Python-based attendance system that uses face recognition to mark attendance, displaying the video feed within a Tkinter GUI. It automatically reads face images from a predefined `images` folder, marks attendance with names and timestamps in a CSV file, prevents duplicate entries for the same day, and clears the video feed when stopped.

## Features
- **Face Recognition**: Identifies faces using the `face_recognition` library and marks attendance.
- **GUI Integration**: Displays the camera feed within the Tkinter interface alongside control buttons.
- **Predefined Image Folder**: Uses an `images` folder in the project directory for face data, eliminating manual folder selection.
- **Attendance Logging**: Records names and timestamps in `attendance.csv`, preventing duplicate entries for the same person on the same day.
- **Clear Video Feed**: The camera feed is cleared (displays a black placeholder) when attendance is stopped.
- **User-Friendly Interface**: Buttons for training the model, starting/stopping attendance, adding images, viewing the CSV, and exiting.

## Prerequisites
- Python 3.8+
- A webcam for capturing video
- Required Python libraries (listed in `requirements.txt`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-recognition-attendance.git
   cd face-recognition-attendance
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create an `images` folder in the project directory and add images of individuals (e.g., `John.jpg`, `Alice.png`). Each image should contain one clear face and be named after the person.

## Usage
1. Run the script:
   ```bash
   python attendance.py
   ```
2. **Train Model**: Click "Train Model" to load faces from the `images` folder.
3. **Start Attendance**: Click "Start Attendance" to display the camera feed and begin recognizing faces. Names appear on the feed, and attendance is logged in `attendance.csv`.
4. **Stop Attendance**: Click "Stop Attendance" to stop the camera and clear the feed (displays a black placeholder).
5. **Add Image**: Add new face images to the `images` folder and retrain the model.
6. **Open CSV**: View the attendance records in `attendance.csv`.
7. **Exit**: Close the application.

## Project Structure
```
face-recognition-attendance/
├── images/                # Folder containing face images (e.g., John.jpg, Alice.png)
├── attendance.py          # Main Python script
├── requirements.txt       # List of dependencies
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
└── attendance.csv        # Generated attendance log (not included in repo)
```

## Dependencies
Listed in `requirements.txt`:
```
opencv-python
face_recognition
pandas
numpy
pillow
```

## Notes
- Ensure the `images` folder exists in the project directory with valid face images.
- Images should be named as `<person_name>.jpg` (or `.png`, `.jpeg`) and contain only one face.
- The webcam must be accessible (default index `0`; change to `1` if needed in `attendance.py`).
- The `attendance.csv` file is created automatically when attendance is marked.

## Troubleshooting
- **Images Folder Not Found**: Create the `images` folder and add images.
- **No Faces Detected**: Ensure images are clear and contain one face each.
- **Camera Issues**: Verify webcam connectivity (try `cv2.VideoCapture(1)` if `0` fails).
- **Feed Not Clearing**: The feed should revert to a black placeholder on stopping; ensure `stop_attendance` is called correctly.

## License
MIT License (feel free to add a `LICENSE` file if desired).