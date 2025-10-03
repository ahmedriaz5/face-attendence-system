import cv2
import os
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from PIL import Image, ImageTk

# ==================== Global Variables ====================
# Predefined folder for images (relative to script directory)
image_folder = os.path.join(os.path.dirname(__file__), "images")
known_encodings = []
known_names = []
camera_running = False
video_thread = None
attended_today = set()
cap = None
blank_image = None  # For blank placeholder when camera is stopped

# Attendance CSV file
attendance_file = "attendance.csv"

# ==================== Attendance Function ====================
def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    new_row = {"Name": name, "Time": dt_string}

    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(attendance_file, index=False)
    status_label.config(text=f"Attendance marked for {name} at {dt_string}", fg="green")

# ==================== Load Images & Train ====================
def train_model():
    global known_encodings, known_names, image_folder

    # Check if the images folder exists
    if not os.path.exists(image_folder):
        messagebox.showerror("Error", f"Images folder '{image_folder}' not found. Please create it and add images.")
        return

    known_encodings = []
    known_names = []

    # Load images from the predefined folder
    for file in os.listdir(image_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_folder, file)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                name = os.path.splitext(file)[0]
                known_names.append(name)
            else:
                print(f"No face found in {file}")

    if len(known_encodings) == 0:
        messagebox.showerror("Error", "No valid faces found in the images folder!")
        return

    status_label.config(text=f"Model trained successfully with {len(known_encodings)} faces!", fg="green")

# ==================== Camera Loop ====================
def _camera_loop():
    global camera_running, known_encodings, known_names, attended_today, cap

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            # Draw rectangle & name on face
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Mark attendance only if not already marked today
            if name != "Unknown" and name not in attended_today:
                mark_attendance(name)
                attended_today.add(name)

        # Convert frame to Tkinter-compatible image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((320, 240), Image.Resampling.LANCZOS)  # Resize for display
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # Keep a reference
        video_label.configure(image=imgtk)

        root.update()  # Update the GUI

# ==================== GUI Button Functions ====================
def start_attendance():
    global camera_running, video_thread, attended_today, cap

    if len(known_encodings) == 0:
        messagebox.showerror("Error", "Please train the model first.")
        return

    # Load attended_today from CSV for today's date
    today = datetime.now().strftime("%Y-%m-%d")
    attended_today = set()
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        today_attendances = df[df['Time'].str.startswith(today)]
        attended_today = set(today_attendances['Name'].unique())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open camera.")
        return

    camera_running = True
    video_thread = threading.Thread(target=_camera_loop)
    video_thread.daemon = True  # Make thread daemon so it exits when app closes
    video_thread.start()
    status_label.config(text="Camera started...", fg="green")

def stop_attendance():
    global camera_running, cap
    camera_running = False
    if cap is not None:
        cap.release()
        cap = None
    # Clear the video feed and show a blank placeholder
    video_label.imgtk = blank_image
    video_label.configure(image=blank_image)
    status_label.config(text="Camera stopped and feed cleared.", fg="red")

def open_csv():
    if os.path.exists(attendance_file):
        os.startfile(attendance_file)
    else:
        messagebox.showinfo("Info", "No attendance file found yet.")

def add_image():
    global image_folder
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)  # Create images folder if it doesn't exist

    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
    if file_path:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(image_folder, file_name)
        os.replace(file_path, dest_path)
        status_label.config(text=f"Image {file_name} added to dataset.", fg="blue")

def exit_app():
    global camera_running, cap
    camera_running = False
    if cap is not None:
        cap.release()
        cap = None
    root.destroy()

# ==================== Tkinter GUI ====================
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("800x600")
root.configure(bg="lightblue")

# Create a blank placeholder image for when camera is stopped
blank_img = Image.new('RGB', (320, 240), color='black')
blank_image = ImageTk.PhotoImage(blank_img)

# Create main frame to hold video and buttons
main_frame = tk.Frame(root, bg="lightblue")
main_frame.pack(pady=10, padx=10, fill="both", expand=True)

# Video feed label (starts with blank image)
video_label = tk.Label(main_frame, image=blank_image, bg="black")
video_label.pack(side="left", padx=10)

# Button frame
button_frame = tk.Frame(main_frame, bg="lightblue")
button_frame.pack(side="right", fill="y", padx=10)

# Title
title_label = tk.Label(button_frame, text="Face Recognition Attendance System",
                       font=("Arial", 16, "bold"), bg="lightblue", fg="darkblue")
title_label.pack(pady=10)

# Buttons
btn_train = tk.Button(button_frame, text="Train Model", width=25, command=train_model)
btn_train.pack(pady=5)

btn_start = tk.Button(button_frame, text="Start Attendance", width=25, command=start_attendance)
btn_start.pack(pady=5)

btn_stop = tk.Button(button_frame, text="Stop Attendance", width=25, command=stop_attendance)
btn_stop.pack(pady=5)

btn_add = tk.Button(button_frame, text="Add Image", width=25, command=add_image)
btn_add.pack(pady=5)

btn_open = tk.Button(button_frame, text="Open Attendance CSV", width=25, command=open_csv)
btn_open.pack(pady=5)

btn_exit = tk.Button(button_frame, text="Exit", width=25, command=exit_app)
btn_exit.pack(pady=5)

# Status label
status_label = tk.Label(button_frame, text="Click 'Train Model' to start...",
                        font=("Arial", 10), bg="lightblue", fg="black")
status_label.pack(pady=20)

root.mainloop()