# DATA ANALYTICS PROJECT
# 
# Project Title: Development of Land Transportation Vehicle Recognition System via YOLOv8
# Members: Canlas, Yul Guiller Q.
#          Formanes, Ericka Joy R.
#          Garchitorena, Grace Anne C.
#          Linao, David C.
#          Sypio, Lorenzo M.
# Year and Section: BSCS-3A
# Date Submitted: 01/23/2024

import datetime
import tkinter as tk
from tkinter import filedialog
import cv2
import os
from PIL import Image, ImageTk
from ultralytics import YOLO
import math

class VehicleRecognitionApp:

    # INITIALIZATION
    def __init__(self, root):
        self.root = root
        self.detected_objects = []
        self.is_detecting = False
        self.model = YOLO("model/YOLOv8-LTVR.pt")

        # For photo usage
        parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # parent folder
        logo_path = os.path.join(parent_folder, "src\\assets", "logo.png") # App logo path
        u_path = os.path.join(parent_folder, "src\\assets", "u.png") # ULTRALYTICS logo path
        p_path = os.path.join(parent_folder, "src\\assets", "p.png") # PYTHON logo path

        # Set initial window size and position it in the center
        window_width = 960
        window_height = 540
        self.set_window_position(window_width, window_height)
        
        # Panel
        self.left_panel = tk.Label(root, width = 320, height = 540, bg='#F1EAD5')
        self.left_panel.place(x = 0, y = 0)
        self.right_panel = tk.Label(root, width = 640, height = 540, bg='#7E0202')
        self.right_panel.place(x = 320 , y = 0)

        # Create a placeholder frame with padding
        self.placeholder_image = ImageTk.PhotoImage(Image.new("RGB", (560, 420), "gray"))
        self.video_frame = tk.Label(root, image=self.placeholder_image)
        self.video_frame.place(x = 360, y = 20)

        # Logo Placement
        logo = Image.open(logo_path)
        logo = logo.resize((200, 200))

        photo = ImageTk.PhotoImage(logo)

        logo_panel = tk.Label(root, image = photo, bg='#F1EAD5')
        logo_panel.image = photo
        logo_panel.place(x = 60, y = 20)

        # Create buttons with padding       
        bfont = ("Arial Bold", 12)

        self.button1 = tk.Button(root, text="DETECT", font=bfont, width = 15, height = 2, bg = "#7E0202", fg = "#FFFFFF", command=self.toggle_detection)
        self.button1.place(x = 80, y = 270)

        self.button2 = tk.Button(root, text="UPLOAD", font=bfont, width = 15, height = 2, bg = "#7E0202", fg = "#FFFFFF", command=self.upload)
        self.button2.place(x = 80, y = 330)

        self.button3 = tk.Button(root, text="EXIT", font=bfont, width = 15, height = 2, bg = "#7E0202", fg = "#FFFFFF", command=self.root.destroy)
        self.button3.place(x = 80, y = 390)

        # Miscellaneous
        self.misc1 = tk.Label(root, text="Developed by", width = 10, bg='#7E0202', fg='#FFFFFF')
        self.misc1.config(font = ('Arial', 10))
        self.misc1.place(x = 339, y = 460)

        self.misc2 = tk.Label(root, text="CANLAS, FORMANES, GARCHITORENA, LINAO, SYPIO", width = 43, bg='#7E0202', fg='#FFFFFF')
        self.misc2.config(font = ('Arial Bold', 10))
        self.misc2.place(x = 420, y = 460)

        self.misc3 = tk.Label(root, text="BSCS-3A", width = 7, bg='#7E0202', fg='#FFFFFF')
        self.misc3.config(font = ('Arial', 10))
        self.misc3.place(x = 340, y = 480)
        
        self.misc4 = tk.Label(root, text="S.Y. 2023-2024", width = 11, bg='#7E0202', fg='#FFFFFF')
        self.misc4.config(font = ('Arial', 10))
        self.misc4.place(x = 340, y = 500)

        misc5 = Image.open(u_path)
        misc5 = misc5.resize((96, 18))

        misc6 = Image.open(p_path)
        misc6 = misc6.resize((30, 30))

        m5 = ImageTk.PhotoImage(misc5)
        m6 = ImageTk.PhotoImage(misc6)

        m5_panel = tk.Label(root, image = m5, bg='#7E0202')
        m5_panel.image = m5
        m5_panel.place(x = 790, y = 487)

        m6_panel = tk.Label(root, image = m6, bg='#7E0202')
        m6_panel.image = m6
        m6_panel.place(x = 900, y = 480)

        self.camera = None
        self.update_id = None

    # Position the window in the center of the screen
    def set_window_position(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        self.root.geometry(f"{width}x{height}+{x}+{y}")

########################################################################################################################
# ------------------------------------------- METHODS FOR DETECTION METHOD ------------------------------------------- #
########################################################################################################################

    def toggle_detection(self):
        self.is_detecting = not self.is_detecting

        if self.is_detecting:
            self.button1.config(text="STOP DETECT")
            self.start_detection()
        else:
            self.button1.config(text="DETECT")
            self.stop_detection()

    def start_detection(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(1)
            self.camera.set(3, 560)
            self.camera.set(4, 420)

        ret, frame = self.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.video_frame.img = img
            self.video_frame.config(image=img)
        
        if self.is_detecting:
            self.update_id = self.root.after(10, self.update_frame)

    def stop_detection(self):
        self.is_detecting = False
        if self.update_id:
            self.root.after_cancel(self.update_id)
        
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        self.video_frame.config(image=self.placeholder_image)

        print("Detected classes:")
        for obj in self.detected_objects:
            print(obj)

        self.detected_objects = []

    def update_frame(self):
        if self.is_detecting:
            ret, frame = self.camera.read()
            if ret:
                results = self.model(frame, stream=True)

                confidence_threshold = 0.6

                for r in results:
                    for box in r.boxes:
                        if box.conf[0] > confidence_threshold:

                            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            current_day = datetime.datetime.now().strftime("%Y-%m-%d") 

                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                            confidence = math.ceil((box.conf[0] * 100)) / 100

                            cls = int(box.cls[0])

                            self.detected_objects.append([self.class_names[cls], current_time, current_day])

                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2

                            cv2.putText(frame, self.class_names[cls], org, font, fontScale, color, thickness)

                resized_frame = cv2.resize(frame, (560, 420))

                img = Image.fromarray(resized_frame)
                img = ImageTk.PhotoImage(image=img)

                self.video_frame.configure(image=img)
                self.video_frame.image = img

            self.update_id = self.root.after(10, self.update_frame)
        else:
            self.stop_detection()

#####################################################################################################################
# ------------------------------------------- METHODS FOR UPLOAD METHOD ------------------------------------------- #
#####################################################################################################################

    def upload(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.camera = cv2.VideoCapture(file_path)
            self.is_detecting = True 

            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.model(frame)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        cls = int(box.cls[0])

                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 2
                        color = (0, 0, 0)
                        thickness = 5

                        cv2.putText(frame, self.class_names[cls], org, font, fontScale, color, thickness)

                resized_frame = cv2.resize(frame, (560, 420))

                img = Image.fromarray(resized_frame)
                img = ImageTk.PhotoImage(image=img)

                self.video_frame.configure(image=img)
                self.video_frame.image = img
                self.video_frame.update()

                if not self.is_detecting:
                    break

            self.camera.release()

def main():
    root = tk.Tk()
    root.title("LAND TRANSPORTATION VEHICLE RECOGNITION")

    app = VehicleRecognitionApp(root)
    app.class_names = ["automobile", "motorcycle", "jeepney", "bus", "truck", "tricycle"]

    root.mainloop()

if __name__ == "__main__":
    main()
