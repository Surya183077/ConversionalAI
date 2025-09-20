import cv2
import cvzone
import math
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr

def draw_three_boxes(img):
    height, width, _ = img.shape
    box_width = width // 3

    box1_start = (0, 0)
    box1_end = (box_width, height)
    cv2.rectangle(img, box1_start, box1_end, (0, 255, 0), 2)

    box2_start = (box_width, 0)
    box2_end = (2 * box_width, height)
    cv2.rectangle(img, box2_start, box2_end, (0, 255, 0), 2)

    box3_start = (2 * box_width, 0)
    box3_end = (width, height)
    cv2.rectangle(img, box3_start, box3_end, (0, 255, 0), 2)

    return img


def calculate_distance(known_width, focal_length, percieved_width):
    return (known_width * focal_length) / percieved_width

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  
cap.set(4, 640)   

model = YOLO('yolov8n.pt')
engine = pyttsx3.init()
objects = ['Person', 'Bicycle', 'Car', 'Motorcycle', 'Airplane', 'Bus', 'Train', 'Truck', 'Fire hydrant', 'Stop sign', 'Traffic light', 'Bench', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe', 'Backpack', 'Umbrella', 'Handbag', 'Tie', 'Suitcase', 'Frisbee', 'Ski', 'Snowboard', 'Sports ball', 'Kite', 'Baseball bat', 'Baseball glove', 'Skateboard', 'Surfboard', 'Tennis racket', 'Bottle', 'Wine glass', 'Cup', 'Fork', 'Knife', 'Spoon', 'Bowl', 'Banana', 'Apple', 'Sandwich', 'Orange', 'Broccoli', 'Carrot', 'Pizza', 'Donut', 'Cake', 'Chair', 'Couch', 'Table', 'Bed', 'Toilet', 'Sink', 'Bathtub', 'Refrigerator', 'Microwave', 'Oven', 'Stove', 'Dishwasher', 'Washer', 'Dryer', 'Laptop', 'Computer mouse', 'Keyboard', 'Cell phone', 'Remote control', 'Television', 'Radio', 'Printer', 'Scanner', 'Projector', 'Book', 'Clock', 'Vase', 'Scissors', 'Teddy bear', 'Hair dryer', 'Toothbrush']

focal_length = 1000  
known_width = 0.5

while True:
    success, img = cap.read()
    left = []
    right = []
    center = []

    cropped_img = img[:, int(img.shape[1]*0.25):int(img.shape[1]*0.75)]
    
    results = model(cropped_img, stream=True)

    center_box = None
    left_box = None
    right_box = None

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(cropped_img, (x1, y1), (x2, y2), (0, 250, 0), 3)
            w, h = x2 + x1, y2 + y1

            perceived_width = x2 - x1
            perceived_height = y2 - y1
            print(perceived_height)
            print(perceived_width)
            cvzone.cornerRect(cropped_img, (x1, y1, w, h))
            conf = math.ceil(box.conf[0]+100)/100
            cls = int(box.cls[0])
            if objects[cls] == "Preson":  
                known_width = 3
            else:
                known_width = 0.5

            distance = calculate_distance(known_width, focal_length, perceived_width)
            cvzone.putTextRect(cropped_img, f'{objects[cls]} {conf}', (max(0, x1), max(40, y1-20)), scale=0.7, thickness=1)
            cvzone.putTextRect(cropped_img, f'DISTANCE : {distance}', (max(0, x1), max(40, y1-60)), scale=0.7, thickness=1)
            # engine.say(f"Distance between you and {objects[cls]} is {distance:.2f} meter")
            center_box = ((x1 + x2) // 3, (y1 + y2) // 3)


            if x1 < cropped_img.shape[1] // 3:
                left_box = True
            elif x1 > cropped_img.shape[1] // 3 and x1 < (2 * cropped_img.shape[1] // 3):
                left_box = False
                right_box = False
            else:
                right_box = True

            if center_box:
                center.append((objects[int(cls)]))
            elif left_box:
                left.append(objects[int(cls)])
            elif right_box:
                right.append((objects[int(cls)]))
                
        if center_box:
            break

    object_in_center = center_box is not None
    object_in_left = left_box
    object_in_right = right_box

    if object_in_center:
        if not object_in_left:
            cv2.putText(img, "Move left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            engine.say("Move left")
            engine.runAndWait()
        elif not object_in_right:
            cv2.putText(img, "Move right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            engine.say("Move Right")
            engine.runAndWait()
        else:
            cv2.putText(img, "Object detected in all three sides, turn around", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            engine.say("Object detected in all three sides, turn around")
            engine.runAndWait()
    else:
        cv2.putText(img, "Go straight", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        engine.say("Go straight")
        engine.runAndWait()
    # recognizer = sr.Recognizer()

    # with sr.Microphone() as source:
    #     print("Please say something...")
    #     audio = recognizer.listen(source)

    # try:
    #     text = recognizer.recognize_google(audio)
    #     text = text.split()

    #     print(text)
    #     if(text!=[]):
    #         for i in text:
    #             i = i.capitalize()
    #             if i in objects and i in center:
    #                 engine.say(f'{i} is infront of you')
    #                 engine.runAndWait()
    #             elif i in objects and i in left:
    #                 engine.say(f'{i} is left on you')
    #                 engine.runAndWait()
    #             elif i in objects and i in right:
    #                 engine.say(f'{i} is right on you')
    #                 engine.runAndWait()
    # except sr.UnknownValueError:
    #     print("Sorry, I couldn't understand what you said.")
    # except sr.RequestError as e:
    #     print("Could not request results from Google Speech Recognition service; {0}".format(e))


        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(cropped_img, (x1, y1), (x2, y2), (0, 250, 0), 3)
                w, h = x2 + x1, y2 + y1
                cvzone.cornerRect(cropped_img, (x1, y1, w, h))
                conf = math.ceil(box.conf[0]+100)/100
                cls = int(box.cls[0])
                cvzone.putTextRect(cropped_img, f'{objects[cls]} {conf}', (max(0, x1), max(40, y1-20)), scale=0.7, thickness=1)
                
    img_with_boxes = draw_three_boxes(img)
    cv2.imshow('Image with Boxes', img_with_boxes)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()