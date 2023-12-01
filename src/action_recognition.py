import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

class VideoCamera:
    def __init__(self):
        pass

    def detect_video_device(self):
        video_device = 255
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                video_device = i
                break
        print("Video Device is: ", video_device)
        return video_device           


class GestureRecognition(VideoCamera):
    
    def __init__(self):
        self.__video_device = self.detect_video_device()
        self.__cap = cv2.VideoCapture(self.__video_device)
        # Curl counter variables
        self.__counter_left_arm = 0 
        self.__counter_right_arm = 0 
        self.__counter_stop = 0 
        self.__counter_advance = 0 

        self.__deb_counter_left_arm = 0 
        self.__deb_counter_right_arm = 0 
        self.__deb_counter_stop = 0 
        self.__deb_counter_advance = 0 

        self.__stage_left_arm = None
        self.__stage_right_arm = None
        self.__stage_stop = None
        self.__stage_advance = None
        self.__action = None    

    # Public methods
    def detec_gesture(self):
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.__cap.isOpened():
                ret, frame = self.__cap.read()
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    # Extract landmarks
                    landmarks = results.pose_landmarks.landmark
                    
                    coordinate = self.__get_coordinates(landmarks)
                    angle = self.__calc_angles(coordinate)
                    self.__identify_geture(angle)
                    self.__print_angles(image, angle, coordinate)

                except Exception as error:
                    print("An exception occurred:", error) # An exception occurred: division by zero


                self.__setup_image_box(image)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )               
                
                cv2.imshow('ERR - IACOS ', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            self.__cap.release()
            cv2.destroyAllWindows()

    # Private methods
    def __calculate_angle(self, a, b, c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        return angle
    
    def __print_angles(self, image, angle, coordinate):

        cv2.putText(image, str(angle["left_elbow"]), 
                    tuple(np.multiply(coordinate["left_elbow"], [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image, str(angle["left_shoulder"]), 
                    tuple(np.multiply(coordinate["left_shoulder"], [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image, str(angle["right_elbow"]), 
                    tuple(np.multiply(coordinate["right_elbow"], [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, str(angle["right_shoulder"]), 
                    tuple(np.multiply(coordinate["right_shoulder"], [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image, str(angle["right_wrist"]), 
                    tuple(np.multiply(coordinate["right_wrist"], [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
    def __get_coordinates(self, landmarks):
        coordinate = {}
        coordinate["left_shoulder"] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        coordinate["left_elbow"] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        coordinate["left_wrist"] = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        coordinate["left_hip"] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
   
        coordinate["right_shoulder"] = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        coordinate["right_elbow"] = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        coordinate["right_wrist"] = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        coordinate["right_hip"] = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        coordinate["z_right_shoulder"] = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        coordinate["z_right_elbow"] = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
        coordinate["z_right_wrist"] = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
        coordinate["z_right_hip"] = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]

        coordinate["z_left_shoulder"] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        coordinate["z_left_elbow"] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
        coordinate["z_left_wrist"] = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
        coordinate["z_left_hip"] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        return coordinate

    def __calc_angles(self, coordinate):
        angle = {}
        angle["left_elbow"] = self.__calculate_angle(coordinate["left_shoulder"], coordinate["left_elbow"], coordinate["left_wrist"])
        angle["left_arm"]  = self.__calculate_angle(coordinate["right_shoulder"], coordinate["left_shoulder"], coordinate["left_wrist"])
        angle["left_shoulder"] = self.__calculate_angle(coordinate["left_hip"], coordinate["left_elbow"], coordinate["left_wrist"])

        angle["right_elbow"] = self.__calculate_angle(coordinate["right_shoulder"], coordinate["right_elbow"], coordinate["right_wrist"])
        angle["right_arm"] = self.__calculate_angle(coordinate["left_shoulder"], coordinate["right_shoulder"], coordinate["right_wrist"])
        angle["right_shoulder"] = self.__calculate_angle(coordinate["right_hip"], coordinate["right_elbow"], coordinate["right_wrist"])

        angle["right_wrist"] = self.__calculate_angle(coordinate["z_right_hip"], coordinate["z_right_shoulder"], coordinate["z_right_wrist"])
        angle["left_wrist"] = self.__calculate_angle(coordinate["z_left_hip"], coordinate["z_left_shoulder"], coordinate["z_left_wrist"])
        return angle

    def __identify_geture(self, angle):
        # Curl counter logic
        if angle["left_elbow"] > 160 and (angle["left_shoulder"] > 85 and angle["left_shoulder"] < 140) and self.__action != "turn_left" :
            if self.__stage_left_arm is None:
                self.__deb_counter_left_arm +=1
            if self.__deb_counter_left_arm > 3:    
                stage_left_arm = self.__action = "turn_left" 
                print(self.__counter_left_arm, " ", stage_left_arm)
                self.__counter_left_arm +=1 
        else:
            self.__stage_left_arm = None
            self.__deb_counter_left_arm = 0


        # Curl counter logic
        if angle["right_elbow"] > 160 and (angle["right_shoulder"] > 85 and angle["right_shoulder"] < 140) and self.__action !=  "turn_right":
            if self.__stage_right_arm is None:
                self.__deb_counter_right_arm +=1
            if self.__deb_counter_right_arm > 3:     
                self.__stage_right_arm = self.__action = "turn_right"
                print(self.__counter_right_arm, " ","turn_right")
                self.__counter_right_arm += 1  
        else:
            self.__stage_right_arm = None
            self.__deb_counter_right_arm = 0  

        # Curl counter logic
        if angle["right_arm"] > 95 and angle["right_arm"] < 110  and (angle["right_wrist"] > 100 and angle["right_shoulder"] < 180) and self.__action != "stop":
            if self.__stage_stop is None:
                self.__deb_counter_stop +=1
            if self.__deb_counter_stop > 3:     
                self.__stage_stop = self.__action = "stop" 
                print(self.__counter_stop, " ", "stop")
                self.__counter_stop +=1  
        else:
            self.__stage_stop = None
            self.__deb_counter_stop = 0     

        if angle["left_arm"] > 70 and angle["left_arm"] < 100  and (angle["left_wrist"] > 70 and angle["left_wrist"] < 100) and self.__action != "advance":
            if self.__stage_advance is None:
                self.__deb_counter_advance +=1
            if self.__deb_counter_advance > 3:    
                self.__stage_advance = self.__action = "advance" 
                print(self.__counter_advance, " ", "advance") 
                self.__counter_advance +=1   
        else:
            self.__stage_advance = None 
            self.__deb_counter_advance = 0 

    def __setup_image_box(self, image):
        cv2.rectangle(image, (0,0), (680,110), (245,117,16), -1)
        
        cv2.putText(image, 'LEFT REPS', (10,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.__counter_left_arm), 
                    (10,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)  

        cv2.putText(image, 'RIGHT REPS', (150,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.__counter_right_arm), 
                    (200,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, 'STOP REPS', (300,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.__counter_stop), 
                    (350,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, 'ADVANCE REPS', (450,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.__counter_advance),    
                    (500,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, 'STAGE', (325,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, self.__action, 
                    (300,100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
  

def main():
    gesture = GestureRecognition()
    gesture.detect_video_device()
    gesture.detec_gesture()


if __name__ == "__main__":
    main()