import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
            
    return angle 



cap = cv2.VideoCapture(0)

# Curl counter variables
counter_left_arm = 0 
counter_right_arm = 0 
counter_stop = 0 
counter_advance = 0 

deb_counter_left_arm = 0 
deb_counter_right_arm = 0 
deb_counter_stop = 0 
deb_counter_advance = 0 

stage_left_arm = None
stage_right_arm = None
stage_stop = None
stage_advance = None
action = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Get coordinates right
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            z_right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            z_right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            z_right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            z_right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]

            z_left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            z_left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            z_left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            z_left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]


            
            # Calculate angle
            angle_left_elbow = calculate_angle(left_shoulder, left_elbow, left_wrist)
            angle_left_arm = calculate_angle(right_shoulder, left_shoulder, left_wrist)
            angle_left_shoulder = calculate_angle(left_hip, left_elbow, left_wrist)

            # Calculate angle
            angle_right_elbow = calculate_angle(right_shoulder, right_elbow, right_wrist)
            angle_right_arm = calculate_angle(left_shoulder, right_shoulder, right_wrist)
            angle_right_shoulder = calculate_angle(right_hip, right_elbow, right_wrist)

            # Calculate angle
            angle_right_wrist = calculate_angle(z_right_hip, z_right_shoulder, z_right_wrist)
            angle_left_wrist = calculate_angle(z_left_hip, z_left_shoulder, z_left_wrist)
            

            
            # Visualize angle
            cv2.putText(image, str(angle_left_elbow), 
                           tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            # Visualize angle
            cv2.putText(image, str(angle_left_shoulder), 
                           tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
              # Visualize angle
            cv2.putText(image, str(angle_right_elbow), 
                           tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            # Visualize angle
            cv2.putText(image, str(angle_right_shoulder), 
                           tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            # Visualize angle
            cv2.putText(image, str(angle_right_wrist), 
                           tuple(np.multiply(right_wrist, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            
            
            # Curl counter logic
            if angle_left_elbow > 160 and (angle_left_shoulder > 85 and angle_left_shoulder < 140) and action is not "turn_left" :
                if stage_left_arm is None:
                    deb_counter_left_arm +=1
                if deb_counter_left_arm > 3:    
                    stage_left_arm = action = "turn_left" 
                    print(counter_left_arm, " ", stage_left_arm)
                    counter_left_arm +=1 
            else:
                stage_left_arm = None
                deb_counter_left_arm = 0


              # Curl counter logic
            if angle_right_elbow > 160 and (angle_right_shoulder > 85 and angle_right_shoulder < 140) and action is not  "turn_right":
                if stage_right_arm is None:
                    deb_counter_right_arm +=1
                if deb_counter_right_arm > 3:     
                    stage_right_arm = action = "turn_right"
                    print(counter_right_arm, " ","turn_right")
                    counter_right_arm += 1  
            else:
                stage_right_arm = None
                deb_counter_right_arm = 0  

              # Curl counter logic
            if angle_right_arm > 95 and angle_right_arm < 110  and (angle_right_wrist > 100 and angle_right_shoulder < 180) and action is not "stop":
                if stage_stop is None:
                    deb_counter_stop +=1
                if deb_counter_stop > 3:     
                    stage_stop = action = "stop" 
                    print(counter_stop, " ", "stop")
                    counter_stop +=1  
            else:
                stage_stop = None
                deb_counter_stop = 0     

            if angle_left_arm > 70 and angle_left_arm < 100  and (angle_left_wrist > 70 and angle_left_wrist < 100) and action is not "advance":
                if stage_advance is None:
                    deb_counter_advance +=1
                if deb_counter_advance > 3:    
                    stage_advance = action = "advance" 
                    print(counter_advance, " ", "advance") 
                    counter_advance +=1   
            else:
                stage_advance = None 
                deb_counter_advance = 0        
                       
        except:
            pass

        print (angle_left_arm, angle_left_wrist)
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (680,110), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'LEFT REPS', (10,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_left_arm), 
                    (10,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)  

        # Rep data
        cv2.putText(image, 'RIGHT REPS', (150,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_right_arm), 
                    (200,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        

        # Rep data
        cv2.putText(image, 'STOP REPS', (300,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_stop), 
                    (350,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        
        # Rep data
        cv2.putText(image, 'ADVANCE REPS', (450,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_advance),    
                    (500,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (325,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, action, 
                    (300,100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

