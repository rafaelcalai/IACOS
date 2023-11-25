import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import keras

class ActionRecognition:

    def __init__(self, __no_sequences=15, __sequence_length=30):
        self.__mp_holistic = mp.solutions.holistic
        self.__mp_drawing = mp.solutions.drawing_utils
        # Actions that we try to detect
        self.__actions = np.array(['stop', 'advance', 'turn_right', 'turn_left'])
        # Path for exported data, numpy arrays
        self.__DATA_PATH = os.path.join('../MP_Data')
        # Videos worth of data
        self.__no_sequences = __no_sequences
        # Videos are going to be 30 frames in length
        self.__sequence_length = __sequence_length
        self.__colors = [(245,117,16), (117,245,16), (16,117,245),(100,100,100)]
        self.__video_device = self.__detect_video_device()


    def video_detection(self):
        cap = cv2.VideoCapture(self.__video_device)
        with self.__mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                image, results = self.__mediapipe_detection(frame, holistic)
                print(results)
                self.__draw_landmarks(image, results)
		        
		        # Naming a window 
                cv2.namedWindow('OpenCV Feed', cv2.WINDOW_NORMAL) 
	  
		        # Using resizeWindow() 
                cv2.resizeWindow('OpenCV Feed', 1200, 800) 
                cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
		
            cap.release()
            cv2.destroyAllWindows()

    
    def collect_images(self, video_device):
        for action in self.__actions: 
            for sequence in range(self.__no_sequences):
                try: 
                    os.makedirs(os.path.join(self.__DATA_PATH, action, str(sequence)))
                except:
                    pass

        cap = cv2.VideoCapture(video_device)
        # Set mediapipe model 
        with self.__mp_holistic.Holistic(min_detection_confidence=0.85, min_tracking_confidence=0.85) as holistic:
            for action in self.__actions:
                # Loop through sequences aka videos
                for sequence in range(self.__no_sequences):
                    # Loop through video length aka sequence length
                    for frame_num in range(self.__sequence_length):
                        # Read feed
                        ret, frame = cap.read()

                        # Make detections
                        image, results = self.__mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        self.__draw_landmarks(image, results)
                    
                        # NEW Apply wait logic
                        if frame_num == 0: 
                            # Naming a window 
                            cv2.namedWindow('OpenCV Feed', cv2.WINDOW_NORMAL) 
    
                            # Using resizeWindow() 
                            cv2.resizeWindow('OpenCV Feed', 1200, 800) 
                            cv2.putText(image, 'STARTING COLLECTION', (50,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, '{} - Video Number {}/{}'.format(action, sequence+1, self.__no_sequences), (50,250), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)       
                            cv2.putText(image, 'Collecting frames for {} Video {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                                                
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(4000)
                        else: 
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                    
                        # NEW Export keypoints
                        keypoints = self.__extract_keypoints(results)
                        npy_path = os.path.join(self.__DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break  
            cap.release()
            cv2.destroyAllWindows()


    def training_model(self):
        label_map = {label:num for num, label in enumerate(self.__actions)}
        sequences, labels = [], []
        for action in self.__actions:
            for sequence in range(self.__no_sequences):
                window = []
                for frame_num in range(self.__sequence_length):
                    res = np.load(os.path.join(self.__DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])        

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        
        log_dir = os.path.join('../Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.__actions.shape[0], activation='softmax'))
        
        res = [.7, 0.2, 0.1]  
        self.__actions[np.argmax(res)]
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(X_train, y_train, epochs=1100, callbacks=[tb_callback])  
        res = model.predict(X_test)

        model.save('action.h5')
        model.load_weights('action.h5')   
        
        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        print(accuracy_score(ytrue, yhat))
    

    def real_time_detection(self):
        sequence = []
        sentence = []
        threshold = 0.95

        model = keras.models.load_model('action.h5')
        cap = cv2.VideoCapture(0)
        
        with self.__mp_holistic.Holistic(min_detection_confidence=0.85, min_tracking_confidence=0.85) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                image, results = self.__mediapipe_detection(frame, holistic)

                self.__draw_landmarks(image, results)
                keypoints = self.__extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]			
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if self.__actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(self.__actions[np.argmax(res)])
                                #print( res[np.argmax(res)])
                        else:
                            sentence.append(self.__actions[np.argmax(res)])
                            #print( res[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = self.__prob_viz(res, self.__actions, image, self.__colors)
                    
                    
                    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                cv2.imshow('Real Time Detection', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows() 
    
                     
    def __detect_video_device(self):
        video_device = 255
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                video_device = i
                break
        print("Video Device is: ", video_device)
        return video_device      
        
        
    def __mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results    
        
        
    def __draw_landmarks(self, image, results):
        self.__mp_drawing.draw_landmarks(image, results.pose_landmarks, self.__mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        self.__mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.__mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.__mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.__mp_holistic.HAND_CONNECTIONS) # Draw right hand connections           
    
    	
    def __prob_viz(self, res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)   
        return output_frame 
