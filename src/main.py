from action_recognition import ActionRecognition
import getopt, sys

def help(long_options):
    for _ in range(25): print("-", end="")
    print("\n\tHelp:")
    for _ in range(25): print("-", end="")
    print("\n\nArgument options:", long_options )
    print("-h --Help:\t\t Print the help instructions.")
    print("-v --Video:\t\t Opens a window with the video and point detections.")
    print("-c --Collect:\t\t Collects n videos and save to be trained.")
    print("-t --Training:\t\t Runs training of videos captured.")
    print("-d --Detection:\t\t Runs real time detection.")
    


    


def main():
    for _ in range(35): print("-", end="")
    print("\nScript to Recognize Actions IACOS")
    for _ in range(35): print("-", end="")

     # Optionspython
    options = "hvctd:"

    # Long options
    long_options = ["Help", "Video", "Collect", "Training", "Detection"] 

    # total arguments
    n = len(sys.argv)
    print("\nTotal arguments passed:", n)
    if(n == 1):
        help(long_options)
    
    # Arguments passed
    print("\nName of Python script:", sys.argv[0])
 
    print("\nArguments passed:", end = " ")
    for i in range(1, n):
        print(sys.argv[i], end = " ")

    argumentList = sys.argv[1:] 
    act = ActionRecognition()
   

    try:
        # Parsing argument
        arguments, _ = getopt.getopt(argumentList, options, long_options)
        

        # checking each argument
        for currentArgument, currentValue in arguments:
    
            if currentArgument in ("-h", "--Help"):
                help(long_options)
                
            elif currentArgument in ("-v", "--Video"):
                act.video_detection()
                
            elif currentArgument in ("-c", "--Collect"):
                act.collect_images()

            elif currentArgument in ("-t", "--Training"):
                act.training_model()

            elif currentArgument in ("-d", "--Detect"):
                act.real_time_detection()      
                
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))


if __name__ == "__main__":
    main()