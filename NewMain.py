"""
----------------------------------------------------------------------------
Project Name       : Full GPU DVS Simulator
File Name          : NewMain.py
Authors            : Benedikte Petersen, Natalia Anna Prokopiuk, Stijn Kolkman
Date Created       : 2024-12-04
Last Modified      : YYYY-MM-DD
Description        : This script simulates events from video frames using a
                     GPU-based event emulator. It processes video input to 
                     generate events suitable for neuromorphic vision processing.

Usage              : Run the script in the v2e conda environment. For details on
                     creating the environment, see:
                     https://github.com/StijnKolkman/Project_cpsp-v2eV3.git

Command            : python NewMain.py
----------------------------------------------------------------------------
"""
from v2ecore.emulatorNew import EventEmulator                       # Import the v2e simulator
import torch                                                        # Torch
import cv2                                                          # To read video using OpenCV
import glob

# Configuration
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #  Probably this allows it to work on a 
video_folder = 'input/*.mov'

# define a emulator (set the settings of the emulator)
emulatorNew = EventEmulator(
    pos_thres          = 0.2,
    neg_thres          = 0.2,
    sigma_thres        = 0.03,
    cutoff_hz          = 200,
    leak_rate_hz       = 1,
    shot_noise_rate_hz = 10,
    device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# **IMPORTANT** make torch static, likely get faster emulation (might also cause memory issue)
torch.set_grad_enabled(False)

# Read video files
video_files = glob.glob(video_folder)
batch_size = len(video_files)                                   #The batch size is equal to the amount of frames
if batch_size == 0:
    print("No video files found in the specified folder.")
    exit()

# Initialize resources and tensors
caps        = []                                                                    #here the videos will be saved
fps           = torch.zeros(batch_size,device=device)                               #tensor containing the fps of overy video
num_of_frames = torch.zeros(batch_size, dtype=torch.int,device=device)              #tensor containing the number of frames of every video
duration      = torch.zeros(batch_size,device=device)                               #tensor containing the duration of every video
delta_t       = torch.zeros(batch_size,device=device)                               #tensor containing the delta_t of every video
current_time  = torch.zeros(batch_size,device=device)                               #tensor containing for the current time in every video

#loop over the video's in the input folder to get the frames and the information (fps, num_of_frames, duration, delta_t, current_time)
print() 
for i, video_file in enumerate(video_files): 
    print(f"Opening video {i+1}: {video_file}")
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened(): 
        print(f"Error opening video file: {video_file}")
        continue

    # Append the VideoCapture object to the list
    caps.append(cap)

    # Get the information
    fps[i] = cap.get(cv2.CAP_PROP_FPS)
    print("FPS: {}".format(fps[i]))
    num_of_frames[i] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Num of frames: {}".format(num_of_frames[i]))
    duration[i] = num_of_frames[i]/fps[i]
    print("Clip Duration: {}s".format(duration[i]))
    delta_t[i] = 1/fps[i]
    print("Delta Frame Tiem: {}s".format(delta_t[i]))
    print() 

new_events = None                                           #Initialise the new_events. Will be filled by the emulator with events
idx        = 0                                                     #Initialise counter
N_frames   = 2                                                #Only Emulate the first N_frames of every video TODO: LATER REMOVE JUST TO MAKE TESTING TAKE LESS TIME!!!
ret        = torch.zeros(batch_size,device=device)                 #Tensor that stores the return value of cap.read()

# TODO: is this the way to make this tensor?
max_height  = 720                                                                        # Example max height
max_width   = 1280                                                                        # Example max width
channels    = 3                                                                            # RGB
frame       = torch.zeros((batch_size, max_height, max_width, channels), device=device)       # tensor containing the frames
luma_frames = torch.zeros((batch_size, max_height, max_width), device = device)         # tensor containing the luma_frames
weights     = torch.tensor([0.299, 0.587, 0.114],device=device).view(1, 1, 1, 3)            # Weights for transfer to grayscale, see:https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray 

while(True): 
    # Capture frame-by-frame
    # TODO: what happens if ret is not True for one of the videos (for example if videos have different size)?
    # Capture a frame from each video
    for i, cap in enumerate(caps): #for all the videos (different cap) do:
        ret[i], frame_read = cap.read()  # Read a frame from the current video
        if ret[i]: # If the frame was read successfully
            frame[i] = torch.from_numpy(frame_read)
        else: 
            frame[i] = None  # Append None if frame not read
            
        if idx < N_frames: #TODO is this correct? it was this (if ret is True and idx < N_frames:). I think we should add something that if all ret[i] are false the programm should stop
            # convert it to Luma frame
            luma_frame = (frame * weights).sum(dim=-1) #The transform from rgb to grayscale (summed over last dimension: channels)

            # Now that the frame is a luma_frame we can start the  
            print("="*50)
            print("Currently calculating batch {} of in total {} batches".format(idx+1, N_frames)) # print function to track the progress
            print("-"*50)
            # emulate events --> emulatorNew.generate_events() calculates the events from the frame
            # **IMPORTANT** The unit of timestamp here is in second, a floating number
            new_events = emulatorNew.generate_events(luma_frame, current_time)
            #TODO change new_events ouput, it should be (b,t,x,y,p)
            
            ### The output events is in a numpy array that has a data type of np.int32
            ### THe shape is (N, 4), each row is one event in the format of (t, x, y, p)
            ### The unit of timestamp here is in microseconds --> TODO: change this, needt to be(b,t,x,y,p)

            # update time
            current_time += delta_t  #sum the two tensors to update the current time of every frame in the batch

            # print atats of the new event --> TODO: change such that it works with multiple frames in the batch
            if new_events is not None: 
                num_events      = new_events.shape[0]
                start_t         = new_events[0, 0]
                end_t           = new_events[-1, 0]
                event_time      = (new_events[-1, 0]-new_events[0, 0])
                event_rate_kevs = (num_events/delta_t)/1e3

                print("Number of Events: {}\n"
                    "Duration: {}s\n"
                    "Start T: {:.5f}s\n"
                    "End T: {:.5f}s\n"
                    "Event Rate: {:.2f}KEV/s".format(
                        num_events, event_time, start_t, end_t,
                        event_rate_kevs))
            idx += 1
            print("="*50)
    else: 
        break

# Release resources
for cap in caps:
    cap.release()
print("Processing complete. Resources released.")