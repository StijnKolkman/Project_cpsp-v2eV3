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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #  This allows it to work on a cpu
print(f"Device: {device}")
video_folder = 'input/*.mov'

# Read video files
video_files = glob.glob(video_folder)                           #List of paths to the videos
batch_size = len(video_files)                                   #The batch size is equal to the amount of videos
if batch_size == 0:
    print("No video files found in the specified folder.")
    exit()

# define a emulator (set the settings of the emulator)
emulatorNew = EventEmulator(
    pos_thres           = 0.01,
    neg_thres           = 0.01,
    sigma_thres         = 0.03,
    cutoff_hz           = 1,
    leak_rate_hz        = 1, 
    batch_size          = batch_size,
    device              = device,
    refractory_period_s = 0.01,
    output_folder       = 'output/',

    # having this will allow H5 output
    dvs_h5              = 'eventsH5',

    # Other possible output format
    dvs_aedat2          = None, #currently doesnt work with batch processing
    dvs_aedat4          = None, #currently doesnt work with batch processing
    dvs_text            = None, #currently doesnt work with batch processing

    # True when the input is in hdr format (then it will skip the linlog conversion step)
    hdr                 = False,

    # If True, photoreceptor noise is added to the simulation
    photoreceptor_noise = False,

    #CSDVS --> This will enable csdvs, this function currently does not work with batch processing, so keep at None
    cs_lambda_pixels    = None,

    #scidvs
    scidvs              = False,

    # Parameter to show an image and to save it to an avi file. If show_dvs_model_state is enabled it will output the frames and will wait for a key press before it continues. Pressing 'x' will quite the process
    show_dvs_model_state = None, #['new_frame','diff_frame'], # options:['all','new_frame', 'log_new_frame','lp_log_frame', 'scidvs_highpass', 'photoreceptor_noise_arr', 'cs_surround_frame','c_minus_s_frame', 'base_log_frame', 'diff_frame'])
    output_height       =200, #output height of the windowg
    output_width        =200,
    save_dvs_model_state= True,

    #define shot noise or not 
    shot_noise_rate_hz  = 1,
    label_signal_noise  = False, #Currently doesnt work with batch processing and also doesnt work in the original code

    #record the state of a single pixel (input is tuple)
    record_single_pixel_states = (1, 10, 20) #example tuple containing pixel info: batch, height, width
)
# **IMPORTANT** make torch static, likely get faster emulation (might also cause memory issue)
torch.set_grad_enabled(False)

# Initialize resources and tensors
caps        = []                                                                    #here the videos will be saved
fps           = 0                                                                   #fps of the videos, should be same for th videos
num_of_frames = 0                                                                   #total number of frames. Should be the same for every video
duration      = 0                                                                   # Duration of the videos, should be the same for every video
delta_t       = 0                                                                   #The time between two frames. IS ASSUMED TO BE THE SAME FOR EVERY VIDEO
current_time  = 0                                                                   #Current time is not a tensor anymore since we can assume that every video has the same size

#loop over the video's in the input folder to get the frames and the information (fps, num_of_frames, duration, delta_t, current_time)
#maybe add something here to check that the videos really have the same fps and delta_t
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS: {}".format(fps))
    num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Num of frames: {}".format(num_of_frames))
    duration = num_of_frames/fps
    print("Clip Duration: {}s".format(duration))
    delta_t = 1/fps                                    
    print("Delta Frame Time: {}s".format(delta_t))
    print() 

new_events = None                                           #Initialise the new_events. Will be filled by the emulator with events
idx        = 0                                              #Initialise counter
N_frames   = 10                                              #Only Emulate the first N_frames of every video TODO: LATER REMOVE JUST TO MAKE TESTING TAKE LESS TIME!!!
ret        = torch.zeros(batch_size,device=device)          #Tensor that stores the return value of cap.read()


max_height  = 720                                                                        # Example max height
max_width   = 1280                                                                       # Example max width
channels    = 3                                                                          # RGB
frame_tensor = torch.zeros((batch_size, max_height, max_width, channels), device=device) # Tensor containing the frames
luma_frame_tensor = torch.zeros((batch_size, max_height, max_width), device=device)      # Tensor containing the luma_frames
weights     = torch.tensor([0.299, 0.587, 0.114],device=device).view(1, 1, 1, 3)         # Weights for transfer to grayscale, see:https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray 


while True:
    all_ret_false = True  # Flag to determine if all videos are done

    for i, cap in enumerate(caps):
        ret, frame_read = cap.read() 
        if ret:
            all_ret_false = False
            frame_tensor[i] = torch.from_numpy(frame_read).float().to(device)  # Convert frame (np array) to tensor
            #print("Size of frame_tensor[{}]: {}".format(i, frame_tensor[i].size()))
        else:
            # Append a zero tensor if video is finished
            frame_tensor[i] = torch.zeros((max_height, max_width), device=device) #Not really needed anymore since it is assumed that all files will have the same number of frames

    if all_ret_false:  # Stop if all videos are done
        break
    #print("Size of frame_tensor: {}".format(frame_tensor.size()))
    luma_frame_tensor = (frame_tensor * weights).sum(dim=-1)  # Compute luma frame
    #print("Size of luma_frame_tensor: {}".format(luma_frame_tensor.size()))
    # Generate events
    print("="*50)
    print(f"Processing batch {idx + 1} of in total {N_frames} batches")
    print("="*50)
    new_events = emulatorNew.generate_events(luma_frame_tensor, current_time)

    # Update current time
    current_time += delta_t

    # printing some events to look at the output
    if current_time > delta_t:
        print('Now going to print some events for you (batch, time, x, y, polarity):')
        print(new_events[0:4,:])

    # Log event statistics for the batch
    if new_events is not None:
        num_events = new_events.shape[0]
        start_t = new_events[0, 1]
        end_t = new_events[-1, 1]
        event_time = (new_events[-1, 1]-new_events[0, 1])
        event_rate_kevs = (num_events/delta_t)/1e3
        print("Number of Events: {}\n"
            "Duration: {}s\n"
            "Start T: {:.5f}s\n"
            "End T: {:.5f}s\n"
            "Event Rate: {:.2f}KEV/s".format(
                num_events, event_time, start_t, end_t,
                event_rate_kevs))

    print("="*50)
    idx += 1
    if idx >= N_frames:  # Limit for testing
        break

# Release resources
for cap in caps:
    cap.release()
print("Processing complete. Resources released.")