import streamlit as st
import pandas as pd

#Setup page : 
st.set_page_config(page_title="Football Analysis Project - by Stolfi Matthias", page_icon="images/logo_ft.png")

# Presentation of the project --------------
st.title("🤖 Football Video Analysis ⚽︎")
st.markdown("""---""")

display_col=st.columns([3,1])
with display_col[0]:
    st.subheader("Intoduction")
with display_col[1]:
    st.link_button(label="Github Repo",url='https://github.com/matthias83100/football-analysis-ai')


st.write("""This is an image processing and AI project. 
         We will talk about object detection with YOLO model fine-tuning, 
         video processing, technics to make the result much better using fundamentals 
         of image processing theory, and of course a lot of Python! If you want to try it 
         you can check the code in my Github repository.
         """)

st.write("""
         Before get right into the work, let's make present our goal: 
         we want to make an object detection tool to track the players and analyse 
         the video in order to make some statistics and calculus about the game 
         (like a real world use case but here we will focus on a 30 seconds sample
          of a Bundesliga game). This project is inspired from the work of "Code in a Jiffy" 
          on Youtube.
         Let's check, below, an overview of the final result, to see what we expect 
         (anotation + tracking of player with most ball control).
        """)

# Final result video -------------------
st.image('images/presentation.png')



# Step by step explanations ------------
#1)=================
st.subheader("1) Training a fine-tuned model with YOLO:")

st.write("""
    In this section we will see how we do the object detection, in our case: 
    the players, the referees and the ball. In order to perform this, we will use some football 
    game photos available on a Roboflow dataset. 
    First of all we need to create an account on Roboflow and get an API key to download 
    the dataset. The dataset was already anoted (that's mean there are bounding boxes around 
    the objects of interest) as we can see (image below) there are not so many photos but it will 
    be enough for out use case! This dataset will help us a lot, because usually fine-tune a
    model can be a very long task just because of those anotaions (imagine you have to do this
         one by one and you have 500 or 1000 images...)
""")
st.image('images/roboflow.png')

st.write("""
    Now we have to create a Jupiter Notebook to prepare and train our fine-tuned model. 
    If you have a GPU, then you can run the code on your local. In my case I'm using Google
    Colab to get the free computation that they provide (we need it because the training is 
    using a lot of compation so the GPU is mandatory).
    The base model we will use is YOLO V5, to be more precise: the yolov5l (large), to 
         avoid problems with the extra large which is not runnable with Colab GPU,
          we can see the results are quite good in the following image:
    we have a very good detection for players and referee but the ball is the 
    main weakness of our model. But don't worry we will talk about the "interpolation method" 
    that will be very powerfull to fix our problem.
""")

st.image('images/eval_model.png')

st.write("""
    Now let's run this model on our video using OpenCV library to apply the prediction
    by looping on each frames. We can see it's a quite good result (following video). But 
    we will change "goalkeeper" into player to avoid miss-predictions 
    (we are assuming it's because our dataset is not big enough)
""")
# Import and display the video
video_yolov5 = open("videos/yolov5l.mp4", "rb")
video_yolov5_bytes = video_yolov5.read()
st.video(video_yolov5_bytes,autoplay=True,loop=True)

#2)=================
st.subheader("2) Trackers:")

st.write("""
We want now to implement a tracking system in order to give to each player an "id" and make 
some statistics and calculus. In video processing and AI, a tracker is an algorithm or model designed 
to follow the movement 
of objects across frames in a video. Once an object, such as a person, car, or ball, is 
detected in an initial frame, the tracker works to continuously locate and identify that 
object in subsequent frames, even as it moves or changes appearance. Trackers are essential 
for applications that require continuous observation of objects, such as surveillance, sports 
analysis, and autonomous driving.
""")
st.write("""
    Let's use a Python library called "supervision" that will give us the tracking 
    algorithm (ByteTrack). We will track only the players and the referee because the ball is alone
         so it's not interesting to track in this use case. (For the ball we will just do a supervison
         detection.) With the tracking we will obtain an id. 
         And basically, the algorithm is tracking the bounding boxes for each frames and it will be an 
         array with all the positions. When we got this we can drow the bounding boxes differently to 
         bring more clarity to the visualisation and add gamified anotations instead of big rectangles.
         As you can see on the video each player got his own number now (in a little rectangle) and an 
         ellipse is representing the bounding box, with a proportional size as before but now it's much 
         more readable.
         We also have the ball with a little triangle at the top. And we will see that we have some frame
         that are not detecting the ball, that's why we will need to make some interpolation later to 
         make it 
         works better.
""")

# Import and display the video
video_track = open("videos/trackers.mp4", "rb")
video_track_bytes = video_track.read()
st.video(video_track_bytes,autoplay=True,loop=True)


st.markdown("""
    If you want more details about ByteTrack Algorithm for Object Tracking here are a few explanations:

**ByteTrack** is a multi-object tracking (MOT) algorithm designed to assign consistent IDs to 
            objects across video frames, even in cases of occlusions or temporary disappearances.
             

1. **Object Detection**: ByteTrack begins by detecting objects in each frame using a detector (here YOLO) to 
            generate bounding boxes and confidence scores.

2. **High- and Low-Confidence Detections**: It splits detections into high-confidence
             (reliable) and low-confidence (less certain) groups to improve tracking stability.

3. **Track Association**:
   - **Hungarian Algorithm**: Matches high-confidence detections 
            with existing tracks based on **Intersection over Union (IoU)** scores.
   - **Low-Confidence Matching**: Uses low-confidence detections 
            to continue tracking objects that may be briefly occluded, increasing reliability.

4. **Updating Tracks**: Updates tracked object positions and IDs based on successful 
            associations. New tracks are created for unmatched detections, and untracked
             objects are removed after a set duration.

5. **Output**: ByteTrack outputs bounding boxes and IDs for each tracked object, maintaining continuity across frames.

**Advantages**: 
- Robust tracking by handling brief occlusions
- Accurate ID continuity across frames
- Efficient and real-time capable

**Limitations**: 
- Needs a reliable detector
- May struggle with fast-moving or visually similar objects 
             
            """)
#3)=================
st.subheader("3) Color/Team assignement:")
st.write("""
    In this section we will use some image image processing technics to get the color of each player
    and make a team assignement. Let's take an example for one player to illustrate the process:
    We will take a bouding box of a player and we want to apply a K-mean algorithm on it to create a 
    "mask" on the player himself and then calculate the average color of his T-shirt without having 
         the values of the background.
    First of all to make the clustering easier we have to crop the image to keep only the top 
         (part of the t-shirt) because in our video we don't have anyone doing an overhead kick!
         Then we can apply the algorithm, to do that we could create our own because it's not a very
         hard clustering method but scikit-learn library already have a very good one, so let's use it.
         When we have the mask we can apply it to the crop image and calculate the average value of 
         pixel colors and we obtain a yellow-green color for the first team and a white for the other one.
    """)

# Display images to show the process
im_col=st.columns(3)
with im_col[0]:
    st.write('Player image')
    st.image('images/player.png',width=150)
with im_col[1]:
    st.write("Top half cropped image")
    st.image('images/player_crop.png')
with im_col[2]:
    st.write('Mask on the player')
    st.image('images/player_mask.png')
    

st.write("""
    Now we can create a function to get the color for each player and with this color we can 
    assign a team color on the first frame and then we will loop on every frame and for each 
    get the color for each player. We can see the result is now fine on the video. We have the 
         ellipse with the good color for each player !
         """)


# Import and display the video
video_color = open("videos/color.mp4", "rb")
video_color_bytes = video_color.read()
st.video(video_color_bytes,autoplay=True,loop=True)


#4)=================
st.subheader("4) Ball interpolation:")

st.write("""
    In this section we will solve the problem of missing ball detections. In fact we have a lot of 
 frames that doesn't have any detection for the ball and this is a problem for our analysis. So let's 
         explain the process: we know the trajectory of the ball is quite "linear" so we will use 
         an interpolation to approximate the position of the detection when a frame has a missing value.
        to do this, let's use pandas library. If we convert the list of each ball positions we know as 
         a pandas DataFrame we can use some functions. We will first do an interpolation when we know
          the previous
         values and the next ones. For the last ones or the first ones if they are missing let's add
         a backfill to make sure there is no missing values. We use the following pandas command:   
         """)

st.code("""
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        """, language='python')

st.write("""
Now we have no more missing values, so we can add a new feature that is the player with the ball
for each frame (with little red triangle in the video). To achieve that we will do a simple method:
We will calculate the distance between the foot position (bottom of player bounding box)
          and the ball. If we are less than 70 pixels we consider a player has the ball.
         Now we can just have to pick the one with the minimal distance. We will se it's not 100% 
         accurate in every cases but it's still work fine most of the time.

         """)

# Import and display the video
video_interp = open("videos/interpolate.mp4", "rb")
video_interp_bytes = video_interp.read()
st.video(video_interp_bytes,autoplay=True,loop=True)


#5)=================
st.subheader("5) Team ball control:")
st.write("""
Now we will create a function to know the ball control for each team. It's very simple for that
         we just need to initialise an empty list and fill it with value for each frame: if a player
         has the ball we can know his team so we just add the team id in the list and if no one has the
         we just add the previous value to give the ball control to the same team. Then we just calculate 
         a percentage with the total of frames and we have a real-time statistic of the ball control!
         We can also add the color in the anotation to make it easier to understand which team is 
         team 1 and which one is team 2, as you can see on the video below. We can also implement the 
         "player ball control" to see the time of each players with the ball in their feet. (Check the 
         table below with time in second per player id)
         """)

# Import and display the video
video_contr = open("videos/ball_control.mp4", "rb")
video_contr_bytes = video_contr.read()
st.video(video_contr_bytes,autoplay=True,loop=True)

# Player ball control
df_ball_control= pd.read_csv('result_player_ball_control.csv')
frame_per_sec=24
df_ball_control['Control time (seconds)']=df_ball_control['Number of controls'].apply(lambda x: round(x/24,2))
df_col=st.columns(2)
with df_col[0]:
    with st.expander("Click to see the control time per player:"):
        st.dataframe(df_ball_control.loc[:,['Player ID','Control time (seconds)']],hide_index=True)



#6)=================
st.subheader("6) Camera movements:")

st.write("""
        In order to track the camera movement we will use cv2 (openCV), first things,
        we will track a "good feature" (with the function cv2.goodFeatureToTrack) and in our use
         case we will simply get the first 20 rows and the last 150 ones because they are not 
         suposed to move except if the camera is moving, that's how we can get a fix image in those 
         2 zones and it will allow us to track if there is any movement. Let's add those features 
         with goodFeatureToTrack
         and then we will use the "Lucas-Kanade algorithm" that will be the cv2.calcOpticalFlowPyrLK 
         function. And if it's detecting a movement then we have to change the "good feature" and that's how
         we will do to track if it's moving or not. But to avoid "fake movements" lets add a minimum distance
         to declare it as a movement. In our case we set 5 pixels, and as you can see it's quite accuarate
         and close to what our eyes see in the video. We will also display the X and Y coordinates to 
         have a precise value of the movements (check video below).         
         """)

# Import and display the video
video_cam = open("videos/camera_move.mp4", "rb")
video_cam_bytes = video_cam.read()
st.video(video_cam_bytes,autoplay=True,loop=True)


st.markdown("""
If you want more details about the theory behind this openCV functions here is a brief explanation:

1. **Principle of Optical Flow**
- Optical flow represents the movement of pixels across frames in a sequence. It aims 
to determine how each pixel shifts from one frame to the next.
- The Lucas-Kanade algorithm assumes that pixel intensities remain constant during 
movement and that displacements are small, which is ideal for videos with slow or gradual 
scene changes.

2. **Small Window Assumption** 
- Lucas-Kanade assumes each point in the image moves within a small window (or region)
 around itself.
- Within each small window, the algorithm assumes all pixels move similarly.
- This simplifies calculations by focusing on local changes and helps estimate point 
displacement accurately without interference from global variations.

""")
#7)=================
st.subheader("7) View Transformer:")

st.write("""
    This one is a little bit more complex and we need it to have a good estimation of the distances
         because in fact the view is not from the top its from the side so there is a perspective 
         angle that we need to correct if we want the good distances. Let me explain you with an image: 
         """)

st.image('images/schema.png')
st.write("""
    As you can see if we select a rectangle on the football field with this view we obtain a trapeze.
         And the half field line in blue is much shorter than the yellow one because of this perspective.
         But thanks to the rules of football every field has the same mesure so we can use the values 
         easy to find on google to create like a ratio and use the function cv2.perspectiveTransformer in order
         to make equal distances. But the limitation with this method is that we can calculate distances 
         only for player that are in this polygone (the trapeze or rectangle). It will be hard to do this for 
         all the field because of camera movement and views (we don't always have the full view of the 
         field).
         So basically we just create new "positions" for the players we track in the zone, and this will
         allow us to calculate some distances and have the result in meters, because the view will be 
         transformed into a rectangle without perspective instead of the trapeze.

         """)

#8)=================
st.subheader("8) Speed estimation:")

st.write("""
    Now let's add our last feature: distance and spead estimation thanks to the view transformer.
         To implement this we will need to select a number of frame (here we choose 5) to calculate 
         the difference of position between fame 1 and frame 5. That will give us the distance and after
         that we know the frame rate is 24 frame per seconds so we can calulate the speed with this value.
         After that we will just plot the values at the bottom of the player and as we can see in the video,
         we have only the player in the "zone" we choosed to do our transformations that have this annotation
         we cannot track all of the player in the field as we said before.

         """)

# Import the videos
video_origin = open("videos/08fd33_4.mp4", "rb")
video_origin_bytes = video_origin.read()

video_final = open("videos/final_video.mp4", "rb")
video_final_bytes = video_final.read()

# Display videos
col_video=st.columns(2)
with col_video[0]:
    st.write('Original video sample:')
    st.video(video_origin_bytes,autoplay=True,loop=True)
with col_video[1]:
    st.write('Final video sample:')
    st.video(video_final_bytes,autoplay=True, loop=True)

st.write("""
         Finally we got something that is working pretty well even if we still have little bugs. We assume
         this is due to the low number of images of our dataset and also because we are using the large model
         and not the extra large. We are supposed to run in without GPU and only with Google Colab.
         And we saw many concepts of the image processing and AI object detection and also how to achieve a 
         real world use case!
         """)


