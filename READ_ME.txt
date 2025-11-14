

🧠 Basic Idea

Capture hand landmarks using MediaPipe.

Extract key points (like fingertip positions).

Train a simple model to classify 10 gestures:

['hello', 'how', 'eat', 'sleep', 'write', 'yes', 'no', 'thankyou', 'good', 'morning']



⚙️ Requirements

Install these libraries first:

pip install opencv-python mediapipe scikit-learn numpy

pip install opencv-python
pip install mediapipe
pip install scikit-learn
pip install numpy


🧾 Step 1: Collect Data (Run once)

This script captures your gestures and saves them as landmark data for training.

press ‘s’ a few times per gesture to record samples.
(Do around 30–50 samples per gesture.)


🧾 Step 2: Train the Model


🧾 Step 3: Real-Time Prediction

It will show:

Your webcam feed.

The predicted sign name in real-time (from the 10 you trained).






KEY WORKING:- 

Step 1: Collect Data, the program will:

Open your webcam window.

Ask you to show a specific gesture (like "hello").

Draw hand landmarks (dots and lines) on your hand in real-time.

Now:

🧩 What you must do:

Make the gesture — for example, raise your hand in the shape that means “hello.”

Once your hand is visible and the blue dots appear (that means MediaPipe detected it),
press the ‘S’ key on your keyboard.
→ That saves one sample (one frame of that gesture) into a CSV file.

Try slightly changing your hand angle or position and press ‘S’ again.
Each press adds one more training example.

Collect 30 to 50 samples per gesture (so 30–50 times pressing ‘S’ while showing that gesture).

When done with that gesture, press ‘Q’ to move to the next gesture name in the list.



Gesture: hello → press S about 40 times → press Q  
Gesture: how   → press S about 40 times → press Q  
Gesture: eat   → press S about 40 times → press Q  
... and so on ...


After you finish all gestures, your file sign_data.csv will contain the training data used in Step 2 (training phase).
