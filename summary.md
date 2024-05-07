# AI Developer (Computer Vision Engineer) test Summary

Candidate: Ramadhan Gerry Akbar

## How does it work?
1. Future improvement parts like speed analysis are tricky because camera setup really influence the analytic performance, like too slow shutter speed leads to always blurry shots while too fast leads to crisp shot for first few thousands RPM but introduces confusion of classifying whether the Beyblade is already stopped or still spinning, especially if the shutter speed and video FPS combinations are matched with Beyblade rotational velocity
2. In main_traditional-cv.py, I have tried several methods to do beyblades detection:
   1. Background subtraction: frames differences, GMG
   2. Circle detection using Houghcircle
   Using all of those methods, I didn’t find any of those methods work reliably on public footages, including due to shadows. Sure, I believe We can tackle with combination of those methods + fine tuning using workaround e.g. digital stabilization for background, limiting the bounding boxes area, bbox’s aspect ratio, IoU, use tracking in case of missing detections, etc. but that will need quite a time

## Assumptions
1. The video was taken coaxial with the battle arena’s center axis
2. Arena, camera, and lighting are placed statically. No noticeable movement is allowed
3. The battle video analytic doesn’t need to be run in real-time conditions (analysis can be executed after the battle is finished)
4. Two Beyblades have visually similar radiuses
5. It’s better if there is no shadow, i.e. project lights from any direction

## Challenges during finishing this task
This Beyblade detection is an object detection from video task but suffers unique challenges, like
1. Proofing of work: finding a video footage for Beyblade battle which complies with all of the assumptions (static setup - camera and the arena, and the camera axis) are tremendously hard. I didn’t find any even small section of any footage which complies with those assumptions.
2. There is a state of the object that will be shown as blurry due to spinning much faster relative to the camera’s shutter speed
3. Sometimes, moves really fast when hit by other object at some occasions
4. Time is really short to do deeper fine-tuning any method, especially fine-tuning which includes supervised training

## How to run
### Setup
1. Create a YAML file just the way you can see in the repo
2. run `docker run -v <source-consisting-yml>:<path-inside-container> –gpus all ranger14/cilikno-ai-dev-comvis:latest` –config-path <path>.yml

### App interface
1. Enroll the arena by single left-click any 3 points throughout the perimeter 
2. Enroll the beyblade’s perimeter by right click first to pause the footage then click 3 points throughout the perimeter just like 1st
3. After a successful 3 points blade perimeter registration, press any key beside ‘q’ key

## Explanations
1. The model I chose for this task: beyblade detection using Grounding Dino with text prompt: "beyblades, spinning beyblades, beyblade" due to reasons I mentioned above
2. There is no any accuracy score yet since I haven’t tested this yet to lots amount of dataset
3. For every battle, there are additional data: spin direction and mass for every beyblade, and there are additional data for every timestamp: center coordinate (X,Y) and rotational speed for every beyblade. The reasons why those are important because key element to win a battle beyblade is increasing impulse on the enemy's beyblade over time and the varriables which impact are mass + rotational+translational speed + restituion