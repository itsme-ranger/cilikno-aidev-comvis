# 1. Circle arena registration
# 2. beyblade radius registration
# 2.a tolerance
# 3. Object stability detection

# Number of frames to pass before changing the frame to compare the current
# frame against
FRAMES_TO_PERSIST = 10
THRESH_TO_PERSIST = 1
EXPERIMENT_NO = 29
RECORD_IT = False
VISUALIZE_IT = True

TIME_PRINT_SEC = 4
# Minimum boxed area for a detected motion to count as actual motion
# Use to filter out noise or small objects
MIN_SIZE_FOR_MOVEMENT = 100
# MIN_SIZE_FOR_MOVEMENT = 2000

# Minimum length of time where no motion is detected it should take
#(in program cycles) for the program to declare that there is no movement
MOVEMENT_DETECTED_PERSISTENCE = 100
MOVE_RANGE_PCNT = 5

SEC_AFTER_POLYGON_CREA = 1.0

# self.vidcap.get(cv2.CAP_PROP_FPS) = 30


import numpy as np
import cv2
from pathlib import Path
import glob
import pickle
import copy
import faiss
import yaml
import math
from datetime import datetime
from typing import overload
from colorthief import ColorThief
import numpy as np
import csv
from shapely.geometry import box
from torchvision.ops import box_convert

# CONF_YML_FPATH = "/Users/haimac/Documents/Projects/lamar-kerja_2024/Kecilin/vol/beyblades-det-conf.yml"
CONF_YML_FPATH = "/root/vol/beyblades-det-conf.yml"

conf_yml = None
with open(CONF_YML_FPATH) as f:
    try:
        conf_yml = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

_VID_PATH = conf_yml['vid_path']
_CIRCLE_3POINTS_COORDS_PKL = conf_yml['circle_3points_coords_pickle']
_SPEED_PLAYBACK = conf_yml['speed_playback']
_ENROLL_IMG_PATH = conf_yml['enroll_img']
_BEYBLADE_RAD_TOLERANCE = conf_yml['beyblade_radius_tolerance']
_RESIZE = conf_yml['resize']
if _RESIZE:
    _FRAME_HEIGHT_PX = conf_yml["resize_conf"]["frame_height_px"]
_GROUNDING_DINO_PARENT_PATH = conf_yml["grounding_dino_parent_path"]
_CSV_TARGET_PATH = conf_yml["csv_target_path"]
_CSV_TRAJECTORY_TARGET_PATH = conf_yml["csv_trajectory_target_path"]

###############
### GROUNDING DINO SETUP
CONFIG_PATH = os.path.join(_GROUNDING_DINO_PARENT_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(_GROUNDING_DINO_PARENT_PATH, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

from groundingdino.util.inference import load_model, load_image, predict, annotate

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

TEXT_PROMPT = "beyblades, spinning beyblades, beyblade"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

from PIL import Image
from matplotlib import cm
import groundingdino.datasets.transforms as T

def img_from_opencv_to_pil(img_opencv):
    cv2_im = cv2.cvtColor(img_opencv,cv2.COLOR_BGR2RGB)
    image_source = Image.fromarray(cv2_im)
    return image_source

def load_image_groundingdino_from_opencv(img_opencv):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # image_source = Image.open(image_path).convert("RGB")
    image_source = img_from_opencv_to_pil(img_opencv)
    # image_source = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed
### GROUNDING DINO SETUP END
###############

def IOU(box1, box2):
	""" We assume that the box follows the format:
		box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
		where (x1,y1) and (x3,y3) represent the top left coordinate,
		and (x2,y2) and (x4,y4) represent the bottom right coordinate """
	x1, y1, x2, y2 = box1	
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	iou = area_inter / area_union
	return iou

dilatation_size = 1
dilation_shape = cv2.MORPH_CROSS
# dilation_shape = cv2.MORPH_RECT
# dilation_shape = cv2.MORPH_ELLIPSE
element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
# dilatation_dst = cv.dilate(src, element)
# dilated = cv2.erode(th3, element, iterations=1)

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return (None, np.inf)
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)
    

class Circle3PointsGenMousePack:
    def __init__(self, mouse_pack_name, orig_height=None, frame_ith=-1) -> None:
        self.stored_points = []
        self.mouse_pack_name = mouse_pack_name
        
        self.frame_ith = frame_ith
        self.frame = None
        self.orig_height = orig_height
        self.aspect_ratio = None
        self.detected_radius = None
        self.detected_center_coord = None
    
    def get_radius(self):
        return self.detected_radius
    
    def get_center(self):
        return self.detected_center_coord
    
    def process_stored_points(self):
        p1 = self.stored_points[0]
        p2 = self.stored_points[1]
        p3 = self.stored_points[2]
        center, radius = define_circle(p1, p2, p3)
        self.detected_radius = radius
        self.detected_center_coord = center
        print(f"process_stored_points radius={radius} center={center}")

sharing_state = {
    'is_arena_enrolled': False,
    'is_paused': False
}
# mouse callback function
# a = optional param, for the sake of writing simplicity
def draw_circle_3points(event, x, y, flags, a: Circle3PointsGenMousePack):
    # if the left mouse button was clicked, record the starting
    # for k,v in sharing_state.items():
    #     print(f"k={k} {v} type={type(v)}")
    if (event == cv2.EVENT_LBUTTONDOWN) & (len(a.stored_points) <= 1):
        if ((not sharing_state['is_arena_enrolled']) and (a.mouse_pack_name == 'arena')) | ((sharing_state['is_arena_enrolled']) and (a.mouse_pack_name == 'beyblades') and sharing_state['is_paused']):
            print(f"Left button of mouse has been single clicked for name={a.mouse_pack_name}")

            a.stored_points.append([x, y])
            print(f"stored points len={len(a.stored_points)}")
        
    elif (event == cv2.EVENT_LBUTTONDOWN) & (len(a.stored_points) == 2):
        if ((not sharing_state['is_arena_enrolled']) and (a.mouse_pack_name == 'arena')) | ((sharing_state['is_arena_enrolled']) and (a.mouse_pack_name == 'beyblades') and sharing_state['is_paused']):
            print(f"Left button of mouse has been single clicked with len==2 for name={a.mouse_pack_name}")
            a.stored_points.append([x, y])
            a.process_stored_points()
            if (not sharing_state['is_arena_enrolled']) and (a.mouse_pack_name == "arena"):
                sharing_state['is_arena_enrolled'] = True

            current_height = a.frame.shape[0]
            # aspect_ratio = orig_height/orig_width

            stored_points_save = copy.deepcopy(a.stored_points)
            for i in range(len(a.stored_points)):
                stored_points_save[i] = list(map(lambda c: round(c*orig_height/current_height), a.stored_points[i]))
            
            print(f"a.stored_points={a.stored_points}")
            print(f"stored points len={len(a.stored_points)}")

            if a.mouse_pack_name == 'arena':
                with open(_CIRCLE_3POINTS_COORDS_PKL, 'wb') as handle:
                    pickle.dump(stored_points_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Dumping pkl to {_CIRCLE_3POINTS_COORDS_PKL} has been successfully executed")
            
            if a.mouse_pack_name == 'beyblades':
                sharing_state['is_paused'] = False
    
    # elif (event == cv2.EVENT_LBUTTONDBLCLK):
    elif (event == cv2.EVENT_RBUTTONDOWN):
        sharing_state['is_paused'] = not sharing_state['is_paused']
        print(f"double click a.mouse_pack_name={a.mouse_pack_name}, sharing_state['is_paused'] changed to ={sharing_state['is_paused']}")
    
    for x,y in a.stored_points:
        # cv2.circle(image, center_coordinates, radius, color (BGR), thickness)
        a.frame = cv2.circle(a.frame, (x, y), 3, (0, 0, 255), -1)

def enroll_empty_arena(event, x, y, flags, a: ImgEnroller):
    if (event == cv2.EVENT_RBUTTONDOWN):
        if (a.is_enrolled):
            print(f"Arena image is already enrolled, processing to enroll new arena image...")
        else:
            print(f"Arena image hasn't been enrolled yet, processing to enroll arena image...")
        a.enroll_img_n_save()
        print(f"Enroll arena success")


def best_pair(new_color, prev_boxes):
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return (None, np.inf)
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

vid_path = Path(_VID_PATH)
print(f"== Processing {vid_path.name} init ==")

# vidcap = cv2.VideoCapture(0)
vidcap = cv2.VideoCapture(_VID_PATH)

_FPS = vidcap.get(cv2.CAP_PROP_FPS)
_MOVE_RANGE_PCNT = MOVE_RANGE_PCNT/100*max(vidcap.get(3), vidcap.get(4))

obj = Circle3PointsGenMousePack("arena", orig_height=vidcap.get(4))
beyblade_obj = Circle3PointsGenMousePack("beyblades", orig_height=vidcap.get(4))

is_stored_points_restored = False
# pickle_fname = str(self.vid_path.parent / self.vid_path.stem)+'.pickle'
try:
    with open(_CIRCLE_3POINTS_COORDS_PKL, 'rb') as handle:
        obj.stored_points = pickle.load(handle)
        if _RESIZE:
            orig_height = vidcap.get(4)
            for i in range(len(obj.stored_points)):
                obj.stored_points[i] = list(map(lambda c: round(c*_FRAME_HEIGHT_PX/orig_height), obj.stored_points[i]))
            obj.process_stored_points()
        if len(obj.stored_points) != 3:
            print(f"len(obj.stored_points)={len(obj.stored_points)} != 3. Resetting obj.stored_points...")
            obj.stored_points = []
            is_stored_points_restored = False    
        else:
            is_stored_points_restored = True
            print(f"{_CIRCLE_3POINTS_COORDS_PKL} have been successfully loaded")
            sharing_state['is_arena_enrolled'] = True
except FileNotFoundError as e:
    print(f"WARNING-pickle! {_CIRCLE_3POINTS_COORDS_PKL} is not found")
    sharing_state['is_arena_enrolled'] = False

imshow_name_main = "Main detection (press 'q' to quit)"
cv2.namedWindow(imshow_name_main)
if not is_stored_points_restored:
    cv2.setMouseCallback(imshow_name_main, draw_circle_3points, obj)
cv2.setMouseCallback(imshow_name_main, draw_circle_3points, beyblade_obj)

# Init frame variables
first_frame = None
next_frame = None

# Init display font and timeout counters
font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0

# len <= FRAME_TO_PERSIST
frames_prior = []
# len <= THRESH_TO_PERSIST
frames_mask_prior = []
frames_mask_priorBS = []

bboxes_prev = []
xs = np.zeros((0,2))
ys = np.zeros((0,2))

prev_frame = None
if enroll_obj.is_enrolled:
    prev_frame = enroll_obj.get_enrolled_img()
    gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    print(f"gray={gray.shape}")
    
    if _RESIZE:
        orig_height = gray.shape[0]
        orig_width = gray.shape[1]
        aspect_ratio = orig_height/orig_width

        # scale_percent = 60 # percent of original size
        # scale_size = _FRAME_HEIGHT_PX/orig_height
        height = _FRAME_HEIGHT_PX
        width = int(height/aspect_ratio)
        dim = (width, height)
        gray = cv2.resize(gray, dim)
    
    frames_prior.append(gray)

is_battle_on=True
winner = None
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=round(_FPS))

prev_boxes = {}
prev_colors = np.zeros((0,3))

time_start_func = datetime.now()
while vidcap.isOpened():
    ret, obj.frame = vidcap.read()

    if ret and is_battle_on:
        obj.frame_ith += 1
        if _RESIZE:
            orig_height = obj.frame.shape[0]
            orig_width = obj.frame.shape[1]
            aspect_ratio = orig_height/orig_width

            # scale_percent = 60 # percent of original size
            # scale_size = _FRAME_HEIGHT_PX/orig_height
            height = _FRAME_HEIGHT_PX
            width = int(height/aspect_ratio)
            dim = (width, height)
            obj.frame = cv2.resize(obj.frame, dim)
            frame = copy.deepcopy(obj.frame)
            frameBS = copy.deepcopy(obj.frame)
            enroll_obj.set_current_frame(obj.frame)
        
        fgmask = fgbg.apply(obj.frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # print(f"fgmask shp={fgmask.shape} frame shape={frameBS.shape}")
        
        beyblade_obj.frame = obj.frame
        for x,y in obj.stored_points:
            # cv2.circle(image, center_coordinates, radius, color (BGR), thickness)
            obj.frame = cv2.circle(obj.frame, (x, y), 3, (0, 0, 255), -1)
        if len(obj.stored_points) == 3:
            radius = obj.get_radius()
            radius = round(radius)
            center = obj.get_center()
            center = (round(center[0]), round(center[1]))
            obj.frame = cv2.circle(obj.frame, center, radius, (255, 0, 0), 2)
        
        image_source, image = load_image_groundingdino_from_opencv(obj.frame)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        OFFSET_ENCIRCLE = 0.1
        detected_beyblade_cnt = 0
        colors = np.zeros((0,3))
        xs_temp = np.array([])
        ys_temp = np.array([])
        bbox_temp = np.zeros((0,4))
        for box in xyxy:
            h_ = box[3]-box[1]
            w_ = box[2]-box[0]
            top_left = [max(0,box[0]-round(w_*OFFSET_ENCIRCLE)), max(0, box[1]-round(h_*OFFSET_ENCIRCLE))]
            bot_right = [min(width,box[2]+round(w_*OFFSET_ENCIRCLE)), min(height, box[3]+round(h_*OFFSET_ENCIRCLE))]
            cropped = frame[top_left[0]:bot_right[0], top_left[1]:bot_right[1]]

            beyblade_circles = None

            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            # print(f"gray={gray.shape}")

            gray = cv2.equalizeHist(gray)
            # Blur it to remove camera noise (reducing false positives)
            gray = cv2.GaussianBlur(gray, (9, 9), 0)
            # print(f"gblur={gray.shape}")

            if beyblade_obj.detected_radius is not None:
                bb_rad = round(beyblade_obj.detected_radius)
                gap = round(_BEYBLADE_RAD_TOLERANCE/100*bb_rad)
                minDis = bb_rad*2 - gap
                minRadius = round(bb_rad - gap/2)
                maxRadius = round(bb_rad + gap/2)
                beyblade_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDis, param1=14, param2=25, minRadius=minRadius, maxRadius=maxRadius)
        
            # this bounding box will counts as beyblade
            if beyblade_circles is not None:
                beyblade_circles = np.round(beyblade_circles[0, :]).astype("int")
                for (x, y, r) in beyblade_circles:
                    # filter by ratio of circle area per rectangle area. If too small, pass
                    if (math.pi*r*r)/w_*h_ < 0.6:
                        continue
                    cv2.circle(obj.frame, (x, y), r, (0, 150, 150), 2)
                    cv2.rectangle(obj.frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

                    col_thief = ColorThief(frame)
                    col = col_thief.get_color(quality=15)
                    colors = np.append(colors, col, axis=0)
                    xs_temp = np.append(xs_temp, x)
                    ys_temp = np.append(ys_temp, y)
                    bbox_temp = np.append(bbox_temp, box, axis=0)
        
        # tracking
        index = faiss.IndexFlatL2(3)   # build the index
        index.add(prev_colors)        
        if obj.frame_ith > 1:
            _, I = index.search(colors, 1)
        else:
            I = range(len(colors))
            # todo: pick the best color by sorting the distance, if colors != 2
        xs = np.append(xs, np.zeros(2), axis=0)
        ys = np.append(ys, np.zeros(2), axis=0)
        bboxes = np.zeros((2,4))
        added_idx = set()
        for i in range(len(colors)):
            if len(added_idx) >= 2:
                break
            if I[i] in added_idx:
                continue
            prev_colors[I[i]] = colors[i]
            added_idx.add(I[i])
            xs[-1][I[i]] = xs_temp[i]
            ys[-1][I[i]] = ys_temp[i]
            bboxes[I[i]] = bboxes[i]
        
        for i in range(2):
            if i not in added_idx:
                xs[-1][i] = np.nan
                ys[-1][i] = np.nan
                box = np.zeros((4))
                box.fil(np.nan)
                bboxes[i] = box
        # Check whether the battle has finished or not
        for i,box in enumerate(bboxes):
            if box[0] == np.nan:
                continue
            if math.dist([xs[-1,i], ys[-1,i]], obj.get_center() > obj.get_radius()):
                winner = (i+1)%2
                is_battle_on = False
                break
            h_ = box[3]-box[1]
            w_ = box[2]-box[0]
            top_left = [max(0,box[0]-round(w_*OFFSET_ENCIRCLE)), max(0, box[1]-round(h_*OFFSET_ENCIRCLE))]
            bot_right = [min(width,box[2]+round(w_*OFFSET_ENCIRCLE)), min(height, box[3]+round(h_*OFFSET_ENCIRCLE))]
            cropped = fgmask[top_left[0]:bot_right[0], top_left[1]:bot_right[1]]

            thresh = cv2.threshold(cropped, 40, 255, cv2.THRESH_BINARY)[1]
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            significant_move = 0
            for c in cnts:
                # Save the coordinates of all found contours
                (x, y, w, h) = cv2.boundingRect(c)
                
                # If the contour is too small, ignore it, otherwise, there's transient
                # movement
                if cv2.contourArea(c) >= MIN_SIZE_FOR_MOVEMENT:
                    significant_move += 1
            
            if significant_move == 0:
                xs_temp = xs[round(-_FPS):,i]
                ys_temp = ys[round(-_FPS):,i]
                max_x = max(xs_temp)y
                min_x = min(xs_temp)
                max_y = max(ys_temp)
                min_y = min(ys_temp)

                if (max_x-min_x<=_MOVE_RANGE_PCNT) and (max_y-min_y<=_MOVE_RANGE_PCNT):
                    winner = (i+1)%2
                    is_battle_on = False
                    break
                    
        
        # If the first frame is nothing, initialise it
        if prev_frame is None:
            prev_frame = gray
            frames_prior.append(prev_frame)
        
        # delay_counter += 1

        ######

        # # Otherwise, set the first frame to compare as the previous frame
        # # But only if the counter reaches the appriopriate value
        # # The delay is to allow relatively slow motions to be counted as large
        # # motions if they're spread out far enough
        # if delay_counter > FRAMES_TO_PERSIST:
        #     delay_counter = 0
        #     first_frame = next_frame

        # Set the next frame to compare (the current frame)
        next_frame = gray

        # Compare the two frames, find the difference
        # frame_delta = cv2.absdiff(prev_frame, next_frame)
        # frame_delta = cv2.absdiff(frames_prior[0], next_frame)
        frame_delta = np.zeros(shape=gray.shape, dtype=np.uint8)
        for iframe in frames_prior:
            temp = cv2.absdiff(iframe, next_frame)
            # eroded = cv2.erode(temp, element, iterations=2)
            eroded = cv2.dilate(temp, element, iterations=2)
            frame_delta = cv2.add(frame_delta, eroded)
        # print(f"frame_delta={frame_delta.shape}")
        # eroded = cv2.erode(frame_delta, element, iterations=1)
        jthresh = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]
        # jthresh = cv2.threshold(eroded, 5, 255, cv2.THRESH_BINARY)[1]
        # print(f"jthresh={jthresh.shape}")
        
        frames_mask_prior.append(jthresh)
        if len(frames_mask_prior) > THRESH_TO_PERSIST:
            frames_mask_prior.pop(0)
        
        # temp_fr = frames_diff_prior[0]
        # for iframe in frames_diff_prior:
        #     temp_fr = cv2.absdiff(temp_fr, iframe)
        #     eroded = cv2.erode(temp_fr, element, iterations=1)
        #     thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        # frames_mask_prior.append(frame_delta)

        thresh = frames_mask_prior[0]
        # print(f"thresh.shape={thresh.shape}")
        for i, ithresh in enumerate(frames_mask_prior):
            # print(f"i={i} ithresh.shape={ithresh.shape} {thresh.shape}")
            thresh = cv2.bitwise_or(ithresh, thresh)
        
        # thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # # Fill in holes via dilate(), and find contours of the thesholds
        # thresh = cv2.dilate(thresh, None, iterations = 2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)
            
            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True
                
                # Draw a rectangle around big enough movements
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        jthresh = cv2.threshold(fgmask, 40, 255, cv2.THRESH_BINARY)[1]
        frames_mask_priorBS.append(jthresh)
        if len(frames_mask_priorBS) > THRESH_TO_PERSIST:
            frames_mask_priorBS.pop(0)
        
        thresh = frames_mask_priorBS[0]
        # print(f"thresh.shape={thresh.shape}")
        for i, ithresh in enumerate(frames_mask_priorBS):
            # print(f"i={i} ithresh.shape={ithresh.shape} {thresh.shape}")
            thresh = cv2.bitwise_or(ithresh, thresh)
        
        # thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # # Fill in holes via dilate(), and find contours of the thesholds
        # thresh = cv2.dilate(thresh, None, iterations = 2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cols = {}
        for i,box in enumerate(boxes):
            box_area = get_box_range(box)
            col = ColorThief(obj.frame[box_area])
            cols[i] = col.get_color(quality=12)
        
        box_tracks = best_pair(cols, prev_boxes)

        is_still_in_battle = {}
        is_total_in_battle = False
        # loop over the contours
        for c in cnts:
            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)
            
            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True
                
                # Draw a rectangle around big enough movements
                cv2.rectangle(frameBS, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center = shapely.Point(x+w/2, y+h/2)

                for k,box in cols.items():
                    box_p = shapely.box(box)
                    if box_p.contains(center):
                        is_still_in_battle[k] = True
        
        # Convert the frame_delta to color for splicing
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
        
        # out.write(image)
        # blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
        jthresh = cv2.cvtColor(jthresh, cv2.COLOR_GRAY2BGR)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        show_d = np.hstack(((frame_delta, frame)))
        # show = np.hstack(((th3_col, image_copy2)))
        eroded = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        show = np.hstack(((eroded, thresh)))
        # show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)
        show = np.vstack([show_d, show])

        prev_frame = next_frame
        frames_prior.append(next_frame)
        if len(frames_prior) > FRAMES_TO_PERSIST:
            frames_prior.pop(1)
        if RECORD_IT:
            vid_writer_out.write(show)
        
        tm_end_infer = datetime.now()

        frame_tm = (vidcap.get(cv2.CAP_PROP_FPS)*TIME_PRINT_SEC)
        if obj.frame_ith % frame_tm == 0:
            real_time_overall_dur = (tm_end_infer-time_start_func).total_seconds()
            print(f"{TIME_PRINT_SEC*obj.frame_ith / frame_tm}s of vid elapsed. real_time_overall_dur={real_time_overall_dur}s.")
            if obj.frame_ith != 0:
                real_time_avg_per_frame = real_time_overall_dur/obj.frame_ith*1000
                real_fps_avg = 1000/real_time_avg_per_frame
                print(f"real_time_avg_per_frame = {real_time_avg_per_frame}ms. real fps average={real_fps_avg}")

        if VISUALIZE_IT:
            cv2.imshow(imshow_name_main, obj.frame)
            cv2.imshow("raw processing", show)
            fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
            cv2.imshow("frameBS", np.hstack((fgmask, frameBS)))
            # Interrupt trigger by pressing q to quit the open CV program
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            if not sharing_state['is_paused']:
                key_pressed = cv2.waitKey(int(1000/fps/_SPEED_PLAYBACK))
            else:
                key_pressed = cv2.waitKey(0)
            # key_pressed = cv2.waitKey(int(1000/fps/_SPEED_PLAYBACK))
            if key_pressed & 0xFF == ord('q'):
                break 
    else:
        break

rpm = np.zeros((xs.shape[0], 2))
rpm.fill(np.nan)
vidcap.release()
# if RECORD_IT:
    # vid_writer_out.release()
    # self.results_markdown += print_n_ret("\n!! vid_writer_out.released !!\n\n")
cv2.destroyAllWindows()

###### csv writing
battle_id = time_start_func.strftime("%Y%M%D%H%M%S")


is_csv_file_exists = False
if os.path.exists(_CSV_TARGET_PATH):
    is_csv_file_exists = True
with open(_CSV_TARGET_PATH, mode='a+') as f:
    csv_writer = csv.writer(f)
    if not is_csv_file_exists:
        csv_writer.writerow(["battle_id","duration", "winner", "dir_1", "dir_2", "mass_1", "mass_2"])
    duration_str = duration.strftime("%H:%M:%S.%f")[:-3]
    csv_writer.writerow([battle_id, duration_str, winner])

is_csv_trajectory_file_exists = False
if os.path.exists(_CSV_TARGET_PATH):
    is_csv_trajectory_file_exists = True
with open(_CSV_TRAJECTORY_TARGET_PATH, mode='a+') as f:
    csv_writer = csv.writer(f)
    if not is_csv_trajectory_file_exists:
        csv_writer.writerow(["battle_id","time_elapsed", "1_x", "1_y", "1_w", "2_x", "2_y", "2_w"])
    for i in range(len(rpm[2])):
        csv_writer.writerow([battle_id, i/_FPS, xs[0][i], ys[0][i], rpm[0][i] xs[1][i], ys[1][i], rpm[1][i]])