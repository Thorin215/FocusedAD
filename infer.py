import os
import cv2
from focusedad.cpm import face_recognition
from focusedad.dpm import generate_prompt
from focusedad.fcm import infer
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

def init_face_models():
    """Initialize face detection and recognition models"""
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device,
        keep_all=True
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

def get_scene_ids():
    """Get scene_ids for all mp4 files in video folder"""
    video_path = os.path.join('demo_data', 'video')
    scene_ids = []
    
    for file_name in os.listdir(video_path):
        if file_name.endswith('.mp4'):
            scene_id = os.path.splitext(file_name)[0]
            scene_ids.append(scene_id)
            
    return scene_ids

def extract_frame(video, output_path, frame_id=0):
    """Extract specified frame from video and save as PNG"""
    try:
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_id >= total_frames:
            print(f"Warning: Requested frame {frame_id} exceeds total frames {total_frames}")
            return False
            
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = video.read()
        
        if ret:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)
            print(f"Successfully extracted frame {frame_id} to: {output_path}")
        else:
            print(f"Unable to read frame {frame_id} from video")
            
        return ret
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False

def load_demo_data(scene_id):
    """Load corresponding data based on scene_id"""
    base_path = os.path.join('demo_data')
    
    # Build relative paths
    video_rel_path = os.path.join('demo_data', 'video', f'{scene_id}.mp4')
    character_folder_rel_path = os.path.join('demo_data', 'character', scene_id)
    frame0_rel_path = os.path.join('demo_data', 'temp', f'{scene_id}_frame0.png')
    
    # Read video file
    video_path = video_rel_path
    video = cv2.VideoCapture(video_path)

    # Extract and save frame0
    output_path = os.path.join("demo_data/temp", f"{scene_id}_frame0.png")
    extract_frame(video, output_path, frame_id=0)

    # Get frame0 resolution
    frame0 = cv2.imread(output_path)
    if frame0 is not None:
        height, width = frame0.shape[:2]
        frame0_resolution = [height, width]  # Store as [width, height] list
    else:
        frame0_resolution = None
        print(f"Warning: Could not read frame0 for resolution: {output_path}")

    # Read character images
    character_images = {}
    if os.path.exists(character_folder_rel_path):
        for img_name in os.listdir(character_folder_rel_path):
            if img_name.endswith('.png'):
                img_path = os.path.join(character_folder_rel_path, img_name)
                character_name = os.path.splitext(img_name)[0]
                img = cv2.imread(img_path)
                character_images[character_name] = img

    # Read text prior file
    text_prior_path = os.path.join(base_path, 'text_prior', f'{scene_id}.txt')
    text_prior = ""
    if os.path.exists(text_prior_path):
        with open(text_prior_path, 'r', encoding='utf-8') as f:
            text_prior = f.read()

    return {
        'scene_id': scene_id,
        'video_path': video_rel_path,
        'character_folder': character_folder_rel_path,
        'frame0_path': frame0_rel_path,
        'frame0_resolution': frame0_resolution,  # Add resolution information
        'character_images': character_images,
        'text_prior': text_prior,
        'video': video
    }

def process_all_videos():
    """Process all video files"""
    scene_ids = get_scene_ids()
    
    for scene_id in scene_ids:
        print(f"\nProcessing scene {scene_id}")
        data = load_demo_data(scene_id)
        
        # Check data loading status and output relative paths
        if data['video'].isOpened():
            print(f"Video {scene_id} loaded successfully")
            print(f"Video path: {data['video_path']}")
            print(f"Character folder: {data['character_folder']}")
            print(f"Frame0 path: {data['frame0_path']}")
            print(f"Frame0 resolution: {data['frame0_resolution']}")  # Print resolution
            print(f"Found {len(data['character_images'])} character images")
            print(f"Text prior length: {len(data['text_prior'])} characters")
            # cpm
            mtcnn, resnet = init_face_models()
            character_results = face_recognition(data['frame0_path'], data['character_folder'], mtcnn, resnet)

            # dpm
            prompt, bboxes = generate_prompt(character_results, data['frame0_resolution'])
            # fcm
            output = infer(prompt, bboxes, data['video_path'])
            print(f"Inference result:",output)
            
            # Release video resources after processing
            data['video'].release()
        else:
            print(f"Failed to load video {scene_id}")

# Usage example
if __name__ == '__main__':
    try:
        process_all_videos()
    except Exception as e:
        print(f"Error occurred: {e}")
