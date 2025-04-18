import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

def init_models():
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

def face_recognition(image_path, character_folder, mtcnn=None, resnet=None):
    """
    Perform face recognition on input image
    
    Args:
        image_path: Path to input image
        character_folder: Path to character images folder
        mtcnn: Pre-loaded MTCNN model (optional)
        resnet: Pre-loaded ResNet model (optional)
    
    Returns:
        list: Recognition results list, format: [{'name': [x1,y1,x2,y2,conf,dist]}, ...]
    """
    # Initialize new models if not provided
    if mtcnn is None or resnet is None:
        mtcnn, resnet = init_models()

    try:
        # Read input image
        image = Image.open(image_path)
        
        # Detect faces
        boxes, probs = mtcnn.detect(image)
        faces = mtcnn(image)
        
        if boxes is None or probs is None:
            print("No faces detected in the input image.")
            return []

        # Get character list
        character_list = [os.path.join(character_folder, f) 
                         for f in os.listdir(character_folder) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not character_list:
            print("No character images found in the specified folder.")
            return []

        results = []
        # Process each detected face
        for face_idx, (face, box, prob) in enumerate(zip(faces, boxes, probs)):
            if prob < 0.7:  # Skip if confidence is too low
                continue
                
            print(f"\nProcessing face {face_idx} with confidence {prob:.3f}")
            
            # Get face embedding vector
            face_embedding = resnet(face.unsqueeze(0))
            
            # Compare with all character images
            best_match = None
            best_dist = float('inf')
            
            for char_path in character_list:
                char_name = os.path.splitext(os.path.basename(char_path))[0]
                
                # Get character face embedding vector
                char_img = Image.open(char_path)
                char_face = mtcnn(char_img)
                
                if char_face is None:
                    print(f"No face detected in character image: {char_name}")
                    continue
                    
                char_embedding = resnet(char_face)
                
                # Calculate distance
                dist = (face_embedding - char_embedding).norm().item()
                
                if dist < best_dist and dist < 1.3:  # Distance threshold 1.3
                    best_dist = dist
                    best_match = (char_name, [
                        float(box[0]), float(box[1]),  # x1, y1
                        float(box[2]), float(box[3]),  # x2, y2
                        float(prob),                   # confidence
                        float(dist)                    # distance
                    ])
            
            if best_match:
                results.append({best_match[0]: best_match[1]})
                print(f"Matched with {best_match[0]}, distance: {best_match[1][5]:.3f}")

        return results

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return []
