import numpy as np
import torch
from functools import partial
from typing import Optional, List   
import sys
sys.path.append('focusedad/VideoRefer')
from videorefer import model_init, mm_infer
from videorefer.mm_utils import process_video

# Add sam2 path
sam2_path = "sam2"
sys.path.insert(0, sam2_path)
from sam2.build_sam import build_sam2_video_predictor

# Global variables to store initialized models
global_model = None
global_processor = None
global_tokenizer = None
global_predictor = None

def initialize_models_once():
    """Lazy loading initialization of models, only initialize on first call"""
    global global_model, global_processor, global_tokenizer, global_predictor
    
    if global_model is None:
        from videorefer.utils import disable_torch_init
        disable_torch_init()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize video understanding model
        global_model, global_processor, global_tokenizer = model_init("checkpoints")
        
        # Critical fix: Add tokenizer attribute to all model submodules
        for module in global_model.modules():
            module.tokenizer = global_tokenizer
        
        # Initialize SAM segmentation model
        global_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "checkpoints/sam2.1_hiera_large.pt",
            device=device
        )

def generate_video_masks(
    video_path: str,
    regions: List[List[float]],
    frame_idx: int = 0
) -> dict:
    """Generate segmentation masks for all frames in video (supports multiple regions)
    
    Args:
        video_path (str): Input video path
        regions (List[List[float]]): List of target regions, each region should be a bbox of 4 floats [x1, y1, x2, y2]
        frame_idx (int): Initial frame index, defaults to 0
        
    Returns:
        dict: Dictionary containing segmentation masks for each frame, format {frame_idx: {obj_id: mask_array}}
        
    Example:
        >>> generate_video_masks("video.mp4", [[0.1, 0.2, 0.5, 0.6], [0.3, 0.4, 0.7, 0.8]])
    """
    # Parameter validation
    for idx, region in enumerate(regions):
        if len(region) != 4:
            raise ValueError(f"Region {idx} must contain 4 coordinate values (x1, y1, x2, y2)")
        x1, y1, x2, y2 = region
        if not all(isinstance(c, (int, float)) for c in region):
            raise TypeError(f"Region {idx} contains non-numeric coordinates")
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Region {idx} has invalid coordinates: must satisfy x1 < x2 and y1 < y2")

    # Initialize inference state
    inference_state = global_predictor.init_state(video_path=video_path)
    global_predictor.reset_state(inference_state)

    # Add all regions to predictor
    for obj_id, region in enumerate(regions, start=1):  # Object IDs start from 1
        box = np.array(region, dtype=np.float32)
        global_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )

    # Generate video masks
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in global_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, obj_id in enumerate(out_obj_ids)
        }

    return video_segments

def uniform_sample(ids, sample_size):
    """Uniform sampling, repeat sampling when sample size is larger than actual frame count"""
    length = len(ids)
    if sample_size > length:
        # When sample size is larger than actual frames, return all frames and repeat to reach sample size
        repeated_ids = ids * (sample_size // length + 1)
        return repeated_ids[:sample_size]
    
    step = length / sample_size
    sampled_ids = [ids[int(i * step)] for i in range(sample_size)]
    return sampled_ids

def infer(prompt: str, regions: List[List[float]], video_path: str, frame_idx: int = 0, sampled_frames: int = 32):
    """Simplified video understanding inference function"""
    try:
        # Ensure models are initialized
        initialize_models_once()
        
        # Generate segmentation masks for all video frames
        video_segments = generate_video_masks(video_path, regions, frame_idx)
        
        print(f"Total video segments: {len(video_segments)}")  # Debug info
        
        # Sampling process
        sampled_ids = uniform_sample(list(video_segments.keys()), sampled_frames)
        print(f"Sampled frames: {len(sampled_ids)}")  # Debug info
        
        # Preprocess video and mask data
        video_tensor, frame_tensor, _, _ = process_video(
            video_path,
            processor=global_processor,
            aspect_ratio='square',
            frame_idx=sampled_ids
        )
        
        # Prepare mask tensor
        mask_list = []
        for obj_id in video_segments[0].keys():
            for sampled_id in sampled_ids:
                mask_list.append(video_segments[sampled_id][obj_id].squeeze(0))
        masks_tensor = torch.Tensor(np.array(mask_list)).unsqueeze(0).to(global_model.device)
        
        # Build annotation indices
        frame_nums = [len(sampled_ids)]
        num_objects = len(regions)
        ann_indices = [[list(range(frame_nums[0])) for _ in range(num_objects)]]
        
        # Execute multimodal inference
        output = mm_infer(
            video_tensor,
            prompt,
            model=global_model,
            tokenizer=global_tokenizer,
            masks=masks_tensor,
            frame=frame_tensor,
            ann_indices=ann_indices,
            frame_nums=frame_nums,
        )
        
        return output
        
    except Exception as e:
        print(f"Error in inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error during inference: {str(e)}"

