from typing import List, Dict

def generate_prompt(detect_characters: List[Dict], resolution: List[int]) -> tuple:
    """Generate prompt and corresponding bboxes"""
    if not detect_characters:  # Empty list case
        return "Describe the [<region>] in detail in the video.", [[0, 0, resolution[1]-10, resolution[0]-10]]
    
    elif len(detect_characters) == 1:  # Single character case
        char_name = list(detect_characters[0].keys())[0]
        bbox = detect_characters[0][char_name][:4]
        prompt = f"The character name of [<region>] is {char_name}. Describe what {char_name} is doing."
        return prompt, [bbox]
    
    else:  # Multiple characters case
        names = []
        bboxes = []
        for char_dict in detect_characters:
            name = list(char_dict.keys())[0]
            bbox = char_dict[name][:4]
            names.append(name)
            bboxes.append(bbox)
        
        character_descriptions = [f"The character name of [<region>] is {name}" for name in names]
        objects_str = ". ".join(character_descriptions)
        names_desc = " and ".join(names)
        prompt = f"There are {len(names)} objects: {objects_str}. Describe the {names_desc} in detail in the video."
        return prompt, bboxes
