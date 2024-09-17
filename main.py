import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import requests
import cv2
import numpy as np
import craft_utils
import imgproc
from craft import CRAFT
from collections import OrderedDict
from scipy.spatial import distance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import spacy
import re
import random

class CRAFTTextExtractor:
    def __init__(self, model_path, cuda=True, text_threshold=0.7, link_threshold=0.4, low_text=0.4, canvas_size=1280, mag_ratio=1.5):
        self.cuda = cuda
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.net = CRAFT()
        self.load_model(model_path)        
        self.net.eval()
    
    def load_model(self, model_path):
        print('Loading weights from checkpoint (' + model_path + ')')
        if self.cuda:
            self.net.load_state_dict(self.copyStateDict(torch.load(model_path, map_location='cuda')))
            self.net = self.net.cuda()
            self.net = nn.DataParallel(self.net, device_ids=[1])
        else:
            self.net.load_state_dict(self.copyStateDict(torch.load(model_path, map_location='cpu')))
    
    def copyStateDict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def preprocess_image(self, image):
        # Resize and normalize the image
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

        if self.cuda:
            x = x.cuda()

        return x, ratio_w, ratio_h

    def extract_text_boxes(self, image_url):
        # Load image from URL
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        image_np = np.array(image)

        x, ratio_w, ratio_h = self.preprocess_image(image_np)

        with torch.no_grad():
            y, _ = self.net(x)

        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, poly=False)

        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        return boxes, polys

    def merge_close_boxes_on_boundary(self, boxes, distance_threshold=15):
        merged_boxes = []
        boxes = [np.array(box, dtype=np.float32) for box in boxes]
        
        while len(boxes) > 0:
            base_box = boxes.pop(0)
            to_merge = [base_box]
            not_merged = []

            for other_box in boxes:
                min_distance = np.min([distance.euclidean(p1, p2) for p1 in base_box for p2 in other_box])

                if min_distance < distance_threshold:
                    to_merge.append(other_box) 
                else:
                    not_merged.append(other_box)
            merged_box = np.vstack(to_merge)  
            x_min, y_min = np.min(merged_box[:, 0]), np.min(merged_box[:, 1])
            x_max, y_max = np.max(merged_box[:, 0]), np.max(merged_box[:, 1])
            merged_boxes.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

            boxes = not_merged

        return merged_boxes

class TextEntityExtractor:
    def __init__(self, craft_model_path, trocr_model_name='microsoft/trocr-large-printed'):
        # Initialize CRAFT model
        self.craft = CRAFTTextExtractor(craft_model_path)
        
        # Initialize TrOCR model and processor
        self.processor = TrOCRProcessor.from_pretrained(trocr_model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Initialize SpaCy
        self.nlp = spacy.load("en_core_web_sm")
        self.entity_unit_map = {
            'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
            'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
            'voltage': {'kilovolt', 'millivolt', 'volt'},
            'wattage': {'kilowatt', 'watt'},
            'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                            'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
        }
        self.unit_short_form_map = {
            'mg': 'milligram','g': 'gram','kg': 'kilogram','mm': 'millimetre','cm': 'centimetre','m': 'metre',
            'km': 'kilometre','lbs': 'pound','oz': 'ounce','ml': 'millilitre','l': 'litre','v': 'volt','w': 'watt',
            'wt': 'weight', 'ft': 'feet'
        }
        self.weight_context = {
            'item_weight': ['net weight', 'weight of', 'total weight', 'weighs', 'gross weight', 'weight', 'wt'],
#             'maximum_weight_recommendation': ['maximum weight', 'max weight', 'weight capacity', 'recommended weight', 'load-bearing', 'up to'],
            'width': ['width of', 'total width', 'overall width', 'product width', 'width', 'wide', 'w'],
            'depth': ['depth of', 'total depth', 'overall depth', 'product depth', 'depth', 'deep', 'd'],
            'height': ['height of', 'total height', 'overall height', 'product height', 'height', 'tall', 'high', 'h'],
            'voltage': ['voltage of', 'voltage rating', 'rated voltage', 'max voltage', 'voltage', 'v'],
            'wattage': ['wattage of', 'wattage rating', 'rated wattage', 'power rating', 'wattage', 'w'],
            'item_volume': ['volume of', 'total volume', 'overall volume', 'product volume', 'capacity', 'vol']
        }
        self.weight_context_2={
            'item_weight': ['net weight', 'weight of', 'total weight', 'weighs', 'gross weight', 'weight'],
            'maximum_weight_recommendation': ['maximum wt', 'max wt', 'recommended wt', 'load-bearing', 'up to'],
            'width': ['width of', 'total width', 'overall width', 'product width', 'width'],
            'depth': ['depth of', 'total depth', 'overall depth', 'product depth', 'depth'],
            'height': ['height of', 'total height', 'overall height', 'product height', 'height'],
            'voltage': ['voltage of', 'voltage rating', 'rated voltage', 'max voltage'],
            'wattage': ['wattage of', 'wattage rating', 'rated wattage', 'power rating'],
            'item_volume': ['volume of', 'total volume', 'overall volume', 'product volume', 'capacity']
        }
        self.add_entity_ruler()

    def add_entity_ruler(self):
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = []
        for entity, units in self.entity_unit_map.items():
            for unit in units:
                patterns.append({"label": entity, "pattern": [{"LIKE_NUM": True}, {"LOWER": unit}]})
        ruler.add_patterns(patterns)

    def replace_short_forms(self, text):
        for short_form, full_form in self.unit_short_form_map.items():
            pattern = re.compile(rf'(\d+)\s*{re.escape(short_form)}\b', re.IGNORECASE)
            text = pattern.sub(lambda match: f"{match.group(1)} {full_form}", text)
            pattern = re.compile(rf'\b{re.escape(short_form)}\b', re.IGNORECASE)
            text = pattern.sub(full_form, text)
        return text

    def reassign_entity_based_on_context(self, doc):
        proximity_threshold = 50  # Adjust as needed
        updated_entities = []
        for ent in doc.ents:
            assigned_entity_type = ent.label_
            for entity_type, keywords in self.weight_context.items():
                for keyword in keywords:
                    # Use regex to find keyword positions
                    for match in re.finditer(r'\b' + re.escape(keyword.lower()) + r'\b', doc.text.lower()):
                        keyword_pos = match.start()
                        if abs(keyword_pos - ent.start_char) <= proximity_threshold:
                            assigned_entity_type = entity_type
                            break
                if assigned_entity_type != ent.label_:
                    break
            updated_entities.append((assigned_entity_type, ent.text))
        return updated_entities

    def extract_text_from_crops(self, image_url, boxes):
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        extracted_texts = []
        for box in boxes:
            x_min = int(min([point[0] for point in box]))
            y_min = int(min([point[1] for point in box]))
            x_max = int(max([point[0] for point in box]))
            y_max = int(max([point[1] for point in box]))
            cropped_img = image.crop((x_min, y_min, x_max, y_max))
            pixel_values = self.processor(images=cropped_img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            extracted_texts.append(generated_text)
        return extracted_texts

    def draw_boxes(self,image, boxes):
        for box in boxes:
            box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        return image
    def merge_close_boxes_on_boundary(self, boxes, distance_threshold=15):
        merged_boxes = []
        boxes = [np.array(box, dtype=np.float32) for box in boxes]
        
        while len(boxes) > 0:
            base_box = boxes.pop(0)
            to_merge = [base_box]
            not_merged = []

            for other_box in boxes:
                min_distance = np.min([distance.euclidean(p1, p2) for p1 in base_box for p2 in other_box])

                if min_distance < distance_threshold:
                    to_merge.append(other_box) 
                else:
                    not_merged.append(other_box)
            merged_box = np.vstack(to_merge)  
            x_min, y_min = np.min(merged_box[:, 0]), np.min(merged_box[:, 1])
            x_max, y_max = np.max(merged_box[:, 0]), np.max(merged_box[:, 1])
            merged_boxes.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

            boxes = not_merged

        return merged_boxes
    def extract_entity(self, url, entity_name):

        entity_unit_map = {
            'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
            'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
            'voltage': {'kilovolt', 'millivolt', 'volt'},
            'wattage': {'kilowatt', 'watt'},
            'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                            'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
        }

        common_units = {
            "item_weight": ["gram", "kilogram"],
            'maximum_weight_recommendation': ["kilogram", "pound"],
            "height": ["centimetre", "metre"],
            "width": ["centimetre", "metre"],
            "depth": ["centimetre", "metre"],
            "voltage": ["volt"],
            "wattage": ["watt"],
            "item_volume": ["millilitre", "litre"]
        }
        
        def contains_number_without_unit(text, entity_name):
            # Check if text contains a number but no unit
            return any(char.isdigit() for char in text) and not any(unit in text for unit in entity_unit_map.get(entity_name, []))

        
        # Step 1: Text Detection
        boxes, _ = self.craft.extract_text_boxes(url)
        
        #output dekhne ke liye
        merged_boxes = self.craft.merge_close_boxes_on_boundary(boxes)
        image_b = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        image_with_boxes_b = self.draw_boxes(np.array(image_b), boxes)
        cv2.imwrite("heheheh.jpg", image_with_boxes_b)
        merged_boxes = self.merge_close_boxes_on_boundary(boxes)
        image_with_boxes_m = self.draw_boxes(np.array(image_b), merged_boxes)
        cv2.imwrite("heheheh_merged.jpg", image_with_boxes_m)
        
        # Step 2: Text Recognition
        recognized_texts = self.extract_text_from_crops(url, merged_boxes)
        combined_text = " ".join(recognized_texts)
        normalized_text = self.replace_short_forms(combined_text)
        #text dekhne ke liye
        print(normalized_text)
        
        # Step 3: Entity Extraction
        doc = self.nlp(normalized_text.lower())
        main_entities = self.reassign_entity_based_on_context(doc)

        # Logic to decide which entity to return
        
        for ent in doc.ents:
            print(f"Entity: {ent.label_}, Value: {ent.text}")
            for updated_entity, updated_text in main_entities:
                if ent.text == updated_text:
                    if ent.label_ == "item_weight" and updated_entity == entity_name:
#                         return f"Entity: {updated_entity}, Value: {updated_text}"
                        return f"{updated_text}"
        # If not found in main_entities, look in doc.ents
        for ent in doc.ents:
            if ent.label_ == entity_name:
#                 return f"Entity: {ent.label_}, Value: {ent.text}"
                return f"{ent.text}"
    
        for ent in doc.ents:
            if ent.label_ == entity_name:
                if contains_number_without_unit(ent.text, entity_name):
                    random_unit = random.choice(common_units[entity_name])
#                     print(f"Missing unit for {entity_name}, assigning {random_unit}")
                    return f"{ent.text} {random_unit}"
                return f"{ent.text}"

        # Handle possible misclassifications for height, width, and depth
        if entity_name == "height":
            for ent in doc.ents:
                if ent.label_ in ["depth", "width"]:  # Misclassified as depth or width
#                     print(f"Potential misclassification: Using {ent.label_} for height")
                    return f"{ent.text}"
        elif entity_name == "width":
            for ent in doc.ents:
                if ent.label_ in ["depth", "height"]:  # Misclassified as depth or height
#                     print(f"Potential misclassification: Using {ent.label_} for width")
                    return f"{ent.text}"
        elif entity_name == "depth":
            for ent in doc.ents:
                if ent.label_ in ["width", "height"]:  # Misclassified as width or height
#                     print(f"Potential misclassification: Using {ent.label_} for depth")
                     return f"{ent.text}"
    
        if entity_name == "depth" or entity_name ==  "width" or entity_name == "height":
            for ent in doc.ents:
                if ent.label_ == "CARDINAL":
                    return f"{ent.text}"

        for ent in doc.ents:
            if contains_number_without_unit(ent.text.lower(), entity_name):
                random_unit = random.choice(common_units[entity_name])
#                 print(f"Missing unit for {entity_name}, assigning {random_unit}")
                return f"{ent.text} {random_unit}"
            
#         def get_random_unit(entity_name):
#             """Get a random unit for the given entity_name."""
#             if entity_name in default_units:
#                 return random.choice(default_units[entity_name])
#             return ""
        
        
        return ""

import csv
import os
import pandas as pd

if __name__=="__main__":
    craft_model_path = 'craft_mlt_25k.pth'
    extractor = TextEntityExtractor(craft_model_path)
    entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
    }
    allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

    def clean_and_extract(text):
        pattern = re.compile(r'([+-]?[0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\s*([a-zA-Z]+)')
        match = pattern.search(text)
        
        if match:
            number = match.group(1) 
            unit = match.group(2)
            if unit in allowed_units:
                return f"{number} {unit}"
        
        return ""


    def process_input_csv(input_file, output_file, checkpoint_interval=10):
        # Check if output file already exists (resume from a checkpoint)
        if os.path.exists(output_file):
            output_df = pd.read_csv(output_file)
            completed_indices = set(output_df['index'].tolist())  # Completed rows
            print(f"Resuming from checkpoint. {len(completed_indices)} rows already processed.")
        else:
            completed_indices = set()

        # Read the input CSV file
        df = pd.read_csv(input_file)
        
        # Open the output file in append mode and use csv.writer
        with open(output_file, mode='a', newline='') as f_output:
            writer = csv.writer(f_output)
            
            # If the file is new, write the header
            if len(completed_indices) == 0:
                writer.writerow(['index', 'prediction'])

            # Iterate over the rows in the input CSV
            for index, row in df.iterrows():
                # if 60000<= index <= 90000:
                if row['index'] in completed_indices:
                    continue  # Skip already processed rows

                # Extract the necessary fields
                image_url = row['image_link']
                entity_name = row['entity_name']

                # Extract the entity value using the extractor (inference)
                entity_value = extractor.extract_entity(image_url, entity_name)

                # Write the result instantly
                writer.writerow([row['index'], clean_and_extract(entity_value)])
                f_output.flush()  # Ensure the data is written immediately

                # Savepoint logic: Check if we've reached the checkpoint interval
                if (index + 1) % checkpoint_interval == 0:
                    print(f"Checkpoint reached at index: {index + 1}. Saving progress...")
                    f_output.flush()  # Ensure all data is written to the file
        
        print("Inference completed and all data saved.")

    process_input_csv("test.csv", "test_out.csv")
