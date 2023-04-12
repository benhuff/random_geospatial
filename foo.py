import json
import os
from PIL import Image
import datetime

def coco_json_to_yolo(coco_file, output_folder):
    """ 
    # Example usage:
    coco_file = "coco_coco.json"        # Path to COCO formatted JSON file
    output_folder = "yolo_label_files"  # Output folder to store .txt files with YOLO format labels
    coco_json_to_yolo(coco_file, output_folder)
    """
    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image in coco_data["images"]:
        image_id = image["id"]
        image_file_name = image["file_name"]
        image_width = image["width"]
        image_height = image["height"]

        image_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

        txt_file_path = os.path.join(output_folder, os.path.splitext(image_file_name)[0] + ".txt")

        with open(txt_file_path, "w") as txt_file:
            for annotation in image_annotations:
                category_id = annotation["category_id"]
                bbox = annotation["bbox"]
                x_min, y_min, bbox_width, bbox_height = bbox
                x_center = x_min + bbox_width / 2
                y_center = y_min + bbox_height / 2

                # Convert bounding box coordinates to YOLO format
                yolo_x = x_center / image_width
                yolo_y = y_center / image_height
                yolo_width = bbox_width / image_width
                yolo_height = bbox_height / image_height

                # Write YOLO format label to txt file
                label = f"{category_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_width:.6f} {yolo_height:.6f}"
                txt_file.write(label + "\n")

    print("Conversion complete!")

def yolo_folders_to_coco(images_folder, annotations_folder, output_file, category_map=None):
    """ 
    # Example usage:
    images_folder = "."             # Folder containing the .png images
    annotations_folder = "."        # Folder containing the .txt files with YOLO format labels
    output_file = "coco_yolo.json"  # Output COCO formatted JSON file
    yolo_folders_to_coco(images_folder, annotations_folder, output_file, category_map={1: 'foo', 2: 'bar'})
    """
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d:%H:%M:%S")

    coco_data = {
        "info": {
            "description": "COCO formatted annotations for VGG Image Annotator (VIA)",
            "version": "1.0",
            "year": 2023,
            "contributor": "Foo Bar",
            "date_created": formatted_date
        },
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpeg")]
    count = 0
    categories_flag = set()
    for image_file in image_files:
        count+=1
        image_id = count

        image_file_path = os.path.join(images_folder, image_file)
        image_width, image_height = get_image_size(image_file_path)

        # Add image information to COCO data
        image_info = {
            "id": image_id,
            "file_name": image_file,
            "width": image_width,
            "height": image_height
        }
        coco_data["images"].append(image_info)

        txt_file_path = os.path.join(annotations_folder, os.path.splitext(image_file)[0] + ".txt")
        
        with open(txt_file_path, "r") as txt_file:
            for line in txt_file:
                label_parts = line.strip().split(" ")
                category_id = int(label_parts[0])
                x_center, y_center, bbox_width, bbox_height = map(float, label_parts[1:])
                x_min = int((x_center - bbox_width / 2) * image_width)
                y_min = int((y_center - bbox_height / 2) * image_height)

                # Recalculate bbox width and height using the new x_min, y_min, bbox_width, bbox_height values
                width = int(bbox_width * image_width)
                height = int(bbox_height * image_height)

                bbox = [x_min, y_min, width, height]

                # Add annotation information to COCO data
                annotation_info = {
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "segmentation": [[x_min, y_min, x_min + width, y_min,
                                      x_min + width, y_min + height,
                                      x_min, y_min + height]],
                    "area": width * height,
                    "iscrowd": 0
                }

                coco_data["annotations"].append(annotation_info)

                # Add category information to COCO data
                if category_id not in categories_flag:
                    if category_map is not None:
                        category_name = category_map[category_id]
                    else:
                        category_name = f"category_{category_id}"
                    category_info = {
                        "id": category_id,
                        "name": category_name,
                        "supercategory": "object"
                    }
                    coco_data["categories"].append(category_info)
                    categories_flag.update([category_id])

    with open(output_file, "w") as f:
        json.dump(coco_data, f)

    print(f"COCO formatted JSON file '{output_file}' generated successfully!")

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size
