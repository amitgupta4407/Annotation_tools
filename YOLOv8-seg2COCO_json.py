import streamlit as st
from PIL import Image
import json



def yolo_to_coco(yolo_lines, image_width, image_height, img_name, category_names):
    coco_data = {
        "info": {
            "description": "my-project-name"
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": cat_id, "name": cat_name} for cat_id, cat_name in category_names.items()]
    }

    coco_data["images"].append(
        {
            "id": 1,
            "width": image_width,
            "height": image_height,
            "file_name": img_name
        }
    )

    image_id = 1  # Starting image ID

    for yolo_line in yolo_lines:
        parts = yolo_line.strip().split()
        # print(len(parts))
        if not parts:
            continue
        category_id = int(parts[0])
        segmentation = [float(x) for x in parts[1:]]
        
        x_values = []
        y_values = []

        for i in range(0,len(segmentation)-2,2):
            x_values.append(segmentation[i])
            y_values.append(segmentation[i+1])


        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)

        area = (max_x - min_x) * (max_y - min_y)
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        #unnormalizing
        ann = []
        for x,y in zip(x_values,y_values):
            ann.append(x*image_width)
            ann.append(y*image_height)

        annotation = {
            "id": len(coco_data["annotations"]),
            "iscrowd": 0,  # Assume single instance
            "image_id": image_id,
            "category_id": category_id + 1,  # Adjust category_id by 1 (0-based to 1-based)
            "segmentation": [ann],  # Use the converted segmentation
            "bbox": bbox,
            "area": area,
        }
        coco_data["annotations"].append(annotation)

    return coco_data


def main():
    st.title("YOLOv8 segment-annotation to COCO JSON annotation Conversion")

    # Add a sidebar for category names
    default_category_names = {
        0: 'RRR', 1: 'HRR', 2: 'BRR', 3: 'DCRR', 4: 'DRR', 5: 'TRR',
        6: 'BPR', 7: 'CRR', 8: 'HPR', 9: 'DPR', 10: 'DCPR',
        11: 'RPR', 12: 'FM', 13: 'FRKRR', 14: 'FRKBR'
    }
    user_category_names = st.sidebar.text_area("Custom Category Names (JSON Format)", json.dumps(default_category_names, indent=4), height= 500)
    category_names = json.loads(user_category_names)

    uploaded_file = st.file_uploader("Upload YOLO Annotation File", type="txt")
    if uploaded_file:
        yolo_lines = uploaded_file.getvalue().decode("utf-8").split("\n")
        
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            image_width, image_height = image.size

            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Image Width:", image_width)
            st.write("Image Height:", image_height)

            coco_data = yolo_to_coco(yolo_lines, image_width, image_height, uploaded_image.name, category_names)
            json_output = json.dumps(coco_data, indent=4)
            
            # Add a download button
            download_filename = "output.json"
            st.download_button("Download JSON", json_output, file_name=download_filename)

            st.text_area("COCO JSON Output", value=json_output, height= 350)


if __name__ == "__main__":
    main()


