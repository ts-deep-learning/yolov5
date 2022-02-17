import json
import numpy as np
import pandas


# change this
JSON_PATH = "Project_1024.json"

def make_bins(x):

    # ['11-20', 'Bright', 'Low', 'Cracked', 12, 'No Disease']
    # starts from index 6 in x

    # Age Group
    dummy = np.zeros(10, dtype=int)
    dummy[int(x[6][0])] = 1
    x[6] = dummy

    # Lighting
    dummy = np.zeros(4, dtype=int)
    if x[7] == "Dark":
        dummy[0] = 1
    elif x[7] == "Overcast":
        dummy[1] = 1
    elif x[7] == "Bright":
        dummy[2] = 1
    else:
        dummy[3] = 1
    x[7] = dummy


    # Weed Density
    dummy = np.zeros(3, dtype=int)
    if x[8] == "Low":
        dummy[0] = 1
    elif x[8] == "Medium":
        dummy[1] = 1
    else:
        dummy[2] = 1
    x[8] = dummy


    #Soil Type:
    dummy = np.zeros(4, dtype=int)
    if x[9] == "Wet":
        dummy[0] = 1
    elif x[9] == "Dry":
        dummy[1] = 1
    elif x[9] == "Cracked":
        dummy[2] = 1
    else:
        dummy[3] = 1
    x[9] = dummy


    # Disease Severity
    dummy = np.zeros(4, dtype=int)
    if x[11] == "No Disease":
        dummy[0] = 1
    elif x[11] == "Low":
        dummy[1] = 1
    elif x[11] == "Medium":
        dummy[2] = 1
    else:
        dummy[3] = 1
    x[11] = dummy

    return x



def get_metadata(json_name):

    f = open(json_name)
    json_data = json.load(f)

    annotations, details = json_data["annotations"], json_data["details"]
    x = []

    for annotation in annotations:
        bboxs = annotation["bbox_info"]
        temp = []
        for bbox in bboxs:
            box_attrs = bbox["box_attr"]

            temp.append(box_attrs["CottonWithDriedFoliage"])
            temp.append(box_attrs["CottonBlurred"])
            temp.append(box_attrs["CottonPure"])
            temp.append(box_attrs["CottonwithDisease"])
            temp.append(box_attrs["CottonWithWeed"])
            temp.append(box_attrs["CottonPartial"])

            # need a break statement because there are cases where there are multiple crops in the image w multiple
            # bounding boxes, we are taking only one
            break

        temp.append(details["age_group"])
        temp.append(details["lighting"])
        temp.append(details["weed_density"])
        temp.append(details["soil_type"])
        temp.append(details["age_of_crop"])
        temp.append(details["disease_severity"])

        # types of age_group = (0-10), (11-20), (21-30), (31-40), (41-50)
        # types of lighting = Bright
        # types of weed_density = Low, Medium, High
        # types of soil_type = Dry, Cracked
        # types of disease_severity = No Disease, Low,

        temp = make_bins(temp)
        x.append(temp)

    f.close()

    x = np.array(x, dtype=object)
    return x

data = get_metadata(JSON_PATH)











'''
Structure:
    
    annotations:
        - original_image_path -> String
        - crop_coordinates -> [, , , ]
        - image_path -> String
        
        - bbox_info -> Dict
            - box_coordinates -> [ , , , ]
            - box_attr -> Dict
                - class -> String,
                - CottonWithDriedFoliage -> Boolean,
                - CottonWithRandomObject -> Boolean,
                - CottonBlurred -> Boolean,
                - CottonShadow -> Boolean,
                - CottonPure -> Boolean,
                - CottonwithDisease -> Boolean,
                - CottonGermination -> Boolean,
                - CottonWithWeed -> Boolean,
                - CottonOccludedByCotton -> Boolean,
                - CottonPartial-> Boolean
                
    details:
        # taking the useful ones only
        
        - state -> String
        - age_group -> String ("11-20")
        - time_of_capture -> String ("2 PM - 4 PM"),
        - lighting -> String ("Bright"),
        - weed_density -> String ("Low"),
        - soil_type": String ("Cracked"),
        - age_of_crop -> int (12),
        - crop -> String ("cotton"),
        - disease_severity -> String ("No Disease"),
                
'''











