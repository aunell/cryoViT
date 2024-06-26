import numpy as np  
import csv
from PIL import Image, ImageDraw
import os


def load_image(image_path):
    image = Image.open(image_path)
    maxi= np.array(image).max()
    image= np.array(image)/maxi
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image = image.convert("L")
    return image

def load_text(text_path):
    with open(text_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        rows={}
        circles = []
        for index, row in enumerate(reader):
            rows[index]={}
            tool_type = row[0]
            rows[index]['tool_type'] = tool_type
            obj= row[1]
            rows[index]['obj'] = obj
            if obj == 'Plasma_membrane' or obj == 'Active_zone' or obj =='Endosome':
                rows[index]['length_obj'] = row[2]
                x_coords, y_coords= list(map(int, row[3].split(","))), list(map(int, row[4].split(",")))
                rows[index]['coords'] = list(zip(x_coords, y_coords))
                if tool_type == 1:
                    rows[index]['diameter'] = row[5]
            else:
                x_coords, y_coords= int(row[2]), int(row[3]) #).split(","))), list(map(int, row[3].split(",")))
                rows[index]['coords'] = (x_coords, y_coords)
                if tool_type == '1':
                    rows[index]['diameter'] = float(row[4]) #.split(","))
                    circles.append(index)
    return rows, circles

def crop_white_box(image_path):
  """Crops a white box from the bottom of an image.

  Args:
      image_path: Path to the image file.

  Returns:
      A new image object with the white box cropped.
  """
  img = load_image(image_path)
  img_id =image_path.split('/')[-1].split('.')[0]
  img_array = np.array(img)
  for index, row in enumerate(reversed(img_array)):
    if np.mean(row[row != 0]) < 254:
        bottom_index=index
        print(bottom_index) #bottom_index is 310
        break
  cropped_img = img.crop((0, 0, img.width, img.height - bottom_index))
  cropped_img.save(f'/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_{img_id}.png')
  return cropped_img


def draw_line_on_image(image, coordinates):
    draw = ImageDraw.Draw(image)

    # Draw the line
    draw.line(coordinates, fill="white", width=5)

    return image

def draw_circle_on_image(image, center, radius, color="white"):
    draw = ImageDraw.Draw(image)
    left = center[0] - radius
    top = center[1] - radius
    right = center[0] + radius
    bottom = center[1] + radius

    # Draw the circle
    draw.ellipse([(left, top), (right, bottom)], outline=color, fill=color, width=1)

    return image

def draw_square_on_image(image, center, side_length, color="white"):
    draw = ImageDraw.Draw(image)
    left = center[0] - side_length // 2
    top = center[1] - side_length // 2
    right = center[0] + side_length // 2
    bottom = center[1] + side_length // 2

    # Draw the square
    draw.rectangle([(left, top), (right, bottom)], outline=color, fill=color)

    return image

def create_trust_region(labelled_vesicles, padding=5):
    # Unzip the list of points into two lists for x and y coordinates
    x_coords, y_coords = zip(*[coord for coord, _ in labelled_vesicles])
    # Find the minimum and maximum x and y coordinates
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # The corners of the bounding box are (min_x, min_y) and (max_x, max_y)
    bounding_box = [(min_x-padding, min_y-padding), (max_x+padding, max_y+padding)]

    return bounding_box

def create_trust_region_coords(labelled_vesicles, padding=5):
    # Unzip the list of points into two lists for x and y coordinates
    coords = zip([coord for coord, _ in labelled_vesicles])
    # Find the minimum and maximum x and y coordinates
    coords_to_trust=[]
    for coord in coords:
        x, y = coord[0]
        x*=14
        y*=14
        for i in range(x-padding, x+padding):
            for j in range(y-padding, y+padding):
                coords_to_trust.append((i,j))
    # The corners of the bounding box are (min_x, min_y) and (max_x, max_y)
    return coords_to_trust

def create_annotations(input_dir, output_dir):
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Sort the files to ensure matching pairs of image and text files
    files.sort()

    # Iterate over the files in pairs (image file and text file)
    for i in range(0, len(files), 2):
        image_path = os.path.join(directory, files[i])
        text_path = os.path.join(directory, files[i+1])

        image = crop_white_box(image_path)
        text, circles = load_text(text_path)

        for key in text.keys():
            if key not in circles:
                image = draw_line_on_image(image, text[key]['coords'])
            else:
                try:
                    center = text[key]['coords']
                    radius = text[key]['diameter'] 
                    image = draw_circle_on_image(image, center, radius)
                except Exception as e:
                    print("Error is", e)
                    print(text[key])

        # Save the processed image with a new name
        # output_path = os.path.join(output, os.path.splitext(files[i])[0] + '.png')
        # image.save(output_path)

directory = '/pasteur/u/aunell/cryoViT/data/sample_data/l25/'
output = '/pasteur/u/aunell/cryoViT/data/sample_data/processed/'
create_annotations(directory, output)