import cv2


# resize without deforming aspect ratio
def resize_without_deforming_aspect_ratio(im):
    desired_size = 224

    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                            value=color)
        
    return new_im

#function for matrix transpose
def transpose(M):
    return list(map(list, zip(*M)))

def draw_keypoints_numbers(keypoints, image, color=(0, 255, 0)):

    h, _, _ = image.shape
    
    # Define the font type
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Define the font scale (multiplies the font's base size)
    font_scale = 1.5
    # Define the thickness of the text lines
    thickness = 2

    for i in range(len(keypoints)):
        if keypoints[i][2]:
            cv2.putText(image, str(i+1), (int(keypoints[i][0]), int(h-keypoints[i][1])), font, font_scale, color, thickness)
            
    return image  

def draw_cross(image, x, y, size, color):
    # Ensure the coordinates are integers
    x, y = int(x), int(y)
    
    # Draw horizontal line (arm of the cross)
    cv2.line(image, (x - size, y), (x + size, y), color, thickness=3)
    
    # Draw vertical line (arm of the cross)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness=3)
    
    return image    

def draw_points_and_lines(points, lines, image, path, text='', points_color=(0, 0, 255), lines_color=(0, 0, 255), text_color=(0, 0, 255)):
    h, _, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5

    img = image.copy()

    for line in lines:
        pt1 = int(line[0][0]), h-int(line[0][1])
        pt2 = int(line[1][0]), h-int(line[1][1])
        img = cv2.line(img, pt1, pt2, color=lines_color, thickness=5)

    cross_size = h//100 
    cross_size = cross_size if cross_size > 2 else 2 

    for point in points:

        img = draw_cross(
            img,
            x = int(point[0]), y = int(h-point[1]),
            size = cross_size,
            color = points_color
        )

        cv2.putText(img, str(point[2]), (int(point[0]), int(h-point[1])), font, font_scale, color=(0, 255, 0), thickness=2)     
        
    cv2.putText(img, text, (50, 50), font, font_scale, color=text_color, thickness=2)             

    cv2.imwrite(path, img)    