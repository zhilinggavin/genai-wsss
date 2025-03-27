import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def get_intersection_mask(img1, img2):
    # Convert images to grayscale and apply Gaussian blur
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    gray_img1 = cv2.GaussianBlur(gray_img1, (7, 7), 0)
    gray_img2 = cv2.GaussianBlur(gray_img2, (7, 7), 0)

    # Apply thresholding to get binary images
    # _, thresh_img1 = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh_img2 = cv2.threshold(gray_img2, 30, 255, cv2.THRESH_BINARY)
    # Create masks from contours
    mask_img1 = np.zeros_like(gray_img1)
    mask_img2 = np.zeros_like(gray_img2)

    # Find contours for the whole image
    contours_img1, _ = cv2.findContours(gray_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_img2, _ = cv2.findContours(thresh_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(mask_img1, contours_img1, -1, color=255, thickness=cv2.FILLED)
    cv2.drawContours(mask_img2, contours_img2, -1, color=255, thickness=cv2.FILLED)

    combined_mask = cv2.bitwise_and(mask_img1, mask_img2)
    return combined_mask



def get_diff_masks(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2, "Input images must be grayscale" 
                
    # Compute the absolute difference between the blurred images
    diff = cv2.absdiff(img1, img2)
    # diff = np.where(img2 > img1, img2 - img1, 0) TODO: Check if this is better
    # Remove shape artefactes (hollow and decreased edges)
    combined_mask = get_intersection_mask(img1, img2)
    diff = cv2.bitwise_and(diff, diff, mask=combined_mask)

    # # normalise to [0,255] Useless!
    # diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Apply Gaussian Blur to reduce noise
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    # Apply Otsu's thresholding to get a binary mask
    _, diff_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return diff_mask

def get_boxes(img):
    middle_boxes = []
    height, width = img.shape
    left_half, right_half = img[:, :width // 2], img[:, width // 2:]

    left_contour = max(cv2.findContours(left_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, default=None)
    right_contour = max(cv2.findContours(right_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, default=None)
    if left_contour is None or right_contour is None:
        print(f"Failed to find two boxes for input image")
        return None

    def draw_box(contour, offset_x=0):
        x, y, w, h = cv2.boundingRect(contour)
        if offset_x == 0:
            mx_start, mx_end = x + offset_x + int(w * 0.45), x + offset_x + int(w * 0.75)
        else:
            mx_start, mx_end = x + offset_x + int(w * 0.25), x + offset_x + int(w * 0.55)
        my_start, my_end = y + int(h * 0.3), y + int(h * 0.75)
        # ax1.add_patch(plt.Rectangle((mx_start, my_start), mx_end - mx_start, my_end - my_start, edgecolor='blue', facecolor='none', linewidth=2))
        return (mx_start, my_start, mx_end - mx_start, my_end - my_start)

    left_middle_box = draw_box(left_contour)
    right_middle_box = draw_box(right_contour, offset_x=width // 2)

    middle_boxes.append((left_middle_box, right_middle_box))
    middle_boxes = np.concatenate(middle_boxes)
    return middle_boxes

def get_vessel_mask(img,thre=180,k=3,iter=1):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _ , mask1 = cv2.threshold(img, thre, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((k, k), np.uint8)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=iter)
    sure_fg = cv2.dilate(mask2, kernel, iterations=1)
    seg_mask = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, kernel, iterations=3)
    return seg_mask

'''
# TODO: Change conditions for improve more accurate mask
'''
def get_fib_mask(orig_img, manip_img, show=False, middle_boxes=None, debug=False):
    diff_mask = get_diff_masks(orig_img, manip_img)
    # Step 2: Morphological Opening and Dilation
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_fg = cv2.dilate(opening, kernel, iterations=1)

    # Step 3: Connected Component Analysis
    num_labels, labels_im = cv2.connectedComponents(sure_fg)

    # Step 4: Filter Components
    # 4.1 Minimum Component is 3% of the mask area
    # 4.2 Sort components by area in descending order and select the top 8
    mask_area = cv2.countNonZero(sure_fg)
    min_area = mask_area * 0.03 

    selected_mask = np.zeros_like(sure_fg)
    components = []
    for label in range(1, num_labels):
        component_mask = (labels_im == label).astype(np.uint8) * 255
        area = cv2.countNonZero(component_mask)
        if area >= min_area:
            components.append((area, component_mask))
    
    components.sort(key=lambda x: x[0], reverse=True)
    top_components = components[:8]
    output_image = cv2.cvtColor(manip_img.copy(), cv2.COLOR_GRAY2BGR)
    for area, mask in top_components:
        selected_mask = cv2.bitwise_or(selected_mask, mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))# Draw contours with the random color
        cv2.drawContours(output_image, contours, -1, random_color, 2)

    # Step 5: Filter Out Vessels Components in the Middle Box
    # TODO: Change conditions for selection
    # 5.1 Remove Components in the Middle Boxes
    # 5.2 Get Naive Vessel Mask to Remove Vessels
    
    vessel_mask = get_vessel_mask(manip_img,thre=180,k=3,iter=1)
    vessel_area = cv2.countNonZero(vessel_mask)
    middle_boxes = get_boxes(manip_img)
    for area, mask in top_components:        
        if middle_boxes is not None:
            total_boxes_area = 0
            for box in middle_boxes:
                box_mask = np.zeros_like(manip_img, dtype=np.uint8)
                cv2.rectangle(box_mask, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), 255, thickness=cv2.FILLED)

                # Calculate the intersection of the selected mask with the box
                intersection = cv2.bitwise_and(mask, box_mask)
                intersection_area = cv2.countNonZero(intersection)
                box_mask_area = cv2.countNonZero(box_mask)
                mask_area = cv2.countNonZero(mask)
                total_boxes_area += box_mask_area
                
                inter_vessel = cv2.bitwise_and(mask, vessel_mask)
                inter_vessel_area = cv2.countNonZero(inter_vessel)
                
                if intersection_area > 50 and debug:
                    print(f"Intersection area: {intersection_area} pixels, box area: {cv2.countNonZero(box_mask)}, mask area: {mask_area}, inter_vessel/mask: {(inter_vessel_area/mask_area):.2f}, Inter/mask: {(intersection_area / mask_area):.2f}")
                # Check if the intersection area is at least 10% of the box area
                if (intersection_area >= 0.15 * box_mask_area or intersection_area >= 0.28 * mask_area) and box_mask_area > 500 and vessel_area >= 45:
                    # output_image[inter_vessel==255]=[0,255,0]
                    if mask_area < 800 or intersection_area >= 0.55 * mask_area or (vessel_area > 200 and inter_vessel_area >= 0.3 * mask_area and box_mask_area>2000 and vessel_area < box_mask_area):
                    # Mark the unwanted area as red, then remove the mask from the selected mask
                        output_image[intersection == 255] = [255, 0, 0]
                        selected_mask[mask == 255] = 0
                
                cv2.rectangle(output_image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)

    # 5.2 Remove vessels by checking the intersection area with the vessel mask
    remove_vessel = False
    if middle_boxes is not None:
        if debug:
            print(f"Vessel area: {vessel_area}, Total boxes area: {total_boxes_area}, vessel/boxes: {vessel_area / total_boxes_area:.2f}")
        if vessel_area < 0.5 * total_boxes_area and total_boxes_area > 1000 and vessel_area > 200:
            output_image[vessel_mask==255]=[0,255,0]
            for area, mask in top_components:        
                # Calculate the intersection of the selected mask with the box
                inter_vessel = cv2.bitwise_and(mask, vessel_mask)
                inter_vessel_area = cv2.countNonZero(inter_vessel)
                mask_area = cv2.countNonZero(mask)
                
                if inter_vessel_area > 50 and debug:
                    print(f"Vessel Intersection area: {inter_vessel_area} pixels, mask area: {mask_area} Inter/mask: {(inter_vessel_area / cv2.countNonZero(mask)):.2f}")
                # Check if the intersection area is at least 10% of the box area
                if inter_vessel_area >= 0.12 * mask_area and inter_vessel_area > 50:
                    if mask_area < 800 or inter_vessel_area >= 0.3 * mask_area and box_mask_area>2000:
                    # Mark the unwanted area as red, then remove the mask from the selected mask
                        selected_mask[mask==255] = 0
                        remove_vessel = True
            
    # Step 6: Contour Detection on Selected Components
    filter_contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation_mask = np.zeros_like(selected_mask, dtype=np.uint8)
    cv2.drawContours(segmentation_mask, filter_contours, -1, 255, thickness=cv2.FILLED)
    if remove_vessel:
        segmentation_mask[vessel_mask==255]=0
    # Step 7: fill holes
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel, iterations=3)            
            
    if show:
        fig, axes = plt.subplots(2, 4, figsize=(12, 6)) 
        images = [orig_img, manip_img, diff_mask, opening]
        titles = ["Original Image", "Manipulated Image", 'diff (Blurred and filtered)','Opening']
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(title)
        
        images2 = [sure_fg, labels_im, output_image, segmentation_mask]
        titles2 = ['Dilation', 'Connected Components', f'Filtered Contour (num: {len(top_components)})', f'Segmentation Mask (num: {len(filter_contours)})']
        for i, (img, title) in enumerate(zip(images2, titles2)):
            axes[1, i].imshow(img, cmap='gray' if i != 1 else None)
            axes[1, i].set_title(title)

        for ax in axes.flatten():
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    return output_image, segmentation_mask, len(top_components), len(filter_contours)
