import os
import cv2
import csv
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

def process_image(image_path, output_folder, verify, verification_folder, epsilon_factor=0.0005):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:return
    height, width = img.shape
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:return

    all_points = np.vstack([contour for contour in contours])
    x_min, y_min = all_points.min(axis=0)[0]
    x_max, y_max = all_points.max(axis=0)[0]

    sample_width = x_max - x_min
    sample_height = y_max - y_min

    scale_factor = 10.0 / max(sample_width, sample_height)

    x_offset = (10 - sample_width * scale_factor) / 2
    y_offset = (10 - sample_height * scale_factor) / 2
    
    csv_rows = []
    
    if verify:
        plt.figure(dpi=300, figsize=(6,6))
        
    for cid, contour in enumerate(contours):
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        contour_points = []
        for pt in approx:
            x, y = pt[0]
            x_cm = (x - x_min) * scale_factor + x_offset
            y_cm = ((height - y) - y_min) * scale_factor + y_offset
            contour_points.append([x_cm, y_cm])
            csv_rows.append([cid, x_cm, y_cm])

        if len(contour_points) > 0:
            first_point = contour_points[0]
            last_point = contour_points[-1]
            if first_point[0] != last_point[0] or first_point[1] != last_point[1]:
                contour_points.append(first_point)
        
        if verify:
            xs = [p[0] for p in contour_points]
            ys = [p[1] for p in contour_points]
            plt.plot(xs, ys, 'b-', linewidth=1)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_csv = os.path.join(output_folder, base_name + ".csv")
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["contour_id", "x_cm", "y_cm"])
        for row in csv_rows:
            writer.writerow(row)
    
    if verify:
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis('off')
        ver_path = os.path.join(verification_folder, base_name + ".png")
        plt.savefig(ver_path, bbox_inches="tight", pad_inches=0)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Batch process binary images to generate contour CSV files for CAD import.")
    default_input = os.path.join(os.getcwd(), "Dataset", "Image_Data")
    parser.add_argument("--input_folder", type=str, default=default_input, help="Folder containing binary images")
    parser.add_argument("--output_folder", type=str, default="Contour_Output", help="Folder to save CSV files")
    parser.add_argument("--verify", action="store_true", default=True, help="Generate verification images")
    parser.add_argument("--epsilon", type=float, default=0.0005, help="Epsilon factor for contour approximation (smaller = more precise)")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    verification_folder = None
    if args.verify:
        verification_folder = os.path.join(args.output_folder, "Verification_Images")
        if not os.path.exists(verification_folder):
            os.makedirs(verification_folder)
    
    valid_ext = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if os.path.splitext(f)[1].lower() in valid_ext]
    
    for image_path in tqdm(files, desc="Processing images"):
        process_image(
            image_path, 
            args.output_folder, 
            args.verify, 
            verification_folder,
            epsilon_factor=args.epsilon
        )

if __name__ == "__main__":
    main()