import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_synthetic_image(image_size, square_pos, circle_pos, shape_intensity, background_intensity):
    image = np.full(image_size, background_intensity, dtype=np.uint8)
    # Draw square
    top_left = (square_pos[0] - 20, square_pos[1] - 20)
    bottom_right = (square_pos[0] + 20, square_pos[1] + 20)
    cv2.rectangle(image, top_left, bottom_right, shape_intensity, -1)
    # Draw circle
    cv2.circle(image, circle_pos, 20, shape_intensity, -1)
    return image

def create_ground_truth_edge_map(image_size, square_pos, circle_pos):
    edge_map = np.zeros(image_size, dtype=np.uint8)
    # Draw square edges
    top_left = (square_pos[0] - 20, square_pos[1] - 20)
    bottom_right = (square_pos[0] + 20, square_pos[1] + 20)
    cv2.rectangle(edge_map, top_left, bottom_right, 255, 1)  # thickness=1
    # Draw circle edges
    cv2.circle(edge_map, circle_pos, 20, 255, 1)  # thickness=1
    return edge_map

def add_noise(image, noise_level):
    noisy_image = image.copy()
    # Calculate the number of black pixels to add as "pepper" noise
    num_pepper = int(noise_level * image.size)
    # Generate random coordinates for pepper noise (black pixels)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0  # Set pixels to black
    return noisy_image

def apply_edge_detection(image, canny_thresholds, sobel_threshold, laplacian_threshold):
    edges_canny = cv2.Canny(image, canny_thresholds[0], canny_thresholds[1])
    edges_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
    edges_sobel = cv2.convertScaleAbs(edges_sobel)
    _, edges_sobel = cv2.threshold(edges_sobel, sobel_threshold, 255, cv2.THRESH_BINARY)
    edges_laplacian = cv2.Laplacian(image, cv2.CV_64F)
    edges_laplacian = cv2.convertScaleAbs(edges_laplacian)
    _, edges_laplacian = cv2.threshold(edges_laplacian, laplacian_threshold, 255, cv2.THRESH_BINARY)
    return edges_canny, edges_sobel, edges_laplacian

def compute_metrics(ground_truth, detected_edges):
    gt_edges = (ground_truth > 0).astype(np.uint8)
    detected_edges = (detected_edges > 0).astype(np.uint8)
    TP = np.sum((gt_edges == 1) & (detected_edges == 1))
    FP = np.sum((gt_edges == 0) & (detected_edges == 1))
    FN = np.sum((gt_edges == 1) & (detected_edges == 0))
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_score = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
    return {'Precision': Precision, 'Recall': Recall, 'F1_score': F1_score}

def run_experiment(experiment_id, square_pos=(50, 50), circle_pos=(150, 150)):
    image_size = (200, 200)
    shape_intensity = np.random.randint(100, 255)
    background_intensity = np.random.randint(0, 100)
    noise_level = np.random.uniform(0.1, 0.75)
    canny_thresholds = (np.random.randint(50, 100), np.random.randint(150, 200))
    sobel_threshold = np.random.randint(50, 100)
    laplacian_threshold = np.random.randint(50, 100)
    
    image = create_synthetic_image(image_size, square_pos, circle_pos, shape_intensity, background_intensity)
    ground_truth_edges = create_ground_truth_edge_map(image_size, square_pos, circle_pos)
    image_noisy = add_noise(image, noise_level)
    edges_canny, edges_sobel, edges_laplacian = apply_edge_detection(image_noisy, canny_thresholds, sobel_threshold, laplacian_threshold)
    metrics_canny = compute_metrics(ground_truth_edges, edges_canny)
    metrics_sobel = compute_metrics(ground_truth_edges, edges_sobel)
    metrics_laplacian = compute_metrics(ground_truth_edges, edges_laplacian)
    
    return {
        'experiment_id': experiment_id,
        'image': image_noisy,
        'edges_canny': edges_canny,
        'edges_sobel': edges_sobel,
        'edges_laplacian': edges_laplacian,
        'metrics_canny': metrics_canny,
        'metrics_sobel': metrics_sobel,
        'metrics_laplacian': metrics_laplacian,
        'ground_truth_edges': ground_truth_edges,
        'parameters': {
            'shape_intensity': shape_intensity,
            'background_intensity': background_intensity,
            'noise_level': noise_level,
            'canny_thresholds': canny_thresholds,
            'sobel_threshold': sobel_threshold,
            'laplacian_threshold': laplacian_threshold
        }
    }

# Set seed for reproducibility
np.random.seed(42)
experiments = [run_experiment(i + 1) for i in range(3)]

metrics_list = []
for exp in experiments:
    metrics_list.append({'Experiment': exp['experiment_id'], 'Method': 'Canny', **exp['metrics_canny']})
    metrics_list.append({'Experiment': exp['experiment_id'], 'Method': 'Sobel', **exp['metrics_sobel']})
    metrics_list.append({'Experiment': exp['experiment_id'], 'Method': 'Laplacian', **exp['metrics_laplacian']})

metrics_df = pd.DataFrame(metrics_list)

# Generate the original synthetic image without any noise
original_image = create_synthetic_image((200, 200), (50, 50), (150, 150), shape_intensity=200, background_intensity=50)
original_edges_canny, original_edges_sobel, original_edges_laplacian = apply_edge_detection(
    original_image, canny_thresholds=(50, 150), sobel_threshold=70, laplacian_threshold=80
)

fig, axs = plt.subplots(len(experiments) + 2, 5, figsize=(15, 3 * (len(experiments) + 2)))

images = [original_image, original_edges_canny, original_edges_sobel, original_edges_laplacian, create_ground_truth_edge_map((200, 200), (50, 50), (150, 150))]
titles = ['Original Image', 'Canny Edges', 'Sobel Edges', 'Laplacian Edges', 'Ground Truth Edges']
for col, (img, title) in enumerate(zip(images, titles)):
    axs[0, col].imshow(img, cmap='gray')
    axs[0, col].set_title(title)
    axs[0, col].axis('off')
axs[0, 0].set_ylabel("Original", rotation=0, size='medium', labelpad=50, ha='right', va='center')

for idx, exp in enumerate(experiments):
    images = [exp['image'], exp['edges_canny'], exp['edges_sobel'], exp['edges_laplacian'], exp['ground_truth_edges']]
    for col, img in enumerate(images):
        axs[idx + 1, col].imshow(img, cmap='gray')
        axs[idx + 1, col].axis('off')
    params = exp['parameters']
    axs[idx + 1, 0].text(
        -0.35, 0.5,
        f'Exp {exp["experiment_id"]}\n'
        f'Shape Intensity: {params["shape_intensity"]}\n'
        f'Background: {params["background_intensity"]}\n'
        f'Noise Level: {params["noise_level"]:.2f}\n'
        f'Canny Thresh: {params["canny_thresholds"]}\n'
        f'Sobel Thresh: {params["sobel_threshold"]}\n'
        f'Laplacian Thresh: {params["laplacian_threshold"]}',
        transform=axs[idx + 1, 0].transAxes,
        fontsize=8, va='center', ha='right', rotation=0, color="black"
    )

metrics_df_sorted = metrics_df.sort_values(by=['Method', 'Experiment'])

methods = ['Canny', 'Sobel', 'Laplacian']
for col, method in enumerate(methods):
    method_data = metrics_df_sorted[metrics_df_sorted['Method'] == method]
    axs[-1, col + 1].bar(method_data['Experiment'], method_data['F1_score'], color='gray')
    axs[-1, col + 1].set_title(f'{method} F1 Scores')
    axs[-1, col + 1].set_ylim(0, 1)
    axs[-1, col + 1].set_xlabel('Experiment')
    axs[-1, col + 1].set_ylabel('F1 Score')
    for i, v in enumerate(method_data['F1_score']):
        axs[-1, col + 1].text(i + 1, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

axs[-1, 0].axis('off')
axs[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("edge_detection_experiment_results_with_labels.png")
plt.show()
