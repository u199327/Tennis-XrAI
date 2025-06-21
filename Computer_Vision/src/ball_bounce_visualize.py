import cv2
import numpy as np
import matplotlib.pyplot as plt
import bisect
import seaborn as sns
from io import BytesIO
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from denoising_filters import detect_outliers, apply_gaussian_filter

class BallBounceVisualizer:
    def __init__(self, court_detector, ball_positions, player_1_strokes_indices=None, player_2_strokes_indices=None):
        self.court_detector = court_detector
        self.ball_positions = np.array(ball_positions, dtype=np.float32)
        self.original_court = self.court_detector.court_reference.court.copy()
        self.court = self.original_court.copy()

        # Remove outliers
        non_outlier_indices = detect_outliers(self.ball_positions)
        self.ball_positions = self.ball_positions[non_outlier_indices]

        # Smooth ball trajectory
        self.smoothed_ball_positions = apply_gaussian_filter(self.ball_positions)

        # Detect bounce indices
        self.bounce_indices  = self.detect_bounce_indices(self.smoothed_ball_positions[:, 1], order=15)

    def transform_coordinates(self, x, y):
        """Transform ball coordinates to match the court's perspective using homography."""
        point = np.array([[x, y]], dtype='float32')
        point = np.array([point])

        transformed_point = cv2.perspectiveTransform(point, self.court_detector.game_warp_matrix[-1])

        transformed_x, transformed_y = transformed_point[0][0]
        return transformed_x, transformed_y

    def smooth_positions(self, positions, window_length=11, polyorder=2):
        """
        Apply Savitzky-Golay filter to smooth the ball positions.
        :param positions: array-like of ball positions
        :param window_length: length of the filter window (must be odd)
        :param polyorder: order of the polynomial used to fit the samples
        :return: smoothed positions
        """
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(positions))

        # Apply Savitzky-Golay filter
        smoothed_x = savgol_filter(positions[:, 0], window_length, polyorder)
        smoothed_y = savgol_filter(positions[:, 1], window_length, polyorder)

        return np.vstack((smoothed_x, smoothed_y)).T

    def detect_bounce_indices(self, y_positions, order=10):
        """
        Detect local minima and maxima in y-coordinate array,
        then filter them based on their position relative to the net.
        :param y_positions: array-like of y positions
        :param order: number of neighbors to compare on each side
        :return: sorted array of indices where local minima or maxima likely occurred
        """
        y_array = np.array(y_positions, dtype=np.float32)
        valid_indices = ~np.isnan(y_array)
        cleaned_y = y_array[valid_indices]
        valid_positions = np.where(valid_indices)[0]

        net_start, net_end = self.court_detector.court_reference.net
        net_y = net_start[1]

        minima_indices = argrelextrema(cleaned_y, np.less, order=order)[0]
        maxima_indices = argrelextrema(cleaned_y, np.greater, order=order)[0]

        filtered_maxima_indices = [idx for idx in maxima_indices if cleaned_y[idx] < net_y]
        filtered_minima_indices = [idx for idx in minima_indices if cleaned_y[idx] > net_y]

        combined_indices = np.sort(np.concatenate((filtered_maxima_indices, filtered_minima_indices))).astype(int)
        return valid_positions[combined_indices]

    def visualize_bounce_heatmap(self, output_path='output/bounce_heatmap.png', bins=100):
        """
        Visualizes a heatmap of bounce positions on the court.

        :param output_path: Path to save the output heatmap
        :param bins: Number of bins for the histogram
        """
        court_img = self.original_court.copy()
        if len(court_img.shape) == 2 or court_img.shape[2] == 1:
            court_img = cv2.cvtColor(court_img, cv2.COLOR_GRAY2BGR)

        net_start, net_end = self.court_detector.court_reference.net
        cv2.line(court_img, net_start, net_end, (255, 255, 255), 5)

        bounce_coords = [
            self.transform_coordinates(self.smoothed_ball_positions[idx][0], self.smoothed_ball_positions[idx][1])
            for idx in self.bounce_indices
        ]
        bounce_coords = np.array(bounce_coords)

        if len(bounce_coords) == 0:
            print("No bounce coordinates found.")
            return

        plt.figure(figsize=(10, 6))

        sns.kdeplot(x=bounce_coords[:, 0], y=bounce_coords[:, 1], cmap="Reds", fill=True, alpha=0.5, bw_adjust=0.35, thresh=0.05)

        plt.imshow(cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB), extent=[0, court_img.shape[1], court_img.shape[0], 0])
        plt.title('Bounce Heatmap')
        plt.axis('off')

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def visualize_bounces_image(self, output_path='output/bounce_visualization.png'):
        court_img = self.original_court.copy()
        court_img = cv2.cvtColor(court_img, cv2.COLOR_GRAY2BGR)

        net_start, net_end = self.court_detector.court_reference.net
        cv2.line(court_img, net_start, net_end, (255, 255, 255), 5)

        for idx, (x, y) in enumerate(self.smoothed_ball_positions):
            transformed_x, transformed_y = self.transform_coordinates(x, y)
            center = (int(transformed_x), int(transformed_y))

            if idx not in self.bounce_indices:
                cv2.circle(court_img, center, 10, (0, 255, 0), -1)
            else:
                size = 35
                cv2.line(court_img, (center[0] - size, center[1] - size),
                        (center[0] + size, center[1] + size), (0, 0, 255), 4)
                cv2.line(court_img, (center[0] - size, center[1] + size),
                        (center[0] + size, center[1] - size), (0, 0, 255), 4)
                cv2.putText(court_img, str(idx + 1), (center[0] + 15, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            if idx % 9 == 0:
                cv2.putText(court_img, str(idx + 1), (center[0] + 15, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        visualized_court_rgb = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(visualized_court_rgb)
        plt.title('Ball Bounce Visualization')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()


    def visualize_bounce_depth_distribution(self, output_path='output/bounce_depth_distribution.png'):
        court_img = self.original_court.copy()
        if len(court_img.shape) == 2 or court_img.shape[2] == 1:
            court_img = cv2.cvtColor(court_img, cv2.COLOR_GRAY2BGR)

        net_start, net_end = self.court_detector.court_reference.net
        net_y = int((net_start[1] + net_end[1]) / 2)

        top_baseline_y = int(self.court_detector.court_reference.baseline_top[0][1])
        bottom_baseline_y = int(self.court_detector.court_reference.baseline_bottom[0][1])

        top_third = (net_y - top_baseline_y) // 3
        bottom_third = (bottom_baseline_y - net_y) // 3

        sections = {
            "Top-Deep": (top_baseline_y, top_baseline_y + top_third),
            "Top-Medium": (top_baseline_y + top_third, top_baseline_y + 2 * top_third),
            "Top-Short": (top_baseline_y + 2 * top_third, net_y),
            "Bottom-Short": (net_y, net_y + bottom_third),
            "Bottom-Medium": (net_y + bottom_third, net_y + 2 * bottom_third),
            "Bottom-Deep": (net_y + 2 * bottom_third, bottom_baseline_y)
        }

        bounce_coords = [
            self.transform_coordinates(self.smoothed_ball_positions[idx][0], self.smoothed_ball_positions[idx][1])
            for idx in self.bounce_indices
        ]
        bounce_coords = np.array(bounce_coords)

        if len(bounce_coords) == 0:
            print("No bounce coordinates found.")
            return

        section_counts = {k: 0 for k in sections}
        for _, y in bounce_coords:
            for section, (y_min, y_max) in sections.items():
                if y_min <= y < y_max:
                    section_counts[section] += 1
                    break

        top_total = sum(section_counts[sec] for sec in ["Top-Short", "Top-Medium", "Top-Deep"])
        bottom_total = sum(section_counts[sec] for sec in ["Bottom-Short", "Bottom-Medium", "Bottom-Deep"])

        section_percents = {}
        for section in sections:
            total = top_total if section.startswith("Top") else bottom_total
            section_percents[section] = (section_counts[section] / total * 100) if total > 0 else 0.0

        left_x = int(self.court_detector.court_reference.left_court_line[0][0])
        right_x = int(self.court_detector.court_reference.right_court_line[1][0])

        for y_min, y_max in sections.values():
            cv2.line(court_img, (left_x, y_min), (right_x, y_min), (0, 255, 255), 2)
        cv2.line(court_img, (left_x, court_img.shape[0] - 1), (right_x, court_img.shape[0] - 1), (255, 255, 255), 2)

        for x, y in bounce_coords:
            x = int(x)
            y = int(y)
            cv2.drawMarker(court_img, (x, y), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=60, thickness=15)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_thickness = 3
        text_color = (0, 255, 255)

        for section, (y_min, y_max) in sections.items():
            label = f"{section.split('-')[1]}: {section_percents[section]:.1f}%"
            y_center = (y_min + y_max) // 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = (court_img.shape[1] // 2) - (text_size[0] // 2)
            text_y = y_center + (text_size[1] // 2)
            cv2.putText(court_img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        court_rgb = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(court_rgb)
        plt.title('Bounce Depth Distribution')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()


    
    def fit_parabola(self, x, y):
        """Fit a quadratic curve to the given x and y coordinates."""
        coefficients = np.polyfit(x, y, 2)
        return coefficients

    def plot_trajectory_on_court_image(self, output_path='output/trajectory_on_court.png'):
        court_img = self.original_court.copy()
        if len(court_img.shape) == 2 or court_img.shape[2] == 1:
            court_img = cv2.cvtColor(court_img, cv2.COLOR_GRAY2BGR)

        net_start, net_end = self.court_detector.court_reference.net
        cv2.line(court_img, net_start, net_end, (255, 255, 255), 5)

        for bounce_idx in self.bounce_indices:
            start_idx = max(0, bounce_idx - 20)
            segment = self.smoothed_ball_positions[start_idx:bounce_idx]

            if len(segment) < 3:
                continue

            transformed = [self.transform_coordinates(x, y) for (x, y) in segment]
            transformed_x = [pt[0] for pt in transformed]
            transformed_y = [pt[1] for pt in transformed]

            coeffs = self.fit_parabola(transformed_x, transformed_y)
            x_range = np.linspace(transformed_x[0], transformed_x[-1], 100)
            y_range = np.polyval(coeffs, x_range)

            curve_pts = np.stack([x_range, y_range], axis=1).astype(np.int32)
            for j in range(1, len(curve_pts)):
                pt1 = tuple(curve_pts[j - 1])
                pt2 = tuple(curve_pts[j])
                cv2.line(court_img, pt1, pt2, (0, 165, 255), 4)

        court_rgb = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(court_rgb)
        plt.title('Trajectories Before Bounces on Court')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
