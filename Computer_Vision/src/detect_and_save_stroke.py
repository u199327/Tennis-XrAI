import os
import time
import argparse
import joblib
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from detection import DetectionModel, center_of_box
from pose import PoseExtractor
from smooth import Smooth
from ball_detection import BallDetector
from stats_utils import Statistics
from stroke_recognition import ActionRecognition
from utils import get_video_properties, get_dtype, get_stickman_line_connection
from court_detection import CourtDetector
from ball_bounce_visualize import BallBounceVisualizer
import matplotlib.pyplot as plt


def get_stroke_predictions(video_path, stroke_recognition, strokes_frames, player_boxes):
    """
    Get the stroke prediction for all sections where we detected a stroke
    """
    predictions = {}
    cap = cv2.VideoCapture(video_path)
    fps, length, width, height = get_video_properties(cap)
    video_length = 2
    # For each stroke detected trim video part and predict stroke
    for frame_num in strokes_frames:
        # Trim the video (only relevant frames are taken)
        starting_frame = max(0, frame_num - int(video_length * fps * 2 / 3))
        cap.set(1, starting_frame)
        i = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            stroke_recognition.add_frame(frame, player_boxes[starting_frame + i])
            i += 1
            if i == int(video_length * fps):
                break
        # predict the stroke
        probs, stroke = stroke_recognition.predict_saved_seq()
        predictions[frame_num] = {'probs': probs, 'stroke': stroke}
    cap.release()
    return predictions


def find_strokes_indices(player_1_boxes, player_2_boxes, ball_positions, skeleton_df, verbose=0):
    """
    Detect strokes frames using location of the ball and players
    """
    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]
    smooth_x = signal.savgol_filter(ball_x, 3, 2)
    smooth_y = signal.savgol_filter(ball_y, 3, 2)

    # Ball position interpolation
    x = np.arange(0, len(smooth_y))
    indices = [i for i, val in enumerate(smooth_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(smooth_y, indices)
    y2 = np.delete(smooth_x, indices)
    ball_f2_y = interp1d(x, y1, kind='cubic', fill_value="extrapolate")
    ball_f2_x = interp1d(x, y2, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(smooth_y)), smooth_y, 'o', xnew,
                 ball_f2_y(xnew), '-r')
        plt.legend(['data', 'inter'], loc='best')
        plt.show()

    # Player 2 position interpolation
    player_2_centers = np.array([center_of_box(box) for box in player_2_boxes])

    player_2_x, player_2_y = player_2_centers[:, 0], player_2_centers[:, 1]
    player_2_x = signal.savgol_filter(player_2_x, 3, 2)
    player_2_y = signal.savgol_filter(player_2_y, 3, 2)
    x = np.arange(0, len(player_2_y))
    indices = [i for i, val in enumerate(player_2_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(player_2_y, indices)
    y2 = np.delete(player_2_x, indices)
    player_2_f_y = interp1d(x, y1, fill_value="extrapolate")

    player_2_f_x = interp1d(x, y2, fill_value="extrapolate")
    xnew = np.linspace(0, len(player_2_y), num=len(player_2_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(player_2_y)), player_2_y, 'o', xnew, player_2_f_y(xnew), '--g')
        plt.legend(['data', 'inter_cubic', 'inter_lin'], loc='best')
        plt.show()

    coordinates = ball_f2_y(xnew)
    # Find all peaks of the ball y index
    peaks, _ = find_peaks(coordinates)
    if verbose:
        plt.plot(coordinates)
        plt.plot(peaks, coordinates[peaks], "x")
        plt.show()

    neg_peaks, _ = find_peaks(coordinates * -1)
    if verbose:
        plt.plot(coordinates)
        plt.plot(neg_peaks, coordinates[neg_peaks], "x")
        plt.show()

    # Get bottom player wrists positions
    left_wrist_index = 9
    right_wrist_index = 10
    skeleton_df = skeleton_df.fillna(-1)
    left_wrist_pos = skeleton_df.iloc[:, [left_wrist_index, left_wrist_index + 15]].values
    right_wrist_pos = skeleton_df.iloc[:, [right_wrist_index, right_wrist_index + 15]].values

    dists = []
    # Calculate dist between ball and bottom player
    for i, player_box in enumerate(player_1_boxes):
        if player_box[0] is not None:
            player_center = center_of_box(player_box)
            ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
            box_dist = np.linalg.norm(player_center - ball_pos)
            right_wrist_dist, left_wrist_dist = np.inf, np.inf
            if right_wrist_pos[i, 0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)
            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)
            dists.append(min(box_dist, right_wrist_dist, left_wrist_dist))
        else:
            dists.append(None)
    dists = np.array(dists)

    dists2 = []
    # Calculate dist between ball and top player
    for i in range(len(player_2_centers)):
        ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
        box_center = np.array([player_2_f_x(i), player_2_f_y(i)])
        box_dist = np.linalg.norm(box_center - ball_pos)
        dists2.append(box_dist)
    dists2 = np.array(dists2)

    strokes_1_indices = []
    # Find stroke for bottom player by thresholding the dists
    for peak in peaks:
        player_box_height = max(player_1_boxes[peak][3] - player_1_boxes[peak][1], 130)
        if dists[peak] < (player_box_height * 4 / 5):
            strokes_1_indices.append(peak)

    strokes_2_indices = []
    # Find stroke for top player by thresholding the dists
    for peak in neg_peaks:
        if dists2[peak] < 100:
            strokes_2_indices.append(peak)

    # Assert the diff between to consecutive strokes is below some threshold
    while True:
        diffs = np.diff(strokes_1_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists[strokes_1_indices[i]], dists[strokes_1_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_1_indices = np.delete(strokes_1_indices, to_del)
        if len(to_del) == 0:
            break

    # Assert the diff between to consecutive strokes is below some threshold
    while True:
        diffs = np.diff(strokes_2_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists2[strokes_2_indices[i]], dists2[strokes_2_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_2_indices = np.delete(strokes_2_indices, to_del)
        if len(to_del) == 0:
            break

    # Assume bounces frames are all the other peaks in the y index graph
    bounces_indices = [x for x in peaks if x not in strokes_1_indices]
    if verbose:
        plt.figure()
        plt.plot(coordinates)
        plt.plot(strokes_1_indices, coordinates[strokes_1_indices], "or")
        plt.plot(strokes_2_indices, coordinates[strokes_2_indices], "og")
        plt.legend(['data', 'player 1 strokes', 'player 2 strokes'], loc='best')
        plt.show()

    return strokes_1_indices, strokes_2_indices, bounces_indices, player_2_f_x, player_2_f_y

def find_strokes_indices_v2(player_1_boxes, player_2_boxes, ball_positions, skeleton_df_1, skeleton_df_2, verbose=0):
    """
    Detect strokes frames using location of the ball and players
    """
    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]
    smooth_x = signal.savgol_filter(ball_x, 3, 2)
    smooth_y = signal.savgol_filter(ball_y, 3, 2)

    # Ball position interpolation
    x = np.arange(0, len(smooth_y))
    indices = [i for i, val in enumerate(smooth_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(smooth_y, indices)
    y2 = np.delete(smooth_x, indices)
    ball_f2_y = interp1d(x, y1, kind='cubic', fill_value="extrapolate")
    ball_f2_x = interp1d(x, y2, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(smooth_y)), smooth_y, 'o', xnew,
                 ball_f2_y(xnew), '-r')
        plt.legend(['data', 'inter'], loc='best')
        plt.show()

    coordinates = ball_f2_y(xnew)
    # Find all peaks of the ball y index
    peaks, _ = find_peaks(coordinates)
    if verbose:
        plt.plot(coordinates)
        plt.plot(peaks, coordinates[peaks], "x")
        plt.show()

    neg_peaks, _ = find_peaks(coordinates * -1)
    if verbose:
        plt.plot(coordinates)
        plt.plot(neg_peaks, coordinates[neg_peaks], "x")
        plt.show()

    # Get bottom player wrists positions
    left_wrist_index = 9
    right_wrist_index = 10
    skeleton_df_1 = skeleton_df_1.fillna(-1)
    left_wrist_pos = skeleton_df_1.iloc[:, [left_wrist_index, left_wrist_index + 15]].values
    right_wrist_pos = skeleton_df_1.iloc[:, [right_wrist_index, right_wrist_index + 15]].values

    dists = []
    # Calculate dist between ball and bottom player
    for i, player_box in enumerate(player_1_boxes):
        if player_box[0] is not None:
            player_center = center_of_box(player_box)
            ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
            box_dist = np.linalg.norm(player_center - ball_pos)
            right_wrist_dist, left_wrist_dist = np.inf, np.inf
            if right_wrist_pos[i, 0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)
            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)
            dists.append(min(box_dist, right_wrist_dist, left_wrist_dist))
        else:
            dists.append(None)
    dists = np.array(dists)


    left_wrist_index = 9
    right_wrist_index = 10
    skeleton_df_2 = skeleton_df_2.fillna(-1)
    left_wrist_pos = skeleton_df_2.iloc[:, [left_wrist_index, left_wrist_index + 15]].values
    right_wrist_pos = skeleton_df_2.iloc[:, [right_wrist_index, right_wrist_index + 15]].values
    dists2 = []

    for i, player_box in enumerate(player_2_boxes):
        if player_box[0] is not None:
            player_center = center_of_box(player_box)
            ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
            box_dist = np.linalg.norm(player_center - ball_pos)
            right_wrist_dist, left_wrist_dist = np.inf, np.inf
            if right_wrist_pos[i, 0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)
            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)
            dists2.append(min(box_dist, right_wrist_dist, left_wrist_dist))
        else:
            dists2.append(None)
    dists2 = np.array(dists2)

    strokes_1_indices = []
    # Find stroke for bottom player by thresholding the dists
    for peak in peaks:
        player_box_height = max(player_1_boxes[peak][3] - player_1_boxes[peak][1], 130)
        if dists[peak] < (player_box_height * 4 / 5):
            strokes_1_indices.append(peak)

    strokes_2_indices = []
    # Find stroke for top player by thresholding the dists
    for peak in neg_peaks:
        player_box_height = max(player_2_boxes[peak][3] - player_2_boxes[peak][1], 130)
        if dists2[peak] < (player_box_height * 4 / 5):
            strokes_2_indices.append(peak)

    while True:
        diffs = np.diff(strokes_1_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists[strokes_1_indices[i]], dists[strokes_1_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_1_indices = np.delete(strokes_1_indices, to_del)
        if len(to_del) == 0:
            break

    while True:
        diffs = np.diff(strokes_2_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists2[strokes_2_indices[i]], dists2[strokes_2_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_2_indices = np.delete(strokes_2_indices, to_del)
        if len(to_del) == 0:
            break

    bounces_indices = [x for x in peaks if x not in strokes_1_indices]
    if verbose:
        plt.figure()
        plt.plot(coordinates)
        plt.plot(strokes_1_indices, coordinates[strokes_1_indices], "or")
        plt.plot(strokes_2_indices, coordinates[strokes_2_indices], "og")
        plt.legend(['data', 'player 1 strokes', 'player 2 strokes'], loc='best')
        plt.show()

    return strokes_1_indices, strokes_2_indices, bounces_indices

def mark_player_box(frame, boxes, frame_num):
    box = boxes[frame_num]
    if box[0] is not None:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
    return frame


def mark_skeleton(skeleton_df, img, img_no_frame, frame_number):
    """
    Mark the skeleton of the bottom player on the frame
    """
    circle_color, line_color = (0, 0, 255), (255, 0, 0)
    stickman_pairs = get_stickman_line_connection()

    skeleton_df = skeleton_df.fillna(-1)
    values = np.array(skeleton_df.values[frame_number], int)
    points = list(zip(values[5:17], values[22:]))
    for point in points:
        if point[0] >= 0 and point[1] >= 0:
            xy = tuple(np.array([point[0], point[1]], int))
            cv2.circle(img, xy, 2, circle_color, 2)
            cv2.circle(img_no_frame, xy, 2, circle_color, 2)

    for pair in stickman_pairs:
        partA = pair[0] - 5
        partB = pair[1] - 5
        if points[partA][0] >= 0 and points[partA][1] >= 0 and points[partB][0] >= 0 and points[partB][1] >= 0:
            cv2.line(img, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
            cv2.line(img_no_frame, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
    return img, img_no_frame


def add_data_to_video(input_video, court_detector, players_detector, ball_detector,
                      strokes_predictions_1, strokes_predictions_2,
                      skeleton_df_1, skeleton_df_2,
                      show_video, with_frame, output_folder, output_file,
                      p1, p2, f_x, f_y):

    player1_boxes = players_detector.player_1_boxes
    player2_boxes = players_detector.player_2_boxes

    if skeleton_df_1 is not None:
        skeleton_df_1 = skeleton_df_1.fillna(-1)

    if skeleton_df_2 is not None:
        skeleton_df_2 = skeleton_df_2.fillna(-1)

    cap = cv2.VideoCapture(input_video)

    fps, length, width, height = get_video_properties(cap)
    final_width = width * 2 if with_frame == 2 else width

    out = cv2.VideoWriter(os.path.join(output_folder, output_file + '.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (final_width, height))

    frame_number = 0
    orig_frame = 0

    if show_video:
        fig, ax = plt.subplots()
        img_display = ax.imshow(np.zeros((height, final_width, 3), dtype=np.uint8))
        plt.ion()

    while True:
        orig_frame += 1
        print(f'Creating new videos frame {orig_frame}/{length}', '\r', end='')
        if not orig_frame % 100:
            print('')

        ret, img = cap.read()
        if not ret:
            break

        img_no_frame = np.ones_like(img) * 255

        img = court_detector.add_court_overlay(img, overlay_color=(0, 0, 255), frame_num=frame_number)
        img_no_frame = court_detector.add_court_overlay(img_no_frame, overlay_color=(0, 0, 255), frame_num=frame_number)

        img = mark_player_box(img, player1_boxes, frame_number)
        img = mark_player_box(img, player2_boxes, frame_number)
        img_no_frame = mark_player_box(img_no_frame, player1_boxes, frame_number)
        img_no_frame = mark_player_box(img_no_frame, player2_boxes, frame_number)

        img = ball_detector.mark_positions(img, frame_num=frame_number)
        img_no_frame = ball_detector.mark_positions(img_no_frame, frame_num=frame_number, ball_color='black')

        if skeleton_df_1 is not None:
            img, img_no_frame = mark_skeleton(skeleton_df_1, img, img_no_frame, frame_number)
        if skeleton_df_2 is not None:
            img, img_no_frame = mark_skeleton(skeleton_df_2, img, img_no_frame, frame_number)

        for i in range(-10, 10):
            frame_idx = frame_number + i

            if frame_idx in strokes_predictions_1:
                stroke_1 = strokes_predictions_1[frame_idx]
                cv2.putText(img, f'P1 Stroke: {stroke_1}',
                            (70, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            if frame_idx in strokes_predictions_2:
                stroke_2 = strokes_predictions_2[frame_idx]
                cv2.putText(img, f'P2 Stroke: {stroke_2}',
                            (70, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        for i in range(-5, 10):
            if frame_number + i in p1:
                cv2.putText(img, 'P1 Stroke Detected',
                            (int(f_x(frame_number)) - 30, int(f_y(frame_number)) - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if frame_number + i in p2:
                cv2.putText(img, 'P2 Stroke Detected',
                            (int(f_x(frame_number)) - 30, int(f_y(frame_number)) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        if show_video:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_display.set_data(img_rgb)
            plt.pause(0.001)

        final_frame = img_no_frame if with_frame == 0 else img
        if with_frame == 2:
            final_frame = np.concatenate([img, img_no_frame], axis=1)

        out.write(final_frame)
        frame_number += 1

    print(f'\nNew video created: {output_file}.avi')
    cap.release()
    out.release()
    if show_video:
        plt.ioff()
        plt.show()


def create_top_view(court_detector, detection_model):
    """
    Creates top view video of the gameplay
    """
    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter('output/top_view.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (v_width, v_height))
    smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)

    for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
        frame = court.copy()
        frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 10, (0, 0, 255), 15)
        if feet_pos_2[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 10, (0, 0, 255), 15)
        out.write(frame)
    out.release()
    #cv2.destroyAllWindows()



def append_strokes_to_csv(input_file, smoothed_csv, stroke_indices, output_csv, fps):
    df_smooth = pd.read_csv(smoothed_csv)

    df_filtered = df_smooth.iloc[stroke_indices].copy()

    df_filtered.insert(0, "Filename", input_file)
    df_filtered.insert(1, "Strike_Second", df_filtered.index / fps)  

    df_filtered.to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False)

def predict_strokes(rf_clf,label_encoder ,df_smooth, stroke_indices, player_name):
    if df_smooth is not None:
        print(f"Predicted strokes for {player_name}:")
        for index in stroke_indices:
            if index < len(df_smooth):  
                skeleton_features = df_smooth.iloc[index].values.reshape(1, -1)  
                prediction = rf_clf.predict(skeleton_features)[0]  
                prediction_label = label_encoder.inverse_transform([prediction])[0]
                
                print(f"Frame {index}: {prediction_label}")

def predict_strokes_window(rf_clf,label_encoder ,df_smooth, stroke_indices, player_name, window_size=5):
    frame_predictions = {}  

    if df_smooth is not None:
        
        for index in stroke_indices:
            start = max(0, index - window_size // 2)
            end = min(len(df_smooth), index + window_size // 2 + 1)

            predictions = []
            for i in range(start, end):
                skeleton_features = df_smooth.iloc[i].values.reshape(1, -1) 
                prediction = rf_clf.predict(skeleton_features)[0]
                prediction_label = str(label_encoder.inverse_transform([prediction])[0]) 

                predictions.append(prediction_label)

            most_common_prediction = Counter(predictions).most_common(1)[0][0]
            print(f"Frame {index} (window {start}-{end}): {most_common_prediction}")

            frame_predictions[index] = {
                "stroke": most_common_prediction,
                
            }

    return frame_predictions
def create_top_view(court_detector, detection_model):
    """
    Creates top view video of the gameplay
    """
    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter('output/top_view.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (v_width, v_height))
    # players location on court
    smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)

    for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
        frame = court.copy()
        frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 10, (0, 0, 255), 15)
        if feet_pos_2[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 10, (0, 0, 255), 15)
        out.write(frame)
    out.release()
    #cv2.destroyAllWindows()

def create_heatmap_dual(player1_positions, player2_positions, court_reference_img, 
                        output_path="output/combined_heatmap.png", 
                        player1_name="Player 1", player2_name="Player 2"):
    """
    Creates a combined heatmap of two players' positions over the court.

    :param player1_positions: List of (x, y) tuples for Player 1
    :param player2_positions: List of (x, y) tuples for Player 2
    :param court_reference_img: Background image (court)
    :param output_path: Path to save the output heatmap
    :param player1_name: Label for Player 1
    :param player2_name: Label for Player 2
    """
    # Extract valid coordinates
    p1_x = [p[0] for p in player1_positions if p[0] is not None]
    p1_y = [p[1] for p in player1_positions if p[1] is not None]
    p2_x = [p[0] for p in player2_positions if p[0] is not None]
    p2_y = [p[1] for p in player2_positions if p[1] is not None]

    if not p1_x and not p2_x:
        print("No valid positions to create combined heatmap.")
        return

    # Prepare court image
    heatmap_img = court_reference_img.copy()
    if len(heatmap_img.shape) == 2:
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_GRAY2BGR)

    plt.figure(figsize=(10, 6))

    # Plot both players with different colormaps
    if p1_x and p1_y:
        sns.kdeplot(x=p1_x, y=p1_y, cmap="Reds", fill=True, alpha=0.5, bw_adjust=0.7, thresh=0.05, label=player1_name)
    if p2_x and p2_y:
        sns.kdeplot(x=p2_x, y=p2_y, cmap="Blues", fill=True, alpha=0.5, bw_adjust=0.7, thresh=0.05, label=player2_name)

    plt.imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), extent=[0, heatmap_img.shape[1], heatmap_img.shape[0], 0])
    plt.title(f'Combined Heatmap: {player1_name} & {player2_name}')
    plt.axis('off')
    plt.legend(loc='upper right')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Combined heatmap saved to {output_path}")


def video_process(video_path, show_video=False, include_video=True,
                  stickman=True, stickman_box=True, court=True,
                  output_file='output', output_folder='output',
                  smoothing=True, top_view=True):
    """
    Takes videos of one person as input, and calculate the body pose and face landmarks, and saves them as csv files.
    Also, output a result videos with the keypoints marked.
    :param court:
    :param video_path: str, path to the videos
    :param show_video: bool, show processed videos while processing (default = False)
    :param include_video: bool, result output videos will include the original videos as well as the
    keypoints (default = True)
    :param stickman: bool, calculate pose and create stickman using the pose data (default = True)
    :param stickman_box: bool, show person bounding box in the output videos (default = False)
    :param output_file: str, output file name (default = 'output')
    :param output_folder: str, output folder name (default = 'output') will create new folder if it does not exist
    :param smoothing: bool, use smoothing on output data (default = True)
    :return: None
    """
    dtype = get_dtype()

    # initialize extractors
    court_detector = CourtDetector()
    detection_model = DetectionModel(dtype=dtype)
    pose_extractor_1 = PoseExtractor(person_num=1, box=stickman_box, dtype=dtype) if stickman else None
    pose_extractor_2 = PoseExtractor(person_num=1, box=stickman_box, dtype=dtype) if stickman else None

    #stroke_recognition = ActionRecognition('storke_classifier_weights.pth')
    #stroke_recognition = ActionRecognition('saved_state_thetis_3e-05_0.003__epoch_19')
    #stroke_recognition = ActionRecognition('saved_state_thetis_GRU_3e-05_0.003__epoch_19', "GRU")
    rf_clf = joblib.load("saved states/random_forest_model.pkl")
    label_encoder = joblib.load("saved states/label_encoder.pkl")

    ball_detector = BallDetector('saved states/tracknet_weights_2_classes.pth', out_channels=2)

    # Load videos from videos path
    video = cv2.VideoCapture(video_path)

    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # frame counter
    frame_i = 0

    # time counter
    total_time = 0

    # Loop over all frames in the videos
    while True:
        start_time = time.time()
        ret, frame = video.read()
        frame_i += 1

        if ret:
            
            if frame_i == 1:
                court_detector.detect(frame)
                print(f'Court detection {"Success" if court_detector.success_flag else "Failed"}')
                print(f'Time to detect court :  {time.time() - start_time} seconds')
                start_time = time.time()

            court_detector.track_court(frame)
            
            # detect players
            detection_model.detect_player_1(frame.copy(), court_detector)
            detection_model.detect_player_2(frame.copy(), court_detector)
            #print("Player2", detection_model.detect_top_persons(frame, court_detector, frame_i))
            #detection_model.find_player_2_box()
            # Create stick man figure (pose detection)
            if stickman:
                pose_extractor_1.extract_pose(frame, detection_model.player_1_boxes)
                pose_extractor_2.extract_pose(frame, detection_model.player_2_boxes)

            ball_detector.detect_ball(frame)

            total_time += (time.time() - start_time)
            print('Processing frame %d/%d  FPS %04f' % (frame_i, length, frame_i / total_time), '\r', end='')
            if not frame_i % 100:
                print('')
        else:
            break
    print('Processing frame %d/%d  FPS %04f' % (length, length, length / total_time), '\n', end='')
    print('Processing completed')
    video.release()
    #cv2.destroyAllWindows()
   
    # Save landmarks in csv files
    df_1 = None
    df_2 = None
    # Save stickman data
    if stickman:
        df_1 = pose_extractor_1.save_to_csv(output_folder, filename="stickman_player1.csv")
        df_2 = pose_extractor_2.save_to_csv(output_folder,filename="stickman_player2.csv")


    # smooth the output data for better results
    if smoothing:
        smoother_1 = Smooth()
        df_smooth_1 = smoother_1.smooth(df_1)
        smoother_1.save_to_csv(output_folder, filename ="stickman_player1_smoothed.csv")
        
        smoother_2 = Smooth()
        df_smooth_2 = smoother_2.smooth(df_2)
        smoother_2.save_to_csv(output_folder, filename ="stickman_player2_smoothed.csv")
        
    
    
    _, _, _, f2_x, f2_y = find_strokes_indices(
        detection_model.player_1_boxes,
        detection_model.player_2_boxes,
        ball_detector.xy_coordinates,
        df_smooth_1)

    player_1_strokes_indices, player_2_strokes_indices, bounces_indices = find_strokes_indices_v2(
    detection_model.player_1_boxes,
    detection_model.player_2_boxes,
    ball_detector.xy_coordinates,
    df_smooth_1, df_smooth_2)
    
    if top_view:
        # Extract ball bounce positions
        bounce_positions = [(int(x), int(y)) for x, y in zip(ball_detector.xy_coordinates[:, 0], ball_detector.xy_coordinates[:, 1]) if x is not None and y is not None]

        # Visualize ball bounces
        visualizer = BallBounceVisualizer(court_detector, bounce_positions)
        visualizer.visualize_bounces_image(output_path=f"{output_folder}/bounce_visualization.png")
        visualizer.visualize_bounce_heatmap()
        visualizer.plot_trajectory_on_court_image()
        visualizer.visualize_bounce_depth_distribution()

        
        create_top_view(court_detector, detection_model)

        player1_positions, player2_positions = detection_model.calculate_feet_positions(court_detector)
        court_img = court_detector.court_reference.court.copy()
        court_img = cv2.line(court_img, *court_detector.court_reference.net, 255, 5)

        # Use the new dual-player heatmap
        create_heatmap_dual(player1_positions, player2_positions, court_img, output_path="output/combined_heatmap.png")


    print("Player 1 indices:", player_1_strokes_indices)
    print("Player 2 indices:", player_2_strokes_indices)

    append_strokes_to_csv(video_path, "output/stickman_player1_smoothed.csv", player_1_strokes_indices, "strokes_data.xlsx", fps)
    append_strokes_to_csv(video_path, "output/stickman_player2_smoothed.csv", player_2_strokes_indices, "strokes_data.xlsx", fps)

    # Predict strokes for both players
    predict_strokes(rf_clf, label_encoder, df_smooth_1, player_1_strokes_indices, "Player 1")
    predict_strokes(rf_clf, label_encoder, df_smooth_2, player_2_strokes_indices, "Player 2")

    print("Prediction with Window")
    predictions_player_1 = predict_strokes_window(rf_clf, label_encoder, df_smooth_1, player_1_strokes_indices, "Player 1")
    predictions_player_2= predict_strokes_window(rf_clf, label_encoder, df_smooth_2, player_2_strokes_indices, "Player 2")
    
    
    ball_detector.bounces_indices = bounces_indices
    ball_detector.coordinates = (f2_x, f2_y)
    '''
    predictions = get_stroke_predictions(video_path, stroke_recognition,
                                         player_1_strokes_indices, detection_model.player_1_boxes)
    '''
    #heatmap = statistics.get_player_position_heatmap()
    #statistics.display_heatmap(heatmap, court_detector.court_reference.court, title='Heatmap')
    #statistics.get_players_dists()

    add_data_to_video(input_video=video_path, court_detector=court_detector, players_detector=detection_model,
                      ball_detector=ball_detector, strokes_predictions_1=predictions_player_1, strokes_predictions_2=predictions_player_2, skeleton_df_1=df_smooth_1,
                      skeleton_df_2 = df_smooth_2,
                      show_video=show_video, with_frame=1, output_folder=output_folder, output_file=output_file,
                      p1=player_1_strokes_indices, p2=player_2_strokes_indices, f_x=f2_x, f_y=f2_y)
    
    # ball_detector.show_y_graph(detection_model.player_1_boxes, detection_model.player_2_boxes)


def main():
    parser = argparse.ArgumentParser(description="Process a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    
    args = parser.parse_args()

    s = time.time()
    video_process(video_path=args.video_path, show_video=True, stickman=True, stickman_box=False, 
                  smoothing=True, court=True, top_view=True)
    print(f'Total computation time: {time.time() - s} seconds')


if __name__ == "__main__":
    main()
