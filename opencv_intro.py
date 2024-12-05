import cv2
import numpy as np
import argparse
import pandas as pd

# Load smoothed predictions
predictions_df = pd.read_csv('smoothed_predictions.csv')  # Use smoothed predictions

# Define active frames (frames where the ball is being hit or in motion)
active_frames = predictions_df[predictions_df['value'] == 1]['frame'].tolist()  # Assuming 'value' column indicates activity


def process_frame(frame, frame_number, apply_filters=True, ball_color_transition=False, ball_trail=None):
    """
    Process a single frame of the video with optional filters, tracking, and ball trail visualization.
    """
    # Preserve the original frame size
    output_frame = frame.copy()

    if apply_filters:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
        output_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if ball_color_transition or ball_trail:
        # Add prediction overlay
        prediction = predictions_df[predictions_df['frame'] == frame_number]

        if not prediction.empty:
            try:
                # Use predicted values if available
                predicted_x = int(prediction['predicted_x'])
                predicted_y = int(prediction['predicted_y'])
            except KeyError:
                # Default to the center of the frame if columns are missing
                print(f"Warning: Missing 'predicted_x' or 'predicted_y' columns. Defaulting to frame center for frame {frame_number}.")
                predicted_x = output_frame.shape[1] // 2  # Center horizontally
                predicted_y = output_frame.shape[0] // 2  # Center vertically
        else:
            # No prediction for this frame, also default to center
            print(f"Warning: No prediction for frame {frame_number}. Defaulting to frame center.")
            predicted_x = output_frame.shape[1] // 2
            predicted_y = output_frame.shape[0] // 2

        # Draw the ball at the predicted or default position
        color = (0, 255, 0)  # Green for predicted position
        cv2.circle(output_frame, (predicted_x, predicted_y), radius=10, color=color, thickness=-1)

        # Update the ball trail
        if ball_trail is not None:
            ball_trail.append((predicted_x, predicted_y))
            for i, point in enumerate(reversed(ball_trail[-30:])):  # Keep the last 30 points for the trail
                fade_color = (255 - i * 8, 255 - i * 8, 255 - i * 8)  # Fading effect
                cv2.circle(output_frame, point, radius=5, color=fade_color, thickness=-1)

    return output_frame


def apply_transition(prev_frame, current_frame, alpha):
    """
    Applies a fade transition between two frames.
    """
    return cv2.addWeighted(prev_frame, 1 - alpha, current_frame, alpha, 0)

def main():
    parser = argparse.ArgumentParser(description='Process video with effects, tracking, and transitions')
    parser.add_argument('input_video', type=str, help='Path to the input video file')
    parser.add_argument('--filters', action='store_true', help='Apply black-and-white filter to the video')
    parser.add_argument('--track_ball', action='store_true', help='Track ball color and transition color on movement')
    parser.add_argument('--trail', action='store_true', help='Add a trail to the ball movement')
    parser.add_argument('--transition', action='store_true', help='Apply fade-in and fade-out transitions')
    parser.add_argument('--speed_up', type=float, default=1.0, help='Factor to speed up the output video')
    args = parser.parse_args()

    input_video = cv2.VideoCapture(args.input_video)
    if not input_video.isOpened():
        print(f"Error: Unable to open the video file {args.input_video}.")
        return

    original_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(input_video.get(cv2.CAP_PROP_FPS))
    adjusted_fps = int(original_fps * args.speed_up)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output_with_transitions.mp4', fourcc, adjusted_fps, (original_width, original_height))

    frame_number = 0
    prev_frame = None
    ball_trail = []

    while True:
        ret, frame = input_video.read()
        if not ret:
            break

        if frame_number in active_frames:
            processed_frame = process_frame(
                frame,
                frame_number,
                apply_filters=args.filters,
                ball_color_transition=args.track_ball,
                ball_trail=ball_trail if args.trail else None
            )

            if args.transition and prev_frame is not None:
                for alpha in np.linspace(0, 1, int(original_fps // 4)):  # Quarter-second fade
                    transition_frame = apply_transition(prev_frame, processed_frame, alpha)
                    output_video.write(transition_frame)

            output_video.write(processed_frame)
            prev_frame = processed_frame

            cv2.imshow('Processed Video', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_number += 1

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
