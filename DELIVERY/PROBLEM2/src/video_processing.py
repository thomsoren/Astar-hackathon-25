#!/usr/bin/env python3
"""
video_processing.py

Utility functions for reading and processing video streams.
"""

import cv2
import os

def open_video(source):
    """
    Opens a video file or camera stream for reading.
    :param source: Path to the video file, or an integer for webcam device index.
    :return: cv2.VideoCapture object if successful, None otherwise.
    """
    if isinstance(source, int):
        # Assuming it's a webcam index
        cap = cv2.VideoCapture(source)
    else:
        if not os.path.exists(source):
            print(f"[ERROR] Video source not found: {source}")
            return None
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return None
    return cap


def read_frames(cap, skip=0):
    """
    Generator function to read frames from a cv2.VideoCapture object in a loop.

    :param cap: cv2.VideoCapture object.
    :param skip: Number of frames to skip between yields. If 0, read every frame.
    :yield: (frame_index, frame) for each frame captured.
    """
    frame_index = 0
    while True:
        # If skip > 0, read and discard a few frames:
        for _ in range(skip):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            break
        yield frame_index, frame


def extract_frames_from_video(video_path, output_dir,
                              skip=0, limit=None):
    """
    Extract frames from a video file and save them as images.
    :param video_path: Path to the video file.
    :param output_dir: Folder to save the extracted frames.
    :param skip: Number of frames to skip between each extracted frame.
    :param limit: Limit of total frames to extract. None for no limit.
    :return: Count of frames saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = open_video(video_path)
    if not cap:
        return 0

    frame_count = 0
    def frame_generator(cap):
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            yield {
                'image': frame,
                'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC),
                'frame_num': int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            }
    for idx, frame_data in enumerate(frame_generator(cap)):
        # If we've reached a limit, break.
        if limit is not None and frame_count >= limit:
            break

        frame = frame_data['image']
        frame_name = f"frame_{idx:06d}.jpg"
        out_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(out_path, frame)
        frame_count += 1

    cap.release()
    print(f"[INFO] Extracted {frame_count} frames from {video_path}")
    return frame_count


def get_video_metadata(video_path):
    """
    Retrieves metadata about a video file such as frame count, FPS, width, height.

    :param video_path: Path to the video file.
    :return: A dictionary with keys 'frame_count', 'fps', 'width', 'height', or None on failure.
    """
    cap = open_video(video_path)
    if not cap:
        return None

    metadata = {}
    metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
    metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    return metadata


if __name__ == "__main__":
    # Simple demo usage:
    # For example, extracting frames from a sample video.
    sample_video = "../data/video/Noen få enkle varer 720P.mp4"
    out_dir = "../data/extracted_frames"

    meta = get_video_metadata(sample_video)
    if meta:
        print("[INFO] Video metadata:", meta)

    extracted = extract_frames_from_video(sample_video, out_dir, skip=5, limit=50)
    print(f"[INFO] Done. {extracted} frames were saved.")
