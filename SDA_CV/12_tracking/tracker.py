import os

import numpy as np

from detection import detection_cast, draw_detections, extract_detections
from metrics import iou_score


class Tracker:
    """Generate detections and build tracklets."""

    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Write code here
        # Use extract_detections and new_label

        detections = []
        for label, xmin, ymin, xmax, ymax in extract_detections(frame):
            detections.append([self.new_label(), xmin, ymin, xmax, ymax])

        return detection_cast(detections)

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        
        detections = []
        
        label_idxs = set()
        lookup_tail_size = min(len(self.detection_history), self.lookup_tail_size)
        history = self.detection_history[-lookup_tail_size:]
        for history_detections in history[::-1]:
            for detection in history_detections:
                label_idx = detection[0]
                if label_idx not in label_idxs:
                    label_idxs.add(label_idx)
                    detections.append(detection)

        return detection_cast(detections)

    def bind_tracklet(self, detections):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        prev_detections = self.prev_detections

        # Write code here

        # Step 1: calc pairwise detection IOU
        IoUs = []
        for i_cur in range(len(detections)):
            for i_prev in range(len(prev_detections)):
                bbox1 = detections[i_cur][1:]
                bbox2 = prev_detections[i_prev][1:]
                iou = iou_score(bbox1, bbox2)
                IoUs.append([iou, i_cur, i_prev])              

        # Step 2: sort IOU list
        IoUs = sorted(IoUs, reverse=True)

        i_cur_used, i_prev_used = set(), set()
        # Step 3: fill detections[:, 0] with best match
        # One matching for each id
        for iou, i_cur, i_prev in IoUs:
            if i_cur in i_cur_used or i_prev in i_prev_used or iou < 0.5:
                continue
            
            detections[i_cur, 0] = i_prev
            i_cur_used.add(i_cur)
            i_prev_used.add(i_prev)

        # Step 4: assign new tracklet id to unmatched detections
        for i_cur in range(len(detections)):
            if i_cur not in i_cur_used:
                detections[i_cur, 0] = self.new_label()

        return detection_cast(detections)

    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
