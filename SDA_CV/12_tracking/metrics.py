def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    xmin, ymin, xmax, ymax = zip(bbox1, bbox2)
    if xmin[0] >= xmin[1]:
        xmin, ymin, xmax, ymax = zip(bbox2, bbox1)
    
    if xmax[0] <= xmin[1] or ymax[1] <= ymin[0] or ymax[0] <= ymin[1]:
        return 0

    intersect = abs((min(ymax) - max(ymin)) * (min(xmax) - max(xmin)))
    common_square = -intersect 
    for i in range(2):
        common_square += abs((xmax[i] - xmin[i]) * (ymax[i] - ymin[i]))
    
    return 0 if common_square == 0 else intersect / common_square


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj = { detection[0] : detection[1:] for detection in frame_obj }
        frame_hyp = { detection[0] : detection[1:] for detection in frame_hyp }
        
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_idx, hyp_idx in matches.items():
            if not (obj_idx in frame_obj.keys() and hyp_idx in frame_hyp.keys()):
                continue

            iou = iou_score(frame_obj[obj_idx], frame_hyp[hyp_idx])
            if iou < threshold:
                continue
            
            dist_sum += iou
            match_count += 1
            
            frame_obj.pop(obj_idx)
            frame_hyp.pop(hyp_idx)

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ious = []
        for obj_idx, obj_bbox in frame_obj.items():
            for hyp_idx, hyp_bbox in frame_hyp.items():
                iou = iou_score(obj_bbox, hyp_bbox)
                if iou >= threshold:
                    ious.append([iou, obj_idx, hyp_idx])
        
        ious = sorted(ious, reverse=True)

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        new_matches = {}
        for iou, obj_idx, hyp_idx in ious:
            if obj_idx not in frame_obj.keys():
                continue
            
            if obj_idx in matches.keys():
                matches[obj_idx] = hyp_idx
            else:
                dist_sum += iou
                match_count += 1
                new_matches[obj_idx] = hyp_idx
            
            frame_obj.pop(obj_idx)
            frame_hyp.pop(hyp_idx)

        # Step 5: Update matches with current matched IDs
        for obj_idx, hyp_idx in new_matches.items():
            matches[obj_idx] = hyp_idx

    # Step 6: Calculate MOTP
    MOTP = 0 if match_count == 0 else dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    objects = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        objects += len(frame_obj)
        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj = { detection[0] : detection[1:] for detection in frame_obj }
        frame_hyp = { detection[0] : detection[1:] for detection in frame_hyp }

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_idx, hyp_idx in matches.items():
            if not (obj_idx in frame_obj.keys() and hyp_idx in frame_hyp.keys()):
                continue

            iou = iou_score(frame_obj[obj_idx], frame_hyp[hyp_idx])
            if iou < threshold:
                continue
            
            dist_sum += iou
            match_count += 1
            
            frame_obj.pop(obj_idx)
            frame_hyp.pop(hyp_idx)

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ious = []
        for obj_idx, obj_bbox in frame_obj.items():
            for hyp_idx, hyp_bbox in frame_hyp.items():
                iou = iou_score(obj_bbox, hyp_bbox)
                if iou >= threshold:
                    ious.append([iou, obj_idx, hyp_idx])
        
        ious = sorted(ious, reverse=True)

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections        
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        new_matches = {}
        for iou, obj_idx, hyp_idx in ious:
            if obj_idx not in frame_obj.keys():
                continue
            
            if obj_idx in matches.keys():
                mismatch_error += 1
                matches[obj_idx] = hyp_idx
            else:
                dist_sum += iou
                match_count += 1
                new_matches[obj_idx] = hyp_idx
            
            frame_obj.pop(obj_idx)
            frame_hyp.pop(hyp_idx)
        
        # Step 6: Update matches with current matched IDs
        for obj_idx, hyp_idx in new_matches.items():
            matches[obj_idx] = hyp_idx
        
        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        false_positive += len(frame_hyp)
        # All remaining objects are considered misses
        missed_count += len(frame_obj)

    # Step 8: Calculate MOTP and MOTA
    MOTP = 0 if match_count == 0 else dist_sum / match_count
    errors = false_positive + missed_count + mismatch_error
    MOTA = 0 if objects == 0 else 1 - errors / objects

    return MOTP, MOTA
