from unittest import getTestCaseNames
import numpy as np
import matplotlib.pyplot as plt
import json
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    # Compute intersection
    #First we compute the intersection box
    x_min = max(prediction_box[0], gt_box[0])
    y_min = max(prediction_box[1], gt_box[1])
    x_max = min(prediction_box[2], gt_box[2])
    y_max = min(prediction_box[3], gt_box[3])
    
    #check if no intersection
    if (x_max < x_min) or (y_max < y_min):
        return 0.0
    #Compute the area of intersection
    x_diff = abs(x_max - x_min)
    y_diff = abs(y_max - y_min)
    inter_area = x_diff * y_diff

    
    # Compute union
    pbox_xdiff = abs(prediction_box[2] - prediction_box[0])
    pbox_ydiff = abs(prediction_box[3] - prediction_box[1])
    
    gtbox_xdiff = abs(gt_box[2] - gt_box[0])
    gtbox_ydiff = abs(gt_box[3] - gt_box[1])
    
    pbox_area = pbox_xdiff * pbox_ydiff

    gt_area = gtbox_xdiff * gtbox_ydiff

    
    union = pbox_area + gt_area - (inter_area)
    
    #compute iou
    iou = inter_area / union
    
    
    assert iou >= 0 and iou <= 1
    
    return iou

def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    return num_tp / (num_tp + num_fp)



def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fp == 0:
        return 0
    return num_tp / (num_tp + num_fn)



def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    optimal_matches = {}
    match_gt = []
    match_pred = []
                
    for gt_box in gt_boxes:
        best_iou = -np.inf
        for pred_box in prediction_boxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                if iou > best_iou:
                    best_iou = iou
                    optimal_matches[str(gt_box)] = (pred_box, best_iou)
    
    # Sort all matches on IoU in descending order
    dict(sorted(optimal_matches.items(), key=lambda item: item[1][1]))

    # Find all matches with the highest IoU threshold
    for key, value in optimal_matches.items():
        match_gt.append(np.fromstring(key[1:-1], sep=' '))
        match_pred.append(value[0])

    #print('match_pred: ',match_pred)
    #print('match_gt', match_gt)

    return  np.array(match_pred, dtype='object'), np.array(match_gt,  dtype='object')


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    
    match_pred, match_gt = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    true_positive = len(match_pred)
    false_positive = len(prediction_boxes) - true_positive
    false_negative = len(gt_boxes) - true_positive
    
    return {'true_pos': true_positive, 'false_pos': false_positive, 'false_neg': false_negative}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    tp = 0
    fp = 0
    fn = 0
    for i in range(len(all_prediction_boxes)):
        result_per_image = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        tp += result_per_image['true_pos']
        fp += result_per_image['false_pos']
        fn += result_per_image['false_neg']
    precision = calculate_precision(tp, fp, fn)
    recall = calculate_recall(tp, fp, fn)
    return (precision, recall)

def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = [] 
    recalls = []
    for i in range(len(confidence_thresholds)):
        confidence_pred_boxes = []
        # threshold = confidence_thresholds[i]
        for j in range(len(confidence_scores)):
            img_score_pred = []
            for score_id in range(len(confidence_scores[j])):
                score = confidence_scores[j][score_id]
                if score > confidence_thresholds[i]: 
                    img_score_pred.append(all_prediction_boxes[j][score_id])
            confidence_pred_boxes.append(img_score_pred)
        precision, recall = calculate_precision_recall_all_images(confidence_pred_boxes, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    pred_max_sum = 0
    for i in range(len(recall_levels)):
        pred_max = 0
        for j in range(recalls.shape[0]):
            if (precisions[j] > pred_max) and (recalls[j] >= recall_levels[i]):
                pred_max = precisions[j]
        pred_max_sum += pred_max

    # YOUR CODE HERE
    average_precision = pred_max_sum / len(recall_levels)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
