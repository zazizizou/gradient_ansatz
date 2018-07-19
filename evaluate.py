from shapes import Rectangle
import xml.etree.ElementTree as ET


def inter_area(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # (origin in upper left)
    xA = max(boxA.xmin, boxB.xmin)
    yA = max(boxA.ymin, boxB.ymin)
    xB = min(boxA.xmax, boxB.xmax)
    yB = min(boxA.ymax, boxB.ymax)

    interArea = max(0, xB - xA) * max(0, yB - yA)

    return interArea


def intersection_over_union(boxA, boxB):
    """
    computes intersection over union: (Area of Overlap) / (Area of Union)
    boxA = (x,y w, h), where (x,y) is the upper left corner of the
    rectangle and w and h are with and height of the rectangle
    """
    interArea = inter_area(boxA, boxB)

    boxAArea = (boxA.xmax - boxA.xmin) * (boxA.ymax - boxA.ymin)
    boxBArea = (boxB.xmax - boxB.xmin) * (boxB.ymax - boxB.ymin)

    union = float(boxAArea + boxBArea - interArea)

    # print("union:", union)
    # print("intersection", interArea)

    if union == 0:
        return 0.0
    else:
        return interArea / union


def eval_precision_recall(true_lb, pred_lb, iou_threshold=0.5):

    tp = 0
    fp = 0
    fn = 0
    tr_detected = [False] * len(true_lb)

    for pr in pred_lb:
        for idx, tr in enumerate(true_lb):
            iou = intersection_over_union(pr, tr)
            #if iou > 0:
            #    print(iou)
            if iou >= iou_threshold:
                if not tr_detected[idx]:
                    tp += 1
                    tr_detected[idx] = True
                    break
                else:
                    fp += 1

    fn = len(tr_detected) - sum(tr_detected)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("precision:", precision)
    print("recall:", recall)
    return precision, recall


def get_ground_truth(label):
    ann = label["annotations"]
    rect_lb = [(Rectangle(lm["x"], lm["y"], lm["x"]+lm["width"], lm["y"]+lm["height"]))
               for lm in ann]
    return rect_lb


def get_ground_truth_from_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    rect_lb = []
    for bb in root.iter("bndbox"):
        for child in bb:
            if child.tag == "xmin":
                xmin = int(child.text)
            elif child.tag == "ymin":
                ymin =int(child.text)
            elif child.tag == "xmax":
                xmax = int(child.text)
            elif child.tag == "ymax":
                ymax = int(child.text)
        rect_lb.append(Rectangle(xmin, ymin, xmax, ymax))

    return rect_lb














