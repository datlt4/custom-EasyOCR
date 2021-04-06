import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
import numpy as np
import math

from utils.craft import CRAFT
from utils.model.modules import RefineNet, CTCLabelConverter
from utils.model.vgg_model import OCR
import utils.craft_utils
import utils.imgproc
import utils.deep_ocr_utils

from collections import OrderedDict
import imgaug as ia
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def show_polys(image, polys=[], mode="xywh", index=[]):
    Polygons = []
    for poly in polys:
        if len(poly) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = poly if not index else [
                poly[i] for i in index]
        elif len(poly) == 4:
            if mode == "xyxy":
                l, t, r, b = poly if not index else [poly[i] for i in index]
                x1, y1, x2, y2, x3, y3, x4, y4 = l, t, r, t, r, b, l, b
            elif mode == "xywh":
                x, y, w, h = poly if not index else [poly[i] for i in index]
                x1, y1, x2, y2, x3, y3, x4, y4 = x, y, x+w, y, x+w, y+h, x, y+h
        else:
            ia.imshow
        Polygons.append(Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)]))
    psoi = PolygonsOnImage(Polygons, shape=image.shape)
    ia.imshow(psoi.draw_on_image(image, alpha_face=0.2, size_points=7))


def compute_accuracy(ground_truth, predictions, mode='full_sequence'):
    """
    Computes accuracy:
        :param ground_truth:
        :param predictions:
        :param display: Whether to print values to stdout
        :param mode: if 'per_char' is selected then
                    single_label_accuracy = correct_predicted_char_nums_of_single_sample / single_label_char_nums
                    avg_label_accuracy = sum(single_label_accuracy) / label_nums
                    if 'full_sequence' is selected then
                    single_label_accuracy = 1 if the prediction result is exactly the same as label else 0
                    avg_label_accuracy = sum(single_label_accuracy) / label_nums
        :return: avg_label_accuracy
        """
    if mode == 'per_char':
        accuracy = []
        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            # total_count = len(label)
            if len(prediction) == 0 or len(label) == 0:
                accuracy.append(0.0)
            elif len(prediction) > len(label):
                accuracy.append(1.0 - edit_distance(label,
                                prediction) / len(prediction))
            else:
                accuracy.append(
                    1.0 - edit_distance(label, prediction) / len(label))
        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    elif mode == 'full_sequence':
        try:
            correct_count = 0
            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                if prediction == label:
                    correct_count += 1
            avg_accuracy = correct_count / len(ground_truth)
        except ZeroDivisionError:
            if not predictions:
                avg_accuracy = 1
            else:
                avg_accuracy = 0
    else:
        raise NotImplementedError(
            'Other accuracy compute mode has not been implemented')
    return avg_accuracy


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def crop_and_wrap(image, poly):
    x1, y1, x2, y2, x3, y3, x4, y4 = poly
    widthA = int(distance((x1, y1), (x2, y2)))
    widthB = int(distance((x3, y3), (x4, y4)))
    heightA = int(distance((x1, y1), (x4, y4)))
    heightB = int(distance((x3, y3), (x2, y2)))
    maxWidth = max(widthA, widthB)
    maxHeight = max(heightA, heightB)

    src = np.array(poly, dtype="float32").reshape(4, 2)
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1,
                   maxHeight-1], [0, maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warp


def crop_and_wrap_polys(image, polys):
    return [crop_and_wrap(image, poly) for poly in polys]


class CRAFT_Detector():
    def __init__(self, config):
        self.net = CRAFT()     # initialize
        self.device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_name)
        self.config = config
        print('Loading weights from checkpoint (' +
              self.config["trained_model"] + ')')
        if self.device_name == "cuda":
            self.net.load_state_dict(copyStateDict(
                torch.load(self.config["trained_model"])))
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
        else:
            self.net.load_state_dict(copyStateDict(torch.load(
                self.config["trained_model"], map_location='cpu')))
        self.net.eval()

        if config["refiner_model"]:
            self.refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' +
                  config["refiner_model"] + ')')
            if self.device_name == "cuda":
                self.refine_net.load_state_dict(
                    copyStateDict(torch.load(config["refiner_model"])))
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(copyStateDict(
                    torch.load(config["refiner_model"], map_location='cpu')))
            self.refine_net.eval()
        else:
            self.refine_net = None

    def get_textbox(self, image):  # in BGR format
        result = []
        estimate_num_chars = self.config["optimal_num_chars"] is not None
        # bboxes, polys = test_net(canvas_size, mag_ratio, detector, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars)
        img_resized, target_ratio, _ = utils.imgproc.resize_aspect_ratio(
            image, self.config["canvas_size"], interpolation=cv2.INTER_LINEAR, mag_ratio=self.config["mag_ratio"])
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = utils.imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        x = x.to(self.device)

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        # Post-processing
        boxes, polys, mapper = utils.craft_utils.getDetBoxes(
            score_text, score_link, self.config["text_threshold"], self.config["link_threshold"], self.config["low_text"], self.config["poly"])

        # coordinate adjustment
        boxes = utils.craft_utils.adjustResultCoordinates(
            boxes, ratio_w, ratio_h)
        polys = utils.craft_utils.adjustResultCoordinates(
            polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]

        if estimate_num_chars:
            polys = [p for p, _ in sorted(self.config["polys"], key=lambda x: abs(
                self.config["optimal_num_chars"] - x[1]))]

        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            result.append(poly)

        return result

    def group_text_box(self, polys):
        horizontal_list, free_list, combined_list, merged_list = [], [], [], []
        for poly in polys:
            slope_up = (poly[3] - poly[1]) / \
                np.maximum(10, (poly[2] - poly[0]))
            slope_down = (poly[5] - poly[7]) / \
                np.maximum(10, (poly[4] - poly[6]))
            if max(abs(slope_up), abs(slope_down)) < self.config["slope_ths"]:
                x_max = max([poly[0], poly[2], poly[4], poly[6]])
                x_min = min([poly[0], poly[2], poly[4], poly[6]])
                y_max = max([poly[1], poly[3], poly[5], poly[7]])
                y_min = min([poly[1], poly[3], poly[5], poly[7]])
                horizontal_list.append(
                    [x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min])
            else:
                height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
                width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])
                margin = int(
                    1.44 * self.config["add_margin"] * min(width, height))
                theta13 = abs(
                    np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))))
                theta24 = abs(
                    np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))))
                # do I need to clip minimum, maximum value here?
                x1 = poly[0] - np.cos(theta13) * margin
                y1 = poly[1] - np.sin(theta13) * margin
                x2 = poly[2] + np.cos(theta24) * margin
                y2 = poly[3] - np.sin(theta24) * margin
                x3 = poly[4] + np.cos(theta13) * margin
                y3 = poly[5] + np.sin(theta13) * margin
                x4 = poly[6] - np.cos(theta24) * margin
                y4 = poly[7] + np.sin(theta24) * margin
                free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

        # combine box
        new_box = []
        for poly in horizontal_list:
            if len(new_box) == 0:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                new_box.append(poly)
            else:
                # comparable height and comparable y_center level up to ths*height
                if abs(np.mean(b_ycenter) - poly[4]) < self.config["ycenter_ths"] * np.mean(b_height):
                    b_height.append(poly[5])
                    b_ycenter.append(poly[4])
                    new_box.append(poly)
                else:
                    b_height = [poly[5]]
                    b_ycenter = [poly[4]]
                    combined_list.append(new_box)
                    new_box = [poly]
        combined_list.append(new_box)

        # merge list use sort again
        for boxes in combined_list:
            if len(boxes) == 1:  # one box per line
                box = boxes[0]
                margin = int(self.config["add_margin"]
                             * min(box[1] - box[0], box[5]))
                merged_list.append(
                    [box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])
            else:  # multiple boxes per line
                boxes = sorted(boxes, key=lambda item: item[0])
                merged_box, new_box = [], []
                for box in boxes:
                    if len(new_box) == 0:
                        b_height = [box[5]]
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        # merge boxes
                        if (abs(np.mean(b_height) - box[5]) < self.config["height_ths"] * np.mean(b_height)) and (abs(box[0] - x_max) < self.config["width_ths"] * (box[3] - box[2])):
                            b_height.append(box[5])
                            x_max = box[1]
                            new_box.append(box)
                        else:
                            b_height = [box[5]]
                            x_max = box[1]
                            merged_box.append(new_box)
                            new_box = [box]
                if len(new_box) > 0:
                    merged_box.append(new_box)

                for mbox in merged_box:
                    if len(mbox) != 1:  # adjacent box in same line
                        # do I need to add margin here?
                        x_min = min(mbox, key=lambda x: x[0])[0]
                        x_max = max(mbox, key=lambda x: x[1])[1]
                        y_min = min(mbox, key=lambda x: x[2])[2]
                        y_max = max(mbox, key=lambda x: x[3])[3]
                        box_width = x_max - x_min
                        box_height = y_max - y_min
                        margin = int(
                            self.config["add_margin"] * (min(box_width, box_height)))
                        merged_list.append(
                            [x_min - margin, x_max + margin, y_min - margin, y_max + margin])
                    else:  # non adjacent box in same line
                        box = mbox[0]
                        box_width = box[1] - box[0]
                        box_height = box[3] - box[2]
                        margin = int(
                            self.config["add_margin"] * (min(box_width, box_height)))
                        merged_list.append(
                            [box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])

        # may need to check if box is really in image
        if self.config["min_size"]:
            horizontal_list = [i for i in horizontal_list if max(
                i[1]-i[0], i[3]-i[2]) > self.config["min_size"]]
            free_list = [i for i in free_list if max(utils.craft_utils.diff(
                [c[0] for c in i]), utils.craft_utils.diff([c[1] for c in i])) > self.config["min_size"]]

        return merged_list, free_list


class Deep_OCR():
    def __init__(self, config):
        """ model configuration """
        self.config = config
        self.device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_name)
        self.converter = CTCLabelConverter(self.config["character"], self.device)

        self.ocr_model = OCR(self.config)
        self.ocr_model = torch.nn.DataParallel(self.ocr_model).to(self.device)
        state_dict = torch.load(self.config["deep_ocr"], map_location=torch.device(self.device))
        for name, param in self.ocr_model.named_parameters():
            if name not in state_dict: print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print('{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        print(f"Loading weight ({self.config['deep_ocr']})")
        self.ocr_model.load_state_dict(state_dict, strict=False)

        """ setup loss """
        # self.criterion = torch.nn.CTCLoss(zero_infinity=True).to(self.device)

    def preprocess(self, batch):
        batch = filter(lambda x: x is not None, batch)
        batch = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in batch]

        if self.config["PAD"]:  # same concept with 'Rosetta' paper
            input_channel = 3
            transform = utils.deep_ocr_utils.NormalizePAD((input_channel, self.config["imgH"], self.config["imgW"]))
            resized_images = []
            for image in batch:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.config["imgH"] * ratio) > self.config["imgW"]:
                    resized_w = self.config["imgW"]
                else:
                    resized_w = math.ceil(self.config["imgH"] * ratio)
                resized_image = image.resize((resized_w, self.config["imgH"]), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        else:
            transform = utils.deep_ocr_utils.ResizeNormalize((self.config["imgW"], self.config["imgH"]))
            image_tensors = [transform(image) for image in batch]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors

    def predict(self, images_ndarray):
        preds_str = []
        i = 0
        flag = True
        while flag:
            if len(images_ndarray) - i < self.config["batch_size"]:
                batch = images_ndarray[i:]
                flag = False
            else:
                batch = images_ndarray[i:i+self.config["batch_size"]]
                i = i + self.config["batch_size"]
                flag = True

            image_tensors = self.preprocess(batch)
            batch_size = image_tensors.size(0)
            # length_of_data = length_of_data + batch_size
            images = image_tensors.to(self.device)
            text_for_pred = torch.LongTensor(batch_size, config["batch_max_length"] + 1).fill_(0).to(self.device)
            # text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=opt.batch_max_length)

            preds = self.ocr_model(images, text_for_pred)
        
            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            for text in self.converter.decode(preds_index.data, preds_size.data):
                preds_str.append(text)

        return preds_str

if __name__ == '__main__':
    config = {"trained_model": "weights/craft_mlt_25k.pth", "text_threshold": 0.7,
              "refiner_model": "weights/craft_refiner_CTW1500.pth",
              "poly": False, "min_size": 20, "low_text": 0.4, "link_threshold": 0.4,
              "canvas_size": 2560, "mag_ratio": 1.0, "slope_ths": 0.5, "ycenter_ths": 0.5,
              "height_ths": 0.5, "width_ths": 0.5, "add_margin": 0.1, "optimal_num_chars": None,
              "deep_ocr": "weights/deep_ocr_benchmark_weights.pth",
              "character": 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ōĀðü€öÜÖÐ²™ūāŌ°Ū',
              # "Transformation": None, "FeatureExtraction": "VGG", "SequenceModeling": "BiLSTM", "Prediction": "CTC",
              "imgH": 32, "imgW": 512, "batch_size": 16, "batch_max_length": 256, "PAD": True,
              "input_channel": 3, "output_channel": 512, "hidden_size": 256
              }

    detector = CRAFT_Detector(config)
    deep_ocr = Deep_OCR(config)

    from tqdm import tqdm

    for img_path in tqdm(["000.jpg", "0012.jpg", "015.jpg", "413a756af9470b195256.jpg"]):
        image = utils.imgproc.loadImage(img_path)
        polys = detector.get_textbox(image)
        line_text = crop_and_wrap_polys(image, polys)
        preds_txt = deep_ocr.predict(line_text)
        image_draw = utils.deep_ocr_utils.draw_ocr(image, polys, preds_txt)
        show_polys(image_draw)

    import ipdb; ipdb.set_trace()

    # merged_list, free_list = detector.group_text_box(polys)
    # show_polys(image, polys)
    # show_polys(image, merged_list, mode="xyxy", index=[0, 2, 1, 3])
