import cv2
import time
import argparse

import torch
import model.detector
import utils.utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='C:\\Users\\Koren\\Git\\Yolo-FastestV2\\data\\armour.data', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='C:\\Users\\Koren\\Git\\Yolo-FastestV2\\modelzoo\\armour-model.pth', 
                        help='The path of the .pth model to be transformed')
    
    opt = parser.parse_args()

    cfg = utils.utils.load_datafile(opt.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    model.eval()

    #加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
            
    capture = cv2.VideoCapture(0)
    
    while True:
        start = time.perf_counter()

        # 从摄像头读取帧
        ret, ori_img = capture.read()
        cv2.imshow('Camera', ori_img)

        #数据预处理
        res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0,3, 1, 2))
        img = img.to(device).float() / 255.0

        #模型推理
        preds = model(img)

        #特征图后处理
        output = utils.utils.handel_preds(preds, cfg, device)
        output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)

        h, w, _ = ori_img.shape
        scale_h, scale_w = h / cfg["height"], w / cfg["width"]

        #绘制预测框
        for box in output_boxes[0]:
            box = box.tolist()

            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
            x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
            cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("test_result", ori_img)

        end = time.perf_counter()
        print("forward time:%fms"%((end - start) * 1000.))

        # 按'q'退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()