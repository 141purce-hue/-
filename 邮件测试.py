#coding:utf-8
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import torch
import serial
import time
import serial.tools.list_ports
#import webbrowser
#import re
#import os
from pyzbar.pyzbar import decode
from torch.distributed.nn import all_reduce

# 导出onnx模型
# model = YOLO("MyModels/best.pt")
# model.export(format="onnx")

# 用于存储已经扫描过的条形码
scanned_barcodes = set()

# 读取文本文件内容并建立条形码到名称的映射
barcode_to_name = {}
try:
    with open(r"F:\比赛\邮件识别\识别地址码.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            if ":" in line:
                parts = line.split(":")
                barcode = parts[0].strip()  # 提取编号部分作为条形码
                name = parts[1].strip()  # 提取文本部分作为名称
                #area = parts[2].strip()  #提取文本部分作为地区
                barcode_to_name[barcode] = name
    print(barcode_to_name)
except FileNotFoundError:
    print("Error: 未找到识别地址码.txt文件，请检查文件路径。")


class YOLOv8:
    """YOLOv8目标检测模型类，用于处理推理和可视化操作。"""
    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres,classes):
        """
        初始化YOLOv8类的实例。
        参数:
            onnx_model: ONNX模型的路径。
            input_image: 输入图像的路径。
            confidence_thres: 过滤检测的置信度阈值。
            iou_thres: 非极大抑制的IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # 检测物体的类别名称 # 字典存储类别名称
        # self.classes = {0:"bottle",1:"mouse",2:"staple",3:"staple",4:"staple",5:"staple"}
        # self.classes = {0: "Battery", 1: "Candy", 2: "Eyeglass cleaner"}
        self.classes=classes
        print(self.classes)
        # 为类别生成颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 初始化ONNX会话
        self.initialize_session(self.onnx_model)

    def draw_detections(self, img, box, score, class_id):
        """
        根据检测到的对象在输入图像上绘制边界框和标签。
        参数:
            img: 要绘制检测的输入图像。
            box: 检测到的边界框。
            score: 对应的检测得分。
            class_id: 检测到的对象的类别ID。
        返回:
            None
        """
        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取类别ID对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类名和得分的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制填充的矩形作为标签文本的背景
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 条形码识别部分
        decoded_objects = decode(img)
        for obj in decoded_objects:
            barcode = obj.data.decode("utf-8")
            if barcode not in scanned_barcodes:
                scanned_barcodes.add(barcode)
                # 绘制多边形框在二维码周围
                points = obj.polygon
                if len(points) == 4:  # 通常二维码有4个角
                    pts = np.int32(points).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # 获取解码后的数据对应的名称
                name = barcode_to_name.get(barcode, "未找到对应名称")

                # 在图像上绘制解码后的数据文本
                cv2.putText(img, name, (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def preprocess(self):
        """
        在进行推理之前，对输入图像进行预处理。
        返回:
            image_data: 预处理后的图像数据，准备好进行推理。
        """

        # 使用OpenCV读取输入图像(h,w,c)
        # self.img = cv2.imread(self.input_image)
        ret, frame = cap.read()
        # cv2.imshow('Camera', frame)  # 显示画面
        self.img = frame

        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像颜色空间从BGR转换为RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 将图像调整为匹配输入形状(640,640,3)
        img = cv2.resize(img, (self.input_width, self.input_height))

        # 将图像数据除以255.0进行归一化
        image_data = np.array(img) / 255.0

        # 转置图像，使通道维度成为第一个维度(3,640,640)
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先

        # 扩展图像数据的维度以匹配期望的输入形状(1,3,640,640)
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # 返回预处理后的图像数据
        return image_data

    def postprocess(self, input_image, output):
        """
        对模型的输出进行后处理，以提取边界框、分数和类别ID。
        参数:
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。
        返回:
            numpy.ndarray: 输入图像，上面绘制了检测结果。

        """

        # 转置并压缩输出以匹配期望的形状：(8400, 84)
        outputs = np.transpose(np.squeeze(output[0]))
        # 获取输出数组的行数
        rows = outputs.shape[0]
        # 存储检测到的边界框、分数和类别ID的列表
        boxes = []
        scores = []
        class_ids = []
        # 计算边界框坐标的比例因子
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # 遍历输出数组的每一行
        for i in range(rows):
            # 从当前行提取类别的得分
            classes_scores = outputs[i][4:]
            # 找到类别得分中的最大值
            max_score = np.amax(classes_scores)

            # 如果最大得分大于或等于置信度阈值
            if max_score >= self.confidence_thres:
                # 获取得分最高的类别ID
                class_id = np.argmax(classes_scores)

                # 从当前行提取边界框坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算边界框的缩放坐标
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # 将类别ID、得分和边界框坐标添加到相应的列表中
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # 应用非极大抑制以过滤重叠的边界框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        # print("indices:",indices)
        # 遍历非极大抑制后选择的索引
        box_nms=[]
        score_nms=[]
        class_id_nms=[]
        for i in indices:
            # 获取与索引对应的边界框、得分和类别ID
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            box_nms.append(box)
            score_nms.append(score)
            class_id_nms.append(class_id)
            # 在输入图像上绘制检测结果
            self.draw_detections(input_image, box, score, class_id)
        # 返回修改后的输入图像
        # return input_image,class_ids,scores,boxes
        return input_image, class_id_nms,score_nms,box_nms
    def initialize_session(self, onnx_model):
        """
        初始化ONNX模型会话。
        :return:
        """
        if torch.cuda.is_available():
            print("Using CUDA")
            providers = ["CUDAExecutionProvider"]
        else:
            print("Using CPU")
            providers = ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 使用ONNX模型创建推理会话，并指定执行提供者
        self.session = ort.InferenceSession(onnx_model,
                                            session_options=session_options,
                                            providers=providers)
        return self.session

    def main(self):
        """
        使用ONNX模型进行推理，并返回带有检测结果的输出图像。

        我从小喜欢研究电路系统，所以根据兴趣，我大一就加入了集成电路工作室。


        返回:

            output_img: 带有检测结果的输出图像。
        """
        # 获取模型的输入
        model_inputs = self.session.get_inputs()
        # 保存输入的形状，稍后使用
        # input_shape：(1,3,640,640)
        # self.input_width:640,self.input_height:640
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # 对图像数据进行预处理
        img_data = self.preprocess()
        # 使用预处理后的图像数据运行推理,outputs:(1,84,8400)  8400 = 80*80 + 40*40 + 20*20
        outputs = self.session.run(None, {model_inputs[0].name: img_data})
        # 对输出进行后处理以获取输出图像
        return self.postprocess(self.img, outputs)  # 输出图像


# 串口发送数据
ser = None


def send_data(data):
    global ser

    if  ser!=None and ser.is_open:
        ser.write(data.encode('utf-8'))  # 将字符串转换为字节并发送
        print(f"发送数据: {data}")
    else:
        print("串口未打开")


#haucahubxbcacywd

def serial_init():  #串口初始化
    global ser
    ports = serial.tools.list_ports.comports()
    # 打开串口（这里假设使用的是 COM3，波特率为115200）
    for port in ports:
        if 'CH340' in port.description or 'USB' in port.description:
            ser = serial.Serial(port.device, 115200, timeout=1)  # 替换为实际串口名
    # 确认串口已打开
    if ser:
        if ser.is_open:
            print(f"串口 {ser.name} 已打开")
            return True
        else:
            return False
    else:
        print("串口未打开")

def type_of_detect_data(data_brx):
    send_back = 'CD:'
    for key,value in data_brx.items():
        send_back=send_back+str(key)+':'
        send_back=send_back+value+';'
    send_back = send_back + '\#\#'
    print(send_back)
    return  send_back

"""
def type_of_detcter_data(data_brx):
    send_back = 'CD:'
    for key,value in data_brx.items():
        send_back = send_back+str(key)+':'
        send_back=send_back+value+';'
    send_back = send_back + '\#\#'
    print(send_back)
    return send_back
"""




def combine_serial_data(class_ids_rx,boxes_rx ,class_id_rx):  #将发送的数据转换为字符格式并拼接为一串字符
    # print("cb",class_ids_rx)
    # print("b", boxes_rx)
    send_back = 'AB:'
    boxes_rx_center=[]
    for i in range(len(boxes_rx)):
        ko = []
        ko.append(int(boxes_rx[i][0]+  boxes_rx[i][2]/2))
        ko.append(int(boxes_rx[i][1] + boxes_rx[i][3] / 2))
        ko.append(boxes_rx[i][2])
        ko.append(boxes_rx[i][3])
        boxes_rx_center.append(ko)
    # print("前",boxes_rx)
    # print("后",boxes_rx_center):
        send_back = send_back + str(class_id_rx) + ','
        for j in boxes_rx_center[i]:
            send_back = send_back + str(j) + ','
        send_back = send_back + ':'
    send_back = send_back + '\#\#'
    # print(send_back)
    return send_back




def filter(rx_class):  #滤波算法，测试5次，有三次及以上相同则认为没有误判，返回这五次判断中需要串口发送的数据。
    print("rx_class:", rx_class)
    list_of_rx_class = [set(sublist) for sublist in rx_class]
    for i in list_of_rx_class:
        if list_of_rx_class.count(i)>=3:
            print(i)
            for t in range(len(list_of_rx_class)-1,-1,-1):
                if i==list_of_rx_class[t]:
                    print("return",t)
                    return t



if __name__ == "__main__":
    Confidence=0.5 #置信度
    IOU=0.8      #抑制值
    Camera_type=1#摄像头配置
    Classes={0:"black",1:"bar code",2:"letter"}
    onnx_model_name = ("youjian2.onnx")  # 模型名称
    Test_result={}  #本次测试的类别和数量
    # 串口初始化
    serial_init()
    # 创建用于处理命令行参数的解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=onnx_model_name, help="请输入您的ONNX模型路径.")
    parser.add_argument("--img", type=str, default=None, help="输入图像的路径.")
    parser.add_argument("--conf-thres", type=float, default=Confidence, help="置信度阈值.")
    parser.add_argument("--iou-thres", type=float, default=IOU, help="IoU（交并比）阈值.")
    args = parser.parse_args()
    detection = YOLOv8(args.model, args.img, args.conf_thres, args.iou_thres,Classes)
    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)  # 0 通常是默认摄像头的标识
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap = cv2.VideoCapture(0)
    class_mapping={0:"huazhong",1:"huanan",2:"huabei",3:"huadon",4:"letter",5:"black"}
    # time.sleep(3)
    rx=type_of_detect_data(class_mapping) #发送给主控识别物体的类别
    send_data(rx)

    while True:
        area_ID = None
        class_id_rx = None
        # 通过摄像头获取画面并显示
        ret, frame = cap.read()
        if not ret:
            break
        output_image,class_ids_rx,scores_rx,boxes_rx = detection.main()
        barcodes = decode(frame)#调用摄像头去识别摄像头内容，图片或者视频的每一帧
        if barcodes:#如果条形码存在则进入if循环当中
            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                if barcode_data in barcode_to_name:
                    name = barcode_to_name[barcode_data]
                    print(f"条形码:{barcode_data},名称:{name}")
                    if "华中地区" in name:
                        area_ID = 0
                    if "华南地区" in name:
                        area_ID = 1
                    if "华北地区" in name:
                        area_ID = 2
                    if "华东地区" in name:
                        area_ID = 3
                else:
                    print(f"条形码:{barcode_data},未识别到地址")
        else:
            print("未识别到条形码")

        if class_ids_rx : #如果检测到物体
            print("class:",class_ids_rx)
            #print(type(class_ids_rx))
            print("score:",scores_rx)
            print("box:",boxes_rx)
            #print('area_ID:',area_ID)
            if 0 in class_ids_rx:
                class_id_rx=5
                print("手眼标定")
            if 1 in class_ids_rx :
                if(area_ID==None):
                   class_id_rx=4
                   print("条形码不完全")
                else:
                    class_id_rx=area_ID
                #print(class_id_rx,type(class_id_rx))

            #print('rx:',rx)
            if class_id_rx == 4 :
                time.sleep(0.186)#当出现代码模糊识别的内容时，AB输出的值为4，那么当扫描速度快出现条形码内容不完全，增加一个0.185s的延时

            rx = combine_serial_data(class_ids_rx, boxes_rx, class_id_rx)
            send_data(rx) #串口发送数据

             # 串口发送数据
             # temp = class_ids_rx
            # for i in temp:
            #     num = temp.count(i)
            #     Test_result[Classes[i]] = num
            # print("test_result:", Test_result)

        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", output_image)
        cv2.imwrite('Output.jpg', output_image)

        # 按 'q' 键退出循环
        #class_id_rx
        if cv2.waitKey(1) & 0xFF == ord('A'):
            break

    # 释放摄像头资源并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

