#encoding=utf-8
#sdsdsd
import argparse
import os

import cv2
import numpy as np


# 保存视频中的匹配模板
def save_match_capture(path:str, video_file:str):
    # 打开视频文件
    # video_file = "/home/zm/Documents/project/sports_videos/偏右视角，姿势正确.MP4"
    video = cv2.VideoCapture(video_file)

    # 读取视频的某一帧
    ret, frame = video.read()
    # frame, scale_factor= resize_image(frame, (1366, 768))

    # 显示视频帧并选择选框
    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)

    # 提取选框的坐标信息
    x, y, w, h = bbox

    # 裁剪选框内的图像
    roi = frame[y:y+h, x:x+w]

    # 显示裁剪后的图像
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

    # 保存裁剪后的图像
    cv2.imwrite(path, roi)

    # 释放视频对象和窗口
    video.release()
    cv2.destroyAllWindows()

# 查看匹配效果
def match_frame(path:str, frame, match_all:bool):
    mapping = {
        'back_up.jpg': 0,
        'back_down.jpg': 1,
        'leg_up.jpg': 3,
        'leg_down.jpg': 2,
        'wheel.jpg': 4
    }
    points = [[-1,-1] for i in range(5)]
    # 定义模板图像
    if match_all:
        template_path = os.path.dirname(path)
        files = os.listdir(template_path)
        templates = [cv2.imread(os.path.join(template_path, file), 0) for file in files]
        # templates = [cv2.resize(t, (int(t.shape[1]*frame.shape[1]/1366), int(t.shape[0]*frame.shape[0]/768))) for t in templates]
    else:
        template = cv2.imread(path, 0)
        # template = cv2.resize(template, (int(template.shape[1]*frame.shape[1]/1366), int(template.shape[0]*frame.shape[0]/768)))

    # 将当前帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if match_all:
        for i, template in enumerate(templates):
            # 进行模板匹配
            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

            # 获取匹配结果的位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            h, w = template.shape
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center = [int((top_left[0]+bottom_right[0])/2), int((top_left[1]+bottom_right[1])/2)]
            points[mapping[files[i]]] = center

            # 在视频帧上绘制矩形框显示匹配结果
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            put_text(frame, top_left, bottom_right)
            

    else:
        # 进行模板匹配
        result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

        # 获取匹配结果的位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        h, w = template.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center = [int((top_left[0]+bottom_right[0])/2), int((top_left[1]+bottom_right[1])/2)]
        points[mapping[os.path.basename(path)]]= center

        # 在视频帧上绘制矩形框显示匹配结果
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        put_text(frame, top_left, bottom_right)
    
    return points


# 查看匹配效果
def match_template(path:str, video_file:str, match_all:bool):
    # 定义图像缩放的比例
    # target_resolution = (384, 192)
    target_resolution = (768, 384)
    # target_resolution = (1366, 768)

    # 定义模板图像
    if match_all:
        template_path = os.path.dirname(path)
        files = os.listdir(template_path)
        templates = [cv2.imread(os.path.join(template_path, file), 0) for file in files]
        templates = [cv2.resize(t, (int(t.shape[1]*target_resolution[0]/1366), int(t.shape[0]*target_resolution[1]/768))) for t in templates]
    else:
        template = cv2.imread(path, 0)
        template = cv2.resize(template, (int(template.shape[1]*target_resolution[0]/1366), int(template.shape[0]*target_resolution[1]/768)))

    # 打开视频文件
    # video_file = "/home/zm/Documents/project/sports_videos/偏右视角，姿势正确.MP4"
    video = cv2.VideoCapture(video_file)

    # 获取视频的帧率
    fps = video.get(cv2.CAP_PROP_FPS)

    # 创建窗口并命名
    cv2.namedWindow(f"Video[{target_resolution[0]}x{target_resolution[1]}]")

    # 播放视频
    while True:
        # 读取视频的一帧
        ret, frame = video.read()
        if not ret:
            # 视频读取完毕
            break
        frame, scale_factor= resize_image(frame, target_resolution)

        # 将当前帧转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if match_all:
            for template in templates:
                # 进行模板匹配
                result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

                # 获取匹配结果的位置
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                top_left = max_loc
                h, w = template.shape

                # 在视频帧上绘制矩形框显示匹配结果
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                put_text(frame, top_left, bottom_right)

        else:
            # 进行模板匹配
            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

            # 获取匹配结果的位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            h, w = template.shape

            # 在视频帧上绘制矩形框显示匹配结果
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            put_text(frame, top_left, bottom_right)



        # 显示当前帧
        cv2.imshow(f"Video[{target_resolution[0]}x{target_resolution[1]}]", frame)

        # 按键盘上的空格键暂停/继续播放视频
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == ord(' '):
            while True:
                key = cv2.waitKey(1) or 0xFF
                cv2.imshow(f"Video[{target_resolution[0]}x{target_resolution[1]}]", frame)
                if key == ord(' '):
                    break
        elif key == 27:
            # 按下ESC键退出程序
            break

    # 释放视频对象和窗口
    video.release()
    cv2.destroyAllWindows()

# 对图片进行缩放
def resize_image(image, target_resolution):
    # 获取图片分辨率
    original_resolution = (image.shape[1], image.shape[0])
    
    # 计算裁剪后的左上角坐标和宽高
    crop_width = original_resolution[0] % 32
    crop_height = original_resolution[1] % 32
    crop_x = crop_width // 2
    crop_y = crop_height // 2
    crop_width = original_resolution[0] - crop_width
    crop_height = original_resolution[1] - crop_height
    
    # 进行裁剪
    cropped_image = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    
    # 缩放图片
    resized_image = cv2.resize(cropped_image, target_resolution)
    
    # 计算缩放因子
    scale_factor = (
        target_resolution[0] / original_resolution[0],
        target_resolution[1] / original_resolution[1]
    )
    
    return resized_image, scale_factor

def put_text(frame, top_left, bottom_right):
    # 定义文本内容和坐标
    center = int((top_left[0] + bottom_right[0])/2.0), int((top_left[1] + bottom_right[1])/2.0)
    # print(center)
    text = 'COORD:({}, {})'.format(center[0], center[1])
    coordinates = center

    # 设置字体类型、大小、颜色和粗细
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.8*frame.shape[1]/1366
    font_color = (255, 255, 255)
    font_thickness = 1

    # 在图像上显示文本
    cv2.putText(frame, text, coordinates, font, font_size, font_color, font_thickness)


def compute_angle(A:np.array, B:np.array, C:np.array, D:np.array)->float:
    """
    计算AB和CD的夹角
    """
    # 计算向量 AB 和 CD
    AB = B - A
    CD = D - C
    # 计算夹角的余弦值
    cosine_angle = np.dot(AB, CD) / (np.linalg.norm(AB) * np.linalg.norm(CD))
    # 计算夹角的弧度值
    angle_radians = np.arccos(cosine_angle)
    # 将弧度值转换为角度值
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def compute_dist(P:np.array, A:np.array, B:np.array=None)->float:
    """
    计算两点之间的距离或点到直线的距离
    """
    if B is None:
        return np.linalg.norm(P - A)
    else:
        # 计算直线向量和点到直线的向量
        line_vector = B - A
        point_vector = P - A

        # 计算直线向量的单位向量
        line_unit_vector = line_vector / np.linalg.norm(line_vector)

        # 计算点到直线的投影向量
        projection_vector = np.dot(point_vector, line_unit_vector) * line_unit_vector

        # 计算点到直线的距离
        distance = np.linalg.norm(point_vector - projection_vector)
        return distance
    
# 运动计算    
def compute(points):
    """
    坐姿要求：
        1.腰背与靠垫紧贴
        2.脚面钩住转动轴
        3.膝盖窝尽量卡住坐垫边缘  
         
    动作过程： 
        4.顺时针80度～65度之间作为起始位置
        5.以膝关节为定点旋转至180度～170度以上后再回到起始位置
        6.运动过程中背部、膝盖窝、大腿位置不发生改变
        
    Args:
        points (np.array): 9个关键点坐标，前5个背部上下、大腿上下和转动轴，后4个为颈、腰、膝盖、脚踝
    """    
    # requirement 1
    deg1 = compute_angle(points[1], points[0], points[6], points[5])
    dist1 = compute_dist(points[5], points[1], points[0])
    # requirement 2
    dist2 = compute_dist(points[4], points[8], points[7])
    # requirement 3
    dist3 = compute_dist(points[7], points[3])
    # requirement 4
    deg4 = compute_angle(points[7], points[6], points[7], points[8])
    # requirement 5
    # requirement 6
    dist6 = compute_dist(points[2], points[6]) # 大腿
    return {
        'deg1': deg1,
        'dist1': dist1,
        'dist2': dist2,
        'dist3': dist3,
        'deg4': deg4,
        'dist6': dist6
    }

# 逻辑判断
def decide(degdists):
    """
    坐姿要求：
        1.腰背与靠垫紧贴
        2.脚面钩住转动轴
        3.膝盖窝尽量卡住坐垫边缘  
         
    动作过程： 
        4.顺时针80度～65度之间作为起始位置
        5.以膝关节为定点旋转至180度～170度以上后再回到起始位置
        6.运动过程中背部、膝盖窝、大腿位置不发生改变
        
    Args:
        degdists (list): 维护三个compute返回值，第一个为当前帧
    """   
    percent = 0.2
    # requirement 1
    ret1 = degdists[0]['dist1'] < 500.0 and \
        degdists[0]['deg1'] < 5.0
    # requirement 2 
    ret2 = degdists[0]['dist2'] < 400.0
    # requirement 3
    ret3 = degdists[0]['dist3'] < 400.0
    # requirement 4
    ret4 = None
    if len(degdists) < 3 or \
        (degdists[0]['deg4'] > degdists[1]['deg4'] and degdists[1]['deg4'] < degdists[2]['deg4']):
        ret4 = degdists[0]['deg4'] > 65.0 and degdists[0]['deg4'] < 80.0
    # requirement 5
    ret5 = None
    if len(degdists) >= 3 and \
        degdists[0]['deg4'] < degdists[1]['deg4'] and degdists[1]['deg4'] > degdists[2]['deg4']:
        ret5 = degdists[0]['deg4'] > 170.0 and degdists[0]['deg4'] < 180.0
    # requirement 6
    ret6 = True
    if len(degdists) >= 3:
        ret6 =  abs(degdists[0]['dist1'] - degdists[2]['dist1']) < percent * degdists[2]['dist1'] and \
                abs(degdists[0]['deg1'] - degdists[2]['deg1']) < 5.0 and \
                abs(degdists[0]['dist3'] - degdists[2]['dist3']) < percent * degdists[2]['dist3'] and \
                abs(degdists[0]['dist6'] - degdists[2]['dist6']) < percent * degdists[2]['dist6']
    return [ret1, ret2, ret3, ret4, ret5, ret6]

# 打印文字和底纹
def put_text_shading(image, text, pos, font, font_size, font_color, font_thickness, bg_color):
    # 获取文本框大小
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    # 计算底纹框的位置和大小
    bg_box = (pos[0]-10, pos[1]-60, text_size[0]+20, text_size[1]+20)
    # 绘制底纹框
    cv2.rectangle(image, (bg_box[0], bg_box[1]), (bg_box[0] + bg_box[2], bg_box[1] + bg_box[3]), bg_color, -1)
    # 绘制文本
    cv2.putText(image, text, pos, font, font_size, font_color, font_thickness)
    

# 输出逻辑判断结果
def put_decision(canvas, degdist, result):
    # 设置字体类型、大小、颜色和粗细
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.8
    font_color = (255, 255, 255)
    font_thickness = 1
    text_dist = 30
    text_num = 11
    pos_y_start = 50
    pos_x = np.ones((text_num, 1)) * text_dist
    pos_y = np.arange(pos_y_start, pos_y_start+text_num*text_dist, text_dist).reshape((text_num, 1))
    pos = np.concatenate((pos_x, pos_y), axis=1).astype(int).tolist()
    
    #设置底纹颜色
    bg_color = (0, 0, 255)  # 底纹颜色 (红色)



    # 在图像上显示文本
    put_text_shading(canvas, f"1_1.Angle between the back and the backrest:{degdist['deg1']:.2f}", pos[0], font, font_size, font_color, font_thickness, bg_color)
    put_text_shading(canvas, f"1_2.Dist between the back and the backrest:{degdist['dist1']:.2f}", pos[1], font, font_size, font_color, font_thickness, bg_color)
    put_text_shading(canvas, f"2.Dist from the rotation axis to the lower leg:{degdist['dist2']:.2f}", pos[2], font, font_size, font_color, font_thickness, bg_color)
    put_text_shading(canvas, f"3.Dist between the knee recess and the edge of the seat cushion:{degdist['dist3']:.2f}", pos[3], font, font_size, font_color, font_thickness, bg_color)
    put_text_shading(canvas, f"4.Angle between the thigh and the calf:{degdist['deg4']:.2f}", pos[4], font, font_size, font_color, font_thickness, bg_color)

    put_text_shading(canvas, f"Is requirement 1 satisfied? {result[0]}", pos[5], font, font_size, font_color, font_thickness, bg_color)
    put_text_shading(canvas, f"Is requirement 2 satisfied? {result[1]}", pos[6], font, font_size, font_color, font_thickness, bg_color)
    put_text_shading(canvas, f"Is requirement 3 satisfied? {result[2]}", pos[7], font, font_size, font_color, font_thickness, bg_color)
    put_text_shading(canvas, f"Is requirement 4 satisfied? {result[3]}", pos[8], font, font_size, font_color, font_thickness, bg_color)
    put_text_shading(canvas, f"Is requirement 5 satisfied? {result[4]}", pos[9], font, font_size, font_color, font_thickness, bg_color)
    put_text_shading(canvas, f"Is requirement 6 satisfied? {result[5]}", pos[10], font, font_size, font_color, font_thickness, bg_color)
    
    return canvas


def main(args):
    # 在这里编写你的代码逻辑
    path = args.path
    video = args.video
    if not os.path.exists(video):
        raise Exception("video file not exists")
    if args.capture:
        save_match_capture(path, video)
    if args.match:
        if not os.path.exists(path):
            raise Exception("template file not exists")
        match_all = args.all
        match_template(path, video, match_all)

    print("finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="命令行参数解析")
    # 添加命令行参数
    parser.add_argument("-p", "--path", type=str, default='./', help="模板路径，.../back_down|back_up|leg_down|leg_up|wheel.jpg")
    parser.add_argument("-v", "--video", type=str, required=True, help="视频路径")
    parser.add_argument("-a", "--all", action="store_true", help="匹配所有模板")
    parser.add_argument("-c", "--capture", action="store_true", help="存储模板")
    parser.add_argument("-m", "--match", action="store_true", help="模板匹配")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用主函数
    main(args)

# python tools.py -p ./images/template1/wheel.jpg -v /home/zm/Documents/project/sports_videos/偏右视角，姿势正确.MP4 --match
# python tools.py -p ./images/template/template3 -v ./videos/test/姿势正确.mp4 -c 

