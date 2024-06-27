import cv2
import os

def extract_frames(video_path, output_folder, num_frames=6):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算采样间隔
    interval = total_frames // num_frames

    # 采样并保存帧
    for i in range(num_frames):
        frame_number = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_folder, f"frame_{i}.jpg")
            cv2.imwrite(output_path, frame)
        else:
            print(f"Failed to extract frame at {frame_number}")

    # 释放视频捕获对象
    cap.release()

# 示例用法
video_path = '/data/go_vocation/data/bottle.mp4'
output_folder = '/data/go_vocation/data/bottle'
extract_frames(video_path, output_folder, num_frames=12)
