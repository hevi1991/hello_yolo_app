import cv2

def read_video(file_path):
  """读取视频所有的帧

  Args:
      file_path (str): 视频文件路径

  Returns:
      MatLike[]: 视频帧数组
  """
  
  if not file_path:
    raise ValueError("文件路径为空")
  
  video_capture =cv2.VideoCapture(file_path)
  
  # 存放每一帧
  frames = []
  
  while True:
    # 是否有返回，帧
    isReturn, frame = video_capture.read()

    if not isReturn:
      break
    else:
      frames.append(frame)
      
  # 释放摄像头
  video_capture.release()
  
  return frames

def save_video(frames, save_path):
  """将所有帧保存为视频

  Args:
      frames (_type_): 帧图片
      save_path (_type_): 保存路径
  """
  if not frames:
    raise ValueError("帧为空")
  
  # 编码器
  fourcc = cv2.VideoWriter_fourcc(*'avc1')
  # 视频大小
  frame_size = (frames[0].shape[1], frames[0].shape[0])
  
  output_video = cv2.VideoWriter(save_path, fourcc=fourcc, fps=24, frameSize=frame_size)
  
  for frame in frames:
    output_video.write(frame)
  
  output_video.release()
  
  