from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
import numpy as np


def face_verify():
    result = DeepFace.verify("imgs/huge/20250324095937.png",
                             "imgs/huge/20250324095959.png")
    '''
    {
        'verified': True, 
        'distance': 0.3951474552892865, 
        'threshold': 0.68,
        'model': 'VGG-Face', 
        'detector_backend': 'opencv', 
        'similarity_metric': 'cosine', 
        'facial_areas': {
            'img1': {'x': 80, 'y': 154, 'w': 308, 'h': 308, 'left_eye': (285, 270), 'right_eye': (172, 277)}, 
            'img2': {'x': 288, 'y': 184, 'w': 152, 'h': 152, 'left_eye': (385, 243), 'right_eye': (332, 242)}
        }, 
        'time': 4.36
    }
    '''
    print(result)


def face_recognition():
    '''
    人脸识别需要多次执行人脸验证。该函数会在db_path文件夹中搜索输入图像的身份，并返回一个包含 Pandas 数据帧的列表作为输出。同时，面部数据库中的面部嵌入会被存储在一个 pickle 文件中，以便下次更快地进行搜索。结果的大小将与源图像中出现的人脸数量一致。此外，数据库中的目标图像也可以包含多张人脸。
    '''
    results = DeepFace.find("imgs/20250324154252.png", db_path="imgs/huge")
    for dataFrame in results:
        print(dataFrame)


def embedding():
    embeddings = DeepFace.represent("imgs/huge/20250324095937.png")
    '''
    [
        {
            'embedding': [......],
            'facial_area': {'x': 80, 'y': 154, 'w': 308, 'h': 308, 'left_eye': (285, 270), 'right_eye': (172, 277)},
            'face_confidence': 0.93
        }
    ]
    '''
    output = []
    for obj in embeddings:
        if obj["face_confidence"] >= 0.5:
            output.append(obj["embedding"])

    return np.array(output)


def analyze():
    result = DeepFace.analyze("imgs/huge/20250324095937.png")
    '''
    [
        {
            'emotion': {
                'angry': 0.7332970388233662, 
                'disgust': 1.0652382309572772e-07, 
                'fear': 0.03976623702328652, 
                'happy': 0.32364167273044586, 
                'sad': 0.4743750672787428, 
                'surprise': 0.003194848250132054, 
                'neutral': 98.42572212219238
            }, 
            'dominant_emotion': 'neutral', 
            'region': {'x': 80, 'y': 154, 'w': 308, 'h': 308, 'left_eye': (285, 270), 'right_eye': (172, 277)}, 
            'face_confidence': 0.93, 
            'age': 20, 
            'gender': {'Woman': 0.0006645485882472713, 'Man': 99.99933242797852}, 
            'dominant_gender': 'Man', 
            'race': {'asian': 60.22493839263916, 'indian': 7.956235855817795, 'black': 1.1165719479322433, 'white': 7.318518310785294, 'middle eastern': 1.7460104078054428, 'latino hispanic': 21.637719869613647}, 
            'dominant_race': 'asian'
        }
    ]
    '''
    print(result)


def detectFace():
    # result = DeepFace.detectFace("imgs/20250324102804.png")
    # cv2.imshow("result", result)
    # cv2.waitKey(0)

    result = DeepFace.extract_faces(
        "imgs/20250324102804.png", detector_backend="mtcnn")
    src = cv2.imread("imgs/20250324102804.png", cv2.IMREAD_COLOR)
    for val in result:
        facial_area = val['facial_area']
        src = cv2.rectangle(src, (facial_area['x'], facial_area['y']), (
            facial_area['x']+facial_area['w'], facial_area['y']+facial_area['h']), (0, 255, 0), 2)

    cv2.imshow("result", src)
    cv2.waitKey(0)


def real_time():
    '''
    您也可以将 DeepFace 用于实时视频。Stream 函数会访问您的摄像头，并同时应用人脸识别和面部属性分析。如果该函数能够连续在 5 帧中聚焦到一张人脸，它就会开始分析该帧。然后，它会显示分析结果 5 秒钟。
    '''
    DeepFace.stream(db_path="C:/database")


def real_time2():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        result = DeepFace.analyze(frame, actions=['emotion'])
        print(result[0]['dominant_emotion'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_recognition()
