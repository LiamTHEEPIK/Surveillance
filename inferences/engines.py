import onnxruntime as ort
import cv2
import toml
import numpy as np


configs = toml.load('inferences/configs/config.toml')

detection_session = ort.InferenceSession(configs['detection-model-path'], providers=configs['providers'])
extraction_session = ort.InferenceSession(configs['extraction-model-path'], providers=configs['providers'])

width = configs['segment-width']
height = configs['segment-height']
length = configs['segment-length']

x1 = configs['crop-x1']
x2 = configs['crop-x2']
y1 = configs['crop-y1']
y2 = configs['crop-y2']

std = configs['normalization-std']
mean = configs['normalization-mean']

smoothing_weight = np.ones(configs['smoothing-window']) / configs['smoothing-window']


def normalize(inputs):
    return (inputs - mean) / std


def convert_frame(frame):
    converted_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    converted_frame = converted_frame[y1:y2, x1:x2]

    return cv2.cvtColor(converted_frame, cv2.COLOR_BGR2RGB)


def load_next_segment(capture):
    segment_frames = []

    while len(segment_frames) < length:
        read_success, captured_frame = capture.read()

        if read_success:
            segment_frames.append(convert_frame(captured_frame))
        else:
            return False, None

    return True, convert_segment(segment_frames)


def convert_segment(frames):
    converted_segment = np.stack(frames, axis=0).transpose((3, 0, 1, 2))

    if configs['precision'] == 'fp16':
        return normalize(converted_segment).astype(np.float16)
    else:
        return normalize(converted_segment).astype(np.float32)


def extract_segment_features(segment):
    expanded_segment = np.expand_dims(segment, axis=0)

    extraction_outputs = extraction_session.run(['outputs'], {'inputs': expanded_segment})
    extraction_outputs = extraction_outputs[0]

    return np.squeeze(extraction_outputs, axis=0)


def extract_video_features(video_path):
    features = []
    capture = cv2.VideoCapture(video_path)

    while capture.isOpened():
        load_success, converted_segment = load_next_segment(capture)

        if load_success:
            features.append(extract_segment_features(converted_segment))
        else:
            capture.release()

    return np.stack(features, axis=0)


def convert_features(features):
    if configs['precision'] == 'fp16':
        return features.astype(np.float16)
    else:
        return features.astype(np.float32)


def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))


def detection_by_features(features):
    expanded_features = np.expand_dims(features, axis=0)

    detection_outputs = detection_session.run(['outputs'], {'inputs': expanded_features})
    detection_outputs = detection_outputs[0]

    return sigmoid(np.squeeze(detection_outputs, axis=0))


def score_smoothing(scores):
    return np.convolve(scores, smoothing_weight, mode='same').round(decimals=2)


def expand_scores(scores):
    return np.array(scores).repeat(length, axis=0)


def detection_by_video(video_path):
    extracted_features = extract_video_features(video_path)
    converted_features = convert_features(extracted_features)

    return score_smoothing(detection_by_features(converted_features))


def anomaly_prompt_enhancement(frame, prompt):
    return cv2.putText(frame, prompt, (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 215), thickness=2)


def anomaly_border_enhancement(frame, border):
    frame_x = frame.shape[1]
    frame_y = frame.shape[0]

    return cv2.rectangle(frame, (0, 0), (frame_x, frame_y), (0, 0, 215), thickness=border)


def draw_detection_result(frame, score):
    if score > configs['anomaly-threshold']:
        frame = anomaly_prompt_enhancement(frame, configs['anomaly-prompt'])
        frame = anomaly_border_enhancement(frame, configs['anomaly-border'])

        return cv2.putText(frame, f'{score:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 215), thickness=2)
    else:
        return cv2.putText(frame, f'{score:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 215, 0), thickness=2)
