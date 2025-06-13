import queue
import contextlib
import os
import datetime
import threading

import flask
import pymongo
import cv2
import toml
import snowflake

import inferences.engines as engines
import inferences.realtime as realtime

from apscheduler.schedulers.background import BackgroundScheduler


app = flask.Flask(__name__)

configs = toml.load('servers/configs/config.toml')

connection_uri = configs['db-connection-uri']
connection_max = configs['db-connection-max']

database = pymongo.MongoClient(connection_uri, maxpoolsize=connection_max)

scheduler = BackgroundScheduler()
scheduler.start()

realtime_sessions_lock = threading.Lock()
realtime_sessions = {}

remove_queue = queue.Queue()

frames_interval = configs['frames-interval']
remove_interval = configs['remove-interval']

video_speed = configs['video-speed']
video_width = configs['video-width']
cover_width = configs['cover-width']

video_height = configs['video-height']
cover_height = configs['cover-height']

id_generator = snowflake.SnowflakeGenerator(0)


def save_video_cover(source, output):
    capture = cv2.VideoCapture(source)

    if capture.isOpened():
        read_success, image = capture.read()

        if read_success:
            cv2.imwrite(output, cv2.resize(image, (cover_width, cover_height), interpolation=cv2.INTER_LINEAR))

    capture.release()


def save_detection_result(source, output, scores):
    reader = cv2.VideoCapture(source)
    writer = cv2.VideoWriter(output, cv2.VideoWriter.fourcc(*'h264'), video_speed, (video_width, video_height))

    for score in scores:
        read_success, frame = reader.read()

        if read_success:
            writer.write(engines.draw_detection_result(frame, score))

    reader.release()
    writer.release()

    remove_queue.put(source)


def get_realtime_data(session):
    realtime_frame = session.get_result()

    if realtime_frame is not None:
        encode_success, encoded_frame = cv2.imencode('.jpg', realtime_frame)

        if encode_success:
            return encoded_frame.tobytes()


def generate_realtime_response(session):
    while True:
        response_bytes = realtime.execute_task_in_seconds(get_realtime_data, session, frames_interval)

        if response_bytes is not None:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + response_bytes + b'\r\n'


def release_and_delete_session(session_id):
    if session_id in realtime_sessions:
        session = realtime_sessions[session_id]

        session.release()
        realtime_sessions.pop(session_id)


@scheduler.scheduled_job(trigger='interval', seconds=remove_interval)
def remove_task():
    remove_paths = []

    while not remove_queue.empty():
        remove_paths.append(remove_queue.get())

    for remove_path in remove_paths:
        try:
            with contextlib.suppress(FileNotFoundError):
                os.remove(remove_path)
        except OSError:
            remove_queue.put(remove_path)


@app.post('/api/videoinference')
def video_inference():
    video_id = str(next(id_generator))

    video_source = f'servers/videos/source.{video_id}.mp4'
    video_output = f'servers/videos/result.{video_id}.mp4'
    cover_output = f'servers/covers/result.{video_id}.jpg'

    try:
        flask.request.files['video'].save(video_source)
    except KeyError:
        return flask.abort(400)

    try:
        name = flask.request.form['name']
        note = flask.request.form['note']
    except KeyError:
        return flask.abort(400)

    save_video_cover(video_source, cover_output)

    try:
        scores = engines.detection_by_video(video_source).tolist()
    except ValueError:
        return flask.abort(400)

    save_detection_result(video_source, video_output, engines.expand_scores(scores))

    database.surveillance.videos.insert_one({
        'videoId': video_id,
        'name': name,
        'note': note,
        'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scores': scores,
    })

    return flask.jsonify({'videoId': video_id})


@app.post('/api/videoinference/list')
def get_video_list():
    request_params = flask.request.get_json()

    try:
        number = request_params['pageNumber']
        length = request_params['pageLength']
    except KeyError:
        return flask.abort(400)

    response_videos = []

    try:
        pagination_videos = database.surveillance.videos.find().limit(length).skip(length * (number - 1))
    except ValueError:
        return flask.abort(400)

    total_count = database.surveillance.videos.count_documents({})

    for video in pagination_videos:
        response_videos.append({
            'name': video['name'],
            'note': video['note'],
            'time': video['time'],
            'videoId': video['videoId'],
        })

    return flask.jsonify({'videos': response_videos, 'totalCount': total_count})


@app.get('/api/videoinference/detail/<string:video_id>')
def get_video_detail(video_id):
    video = database.surveillance.videos.find_one({'videoId': video_id})

    if video is None:
        return flask.abort(404)

    return flask.jsonify({
        'videoId': video['videoId'],
        'name': video['name'],
        'note': video['note'],
        'time': video['time'],
        'scores': video['scores'],
    })


@app.get('/api/videoinference/video/<string:video_id>')
def get_result_video(video_id):
    try:
        return flask.send_file(f'videos/result.{video_id}.mp4', mimetype='video/mp4')
    except FileNotFoundError:
        return flask.abort(404)


@app.get('/api/videoinference/cover/<string:video_id>')
def get_result_cover(video_id):
    try:
        return flask.send_file(f'covers/result.{video_id}.jpg', mimetype='image/jpg')
    except FileNotFoundError:
        return flask.abort(404)


@app.post('/api/videoinference/delete')
def delete_videos():
    request_params = flask.request.get_json()

    try:
        video_ids = request_params['videoIds']
    except KeyError:
        return flask.abort(400)

    delete_result = database.surveillance.videos.delete_many({'videoId': {'$in': video_ids}})

    for video_id in video_ids:
        remove_queue.put(f'servers/videos/result.{video_id}.mp4')
        remove_queue.put(f'servers/covers/result.{video_id}.jpg')

    return flask.jsonify({'deletedCount': delete_result.deleted_count})


@app.post('/api/realtimeinference/create')
def create_realtime_session():
    request_params = flask.request.get_json()

    try:
        source = request_params['source']
    except KeyError:
        return flask.abort(400)

    try:
        name = request_params['name']
        note = request_params['note']
    except KeyError:
        return flask.abort(400)

    session_id = str(next(id_generator))

    database.surveillance.sessions.insert_one({
        'source': source,
        'name': name,
        'note': note,
        'sessionId': session_id,
    })

    with realtime_sessions_lock:
        realtime_sessions[session_id] = realtime.RealtimeInferenceSession(source)

    return flask.jsonify({'sessionId': session_id})


@app.get('/api/realtimeinference/session/<string:session_id>')
def generate_realtime_frames(session_id):
    with realtime_sessions_lock:
        try:
            session = realtime_sessions[session_id]
        except KeyError:
            return flask.abort(404)

    return flask.Response(generate_realtime_response(session), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.post('/api/realtimeinference/list')
def get_realtime_sessions():
    request_params = flask.request.get_json()

    try:
        number = request_params['pageNumber']
        length = request_params['pageLength']
    except KeyError:
        return flask.abort(400)

    response_sessions = []

    try:
        pagination_sessions = database.surveillance.sessions.find().limit(length).skip(length * (number - 1))
    except ValueError:
        return flask.abort(400)

    total_count = database.surveillance.sessions.count_documents({})

    for session in pagination_sessions:
        response_sessions.append({
            'source': session['source'],
            'name': session['name'],
            'note': session['note'],
            'sessionId': session['sessionId'],
        })

    return flask.jsonify({'sessions': response_sessions, 'totalCount': total_count})


@app.get('/api/realtimeinference/detail/<string:session_id>')
def get_session_detail(session_id):
    session = database.surveillance.sessions.find_one({'sessionId': session_id})

    if session is None:
        return flask.abort(404)

    return flask.jsonify({
        'name': session['name'],
        'note': session['note'],
        'source': session['source'],
    })


@app.post('/api/realtimeinference/delete')
def delete_realtime_sessions():
    request_params = flask.request.get_json()

    try:
        session_ids = request_params['sessionIds']
    except KeyError:
        return flask.abort(400)

    delete_result = database.surveillance.sessions.delete_many({'sessionId': {'$in': session_ids}})

    for session_id in session_ids:
        with realtime_sessions_lock:
            release_and_delete_session(session_id)

    return flask.jsonify({'deletedCount': delete_result.deleted_count})


@app.get('/api/realtimeinference/sync')
def sync_realtime_sessions():
    sessions = database.surveillance.sessions.find()

    with realtime_sessions_lock:
        for session in realtime_sessions.values():
            session.release()

        realtime_sessions.clear()

        for session in sessions:
            realtime_sessions[session['sessionId']] = realtime.RealtimeInferenceSession(session['source'])

        return flask.jsonify({'sessionCount': len(realtime_sessions)})
