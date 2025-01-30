from flask import Flask, render_template, Response, request, jsonify
import os
from video_processor import VideoProcessor
from model_manager import ModelManager

app = Flask(__name__)
model_manager = ModelManager()
video_processor = VideoProcessor(model_manager)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_upload', methods=['POST'])
def video_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        video_path = os.path.join('uploaded_videos', file.filename)
        os.makedirs('uploaded_videos', exist_ok=True)
        file.save(video_path)
        return jsonify({'success': True, 'video_path': video_path})

@app.route('/video_feed/<path:video_path>')
def video_feed(video_path):
    return Response(
        video_processor.process_video(video_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

