import ffmpeg
probe = ffmpeg.probe("file.mp4")
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)