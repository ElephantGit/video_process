import ffmpeg


def get_video_info(source_video_path):
    probe = ffmpeg.probe(source_video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found!')
        return
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    fps = int(video_stream['r_frame_rate'].split('/')[0])/int(video_stream['r_frame_rate'].split('/')[1])
    video_info = {
        'width' : width,
        'height' : height,
        'fps' : fps ,
    }
    return video_info