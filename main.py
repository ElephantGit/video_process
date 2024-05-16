import os
import copy
from torch import windows
from detect_video import detect
from sort import sort
import argparse
from crop import crop, merge_det_track
from utils.video import get_video_info

from aoss_client.client import Client as CephClient


def parse_args():
    """Parse input arguments."""
    # args for sort
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)

    # args for detect
    parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.7, type=float, help='visualization_threshold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    client = CephClient('/mnt/afs/qinshuo/aoss.conf')
    video_urls = ['pub:s3://videodata/video/20240430/3b32I2t9Jbs.webm']
    for url in video_urls:
        video_filename = url.split('/')[-1]
        download_path = '/'.join(url.split('/')[2:-1])
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        video_path = os.path.join(download_path, video_filename)
        client.download_file(url, video_path)
        det_result_file = detect(url, args)
        # video_name = url.split('/')[-1]
        # det_result_file = os.path.join('det_results', f'{video_name}_det.txt')
        # print(det_result_file)

        track_file, frame_height, frame_width = sort(det_result_file, args)

        video_info = get_video_info(url)
        print(video_info)
        fps = video_info['fps']
        height = video_info['height']
        width = video_info['width']
        video_name = url.split('/')[-1].split('.')[0]
        head = [f'Video ID:\t{video_name}\n']
        head.append(f'H:\t{height}\n')
        head.append(f'W:\t{width}\n')
        head.append(f'FPS:\t{fps}\n')
        head.append(f'\nFRAME INDEX X0 Y0 W H [Landmarks (5 Points)]\n\n')

        tmp_files = crop(track_file, frame_height, frame_width)
        for i, (tmp_file, x1, y1, x2, y2) in enumerate(tmp_files):
            cur_res = []
            cur_res.extend(head)
            res = merge_det_track(det_result_file, tmp_file, x1, y1, x2, y2)

            s_frame = int(res[0].split(' ')[0])
            e_frame = int(res[-2].split(' ')[0])

            cur_res.extend(res)
            with open(f'results/Clip+{video_name}+P{i}+F{s_frame}-{e_frame}.txt', 'w') as f:
                f.writelines(cur_res)
            
