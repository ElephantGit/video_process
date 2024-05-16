import numpy as np
import os
from sort import iou_batch


def parse_det_result(det_result_file):
    with open(det_result_file, 'r') as f:
        lines = f.readlines()
    frame_height, frame_width = [int(x) for x in lines.pop(0).split(',')]
    seq_dets = []
    for line in lines:
      seq_dets.append(np.array(eval(line)))
    seq_dets = np.array(seq_dets)
    return seq_dets, frame_height, frame_width

def parse_track_result(track_file):
    with open(track_file, 'r') as f:
        lines = f.readlines()
    res = []
    for line in lines[:-1]:
        res.append(list(eval(line)))
    return np.array(res)

def merge_det_track(det_file, track_file, cx1, cy1, cx2, cy2):
    dets, _, _ = parse_det_result(det_file)
    # track_res = parse_track_result(track_file)
    track_res = np.loadtxt(track_file, delimiter=',')

    final_res = []

    # import pdb;pdb.set_trace()
    for i, item in enumerate(track_res):
        frame_idx, track_id, x1, y1, x2, y2 = item
        # tracking info
        frame_idx = int(frame_idx)
        track_bbox = item[2:6]

        # detection info
        dets_in_current_frame = dets[dets[:, 0] == frame_idx]
        det_bbox = dets_in_current_frame[:, 2:6]
        det_lmks = dets_in_current_frame[:, 7:]

        # calculate iou of det bbox and track bbox for each track bbox
        iou = iou_batch(det_bbox, track_bbox)
        det_idx = np.argmax(iou)
        lmk = det_lmks[det_idx]
        
        cur_res = "%08d %08d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (frame_idx, i, x1, y1, x2-x1, y2-y1,\
            lmk[0], lmk[1], lmk[2], lmk[3], lmk[4], lmk[5], lmk[6], lmk[7], lmk[8], lmk[9])
        final_res.append(cur_res)
    final_res.append('CROP_BBOX: ' + ' '.join([str(x) for x in [cx1, cy1, cx2, cy2]]))
    return final_res
    # with open(track_file + '.txt', 'w') as f:
    #     f.writelines(final_res)

def crop(track_file, frame_height, frame_width):
    track_res = parse_track_result(track_file)
    track_ids = [int(x) for x in set(list(track_res[:, 1]))]
    out_files = []
    for track_id in track_ids:
        cur_track_id_res = track_res[track_res[:, 1] == track_id]
        if cur_track_id_res.shape[0] < 30:
            continue
        
        out_file_name = track_file.replace('.txt.txt', '_' + str(track_id) + '.txt')
        
        np.savetxt(out_file_name, cur_track_id_res, fmt='%.2f', delimiter=',')
        # import pdb;pdb.set_trace()
        x1 = np.min(cur_track_id_res[:, 2])
        y1 = np.min(cur_track_id_res[:, 3])
        x2 = np.max(cur_track_id_res[:, 4])
        y2 = np.max(cur_track_id_res[:, 5])

        x0 = max(int(x1 - (x2 - x1) / 2), 0)
        y0 = max(int(y1 - (y2 - y1) / 2), 0)
        x3 = min(int(x2 + (x2 - x1) / 2), frame_height)
        y3 = min(int(y2 + (y2 - y1) / 2), frame_width)
        out_h = x3 - x0
        out_w = y3 - y0
        print(out_h, out_w, frame_width, frame_height)
        out_side = min(max(out_h, out_w), min(frame_width, frame_height))

        out_files.append((out_file_name, x0, y0, x0 + out_side, y0 + out_side))

        save_path = os.path.join(f'{track_file}_crop.mp4')
        video_path = track_file.replace('_det.txt.txt', '')
        command = f'ffmpeg -accurate_seek -y -i {video_path} -acodec copy -strict -2 -vf "setpts=PTS-STARTPTS,crop={out_side}:{out_side}:{x0}:{y0}" -vsync 0 {save_path}'
        print(command)
        # command = f'ffmpeg -accurate_seek -y -i {video_path} -acodec copy -strict -2 -vf "setpts=PTS-STARTPTS,crop={out_h}:{out_w}:{x0}:{y0}" -vsync 0 {save_path}'
        # print(command)
        # return command

    return out_files


if __name__ == '__main__':
    track_file="output/_-MpWAXYSSk_crop.webm_det_1.txt"
    det_file='det_results/_-MpWAXYSSk_crop.webm_det.txt'
    merge_det_track(det_file, track_file)

