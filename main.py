import cv2
import json
import os
import argparse


def frame_to_matrix(frame, width, height, threshold):
    # 缩放
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

    # 灰度
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 二值化：亮的记 1，暗的记 0
    _, binary = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY)

    # 转成 Python 列表
    matrix = binary.tolist()
    return matrix


def matrix_to_string(matrix):
    # 18x10 -> 一行 180 个 0/1 字符
    return ''.join(str(v) for row in matrix for v in row)


def sample_video_to_matrices(video_path, out_dir, width, height, threshold, fps_limit=None, max_frames=None):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    # 控制抽帧频率
    if fps_limit is None or fps_limit <= 0 or fps_limit >= src_fps:
        frame_step = 1
        out_fps = src_fps
    else:
        frame_step = max(1, round(src_fps / fps_limit))
        out_fps = src_fps / frame_step

    frame_strings = []
    frame_index = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step == 0:
            matrix = frame_to_matrix(frame, width, height, threshold)
            frame_str = matrix_to_string(matrix)
            frame_strings.append(frame_str)
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_index += 1

    cap.release()

    # 导出 JSON
    json_data = {
        "width": width,
        "height": height,
        "threshold": threshold,
        "fps": out_fps,
        "frame_count": len(frame_strings),
        "frames": frame_strings
    }

    json_path = os.path.join(out_dir, "frames.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False)

    # 导出纯文本：每行一帧，只有 0/1
    txt_path = os.path.join(out_dir, "frames.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for frame_str in frame_strings:
            f.write(frame_str + "\n")

    print("Done.")
    print(f"Saved {len(frame_strings)} frames")
    print(f"Output FPS: {out_fps:.2f}")
    print(f"JSON: {json_path}")
    print(f"TXT : {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert video to low-res binary frame strings for TurboWarp/Scratch")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="matrix_output", help="Output folder")
    parser.add_argument("--width", type=int, default=10, help="Output frame width")
    parser.add_argument("--height", type=int, default=18, help="Output frame height")
    parser.add_argument("--threshold", type=int, default=127, help="Binary threshold: 0-255")
    parser.add_argument("--fps", type=float, default=8, help="Target max output FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on saved frames")
    args = parser.parse_args()

    sample_video_to_matrices(
        video_path=args.video,
        out_dir=args.out,
        width=args.width,
        height=args.height,
        threshold=args.threshold,
        fps_limit=args.fps,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()