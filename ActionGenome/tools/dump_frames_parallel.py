import argparse
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def build_video_to_frames(annotation_dir: str):
    frame_list_path = os.path.join(annotation_dir, "frame_list.txt")
    video_to_frames = defaultdict(list)
    with open(frame_list_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            video, frame = line.split("/", 1)
            video_to_frames[video].append(frame)
    return video_to_frames


def dump_one_video(
    video_name: str,
    keep_frames,
    video_dir: str,
    frame_dir: str,
    all_frames: bool,
    ffmpeg_bin: str,
):
    out_dir = os.path.join(frame_dir, video_name)
    if os.path.exists(out_dir):
        return "skipped", video_name

    os.makedirs(out_dir, exist_ok=False)
    video_path = os.path.join(video_dir, video_name)
    cmd = [
        ffmpeg_bin,
        "-loglevel",
        "panic",
        "-i",
        video_path,
        os.path.join(out_dir, "%06d.png"),
    ]

    try:
        subprocess.run(cmd, check=True)
        if not all_frames:
            keep = set(keep_frames)
            for frame_name in os.listdir(out_dir):
                if frame_name not in keep:
                    os.remove(os.path.join(out_dir, frame_name))
        return "done", video_name
    except Exception as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        return "error", f"{video_name}: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Dump ActionGenome frames in parallel")
    parser.add_argument("--video_dir", required=True, help="Folder containing Charades videos")
    parser.add_argument("--frame_dir", required=True, help="Output folder for frames")
    parser.add_argument(
        "--annotation_dir",
        required=True,
        help="Folder containing annotation files including frame_list.txt",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel ffmpeg workers",
    )
    parser.add_argument(
        "--ffmpeg_bin",
        default="ffmpeg",
        help="Path to ffmpeg binary (default: ffmpeg from PATH)",
    )
    parser.add_argument(
        "--all_frames",
        action="store_true",
        help="Keep all extracted frames instead of only annotated ones",
    )
    args = parser.parse_args()

    os.makedirs(args.frame_dir, exist_ok=True)
    video_to_frames = build_video_to_frames(args.annotation_dir)
    videos = sorted(video_to_frames.keys())
    pending = [v for v in videos if not os.path.exists(os.path.join(args.frame_dir, v))]

    print(f"Total videos in frame_list: {len(videos)}")
    print(f"Already complete/started directories: {len(videos) - len(pending)}")
    print(f"Pending videos to dump now: {len(pending)}")
    print(f"Workers: {args.num_workers}")
    print(f"ffmpeg binary: {args.ffmpeg_bin}")

    if not pending:
        print("No pending videos left.")
        return

    done_count = 0
    skipped_count = 0
    errors = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [
            pool.submit(
                dump_one_video,
                video_name=v,
                keep_frames=video_to_frames[v],
                video_dir=args.video_dir,
                frame_dir=args.frame_dir,
                all_frames=args.all_frames,
                ffmpeg_bin=args.ffmpeg_bin,
            )
            for v in pending
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            status, payload = fut.result()
            if status == "done":
                done_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                errors.append(payload)

    print(f"Done: {done_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {len(errors)}")
    if errors:
        print("Sample errors:")
        for err in errors[:20]:
            print(err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
