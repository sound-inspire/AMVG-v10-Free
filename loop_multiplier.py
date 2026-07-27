# -*- coding: utf-8 -*-
"""
A.M.V.G v2 - loop_multiplier.py
破壊的増殖ループモジュール
アセット素材を音声の長さに達するまで自動でループ結合し、ベース長尺映像トラックを生成する
"""

import os
from moviepy import AudioFileClip, VideoFileClip, ImageClip, concatenate_videoclips, VideoClip

def get_audio_duration(audio_path: str) -> float:
    """音声ファイルの長さを取得する"""
    try:
        with AudioFileClip(audio_path) as audio:
            return float(audio.duration)
    except Exception as e:
        print(f"[Warning] Failed to get audio duration using MoviePy: {e}")
        # librosa フォールバック
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return float(librosa.get_duration(y=y, sr=sr))
        except Exception as e2:
            print(f"[Error] Failed to get duration with librosa: {e2}")
            return 30.0 # 最低限のデフォルト値

def create_base_clip(
    audio_path: str,
    assets_input: str,
    resolution: tuple = (1280, 720),
    slide_seconds: float = 3.0
):
    """
    音声の長さに合わせて背景映像のベースクリップを生成する
    - assets_input がフォルダの場合：中の動画アセットをループ結合
    - assets_input が単一画像ファイルの場合：画像を引き延ばした ImageClip を生成
    """
    target_duration = get_audio_duration(audio_path)
    print(f"[LoopMultiplier] Target duration: {target_duration:.2f} seconds")
    
    # 1. assets_input が単一画像ファイルの場合
    if os.path.isfile(assets_input) and assets_input.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        print(f"[LoopMultiplier] Image detected. Creating VideoClip from: {assets_input}")
        from PIL import Image
        import numpy as np
        img = Image.open(assets_input).convert('RGB')
        img_resized = img.resize(resolution, Image.Resampling.LANCZOS)
        img_np = np.array(img_resized)
        base_clip = ImageClip(img_np, duration=target_duration)
        return base_clip
        
    # 2. assets_input がディレクトリの場合
    elif os.path.isdir(assets_input):
        print(f"[LoopMultiplier] Directory detected. Gathering clips from: {assets_input}")
        
        # フォルダ内の動画・画像ファイルを収集
        files = []
        for f in sorted(os.listdir(assets_input)):
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.png', '.jpg', '.jpeg', '.webp')):
                files.append(os.path.join(assets_input, f))
                
        if not files:
            # 代替の黒背景画像を作成して引き延ばす
            import numpy as np
            black_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8) + 10 # 暗い紺/黒
            base_clip = VideoClip(lambda t: black_frame, duration=target_duration)
            return base_clip

        # 映像クリップ（VideoFileClip または ImageClip）のロード
        clips = []
        for f in files:
            try:
                if f.lower().endswith(('.mp4', '.mov', '.avi')):
                    clip = VideoFileClip(f)
                    # 出力解像度にリサイズ
                    if clip.size != resolution:
                        clip = clip.resized(resolution)
                    clips.append(clip)
                else:
                    # 静止画アセットの場合は指定されたスライド秒数を使用
                    from PIL import Image
                    import numpy as np
                    img = Image.open(f).convert('RGB')
                    img_resized = img.resize(resolution, Image.Resampling.LANCZOS)
                    img_np = np.array(img_resized)
                    clip = ImageClip(img_np, duration=slide_seconds)
                    clips.append(clip)
            except Exception as e:
                print(f"[Warning] Failed to load clip {f}: {e}")

        if not clips:
            raise RuntimeError("No valid clips could be loaded from the assets directory.")

        # クリップの合計時間を計算
        total_clips_duration = sum(float(c.duration) for c in clips)
        print(f"[LoopMultiplier] Found {len(clips)} asset clips. Total raw duration: {total_clips_duration:.2f}s")

        # 楽曲の長さに達するまで結合を繰り返す (破壊的増殖ループ)
        assembled_clips = []
        current_dur = 0.0
        
        # 最低1回はすべてのクリップをアサインし、足りない場合はループで複製
        while current_dur < target_duration:
            for clip in clips:
                # 参照コピーをリストに追加して連結
                assembled_clips.append(clip)
                current_dur += float(clip.duration)
                if current_dur >= target_duration:
                    break
                    
        print(f"[LoopMultiplier] Concatenating assembled clips...")
        # concatenate_videoclips で結合し、曲の長さにトリミング
        final_base = concatenate_videoclips(assembled_clips, method="compose")
        final_base = final_base.subclipped(0, target_duration)
        
        return final_base
    else:
        # assets_input が単一の動画ファイルの場合
        if os.path.isfile(assets_input) and assets_input.lower().endswith(('.mp4', '.mov', '.avi')):
            print(f"[LoopMultiplier] Single video file detected: {assets_input}")
            clip = VideoFileClip(assets_input)
            if clip.size != resolution:
                clip = clip.resized(resolution)
                
            # ループ回数を計算
            n_loops = int(target_duration // clip.duration) + 1
            assembled_clips = [clip] * n_loops
            final_base = concatenate_videoclips(assembled_clips, method="compose")
            final_base = final_base.subclipped(0, target_duration)
            return final_base
        else:
            raise ValueError(f"Invalid assets input: {assets_input}. Must be a file or folder.")
