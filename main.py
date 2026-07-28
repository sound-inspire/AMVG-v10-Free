# -*- coding: utf-8 -*-
"""
A.M.V.G V10 Free - Antigravity MV Generator
繝｡繧､繝ｳ邨ｱ蜷医お繝ｳ繝医Μ繝ｼ繝昴う繝ｳ繝 (CLI  FastAPI Web API / Web UI 繧ｵ繝ｼ繝舌)
笘 V10 Free 譁ｰ讖溯: 讌ｽ譖ｲ繧ｨ繝阪Ν繧ｮ繝ｼ蛹ｵ｡邱 (RMS + onset_strength) 縺ｫ繧医ｋ莠碁㍾螟芽ｪｿ繧ｨ繝輔ぉ繧ｯ繝医お繝ｳ繧ｸ繝ｳ謳ｭ霈
"""

import os
import sys
import uuid
import shutil
import time
import argparse
import tempfile
import threading
import uvicorn
import json
import math
import cv2
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont

from proglog import ProgressBarLogger

# 閾ｪ陬ｽ繝｢繧ｸ繝･繝ｼ繝ｫ縺ｮ繧､繝ｳ繝昴繝
import loop_multiplier
import biometric_overlay
import pixel_melter
import metadata_generator

import traceback

# 最後に発生したフィルタープレビュー時のエラーログを保持するグローバル変数
last_filter_error_log = ""



app = FastAPI(title="AMVG V10 Free API")
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)
print("==================== DEBUG: main.py V10 Free (Kinetic Animation & Multi-Exec Engine) loaded! ====================")

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_json():
    return {}

@app.post("/shutdown")
async def shutdown_server():
    import os
    import threading
    import time
    def kill_server():
        time.sleep(1) # クライアントに成功レスポンスを返す猶予を与える
        os._exit(0)   # プロセスを完全終了
    threading.Thread(target=kill_server, daemon=True).start()
    return {"status": "success", "message": "SYSTEM SHUTDOWN SEQUENCE INITIATED"}

class CustomMoviePyLogger(ProgressBarLogger):
    def __init__(self, update_progress_fn=None):
        super().__init__()
        self.update_progress_fn = update_progress_fn
        self.last_percent = -1

    def callback(self, **changes):
        index = changes.get("index")
        total = changes.get("total")
        
        # もし changes に無ければ self.bars をチェック
        if index is None or total is None:
            for bar, data in self.bars.items():
                if isinstance(data, dict):
                    index = data.get('index', index)
                    total = data.get('total', total)

        if index is not None and total is not None and total > 0:
            pct = int((index / total) * 100)
            if pct != self.last_percent:
                self.last_percent = pct
                overall_val = 70 + int(pct * 0.29)
                msg = f"Phase 3: Exporting final H.264 video ({pct}% - frame {index}/{total})"
                print(f"[{overall_val}%] {msg}", flush=True)
                if self.update_progress_fn:
                    self.update_progress_fn(overall_val, msg)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

progress_store = {}
TEMP_DIR = tempfile.gettempdir()

def get_resource_path(relative_path):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

def min_sec_to_seconds(val) -> float:
    """v5完全互換の安全な時間文字列/数値パース"""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        val_str = str(val).strip().replace(',', '.')
        parts = val_str.split(':')
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(val_str)
    except Exception:
        return 0.0

def draw_japanese_lyric(
    frame_np: np.ndarray, 
    text: str, 
    color_scheme: str = "cyan_magenta",
    style: str = "SIMPLE",
    t: float = 0.0,
    bpm: float = 120.0,
    intensity: float = 1.0,
    start_time: float = 0.0,
    font_name: str = "gothic",
    return_meta: bool = False,
    bpm_offset: float = 0.0
):
    """PILとOpenCVを組み合わせて、キネティックおよびグリッチ効果をかけた歌詞を描画する (v5完全準拠ロジック)"""
    import random
    from PIL import Image, ImageDraw, ImageFont
    h, w, c = frame_np.shape
    
    # 1. 透明なアルファチャンネルを持つ画像を作成
    text_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_img)
    
    # 2. キネティックサイズ計算 (BPM・オフセット完全同期 Beat Pumping - 通常版高精度ロジック)
    beat_duration = 60.0 / max(1.0, bpm)
    time_since_beat = (t - bpm_offset) % beat_duration
    if time_since_beat < 0:
        time_since_beat += beat_duration
    decay_constant = 6.0 / beat_duration
    beat_signal = math.exp(-decay_constant * time_since_beat)
    
    # 基本フォントサイズ (画面の高さにスケール)
    base_size = int(35 * (h / 720.0))
    scale = 1.0 + 0.25 * beat_signal * max(0.2, intensity) if style == "KINETIC_BOUNCE" else 1.0
    font_size = int(base_size * scale)
    
    font = None
    if font_name == "mincho":
        font_candidates = [
            "msmincho.ttc", "YuMincho.ttc", "yumin.ttf", "BIZ-UDMinchoM.ttc", 
            "Hiragino Mincho ProN.ttc", "NotoSerifCJK-Regular.ttc", "NotoSerifJP-Regular.otf"
        ]
    else:
        font_candidates = [
            "msgothic.ttc", "meiryo.ttc", "YuGothM.ttc", 
            "yugoth.ttf", "NotoSansCJK-Regular.ttc", "NotoSansJP-Regular.otf", 
            "Hiragino Sans GB.ttc", "AppleGothic.ttf"
        ]
        
    for fpath in font_candidates:
        try:
            # OS標準のフォントパスをPILに自動探索させる
            font = ImageFont.truetype(fpath, font_size)
            print(f"[Subtitle Font Engine] Loaded font: '{fpath}' ({font.getname()}) for mode: '{font_name}'")
            break
        except (IOError, OSError):
            try:
                # Windows用の絶対パスフォールバック
                full_font_path = os.path.join("C:\\Windows\\Fonts", fpath)
                font = ImageFont.truetype(full_font_path, font_size)
                print(f"[Subtitle Font Engine] Loaded font from Windows path: '{full_font_path}' ({font.getname()}) for mode: '{font_name}'")
                break
            except Exception:
                continue
                
    if font is None:
        font = ImageFont.load_default()
        print(f"[Subtitle Font Engine] Warning: Fallback to default font for mode: '{font_name}'")
        
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)
        
    # 中央下部配置
    tx = (w - text_w) // 2
    ty = int(h * 0.84) - (text_h // 2)
    
    # 配色選択 (v5完全再現)
    main_color = (0, 243, 255, 255)  # シアン
    glow_color = (255, 0, 127, 255)  # マゼンタ
    
    if color_scheme == "green_cyan":
        main_color = (57, 255, 20, 255)
        glow_color = (0, 243, 255, 255)
    elif color_scheme == "magenta_green":
        main_color = (255, 0, 127, 255)
        glow_color = (57, 255, 20, 255)
    elif color_scheme == "white_cyan":
        main_color = (255, 255, 255, 255)
        glow_color = (0, 243, 255, 255)
    elif color_scheme == "pure_white":
        main_color = (255, 255, 255, 255)
        glow_color = (0, 243, 255, 255)

    # ネオン光彩の描画 (NEON_GLOW)
    if style == "NEON_GLOW":
        glow_size = int(font_size * 1.08)
        glow_font = None
        for fpath in font_candidates:
            try:
                glow_font = ImageFont.truetype(fpath, glow_size)
                break
            except (IOError, OSError):
                try:
                    glow_font = ImageFont.truetype(os.path.join("C:\\Windows\\Fonts", fpath), glow_size)
                    break
                except Exception:
                    continue
        if glow_font is None:
            glow_font = font
            
        try:
            g_left, g_top, g_right, g_bottom = draw.textbbox((0, 0), text, font=glow_font)
            gtx = (w - (g_right - g_left)) // 2
            gty = int(h * 0.84) - ((g_bottom - g_top) // 2)
        except AttributeError:
            g_w, g_h = draw.textsize(text, font=glow_font)
            gtx = (w - g_w) // 2
            gty = int(h * 0.84) - (g_h // 2)
            
        for offset in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 0)]:
            draw.text((gtx + offset[0], gty + offset[1]), text, font=glow_font, fill=(glow_color[0], glow_color[1], glow_color[2], 100))

    # 通常のアウトライン（黒縁取り）の描画
    outline_color = (0, 0, 0, 255)
    for offset in [(-2, -2), (2, -2), (-2, 2), (2, 2), (-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((tx + offset[0], ty + offset[1]), text, font=font, fill=outline_color)
        
    # メインの文字を描画
    draw.text((tx, ty), text, font=font, fill=main_color)
    
    # NumPyのRGBA配列に変換
    text_np = np.array(text_img)
    text_pixel_count = int(np.count_nonzero(text_np[:, :, 3] > 10))
    
    # 3. グリッチ/色収差 (GLITCH or CHROME_RGB スタイルの場合)
    if style in ["GLITCH", "CHROME_RGB"] and intensity > 0.05:
        glitch_trigger = math.sin(t * 12.0) * beat_signal
        
        # チャンネルスプリットによる色収差
        if glitch_trigger > 0.25 or (random.random() < 0.15 * intensity):
            shift = int(20.0 * intensity * (glitch_trigger if glitch_trigger > 0 else 0.4))
            shift = max(1, shift)
            
            r_chan = text_np[:, :, 0]
            g_chan = text_np[:, :, 1]
            b_chan = text_np[:, :, 2]
            a_chan = text_np[:, :, 3]
            
            new_r = np.zeros_like(r_chan)
            new_b = np.zeros_like(b_chan)
            new_a = a_chan.copy()
            
            new_r[:, shift:] = r_chan[:, :-shift]
            new_b[:, :-shift] = b_chan[:, shift:]
            
            new_a[:, shift:] = np.maximum(new_a[:, shift:], a_chan[:, :-shift])
            new_a[:, :-shift] = np.maximum(new_a[:, :-shift], a_chan[:, shift:])
            
            text_np[:, :, 0] = new_r
            text_np[:, :, 2] = new_b
            text_np[:, :, 3] = new_a
            
        # 横ラインスライス歪み (GLITCH スタイルのみ)
        if style == "GLITCH" and (glitch_trigger > 0.45 or random.random() < 0.1 * intensity):
            num_slices = random.randint(3, 7)
            for _ in range(num_slices):
                slice_y = random.randint(max(0, ty - 20), min(h - 10, ty + text_h + 20))
                slice_h = random.randint(3, 12)
                slice_shift = random.randint(-30, 30)
                
                slice_y = max(0, min(h - slice_h, slice_y))
                text_np[slice_y:slice_y+slice_h] = np.roll(text_np[slice_y:slice_y+slice_h], slice_shift, axis=1)

    # 4. 元画像とのブレンド (v5直接ブレンド)
    alpha = text_np[:, :, 3:4] / 255.0
    text_rgb = text_np[:, :, 0:3]
    blended = (1.0 - alpha) * frame_np + alpha * text_rgb
    final_np = blended.astype(np.uint8)
    
    if return_meta:
        return final_np, text_pixel_count
    return final_np


# ==========================================
# 邨ｱ蜷医ヱ繧､繝励Λ繧､繝ｳ螳溯｡後さ繧｢
# ==========================================

def run_pipeline(
    session_id: str,
    audio_path: str,
    assets_input: str,
    output_path: str,
    aspect_ratio: str = "16:9",
    fps: int = 30,
    api_key: Optional[str] = None,
    lyrics_path: Optional[str] = None,
    metadata_json_path: Optional[str] = None,
    is_async: bool = False,
    enable_ecg: bool = True,
    enable_grf: bool = True,
    enable_hexdump: bool = True,
    max_melt: float = 1.0,
    glitch_freq: float = 1.0,
    subtitle_color: str = "cyan_magenta",
    primary_model: Optional[str] = None,
    bpm: float = 120.0,
    bpm_offset: float = 0.0,
    slide_measures: int = 4,
    ai_filter_code: Optional[str] = None,
    ai_filter_codes_json: Optional[str] = None,
    filter_exec_mode: str = "multi",
    enable_ai_orchestration: bool = False,
    enable_kinetic_lyric: bool = False,
    lyric_effect_style: str = "SIMPLE",
    subtitle_font: str = "gothic"
):
    try:
        def update_progress(val: int, msg: str):
            if is_async and session_id in progress_store:
                progress_store[session_id]["progress"] = val
                progress_store[session_id]["message"] = msg
            else:
                print(f"[{val}%] {msg}")

        # 1. タイムライン構築
        update_progress(5, "Phase 1: Generating base track using loop multiplier...")
        
        resolution = (720, 1280) if aspect_ratio == "9:16" else (1280, 720)
        
        beats_per_measure = 4
        slide_seconds = (slide_measures * beats_per_measure * 60.0) / max(1.0, bpm)
        print(f"[V5] Slide interval: {slide_measures} measures = {slide_seconds:.2f}s (BPM={bpm})")
        
        base_clip = loop_multiplier.create_base_clip(
            audio_path=audio_path,
            assets_input=assets_input,
            resolution=resolution,
            slide_seconds=slide_seconds
        )
        total_duration = base_clip.duration
        
        # 2. 字幕データのパース（タイムライン連携対応）
        sections = []
        if metadata_json_path and os.path.exists(metadata_json_path):
            update_progress(15, "Loading metadata from existing JSON...")
            with open(metadata_json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                if isinstance(metadata, list):
                    sections = metadata
                elif isinstance(metadata, dict):
                    sections = metadata.get("sections", [])
        elif lyrics_path and os.path.exists(lyrics_path):
            update_progress(20, "Extracting subtitles from lyrics file (absolute sync)...")
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics_content = f.read()
            
            ext = os.path.splitext(lyrics_path)[1].lower()
            sections = []
            
            # SRTファイルまたはSRT形式のテキストの場合
            if ext == '.srt' or "-->" in lyrics_content:
                parsed, _ = metadata_generator.parse_srt(lyrics_content)
                for p in parsed:
                    if p.get("is_marker") or metadata_generator.is_section_marker(p["text"]) or not p["text"].strip():
                        continue
                    sections.append({
                        "start": p["start"],
                        "end": p["end"],
                        "lyric": p["text"]
                    })
            else:
                # TXTファイル（プレーンテキスト）の場合: 曲の長さに応じて均等自動配分
                parsed, _ = metadata_generator.parse_txt_to_lines(lyrics_content, total_duration)
                for p in parsed:
                    if p.get("is_marker") or metadata_generator.is_section_marker(p["text"]) or not p["text"].strip():
                        continue
                    sections.append({
                        "start": p["start"],
                        "end": p["end"],
                        "lyric": p["text"]
                    })
                    
            print(f"[Subtitle Sync] Loaded {len(sections)} raw subtitle sections from {os.path.basename(lyrics_path)}")
        else:
            update_progress(35, "No lyrics provided. Running in offline rendering mode...")
            
        # --- 字幕セクションの完全自動正規化 & キャッシュ ---
        normalized_sections = []
        for idx, sec in enumerate(sections):
            st = sec.get("start_sec") if sec.get("start_sec") is not None else (sec.get("start") or sec.get("time_start"))
            et = sec.get("end_sec") if sec.get("end_sec") is not None else (sec.get("end") or sec.get("time_end"))
            txt = sec.get("lyric") or sec.get("text") or sec.get("lyric_text") or ""
            txt = str(txt).strip()
            
            if not txt:
                continue
                
            start_sec = min_sec_to_seconds(st)
            end_sec = min_sec_to_seconds(et)
            
            if end_sec <= start_sec or (end_sec - start_sec) < 0.1:
                end_sec = start_sec + 0.5  # 最低描画持続時間保証
                
            normalized_sections.append({
                "id": idx + 1,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "lyric": txt
            })
        sections = normalized_sections
        print(f"[Subtitle Sync Verification] Normalized {len(sections)} valid subtitle entries for video rendering.")
        if len(sections) > 0:
            update_progress(35, f"Subtitle sync ready: {len(sections)} entries normalized & bound.")
        elif lyrics_path or metadata_json_path:
            update_progress(35, "Warning: No valid subtitle entries after normalization.")

        update_progress(40, f"Base clip generated ({total_duration:.2f}s). Applying effects...")

        # 3. AIによるエフェクト自動連携（オーケストレーション）のタイムライン解析
        orchestration_plan = None
        if enable_ai_orchestration:
            update_progress(42, "V5 Phase 1.5: Analyzing audio progress via Gemini for orchestration timeline...")
            try:
                available_filters = ["none"]
                if os.path.exists("ai_filters.json"):
                    with open("ai_filters.json", "r", encoding="utf-8") as f:
                        filters_data = json.load(f)
                        available_filters.extend(list(filters_data.keys()))
                
                lyrics_text = ""
                if lyrics_path and os.path.exists(lyrics_path):
                    with open(lyrics_path, "r", encoding="utf-8") as f:
                        lyrics_text = f.read()
                
                orchestration_plan = metadata_generator.generate_orchestration_timeline(
                    audio_path=audio_path,
                    lyrics_content=lyrics_text,
                    api_key=api_key,
                    bpm=bpm,
                    available_filters=available_filters,
                    primary_model=primary_model
                )
                print(f"[V5 Orchestration] Orchestrated timeline generated successfully with {len(orchestration_plan.get('sections', []))} sections.")
            except Exception as e_orch:
                print(f"[V5 Orchestration Warning] Gemini timeline orchestration failed, using fallback timeline: {e_orch}")
                orchestration_plan = metadata_generator.generate_fallback_timeline(total_duration, bpm, available_filters)

        # 3b. Librosa 音声エネルギー解析 (Fallback)
        # 3b. 音声エネルギー解析 (Librosaを使わない安全な実装)
        update_progress(44, "V3 Phase 2a: Analyzing audio energy envelope...")
        
        rms_norm = np.array([0.5])
        hop_seconds_energy = 1.0 / 30

        try:
            import wave
            import subprocess
            
            temp_wav_path = audio_path + "_temp_mono.wav"
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ac", "1", "-ar", "22050",
                "-acodec", "pcm_s16le",
                temp_wav_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            if os.path.exists(temp_wav_path):
                with wave.open(temp_wav_path, "rb") as w_file:
                    n_frames = w_file.getnframes()
                    sample_rate = w_file.getframerate()
                    frames = w_file.readframes(n_frames)
                    y = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                
                try:
                    os.remove(temp_wav_path)
                except Exception as e_del:
                    print(f"[Warning] Failed to delete temp wav: {e_del}")
                
                hop_length = int(sample_rate * hop_seconds_energy)
                frame_length = int(sample_rate * 0.1)
                
                rms_list = []
                for start_sample in range(0, len(y), hop_length):
                    end_sample = min(len(y), start_sample + frame_length)
                    chunk = y[start_sample:end_sample]
                    if len(chunk) == 0:
                        break
                    rms = np.sqrt(np.mean(chunk**2) + 1e-8)
                    rms_list.append(rms)
                    
                if len(rms_list) > 0:
                    _rms = np.array(rms_list)
                    rms_min, rms_max = _rms.min(), _rms.max()
                    if rms_max - rms_min > 1e-5:
                        rms_norm = (_rms - rms_min) / (rms_max - rms_min)
                    else:
                        rms_norm = np.zeros_like(_rms) + 0.5
            else:
                raise FileNotFoundError("Temp WAV file not generated by FFmpeg.")
                
            print(f"[V5 Engine] Safe Wav energy analysis completed. Frames: {len(rms_norm)}")
        except Exception as e_wav:
            print(f"[Warning] Safe Wav analysis failed, falling back to constant energy: {e_wav}")
            rms_norm = np.array([0.5])
            hop_seconds_energy = 1.0 / 30

        compiled_ai_filter = None
        if ai_filter_code and ai_filter_code.strip():
            try:
                compiled_ai_filter = compile(ai_filter_code, "<ai_filter>", "exec")
                print("[V5] Custom AI Filter code compiled successfully.")
            except Exception as e_compile:
                print(f"[V5 Error] Failed to compile AI Filter code: {e_compile}")

        # 3c. 動的変調フレームエフェクトフィルターの実装
        update_progress(50, "V5 Phase 2b: Attaching biometric overlay & energy-reactive pixel melter with kinetic animation...")
        
        # 音声特徴からその時刻のエネルギーを取得する関数
        def get_energy_at(t_val):
            idx = int(t_val / hop_seconds_energy)
            if idx < len(rms_norm):
                return float(rms_norm[idx])
            return 0.5

        verified_subtitles_logged = {}

        def frame_effect_filter(get_frame, t):
            raw_frame_copy = get_frame(t).copy()
            frame = raw_frame_copy.copy()
            
            # 現在時刻 t の演出セクション（オーケストレーション）探索
            active_sec = None
            if orchestration_plan:
                for sec in orchestration_plan.get("sections", []):
                    if sec.get("start") <= t <= sec.get("end"):
                        active_sec = sec
                        break
            
            current_melt_int = 1.0
            current_glitch_int = 1.0
            current_biometric_opacity = 1.0
            current_lyric_style = lyric_effect_style if enable_kinetic_lyric else "SIMPLE"
            current_filter_code = None
            
            if active_sec:
                current_melt_int = active_sec.get("melt_intensity", 0.5)
                current_glitch_int = active_sec.get("glitch_intensity", 0.5)
                current_biometric_opacity = active_sec.get("biometric_opacity", 1.0)
                current_lyric_style = active_sec.get("lyric_effect_style", current_lyric_style)
                
                if enable_ai_orchestration:
                    filter_name = active_sec.get("active_filter", "none")
                    if filter_name != "none":
                        if os.path.exists("ai_filters.json"):
                            try:
                                with open("ai_filters.json", "r", encoding="utf-8") as f_json:
                                    filters_data = json.load(f_json)
                                    if filter_name in filters_data:
                                        current_filter_code = filters_data[filter_name]
                            except Exception:
                                pass

            energy = get_energy_at(t)
            beat_interval = 60.0 / max(1.0, bpm)
            time_since_beat = (t - bpm_offset) % beat_interval
            if time_since_beat < 0:
                time_since_beat += beat_interval
            decay = 12.0
            beat_signal = math.exp(-decay * time_since_beat)
            
            # 生体データのノイズ上書き描画 (BPM同期)
            overlay_frame = biometric_overlay.apply_biometric_overlay(
                frame_np=frame.copy(),
                t=t,
                total_duration=total_duration,
                aspect_ratio=aspect_ratio,
                enable_ecg=enable_ecg,
                enable_grf=enable_grf,
                enable_hexdump=enable_hexdump,
                bpm=bpm,
                bpm_offset=bpm_offset
            )

            if enable_ecg or enable_grf or enable_hexdump:
                if current_biometric_opacity >= 0.99:
                    frame = overlay_frame
                elif current_biometric_opacity > 0:
                    frame = cv2.addWeighted(overlay_frame, current_biometric_opacity, frame, 1.0 - current_biometric_opacity, 0)
            
            # 【Antigravity Phase 2: 静止画の生体駆動化 (BPM完全同期 空間歪曲 / Smooth Beat Pumping)】
            zoom_amp = 0.02 + 0.04 * energy
            zoom_scale = 1.0 + zoom_amp * beat_signal

            # BPMに完全同期したビート位相 (拍数)
            beats_elapsed = (t - bpm_offset) * (bpm / 60.0)
            shake_phase = beats_elapsed * 2.0 * math.pi

            twitch_amp = 1.5 + 8.0 * energy * beat_signal
            wiggle_x = math.sin(shake_phase) * twitch_amp
            wiggle_y = math.sin(shake_phase * 0.5) * twitch_amp

            h_img, w_img = frame.shape[:2]
            center = (w_img / 2.0, h_img / 2.0)
            M = cv2.getRotationMatrix2D(center, 0, zoom_scale)
            M[0, 2] += wiggle_x
            M[1, 2] += wiggle_y

            frame = cv2.warpAffine(frame, M, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            base_floor = 0.05 + 0.30 * energy
            modulated_signal = base_floor + (1.0 - base_floor) * beat_signal * energy

            dynamic_max_melt = max_melt * modulated_signal * current_melt_int
            dynamic_glitch_freq = glitch_freq * modulated_signal * current_glitch_int

            if filter_exec_mode == "single":
                dynamic_max_melt = 0.0
                dynamic_glitch_freq = 0.0

            frame = pixel_melter.melt_frame(
                frame_np=frame,
                t=t,
                total_duration=total_duration,
                max_melt=dynamic_max_melt,
                glitch_freq=dynamic_glitch_freq
            )
            
            # 複数フィルターのパイプライン実行 (マルチセレクト対応)
            filter_list = []
            has_explicit_filter_config = False
            if ai_filter_codes_json is not None:
                try:
                    import json
                    parsed = json.loads(ai_filter_codes_json)
                    if isinstance(parsed, list):
                        has_explicit_filter_config = True
                        filter_list = [c for c in parsed if isinstance(c, str) and c.strip()]
                except Exception:
                    pass

            if not filter_list:
                effective_code = current_filter_code if current_filter_code else compiled_ai_filter
                if effective_code:
                    filter_list = [effective_code]

            active_filters_to_run = filter_list

            if filter_exec_mode == "single1" or filter_exec_mode == "single":
                if len(filter_list) > 0:
                    beats_per_bar = 4.0
                    measures = max(1, slide_measures)
                    cycle_duration = (measures * beats_per_bar * 60.0) / max(1.0, bpm)
                    current_cycle_idx = int(t // cycle_duration) if cycle_duration > 0 else 0
                    
                    energy_level = 0 if energy < 0.40 else 1
                    adaptive_idx = (current_cycle_idx + energy_level) % len(filter_list)
                    active_filters_to_run = [filter_list[adaptive_idx]]
                else:
                    active_filters_to_run = []
            elif filter_exec_mode == "single2":
                if len(filter_list) > 0:
                    beats_per_bar = 4.0
                    measures = max(1, slide_measures)
                    cycle_duration = (measures * beats_per_bar * 60.0) / max(1.0, bpm)
                    current_cycle_idx = int(t // cycle_duration) if cycle_duration > 0 else 0
                    
                    seq_idx = current_cycle_idx % len(filter_list)
                    active_filters_to_run = [filter_list[seq_idx]]
                else:
                    active_filters_to_run = []
            else:
                active_filters_to_run = filter_list

            for code_item in active_filters_to_run:
                if code_item == "# PASS_THROUGH_NO_FILTER":
                    continue
                try:
                    local_filter = compile(code_item, "<ai_filter>", "exec") if isinstance(code_item, str) else code_item
                    local_vars = {}
                    exec(local_filter, {
                        "cv2": cv2,
                        "np": np,
                        "math": math,
                        "random": __import__("random")
                    }, local_vars)

                    if "apply_ai_filter" in local_vars:
                        beats_per_bar = 4.0
                        measures = max(1, slide_measures)
                        cycle_duration = (measures * beats_per_bar * 60.0) / max(1.0, bpm)
                        cycle_t = t % cycle_duration if cycle_duration > 0 else t
                        cycle_progress = (cycle_t / cycle_duration) if cycle_duration > 0 else 0.0

                        bar_duration = (60.0 / max(1.0, bpm)) * beats_per_bar
                        bar_t = t % bar_duration if bar_duration > 0 else t
                        bar_progress = (bar_t / bar_duration) if bar_duration > 0 else 0.0

                        import inspect
                        fn = local_vars["apply_ai_filter"]
                        sig = inspect.signature(fn)
                        kwargs = {
                            "frame": frame,
                            "t": cycle_t,
                            "duration": cycle_duration if cycle_duration > 0 else 8.0,
                            "bpm": bpm,
                            "energy": energy
                        }
                        if "bar_progress" in sig.parameters:
                            kwargs["bar_progress"] = cycle_progress
                        if "cycle_progress" in sig.parameters:
                            kwargs["cycle_progress"] = cycle_progress
                        if "bar_t" in sig.parameters:
                            kwargs["bar_t"] = cycle_t
                        if "cycle_t" in sig.parameters:
                            kwargs["cycle_t"] = cycle_t

                        processed = fn(**kwargs)
                        if processed is not None and isinstance(processed, np.ndarray):
                            frame = processed
                except Exception as e_exec:
                    print(f"[Warning] Multi-filter execution error: {e_exec}")
            
            return frame

        # Phase 2-A: AIフィルター＆ビジュアルエフェクト処理 (背景・エフェクトレイヤー)
        effects_clip = base_clip.transform(frame_effect_filter)
        
                                # Phase 2-B: MoviePy 100% Universal Guaranteed Subtitle Top Layer
        def apply_subtitle_to_frame(frame_in, t_val):
            frame_out = frame_in.copy()
            current_lyric_style = lyric_effect_style if enable_kinetic_lyric else "SIMPLE"
            current_glitch_int = 1.0
            
            if orchestration_plan:
                for o_sec in orchestration_plan.get("sections", []):
                    if o_sec.get("start") <= t_val <= o_sec.get("end"):
                        current_glitch_int = o_sec.get("glitch_intensity", 0.5)
                        current_lyric_style = o_sec.get("lyric_effect_style", current_lyric_style)
                        break
            
            current_section = None
            for sec in sections:
                s_sec = float(sec.get("start_sec", sec.get("start", 0)))
                e_sec = float(sec.get("end_sec", sec.get("end", 0)))
                if s_sec - 0.05 <= t_val <= e_sec + 0.05:
                    current_section = sec
                    break

            if current_section and current_section.get("lyric"):
                lyric_text = str(current_section["lyric"])
                sec_start_sec = float(current_section.get("start_sec", current_section.get("start", 0)))
                sec_id = current_section.get("id", 0)
                
                frame_out, px_count = draw_japanese_lyric(
                    frame_np=frame_out, 
                    text=lyric_text, 
                    color_scheme=subtitle_color,
                    style=current_lyric_style,
                    t=t_val,
                    bpm=bpm,
                    intensity=current_glitch_int,
                    start_time=sec_start_sec,
                    font_name=subtitle_font,
                    return_meta=True,
                    bpm_offset=bpm_offset
                )
                
                if sec_id not in verified_subtitles_logged:
                    verified_subtitles_logged[sec_id] = True
                    print(f"[V9 Universal Subtitle Engine] SUCCESS: Sec #{sec_id} '{lyric_text}' (t={t_val:.2f}s, Rendered Pixels: {px_count}px)")
            
            return frame_out

        # MoviePy v1 / v2 100% Universal Binding
        try:
            def subtitle_overlay_filter(get_frame, t):
                return apply_subtitle_to_frame(get_frame(t), t)
            final_clip = effects_clip.transform(subtitle_overlay_filter)
        except Exception:
            final_clip = effects_clip
        
        # 4. 音声の結合と出力
        update_progress(70, "Phase 3: Attaching audio and exporting final H.264 video...")
        
        from moviepy import AudioFileClip
        audio_clip = AudioFileClip(audio_path)
        final_clip = final_clip.with_audio(audio_clip)

        # レンダリング実行 (CustomMoviePyLogger でリアルタイム進捗をキャッチ)
        custom_logger = CustomMoviePyLogger(update_progress_fn=update_progress)
        final_clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
            logger=custom_logger
        )
        
        # 終了処理
        final_clip.close()
        base_clip.close()
        audio_clip.close()
        
        update_progress(100, "Rendering finished successfully!")
        if is_async:
            progress_store[session_id]["status"] = "completed"
            
    except Exception as e:
        import traceback
        import sys
        err = traceback.format_exc()
        print(f"[Error] Pipeline failure: {err}")
        sys.stdout.flush()
        if is_async and session_id in progress_store:
            # 簡潔にスタックトレースの最後の3行を取得してメッセージに含める
            err_lines = [line.strip() for line in err.splitlines() if line.strip()]
            err_summary = " -> ".join(err_lines[-3:]) if len(err_lines) >= 3 else err
            progress_store[session_id]["status"] = "failed"
            progress_store[session_id]["error"] = err
            progress_store[session_id]["message"] = f"Error: {err_summary}"
        else:
            raise e

# ==========================================
# FastAPI Web API
# ==========================================

def parse_time_str_to_float(time_str: str) -> float:
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            m = int(parts[0])
            s_parts = parts[1].split('.')
            s = int(s_parts[0])
            # mm:ss.hh 蠖｢蠑上�蝣ｴ蜷� hh (1/100遘�) 縺ｪ縺ｮ縺ｧ100縺ｧ蜑ｲ繧�
            ms = int(s_parts[1]) if len(s_parts) > 1 else 0
            return m * 60 + s + (ms / 100.0)
        elif len(parts) == 3: # HH:MM:SS,ms
            h = int(parts[0])
            m = int(parts[1])
            s_parts = parts[2].replace(',', '.').split('.')
            s = int(s_parts[0])
            ms = int(s_parts[1]) if len(s_parts) > 1 else 0
            return h * 3600 + m * 60 + s + (ms / 1000.0)
    except Exception:
        pass
    return 0.0

@app.post("/align")
async def api_align_lyrics(
    audio: UploadFile = File(...),
    lyrics_text: str = Form(...),
    api_key: Optional[str] = Form(default=None),
    primary_model: Optional[str] = Form("gemini-3.5-flash"),
    bpm: float = Form(120.0),
    bpm_offset: float = Form(0.0)
):
    effective_api_key = None
    key_source = "None"
    try:
        # 遨ｺ譁�ｭ励�蝣ｴ蜷医�讓呎ｺ門喧
        api_key_clean = api_key.strip() if api_key else None
        
        # 繧ゅ＠蜈･蜉帙＆繧後◆繧ｭ繝ｼ縺檎┌蜉ｹ縺ｪ蠖｢蠑擾ｼ�IzaSy縺ｧ蟋九∪繧峨↑縺�ｼ峨〒縺ゅｌ縺ｰ縲∫┌隕悶＠縺ｦ迺ｰ蠅�､画焚繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ繧堤匱蜍�
        if api_key_clean and not api_key_clean.startswith("AIzaSy"):
            print(f"[Warning] Provided API key does not start with 'AIzaSy'. Ignoring Web UI Input: {api_key_clean[:8]}...")
            api_key_clean = None
            
        effective_api_key = api_key_clean or os.environ.get("GEMINI_API_KEY")
        key_source = "Web UI Input" if api_key_clean else "OS Environment Variable (GEMINI_API_KEY)"
        
        if not effective_api_key:
            return {"status": "error", "error": "API Key is required for alignment. Please configure Gemini API Key properly."}
            
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="amvg_align_")
        
        audio_ext = os.path.splitext(audio.filename)[1] or ".mp3"
        temp_audio_path = os.path.join(temp_dir, f"align_audio{audio_ext}")
        with open(temp_audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
            
        print(f"[API /align] Starting AI Alignment using {primary_model}... (Key source: {key_source})")
        meta_dict = metadata_generator.generate_metadata_json(
            audio_path=temp_audio_path,
            lyrics_content=lyrics_text,
            api_key=effective_api_key,
            is_srt=False,
            primary_model=primary_model,
            bpm=bpm,
            bpm_offset=bpm_offset
        )
        
        segments = []
        for sec in meta_dict.get("sections", []):
            start_val = parse_time_str_to_float(sec["start"])
            end_val = parse_time_str_to_float(sec["end"])
            
            lyric_text = sec.get("lyric", "")
            segments.append({
                "start": start_val,
                "end": end_val,
                "text": lyric_text,
                "is_marker": metadata_generator.is_section_marker(lyric_text)
            })
            
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {"status": "success", "segments": segments}
        
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        key_preview = "None"
        if effective_api_key:
            key_preview = f"{effective_api_key[:8]}...{effective_api_key[-8:]}" if len(effective_api_key) > 16 else "Short Key"
        print(f"[Error] /align failure: {err}")
        return {"status": "error", "error": f"{str(e)} (Active Key: {key_preview} loaded from {key_source})"}

@app.post("/generate-filter-code")
async def api_generate_filter_code(
    prompt: str = Form(...),
    current_code: Optional[str] = Form(default=None),
    api_key: Optional[str] = Form(default=None),
    primary_model: Optional[str] = Form("gemini-3.5-flash")
):
    effective_api_key = api_key.strip() if api_key else None
    if effective_api_key and not effective_api_key.startswith("AIzaSy"):
        effective_api_key = None
    effective_api_key = effective_api_key or os.environ.get("GEMINI_API_KEY")
    
    if not effective_api_key:
        return {"status": "error", "error": "API Key is required for generating code. Please configure Gemini API Key properly."}
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=effective_api_key)
        
        system_instruction = """
You are an expert Python image processing developer specializing in OpenCV (cv2) and NumPy.
Your task is to write a single Python function named `apply_ai_filter` that takes an image frame and rendering states, and returns the modified frame.

You must strictly output ONLY the Python code block starting with `def apply_ai_filter`. DO NOT wrap it in ```python or any markdown. Return raw code.
No explanations, no markdown formatting. Just pure Python code.

Function Signature:
```python
def apply_ai_filter(frame: np.ndarray, t: float, duration: float, bpm: float, energy: float) -> np.ndarray:
    # frame: HxWx3 BGR/RGB image (numpy.ndarray, dtype=uint8)
    # t: float current time in seconds
    # duration: float total video duration in seconds
    # bpm: float tempo
    # energy: float music energy envelope normalized (0.0 to 1.0)
    # Returns: HxWx3 numpy.ndarray
```

Key Guidance:
1. You can use standard Python libraries: math, random, cv2, numpy (as np).
2. The height and width of the frame can be extracted using `h, w = frame.shape[:2]`.
3. Use the 'energy' argument (0.0 to 1.0) to make the visual effect reactive to music volume/beats.
4. Try to make the visual effects aesthetically pleasing, like cyberpunk, retro scanlines, glitched grids, digital neon lines, noise particles, etc.
5. If you draw text, ensure to keep it clean and handle position calculations dynamically.
6. Make sure the output frame has the exact same dimensions and dtype as the input frame.
"""

        model = genai.GenerativeModel(
            model_name=primary_model or "gemini-3.5-flash",
            system_instruction=system_instruction
        )
        
        prompt_text = f"User Request: {prompt}\n"
        if current_code:
            prompt_text += f"\nModify or Refine the following existing code:\n```python\n{current_code}\n```"
            
        response = model.generate_content(prompt_text)
        code = response.text.strip()
        
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
            
        return {"status": "success", "code": code.strip()}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/detect-bpm")
async def detect_bpm_endpoint(audio: UploadFile = File(...)):
    try:
        import tempfile
        import librosa
        import numpy as np

        temp_dir = tempfile.mkdtemp(prefix="amvg_bpm_")
        audio_ext = os.path.splitext(audio.filename)[1] or ".mp3"
        temp_audio_path = os.path.join(temp_dir, f"bpm_audio{audio_ext}")
        
        with open(temp_audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
            
        y, sr = librosa.load(temp_audio_path, sr=22050, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        if isinstance(tempo, np.ndarray):
            bpm_val = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            bpm_val = float(tempo)
            
        if bpm_val <= 0:
            bpm_val = 120.0
            
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        offset_val = float(beat_times[0]) if len(beat_times) > 0 else 0.0
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "status": "success",
            "bpm": round(bpm_val, 2),
            "offset": round(offset_val, 3)
        }
    except Exception as e:
        import traceback
        print(f"[Error] BPM detection failure: {traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "bpm": 120.0,
            "offset": 0.0
        }

# ==========================================
# AI フィルター保存および自己進化 (自己増殖) API
# ==========================================

FILTERS_FILE = get_resource_path("ai_filters.json")

def load_ai_filters() -> dict:
    if os.path.exists(FILTERS_FILE):
        try:
            with open(FILTERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load ai_filters.json: {e}")
    return {}

def save_ai_filters(filters: dict):
    try:
        with open(FILTERS_FILE, "w", encoding="utf-8") as f:
            json.dump(filters, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[Error] Failed to save ai_filters.json: {e}")

@app.post("/save-filter-code")
async def api_save_filter_code(
    name: str = Form(...),
    code: str = Form(...)
):
    filters = load_ai_filters()
    filters[name] = code
    save_ai_filters(filters)
    return {"status": "success", "message": f"Filter '{name}' saved successfully."}

@app.get("/list-filter-codes")
async def api_list_filter_codes():
    filters = load_ai_filters()
    return {"status": "success", "filters": filters}

@app.post("/delete-filter-code")
async def api_delete_filter_code(
    name: str = Form(...)
):
    filters = load_ai_filters()
    if name in filters:
        del filters[name]
        save_ai_filters(filters)
        return {"status": "success", "message": f"Filter '{name}' deleted successfully."}
    return {"status": "error", "error": f"Filter '{name}' not found."}



@app.post("/generate")
async def api_generate(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    aspect_ratio: str = Form("16:9"),
    fps: int = Form(30),
    assets: List[UploadFile] = File(default=[]),
    api_key: Optional[str] = Form(default=None),
    lyrics: Optional[UploadFile] = File(default=None),
    primary_model: Optional[str] = Form(default=None),
    enable_ecg: str = Form("true"),
    enable_grf: str = Form("true"),
    enable_hexdump: str = Form("true"),
    max_melt: float = Form(1.0),
    glitch_freq: float = Form(1.0),
    subtitle_color: str = Form("cyan_magenta"),
    edited_srt: Optional[str] = Form(default=None),
    bpm: float = Form(120.0),
    bpm_offset: float = Form(0.0),
    slide_measures: int = Form(4),
    ai_filter_code: Optional[str] = Form(default=None),
    ai_filter_codes_json: Optional[str] = Form(default=None),
    filter_exec_mode: str = Form("multi"),
    enable_ai_orchestration: str = Form("false"),
    enable_kinetic_lyric: str = Form("false"),
    lyric_effect_style: str = Form("SIMPLE"),
    subtitle_font: str = Form("gothic")
):
    # 既存セッションのクリーンアップ
    cleanup_old_sessions()
    
    cleaned_api_key = api_key.strip() if api_key else None
    if cleaned_api_key and not cleaned_api_key.startswith("AIzaSy"):
        print(f"[Warning] Provided generate API key does not start with 'AIzaSy'. Ignoring Web UI Input.")
        cleaned_api_key = None
    
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_DIR, f"amvg_v2_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    progress_store[session_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Uploading assets to cache...",
        "error": "",
        "temp_dir": session_dir
    }
    
    # 音楽ファイルの保存
    audio_ext = os.path.splitext(audio.filename)[1] or ".mp3"
    temp_audio_path = os.path.join(session_dir, f"input_audio{audio_ext}")
    with open(temp_audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
        
    # 歌詞ファイルの保存
    temp_lyrics_path = None
    if edited_srt:
        temp_lyrics_path = os.path.join(session_dir, "edited_lyrics.srt")
        with open(temp_lyrics_path, "w", encoding="utf-8") as f:
            f.write(edited_srt)
    elif lyrics and lyrics.filename:
        lyr_ext = os.path.splitext(lyrics.filename)[1] or ".txt"
        temp_lyrics_path = os.path.join(session_dir, f"input_lyrics{lyr_ext}")
        with open(temp_lyrics_path, "wb") as f:
            shutil.copyfileobj(lyrics.file, f)
            
    # 背景アセットの保存
    assets_dir = os.path.join(session_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    saved_paths = []
    for idx, asset in enumerate(assets):
        if not asset.filename:
            continue
        ext = os.path.splitext(asset.filename)[1] or ".png"
        path = os.path.join(assets_dir, f"asset_{idx:04d}{ext}")
        with open(path, "wb") as f:
            shutil.copyfileobj(asset.file, f)
        saved_paths.append(path)
        
    if len(saved_paths) == 1 and saved_paths[0].lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        assets_input = saved_paths[0]
    else:
        assets_input = assets_dir
        
    output_filename = f"output_v2_{session_id}.mp4"
    temp_output_path = os.path.join(session_dir, output_filename)
    
    background_tasks.add_task(
        run_pipeline,
        session_id=session_id,
        audio_path=temp_audio_path,
        assets_input=assets_input,
        output_path=temp_output_path,
        aspect_ratio=aspect_ratio,
        fps=fps,
        api_key=cleaned_api_key,
        lyrics_path=temp_lyrics_path,
        is_async=True,
        enable_ecg=(enable_ecg.lower() == 'true'),
        enable_grf=(enable_grf.lower() == 'true'),
        enable_hexdump=(enable_hexdump.lower() == 'true'),
        max_melt=max_melt,
        glitch_freq=glitch_freq,
        subtitle_color=subtitle_color,
        primary_model=primary_model,
        bpm=bpm,
        bpm_offset=bpm_offset,
        slide_measures=slide_measures,
        ai_filter_code=ai_filter_code,
        ai_filter_codes_json=ai_filter_codes_json,
        filter_exec_mode=filter_exec_mode,
        enable_ai_orchestration=(enable_ai_orchestration.lower() == 'true'),
        enable_kinetic_lyric=(enable_kinetic_lyric.lower() == 'true'),
        lyric_effect_style=lyric_effect_style,
        subtitle_font=subtitle_font
    )
    
    return {"session_id": session_id}

@app.post("/preview_filter")
async def api_preview_filter(
    ai_filter_code: Optional[str] = Form(default=None),
    ai_filter_codes_json: Optional[str] = Form(default=None),
    t: float = Form(1.0),
    bpm: float = Form(120.0),
    energy: float = Form(0.8),
    slide_measures: int = Form(4),
    filter_exec_mode: Optional[str] = Form(default="multi"),
    asset_image: Optional[UploadFile] = File(default=None),
    subtitle_font: Optional[str] = Form(default="gothic")
):
    try:
        frame = None
        if asset_image and asset_image.filename:
            image_bytes = await asset_image.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                fh, fw = frame.shape[:2]
                if fw > 640:
                    scale = 640.0 / fw
                    frame = cv2.resize(frame, (640, int(fh * scale)))

        if frame is None:
            # デフォルトのテスト用フレーム生成 (720x1280, BGR)
            h, w = 720, 1280
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                r = int(20 + 50 * (y / h))
                g = int(10 + 30 * (y / h))
                b = int(40 + 90 * (y / h))
                frame[y, :] = (b, g, r)
            
            # グリッド描画
            for x in range(0, w, 80):
                cv2.line(frame, (x, 0), (x, h), (40, 60, 80), 1)
            for y in range(0, h, 80):
                cv2.line(frame, (0, y), (w, y), (40, 60, 80), 1)

            cv2.putText(frame, "A.M.V.G v5 MULTI-FILTER PREVIEW", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 243, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"STATE: t={t:.2f}s | BPM={bpm:.1f} | MEASURES={slide_measures}", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 127), 2, cv2.LINE_AA)
            cv2.circle(frame, (w // 2, h // 2), 160, (0, 243, 255), 4)

        # AI Filter コードリストのパースと複数適用
        filter_list = []
        if ai_filter_codes_json:
            try:
                import json
                parsed = json.loads(ai_filter_codes_json)
                if isinstance(parsed, list):
                    filter_list = [c for c in parsed if isinstance(c, str) and c.strip()]
            except Exception:
                pass

        if not filter_list and ai_filter_code and ai_filter_code.strip():
            filter_list = [ai_filter_code.strip()]

        if not filter_list:
            raise ValueError("有効な Python AI Filter コードが選択されていません。")

        # 単発モード1(single1), 単発モード2(single2), 複合モード(multi) の適用ロジック
        active_filters_to_run = filter_list

        if filter_exec_mode == "single1" or filter_exec_mode == "single":
            # 単発モード1: 重ね合わせは一切せず、選択されたフィルターの中から小節周期・曲調(Energy)に応じて適宜1つだけを単発適用
            if len(filter_list) > 0:
                beats_per_bar = 4.0
                measures = max(1, slide_measures)
                cycle_duration = (measures * beats_per_bar * 60.0) / max(1.0, bpm)
                current_cycle_idx = int(t // cycle_duration) if cycle_duration > 0 else 0
                
                energy_level = 0 if energy < 0.40 else 1
                adaptive_idx = (current_cycle_idx + energy_level) % len(filter_list)
                active_filters_to_run = [filter_list[adaptive_idx]]
            else:
                active_filters_to_run = []
        elif filter_exec_mode == "single2":
            # 単発モード2: 設定されたBPMに対する設定小節時間(SLIDES CHANGEOVER RATE)ぴったりで順次1つずつ単発切り替え
            if len(filter_list) > 0:
                beats_per_bar = 4.0
                measures = max(1, slide_measures)
                cycle_duration = (measures * beats_per_bar * 60.0) / max(1.0, bpm)
                current_cycle_idx = int(t // cycle_duration) if cycle_duration > 0 else 0
                
                seq_idx = current_cycle_idx % len(filter_list)
                active_filters_to_run = [filter_list[seq_idx]]
            else:
                active_filters_to_run = []
        else:
            # 複合モード: 選択された全てのフィルターを重ね合わせ(スタック)適用
            active_filters_to_run = filter_list

        for f_code in active_filters_to_run:
            if f_code == "# PASS_THROUGH_NO_FILTER":
                continue
            local_vars = {}
            exec_globals = {
                "cv2": cv2,
                "np": np,
                "math": math,
                "random": __import__("random")
            }
            compiled_code = compile(f_code, "<ai_filter_preview>", "exec")
            exec(compiled_code, exec_globals, local_vars)

            if "apply_ai_filter" in local_vars:
                beats_per_bar = 4.0
                measures = max(1, slide_measures)
                cycle_duration = (measures * beats_per_bar * 60.0) / max(1.0, bpm)
                cycle_t = t % cycle_duration
                cycle_progress = cycle_t / cycle_duration

                bar_duration = (60.0 / max(1.0, bpm)) * beats_per_bar
                bar_t = t % bar_duration
                bar_progress = bar_t / bar_duration

                import inspect
                fn = local_vars["apply_ai_filter"]
                sig = inspect.signature(fn)
                kwargs = {
                    "frame": frame.copy(),
                    "t": cycle_t,
                    "duration": cycle_duration if cycle_duration > 0 else 8.0,
                    "bpm": bpm,
                    "energy": energy
                }
                if "bar_progress" in sig.parameters:
                    kwargs["bar_progress"] = cycle_progress
                if "cycle_progress" in sig.parameters:
                    kwargs["cycle_progress"] = cycle_progress
                if "bar_t" in sig.parameters:
                    kwargs["bar_t"] = cycle_t
                if "cycle_t" in sig.parameters:
                    kwargs["cycle_t"] = cycle_t

                processed = fn(**kwargs)
                if processed is not None and isinstance(processed, np.ndarray):
                    frame = processed

        success, encoded_img = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            raise ValueError("Failed to encode output frame to JPEG")
            
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

    except Exception as e:
        global last_filter_error_log
        last_filter_error_log = traceback.format_exc()
        err_detail = str(e)
        h, w = 480, 854
        err_frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(err_frame, "FILTER RUNTIME ERROR", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        err_lines = err_detail.splitlines()
        for idx, line in enumerate(err_lines[:8]):
            cv2.putText(err_frame, line[:75], (30, 110 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 255), 1, cv2.LINE_AA)
            
        _, encoded_img = cv2.imencode(".jpg", err_frame)
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.get("/filters")
async def api_get_filters():
    filters_file = get_resource_path("ai_filters.json")
    if os.path.exists(filters_file):
        try:
            with open(filters_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read filters: {e}")
    return {}

@app.post("/save_filter")
async def api_save_filter(name: str = Form(...), code: str = Form(...)):
    filters_file = get_resource_path("ai_filters.json")
    filters_data = {}
    if os.path.exists(filters_file):
        try:
            with open(filters_file, "r", encoding="utf-8") as f:
                filters_data = json.load(f)
        except Exception:
            filters_data = {}
            
    filters_data[name] = code
    try:
        with open(filters_file, "w", encoding="utf-8") as f:
            json.dump(filters_data, f, ensure_ascii=False, indent=4)
        return {"status": "success", "message": f"Filter '{name}' saved successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save filter: {e}")

@app.post("/rename_filter")
async def api_rename_filter(old_name: str = Form(...), new_name: str = Form(...)):
    filters_file = get_resource_path("ai_filters.json")
    if not os.path.exists(filters_file):
        raise HTTPException(status_code=404, detail="ai_filters.json が存在しません。")

    old_name = old_name.strip()
    new_name = new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="新しいフィルター名が空です。")

    try:
        with open(filters_file, "r", encoding="utf-8") as f:
            filters_data = json.load(f)

        if old_name not in filters_data:
            raise HTTPException(status_code=404, detail=f"フィルター '{old_name}' が見つかりません。")

        # キー名を変更（順序保持）
        new_data = {}
        for k, v in filters_data.items():
            if k == old_name:
                new_data[new_name] = v
            else:
                new_data[k] = v

        with open(filters_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

        return {"status": "success", "message": f"フィルター名を '{old_name}' から '{new_name}' に変更しました。", "filters": new_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename filter: {e}")

@app.post("/auto_repair_filter")
async def api_auto_repair_filter(
    code: str = Form(...),
    api_key: Optional[str] = Form(None),
    primary_model: Optional[str] = Form(None)
):
    global last_filter_error_log
    if not last_filter_error_log:
        raise HTTPException(status_code=400, detail="修正対象のエラーログが存在しません。プレビューでエラーが発生した後に実行してください。")

    active_api_key = api_key if api_key else os.environ.get("GEMINI_API_KEY")
    if not active_api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key が設定されていません。環境変数 GEMINI_API_KEY または画面のAPIキー欄に入力してください。")

    import google.generativeai as genai
    genai.configure(api_key=active_api_key)

    model_name = primary_model if primary_model else "gemini-2.5-flash"

    prompt = f"""
あなたは強力なPython AIアシスタントです。
以下のPython OpenCV画像処理フィルターコードを実行したところ、実行時エラーが発生しました。
エラーメッセージとトレースバックを確認し、正しく動作するようにコードを修正してください。

【制約事項】
- 返答には説明文や挨拶、コードフェンス（```python ... ```）を含めず、**純粋なPythonコードのみ**を出力してください。
- `apply_ai_filter(frame: np.ndarray, t: float, duration: float, bpm: float, energy: float) -> np.ndarray` の関数シグネチャを必ず維持してください。
- 必要なライブラリ（cv2, np, math, randomなど）のインポートを関数内、またはコードの先頭に必ず含めてください。

【エラーメッセージ】
{last_filter_error_log}

【元のエラーコード】
{code}
"""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        repaired_code = response.text.strip()

        if repaired_code.startswith("```"):
            lines = repaired_code.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            repaired_code = "\n".join(lines).strip()

        return {"status": "success", "repaired_code": repaired_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini APIによる自動修正に失敗しました: {e}")


@app.get("/status/{session_id}")
async def api_status(session_id: str):
    if session_id not in progress_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return progress_store[session_id]

def cleanup_old_sessions():
    """1譎る俣莉･荳顔ｵ碁℃縺励◆蜿､縺�ｸ譎ゅそ繝�す繝ｧ繝ｳ繝輔か繝ｫ繝繧貞炎髯､縺吶ｋ"""
    try:
        if not os.path.exists(TEMP_DIR):
            return
        now = time.time()
        for folder_name in os.listdir(TEMP_DIR):
            if folder_name.startswith("amvg_v2_"):
                folder_path = os.path.join(TEMP_DIR, folder_name)
                if os.path.isdir(folder_path):
                    mtime = os.path.getmtime(folder_path)
                    # 1譎る俣 (3600遘�) 莉･荳顔ｵ碁℃縺励※縺�ｋ蝣ｴ蜷�
                    if now - mtime > 3600:
                        shutil.rmtree(folder_path, ignore_errors=True)
                        sid = folder_name.replace("amvg_v2_", "")
                        if sid in progress_store:
                            del progress_store[sid]
                        print(f"[Cleanup] Removed expired session folder: {folder_name}")
    except Exception as e:
        print(f"[Warning] Session cleanup failed: {e}")

@app.get("/download/{session_id}")
async def api_download(session_id: str):
    if session_id not in progress_store:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session_data = progress_store[session_id]
    session_dir = session_data.get("temp_dir")
    output_path = os.path.join(session_dir, f"output_v2_{session_id}.mp4")
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="File is still rendering")
        
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename="amvg_v2_output.mp4"
    )

@app.get("/", response_class=HTMLResponse)
async def api_webui():
    index_path = get_resource_path("index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>AMVG v2 index.html not found</h1>", status_code=404)

# ==========================================
# CLI / 繝｡繧､繝ｳ襍ｷ蜍�
# ==========================================

def main():
    import sys
    import os
    # --noconsole環境でのprintエラー（stdoutがNone）を回避する安全装置
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

    parser = argparse.ArgumentParser(description="Antigravity MV Generator V10 Free")
    
    parser.add_argument("--web", action="store_true", help="Web UIサーバーモードで起動する")
    parser.add_argument("--port", type=int, default=8003, help="Webサーバーのポート番号")
    
    # CLI用
    parser.add_argument("--audio", type=str, help="音声ファイルへのパス")
    parser.add_argument("--assets", type=str, help="背景アセットへのパス")
    parser.add_argument("--output", type=str, default="output_v2.mp4", help="出力ファイル名")
    parser.add_argument("--aspect", type=str, choices=["16:9", "9:16"], default="16:9", help="アスペクト比")
    parser.add_argument("--fps", type=int, default=30, help="出力動画のフレームレート")
    parser.add_argument("--api-key", type=str, help="Gemini API キー")
    parser.add_argument("--lyrics", type=str, help="歌詞テキストまたはSRTへのパス")
    parser.add_argument("--metadata-json", type=str, help="すでに生成済みのメタデータJSONを指定")
    
    args = parser.parse_args()
    
    # 【スマート化】引数が何も指定されていない（ダブルクリック起動）場合は強制的にWebUIモードにする
    if not args.web and not args.audio and not args.assets:
        args.web = True
    
    if args.web:
        import socket
        import webbrowser
        import threading
        import time
        import uvicorn
        
        # 【重要】すでにサーバーがバックグラウンドで稼働しているかチェック
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('127.0.0.1', port)) == 0
                
        # 既に稼働中の場合は、新しいサーバーを立てずにブラウザだけを開いて終了する
        if is_port_in_use(args.port):
            print(f"[System] ポート {args.port} は既に使用されています。既存のセッションを開きます。")
            webbrowser.open(f"http://127.0.0.1:{args.port}")
            sys.exit(0)

        print("[WebUI] Starting A.M.V.G v2 Web Server...")
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://127.0.0.1:{args.port}")
            
        threading.Thread(target=open_browser, daemon=True).start()
        # EXE環境でuvicornを安全に起動するため、"main:app" ではなく app オブジェクトを直接渡す
        uvicorn.run(app, host="127.0.0.1", port=args.port)
        return
        
    print("=" * 60)
    print("[CLI] ANTIGRAVITY MV GENERATOR V10 Free - CLI PIPELINE")
    print("=" * 60)
    
    if not args.audio or not args.assets:
        parser.print_help()
        print("\n[Error] --audio and --assets are required in CLI mode.")
        sys.exit(1)
        
    # APIキーの取得優先度 (引数 -> 環境変数)
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    try:
        run_pipeline(
            session_id="cli_v2",
            audio_path=args.audio,
            assets_input=args.assets,
            output_path=args.output,
            aspect_ratio=args.aspect,
            fps=args.fps,
            api_key=api_key,
            lyrics_path=args.lyrics,
            metadata_json_path=args.metadata_json,
            is_async=False
        )
        print(f"[Success] Render complete! Output file: {args.output}")
    except Exception as e:
        import traceback
        print(f"[Error] Pipeline execution failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    # PyInstallerでのマルチプロセス実行が暴走するのを防ぐ必須コード
    import multiprocessing
    multiprocessing.freeze_support()
    main()
