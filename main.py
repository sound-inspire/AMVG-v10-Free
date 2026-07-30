# -*- coding: utf-8 -*-
"""
A.M.V.G v3 - Antigravity MV Generator
郢晢ｽ｡郢ｧ�､郢晢ｽｳ驍ｨ�ｱ陷ｷ蛹ｻ縺顔ｹ晢ｽｳ郢晏現ﾎ懃ｹ晢ｽｼ郢晄亢縺�ｹ晢ｽｳ郢晢ｿｽ (CLI �ｽ�ｽ FastAPI Web API / Web UI 郢ｧ�ｵ郢晢ｽｼ郢晁��ｽ)
隨假ｿｽ V3 隴�ｽｰ隶匁ｺｯ�ｽ: 隶鯉ｽｽ隴厄ｽｲ郢ｧ�ｨ郢晞亂ﾎ晉ｹｧ�ｮ郢晢ｽｼ陋ｹ�ｽ�ｵ�｡驍ｱ�ｽ (RMS + onset_strength) 邵ｺ�ｫ郢ｧ蛹ｻ�玖滋遒√裟陞溯歓�ｪ�ｿ郢ｧ�ｨ郢晁ｼ斐♂郢ｧ�ｯ郢晏現縺顔ｹ晢ｽｳ郢ｧ�ｸ郢晢ｽｳ隰ｳ�ｭ髴茨ｿｽ
"""

import os
import sys
import uuid
import shutil
import time
import argparse
import tempfile
import threading
import socket
import multiprocessing
import uvicorn
import json
import math
import cv2
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont

from proglog import ProgressBarLogger

# 髢ｾ�ｪ髯ｬ�ｽ郢晢ｽ｢郢ｧ�ｸ郢晢ｽ･郢晢ｽｼ郢晢ｽｫ邵ｺ�ｮ郢ｧ�､郢晢ｽｳ郢晄亢郢�
import loop_multiplier
import biometric_overlay
import pixel_melter
import metadata_generator

import traceback

# 譛蠕後↓逋ｺ逕溘＠縺溘ヵ繧｣繝ｫ繧ｿ繝ｼ繝励Ξ繝薙Η繝ｼ譎ゅ�繧ｨ繝ｩ繝ｼ繝ｭ繧ｰ繧剃ｿ晄戟縺吶ｋ繧ｰ繝ｭ繝ｼ繝舌Ν螟画焚
last_filter_error_log = ""



app = FastAPI(title="AMVG V10 Free UPG API")
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)
print("==================== DEBUG: main.py V10 Free UPG (BPM Sync Engine) (Kinetic Animation & Multi-Exec Engine) loaded! ====================")

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_json():
    return {}

class CustomMoviePyLogger(ProgressBarLogger):
    def __init__(self, update_progress_fn=None):
        super().__init__()
        self.update_progress_fn = update_progress_fn
        self.last_percent = -1

    def callback(self, **changes):
        index = changes.get("index")
        total = changes.get("total")
        
        # 繧ゅ＠ changes 縺ｫ辟｡縺代ｌ縺ｰ self.bars 繧偵メ繧ｧ繝�け
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

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    err_detail = traceback.format_exc()
    print(f"\n[500 Server Exception] Error at endpoint: {request.url.path}\n{err_detail}\n")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc), "detail": err_detail}
    )

progress_store = {}
TEMP_DIR = tempfile.gettempdir()

def cleanup_old_sessions():
    try:
        now = time.time()
        expired = [sid for sid, data in progress_store.items() if isinstance(data, dict) and now - data.get("timestamp", now) > 3600]
        for sid in expired:
            progress_store.pop(sid, None)
            
        # Tempディレクトリ内の過去の amvg_v2_ 一時作業用フォルダの自動開放・クリーンアップ
        temp_dir = tempfile.gettempdir()
        for item in os.listdir(temp_dir):
            if item.startswith("amvg_v2_"):
                item_path = os.path.join(temp_dir, item)
                try:
                    if os.path.isdir(item_path):
                        if now - os.path.getmtime(item_path) > 1800:
                            import shutil
                            shutil.rmtree(item_path, ignore_errors=True)
                    elif os.path.isfile(item_path):
                        if now - os.path.getmtime(item_path) > 1800:
                            os.remove(item_path)
                except Exception:
                    pass
    except Exception as e:
        print(f"[Warn] cleanup_old_sessions error: {e}")

def get_resource_path(relative_path):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

def get_external_data_path(filename: str) -> str:
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)

FILTERS_FILE = get_external_data_path("ai_filters.json")

DEFAULT_FILTERS = {
    "蜴溽判�医ヮ繝ｼ繧ｨ繝輔ぉ繧ｯ繝茨ｼ�": """import cv2
import numpy as np

def apply_ai_filter(frame: np.ndarray, t: float, duration: float, bpm: float, energy: float) -> np.ndarray:
    \"\"\"
    蜴溽判�医ヮ繝ｼ繧ｨ繝輔ぉ繧ｯ繝茨ｼ�: 蜈･蜉帙ヵ繝ｬ繝ｼ繝�繧貞刈蟾･縺帙★縺ｫ縺昴�縺ｾ縺ｾ霑泌唆縺励∪縺吶�
    \"\"\"
    return frame
"""
}

def load_ai_filters() -> dict:
    if not os.path.exists(FILTERS_FILE):
        save_ai_filters(DEFAULT_FILTERS)
        return DEFAULT_FILTERS.copy()
    try:
        with open(FILTERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not data:
                save_ai_filters(DEFAULT_FILTERS)
                return DEFAULT_FILTERS.copy()
            return data
    except Exception as e:
        print(f"[Warning] Failed to load ai_filters.json: {e}")
        return DEFAULT_FILTERS.copy()

def save_ai_filters(filters: dict):
    try:
        with open(FILTERS_FILE, "w", encoding="utf-8") as f:
            json.dump(filters, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[Error] Failed to save ai_filters.json: {e}")

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def min_sec_to_seconds(val) -> float:
    """v5螳悟�莠呈鋤縺ｮ螳牙�縺ｪ譎る俣譁�ｭ怜�/謨ｰ蛟､繝代�繧ｹ"""
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
    bpm_offset: float = 0.0,
    intensity: float = 1.0,
    start_time: float = 0.0,
    font_name: str = "gothic",
    return_meta: bool = False
):
    """PIL縺ｨOpenCV繧堤ｵ�∩蜷医ｏ縺帙※縲√く繝阪ユ繧｣繝�け縺翫ｈ縺ｳ繧ｰ繝ｪ繝�メ蜉ｹ譫懊ｒ縺九￠縺滓ｭ瑚ｩ槭ｒ謠冗判縺吶ｋ (v5螳悟�貅匁侠繝ｭ繧ｸ繝�け)"""
    import random
    from PIL import Image, ImageDraw, ImageFont
    h, w, c = frame_np.shape
    
    # 1. 騾乗�縺ｪ繧｢繝ｫ繝輔ぃ繝√Ε繝ｳ繝阪Ν繧呈戟縺､逕ｻ蜒上ｒ菴懈�
    text_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_img)
    
    # 2. 繧ｭ繝阪ユ繧｣繝�け繧ｵ繧､繧ｺ險育ｮ� (讌ｽ譖ｲ縺ｮ邨ｶ蟇ｾ繝薙�繝医♀繧医� BPM Offset 縺ｫ 100% 螳悟�蜷梧悄)
    beat_duration = 60.0 / max(1.0, bpm)
    t_global = max(0.0, t - bpm_offset)
    time_since_beat = t_global % beat_duration
    decay_constant = 6.0 / beat_duration
    beat_signal = math.exp(-decay_constant * time_since_beat)
    
    # 蝓ｺ譛ｬ繝輔か繝ｳ繝医し繧､繧ｺ (逕ｻ髱｢縺ｮ鬮倥＆縺ｫ繧ｹ繧ｱ繝ｼ繝ｫ)
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
            # OS讓呎ｺ悶�繝輔か繝ｳ繝医ヱ繧ｹ繧単IL縺ｫ閾ｪ蜍墓爾邏｢縺輔○繧�
            font = ImageFont.truetype(fpath, font_size)
            print(f"[Subtitle Font Engine] Loaded font: '{fpath}' ({font.getname()}) for mode: '{font_name}'")
            break
        except (IOError, OSError):
            try:
                # Windows逕ｨ縺ｮ邨ｶ蟇ｾ繝代せ繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ
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
        
    # 荳ｭ螟ｮ荳矩Κ驟咲ｽｮ
    tx = (w - text_w) // 2
    ty = int(h * 0.84) - (text_h // 2)
    
    # 驟崎牡驕ｸ謚� (v5螳悟�蜀咲樟)
    main_color = (0, 243, 255, 255)  # 繧ｷ繧｢繝ｳ
    glow_color = (255, 0, 127, 255)  # 繝槭ぞ繝ｳ繧ｿ
    
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

    # 繝阪が繝ｳ蜈牙ｽｩ縺ｮ謠冗判 (NEON_GLOW)
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

    # 騾壼ｸｸ縺ｮ繧｢繧ｦ繝医Λ繧､繝ｳ�磯ｻ堤ｸ∝叙繧奇ｼ峨�謠冗判
    outline_color = (0, 0, 0, 255)
    for offset in [(-2, -2), (2, -2), (-2, 2), (2, 2), (-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((tx + offset[0], ty + offset[1]), text, font=font, fill=outline_color)
        
    # 繝｡繧､繝ｳ縺ｮ譁�ｭ励ｒ謠冗判
    draw.text((tx, ty), text, font=font, fill=main_color)
    
    # NumPy縺ｮRGBA驟榊�縺ｫ螟画鋤
    text_np = np.array(text_img)
    text_pixel_count = int(np.count_nonzero(text_np[:, :, 3] > 10))
    
    # 3. 繧ｰ繝ｪ繝�メ/濶ｲ蜿主ｷｮ (GLITCH or CHROME_RGB 繧ｹ繧ｿ繧､繝ｫ縺ｮ蝣ｴ蜷�)
    if style in ["GLITCH", "CHROME_RGB"] and intensity > 0.05:
        glitch_trigger = math.sin(t * 12.0) * beat_signal
        
        # 繝√Ε繝ｳ繝阪Ν繧ｹ繝励Μ繝�ヨ縺ｫ繧医ｋ濶ｲ蜿主ｷｮ
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
            
        # 讓ｪ繝ｩ繧､繝ｳ繧ｹ繝ｩ繧､繧ｹ豁ｪ縺ｿ (GLITCH 繧ｹ繧ｿ繧､繝ｫ縺ｮ縺ｿ)
        if style == "GLITCH" and (glitch_trigger > 0.45 or random.random() < 0.1 * intensity):
            num_slices = random.randint(3, 7)
            for _ in range(num_slices):
                slice_y = random.randint(max(0, ty - 20), min(h - 10, ty + text_h + 20))
                slice_h = random.randint(3, 12)
                slice_shift = random.randint(-30, 30)
                
                slice_y = max(0, min(h - slice_h, slice_y))
                text_np[slice_y:slice_y+slice_h] = np.roll(text_np[slice_y:slice_y+slice_h], slice_shift, axis=1)

    # 4. 蜈�判蜒上→縺ｮ繝悶Ξ繝ｳ繝� (v5逶ｴ謗･繝悶Ξ繝ｳ繝�)
    alpha = text_np[:, :, 3:4] / 255.0
    text_rgb = text_np[:, :, 0:3]
    blended = (1.0 - alpha) * frame_np + alpha * text_rgb
    final_np = blended.astype(np.uint8)
    
    if return_meta:
        return final_np, text_pixel_count
    return final_np


# ==========================================
# 驍ｨ�ｱ陷ｷ蛹ｻ繝ｱ郢ｧ�､郢晏干ﾎ帷ｹｧ�､郢晢ｽｳ陞ｳ貅ｯ�｡蠕後＆郢ｧ�｢
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
    subtitle_font: str = "gothic",
    video_encoder: str = "libx264"
):
    try:
        def update_progress(val: int, msg: str):
            if is_async and session_id in progress_store:
                progress_store[session_id]["progress"] = val
                progress_store[session_id]["message"] = msg
            else:
                print(f"[{val}%] {msg}")

        # 1. 繧ｿ繧､繝�繝ｩ繧､繝ｳ讒狗ｯ�
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
        
        # 2. 蟄怜ｹ輔ョ繝ｼ繧ｿ縺ｮ繝代�繧ｹ�医ち繧､繝�繝ｩ繧､繝ｳ騾｣謳ｺ蟇ｾ蠢懶ｼ�
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
            
            # SRT繝輔ぃ繧､繝ｫ縺ｾ縺溘�SRT蠖｢蠑上�繝�く繧ｹ繝医�蝣ｴ蜷�
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
                # TXT繝輔ぃ繧､繝ｫ�医�繝ｬ繝ｼ繝ｳ繝�く繧ｹ繝茨ｼ峨�蝣ｴ蜷�: 譖ｲ縺ｮ髟ｷ縺輔↓蠢懊§縺ｦ蝮�ｭ芽�蜍暮�蛻�
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
            
        # --- 蟄怜ｹ輔そ繧ｯ繧ｷ繝ｧ繝ｳ縺ｮ螳悟�閾ｪ蜍墓ｭ｣隕丞喧 & 繧ｭ繝｣繝�す繝･ ---
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
                end_sec = start_sec + 0.5  # 譛菴取緒逕ｻ謖∫ｶ壽凾髢謎ｿ晁ｨｼ
                
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

        # 3. AI縺ｫ繧医ｋ繧ｨ繝輔ぉ繧ｯ繝郁�蜍暮｣謳ｺ�医が繝ｼ繧ｱ繧ｹ繝医Ξ繝ｼ繧ｷ繝ｧ繝ｳ�峨�繧ｿ繧､繝�繝ｩ繧､繝ｳ隗｣譫�
        orchestration_plan = None
        if enable_ai_orchestration:
            update_progress(42, "V5 Phase 1.5: Analyzing audio progress via Gemini for orchestration timeline...")
            try:
                available_filters = ["none"]
                filters_data = load_ai_filters()
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

        # 3b. Librosa 髻ｳ螢ｰ繧ｨ繝阪Ν繧ｮ繝ｼ隗｣譫� (Fallback)
        # 3b. 髻ｳ螢ｰ繧ｨ繝阪Ν繧ｮ繝ｼ隗｣譫� (Librosa繧剃ｽｿ繧上↑縺�ｮ牙�縺ｪ螳溯｣�)
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

        # 3c. 蜍慕噪螟芽ｪｿ繝輔Ξ繝ｼ繝�繧ｨ繝輔ぉ繧ｯ繝医ヵ繧｣繝ｫ繧ｿ繝ｼ縺ｮ螳溯｣�
        update_progress(50, "V5 Phase 2b: Attaching biometric overlay & energy-reactive pixel melter with kinetic animation...")
        
        # 髻ｳ螢ｰ迚ｹ蠕ｴ縺九ｉ縺昴�譎ょ綾縺ｮ繧ｨ繝阪Ν繧ｮ繝ｼ繧貞叙蠕励☆繧矩未謨ｰ
        def get_energy_at(t_val):
            idx = int(t_val / hop_seconds_energy)
            if idx < len(rms_norm):
                return float(rms_norm[idx])
            return 0.5

        verified_subtitles_logged = {}

        def frame_effect_filter(get_frame, t):
            raw_frame_copy = get_frame(t).copy()
            frame = raw_frame_copy.copy()
            
            # 迴ｾ蝨ｨ譎ょ綾 t 縺ｮ貍泌�繧ｻ繧ｯ繧ｷ繝ｧ繝ｳ�医が繝ｼ繧ｱ繧ｹ繝医Ξ繝ｼ繧ｷ繝ｧ繝ｳ�画爾邏｢
            active_sec = None
            if orchestration_plan:
                for sec in orchestration_plan.get("sections", []):
                    if sec.get("start") <= t <= sec.get("end"):
                        active_sec = sec
                        break
            
            # 蜷�ｼｷ蠎ｦ縺ｮ繝�ヵ繧ｩ繝ｫ繝亥､縺ｨ繧ｹ繧ｿ繧､繝ｫ險ｭ螳�
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
                
                filter_name = active_sec.get("active_filter", "none")
                if filter_name != "none":
                    filters_data = load_ai_filters()
                    if filter_name in filters_data:
                        current_filter_code = filters_data[filter_name]

            energy = get_energy_at(t)
            beat_interval = 60.0 / max(1.0, bpm)
            time_since_beat = (t - bpm_offset) % beat_interval
            decay = 12.0
            beat_signal = math.exp(-decay * time_since_beat)
            
            # 逕滉ｽ薙ョ繝ｼ繧ｿ縺ｮ繝弱う繧ｺ荳頑嶌縺肴緒逕ｻ (BPM蜷梧悄)
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

            # 逕滉ｽ薙が繝ｼ繝舌�繝ｬ繧､(BIOMETRIC OVERLAY)繧偵ヵ繝ｬ繝ｼ繝�縺ｫ蜿肴丐
            if enable_ecg or enable_grf or enable_hexdump:
                if current_biometric_opacity >= 0.99:
                    frame = overlay_frame
                elif current_biometric_opacity > 0:
                    frame = cv2.addWeighted(overlay_frame, current_biometric_opacity, frame, 1.0 - current_biometric_opacity, 0)
            
            # 縲植ntigravity Phase 2: 髱呎ｭ｢逕ｻ縺ｮ逕滉ｽ馴ｧ�虚蛹� (Kinetic Animation / 遨ｺ髢捺ｭｪ譖ｲ)縲�
            zoom_amp = 0.03 + 0.05 * energy
            zoom_scale = 1.0 + zoom_amp * beat_signal
            twitch_amp = 2.0 + 12.0 * energy * beat_signal
            wiggle_x = math.sin(t * 14.3) * twitch_amp + (np.random.rand() - 0.5) * 4.0 * energy
            wiggle_y = math.cos(t * 11.7) * twitch_amp + (np.random.rand() - 0.5) * 4.0 * energy

            # 繧｢繝輔ぅ繝ｳ螟画鋤陦悟�縺ｮ菴懈��医ン繝ｼ繝郁ц蜍輔�莨ｸ邵ｮ繝ｻ謠ｺ繧鯉ｼ�
            h_img, w_img = frame.shape[:2]
            center = (w_img / 2.0, h_img / 2.0)
            M = cv2.getRotationMatrix2D(center, 0, zoom_scale)
            M[0, 2] += wiggle_x
            M[1, 2] += wiggle_y

            # 繝ｪ繧ｺ繝�縺ｫ蠢懊§縺溘ヵ繝ｬ繝ｼ繝�螟牙ｽ｢ (BORDER_REFLECT_101)
            frame = cv2.warpAffine(frame, M, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            base_floor = 0.05 + 0.30 * energy
            modulated_signal = base_floor + (1.0 - base_floor) * beat_signal * energy

            # AI繧ｪ繝ｼ繧ｱ繧ｹ繝医Ξ繝ｼ繧ｷ繝ｧ繝ｳ縺ｾ縺溘�繝�ヵ繧ｩ繝ｫ繝医�繝代Λ繝｡繝ｼ繧ｿ縺ｧ螟芽ｪｿ
            dynamic_max_melt = max_melt * modulated_signal * current_melt_int
            dynamic_glitch_freq = glitch_freq * modulated_signal * current_glitch_int

            # single 繝｢繝ｼ繝画凾縺ｯ莉悶�繝吶�繧ｹ繧ｰ繝ｪ繝�メ貅ｶ蜃ｺ縺ｮ驥崎､�ｹｲ貂峨ｒ繧ｯ繝ｪ繧｢縺ｫ縺励∫ｴ皮ｲ九↑ AI Filter 蜊倡匱縺ｮ縺ｿ繧帝←逕ｨ
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
            
            # 隍�焚繝輔ぅ繝ｫ繧ｿ繝ｼ縺ｮ繝代う繝励Λ繧､繝ｳ螳溯｡� (繝槭Ν繝√そ繝ｬ繧ｯ繝亥ｯｾ蠢�)
            filter_list = []
            if ai_filter_codes_json:
                try:
                    import json
                    parsed = json.loads(ai_filter_codes_json)
                    if isinstance(parsed, list):
                        filter_list = [c for c in parsed if isinstance(c, str) and c.strip()]
                except Exception:
                    pass

            if not filter_list:
                effective_code = current_filter_code if current_filter_code else compiled_ai_filter
                if effective_code:
                    filter_list = [effective_code]

            # 蜊倡匱繝｢繝ｼ繝�1(single1), 蜊倡匱繝｢繝ｼ繝�2(single2), 隍�粋繝｢繝ｼ繝�(multi) 縺ｮ驕ｩ逕ｨ繝ｭ繧ｸ繝�け
            active_filters_to_run = filter_list

            if filter_exec_mode == "single1" or filter_exec_mode == "single":
                # 蜊倡匱繝｢繝ｼ繝�1: 驥阪�蜷医ｏ縺帙�荳蛻�○縺壹�∈謚槭＆繧後◆繝輔ぅ繝ｫ繧ｿ繝ｼ縺ｮ荳ｭ縺九ｉ蟆冗ｯ蜻ｨ譛溘�譖ｲ隱ｿ(Energy)縺ｫ蠢懊§縺ｦ驕ｩ螳�1縺､縺�縺代ｒ蜊倡匱驕ｩ逕ｨ
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
                # 蜊倡匱繝｢繝ｼ繝�2: 險ｭ螳壹＆繧後◆BPM縺ｫ蟇ｾ縺吶ｋ險ｭ螳壼ｰ冗ｯ譎る俣(SLIDES CHANGEOVER RATE)縺ｴ縺｣縺溘ｊ縺ｧ鬆�ｬ｡1縺､縺壹▽蜊倡匱蛻�ｊ譖ｿ縺�
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
                # 隍�粋繝｢繝ｼ繝�: 驕ｸ謚槭＆繧後◆蜈ｨ縺ｦ縺ｮ繝輔ぅ繝ｫ繧ｿ繝ｼ繧帝㍾縺ｭ蜷医ｏ縺�(繧ｹ繧ｿ繝�け)驕ｩ逕ｨ
                active_filters_to_run = filter_list

            for code_item in active_filters_to_run:
                # 譏守､ｺ逧�↑繝繝溘�繧ｳ繝ｼ繝峨′豺ｷ蜈･縺励◆蝣ｴ蜷医�繝輔ぉ繧､繝ｫ繧ｻ繝ｼ繝�
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

        # Phase 2-A: AI繝輔ぅ繝ｫ繧ｿ繝ｼ��ン繧ｸ繝･繧｢繝ｫ繧ｨ繝輔ぉ繧ｯ繝亥�逅� (閭梧勹繝ｻ繧ｨ繝輔ぉ繧ｯ繝医Ξ繧､繝､繝ｼ)
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
                    bpm_offset=bpm_offset,
                    intensity=current_glitch_int,
                    start_time=sec_start_sec,
                    font_name=subtitle_font,
                    return_meta=True
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
        
        # 4. 鬮ｻ�ｳ陞｢�ｰ邵ｺ�ｮ驍ｨ莉咏ｲ狗ｸｺ�ｨ陷�ｽｺ陷会ｿｽ
        update_progress(70, "Phase 3: Attaching audio and exporting final H.264 video...")
        
        from moviepy import AudioFileClip
        audio_clip = AudioFileClip(audio_path)
        final_clip = final_clip.with_audio(audio_clip)
        
        # 繝ｬ繝ｳ繝繝ｪ繝ｳ繧ｰ螳溯｡� (CustomMoviePyLogger 縺ｧ繝ｪ繧｢繝ｫ繧ｿ繧､繝�騾ｲ謐励ｒ繧ｭ繝｣繝�メ)
        custom_logger = CustomMoviePyLogger(update_progress_fn=update_progress)
        target_codec = video_encoder if video_encoder else "libx264"
        print(f"[Export] Starting video export with encoder: '{target_codec}' (FPS={fps})...")

        try:
            final_clip.write_videofile(
                output_path,
                fps=fps,
                codec=target_codec,
                audio_codec="aac",
                ffmpeg_params=["-pix_fmt", "yuv420p"],
                logger=custom_logger
            )
        except Exception as export_err:
            if target_codec != "libx264":
                print(f"[Export Warning] Hardware encoder '{target_codec}' failed ({export_err}). Falling back to CPU encoder 'libx264'...")
                update_progress(75, f"GPU encoder ({target_codec}) unavailable. Falling back to CPU (libx264)...")
                final_clip.write_videofile(
                    output_path,
                    fps=fps,
                    codec="libx264",
                    audio_codec="aac",
                    ffmpeg_params=["-pix_fmt", "yuv420p"],
                    logger=custom_logger
                )
            else:
                raise export_err
        
        # 邨ゆｺ��逅�
        final_clip.close()
        base_clip.close()
        audio_clip.close()
        
        update_progress(100, "Rendering finished successfully!")
        if is_async and session_id in progress_store:
            progress_store[session_id]["status"] = "completed"
            progress_store[session_id]["output_path"] = output_path
            
    except Exception as e:
        import traceback
        import sys
        err = traceback.format_exc()
        print(f"[Error] Pipeline failure: {err}")
        sys.stdout.flush()
        if is_async and session_id in progress_store:
            # 邁｡貎斐↓繧ｹ繧ｿ繝�け繝医Ξ繝ｼ繧ｹ縺ｮ譛蠕後�3陦後ｒ蜿門ｾ励＠縺ｦ繝｡繝�そ繝ｼ繧ｸ縺ｫ蜷ｫ繧√ｋ
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
            # mm:ss.hh 陟厄ｽ｢陟台ｸ奇ｿｽ陜｣�ｴ陷ｷ�ｽ hh (1/100驕假ｿｽ) 邵ｺ�ｪ邵ｺ�ｮ邵ｺ�ｧ100邵ｺ�ｧ陷托ｽｲ郢ｧ�ｽ
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
        # 驕ｨ�ｺ隴�ｿｽ�ｭ蜉ｱ�ｽ陜｣�ｴ陷ｷ蛹ｻ�ｽ隶灘綜�ｺ髢蝟ｧ
        api_key_clean = api_key.strip() if api_key else None
        
        # 郢ｧ繧�ｼ�陷茨ｽ･陷牙ｸ呻ｼ�ｹｧ蠕娯螺郢ｧ�ｭ郢晢ｽｼ邵ｺ讙寂伯陷会ｽｹ邵ｺ�ｪ陟厄ｽ｢陟第得�ｼ�ｽIzaSy邵ｺ�ｧ陝倶ｹ昶穐郢ｧ蟲ｨ竊醍ｸｺ�ｽ�ｼ蟲ｨ縲堤ｸｺ繧�ｽ檎ｸｺ�ｰ邵ｲ竏ｫ笏碁囎謔ｶ��邵ｺ�ｦ霑ｺ�ｰ陟�ｿｽ�､逕ｻ辟夂ｹ晁ｼ斐°郢晢ｽｼ郢晢ｽｫ郢晁�繝｣郢ｧ�ｯ郢ｧ蝣､蛹ｱ陷搾ｿｽ
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

        prompt_text = f"User Request: {prompt}\n"
        if current_code:
            prompt_text += f"\nModify or Refine the following existing code:\n```python\n{current_code}\n```"

        model_candidates = [primary_model, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        response = None
        last_exception = None

        for m_name in model_candidates:
            if not m_name:
                continue
            try:
                model = genai.GenerativeModel(
                    model_name=m_name,
                    system_instruction=system_instruction
                )
                response = model.generate_content(prompt_text)
                if response and response.text:
                    print(f"[Success] Python AI Filter generated with model: {m_name}")
                    break
            except Exception as candidate_err:
                print(f"[Warn] Model '{m_name}' failed for filter generation: {candidate_err}")
                last_exception = candidate_err

        if not response or not response.text:
            raise last_exception or RuntimeError("All Gemini AI models failed to respond.")

        code = response.text.strip()
        
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
            
        return {"status": "success", "code": code.strip()}
    except Exception as e:
        print(f"[Error] /generate-filter-code failed: {e}")
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
# AI 繝輔ぅ繝ｫ繧ｿ繝ｼ菫晏ｭ倥♀繧医�閾ｪ蟾ｱ騾ｲ蛹� (閾ｪ蟾ｱ蠅玲ｮ�) API
# ==========================================



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
    subtitle_font: str = Form("gothic"),
    video_encoder: str = Form("libx264")
):
    # 譌｢蟄倥そ繝�す繝ｧ繝ｳ縺ｮ繧ｯ繝ｪ繝ｼ繝ｳ繧｢繝��
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
    
    # 髻ｳ讌ｽ繝輔ぃ繧､繝ｫ縺ｮ菫晏ｭ�
    audio_ext = os.path.splitext(audio.filename)[1] or ".mp3"
    temp_audio_path = os.path.join(session_dir, f"input_audio{audio_ext}")
    with open(temp_audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
        
    # 豁瑚ｩ槭ヵ繧｡繧､繝ｫ縺ｮ菫晏ｭ�
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
            
    # 閭梧勹繧｢繧ｻ繝�ヨ縺ｮ菫晏ｭ�
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
        subtitle_font=subtitle_font,
        video_encoder=video_encoder
    )
    
    return {"session_id": session_id}

@app.get("/status/{session_id}")
async def api_get_status(session_id: str):
    if session_id in progress_store:
        return progress_store[session_id]
    return {"status": "not_found", "progress": 0, "message": "Session not found"}

@app.get("/download/{session_id}")
async def api_download_result(session_id: str):
    # 1. progress_store からの探索
    if session_id in progress_store and "output_path" in progress_store[session_id]:
        output_path = progress_store[session_id]["output_path"]
        if os.path.exists(output_path):
            return FileResponse(output_path, media_type="video/mp4", filename=os.path.basename(output_path))
    
    # 2. 一時ディレクトリ構造からの直接安全探索
    temp_dir = tempfile.gettempdir()
    candidate_path = os.path.join(temp_dir, f"amvg_v2_{session_id}", f"output_v2_{session_id}.mp4")
    if os.path.exists(candidate_path):
        return FileResponse(candidate_path, media_type="video/mp4", filename=f"AMVG_render_{session_id[:8]}.mp4")
        
    raise HTTPException(status_code=404, detail="File not found")

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
            # 繝�ヵ繧ｩ繝ｫ繝医�繝�せ繝育畑繝輔Ξ繝ｼ繝�逕滓� (720x1280, BGR)
            h, w = 720, 1280
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                r = int(20 + 50 * (y / h))
                g = int(10 + 30 * (y / h))
                b = int(40 + 90 * (y / h))
                frame[y, :] = (b, g, r)
            
            # 繧ｰ繝ｪ繝�ラ謠冗判
            for x in range(0, w, 80):
                cv2.line(frame, (x, 0), (x, h), (40, 60, 80), 1)
            for y in range(0, h, 80):
                cv2.line(frame, (0, y), (w, y), (40, 60, 80), 1)

            cv2.putText(frame, "A.M.V.G v5 MULTI-FILTER PREVIEW", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 243, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"STATE: t={t:.2f}s | BPM={bpm:.1f} | MEASURES={slide_measures}", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 127), 2, cv2.LINE_AA)
            cv2.circle(frame, (w // 2, h // 2), 160, (0, 243, 255), 4)

        # AI Filter 繧ｳ繝ｼ繝峨Μ繧ｹ繝医�繝代�繧ｹ縺ｨ隍�焚驕ｩ逕ｨ
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
            raise ValueError("譛牙柑縺ｪ Python AI Filter 繧ｳ繝ｼ繝峨′驕ｸ謚槭＆繧後※縺�∪縺帙ｓ縲�")

        # 蜊倡匱繝｢繝ｼ繝�1(single1), 蜊倡匱繝｢繝ｼ繝�2(single2), 隍�粋繝｢繝ｼ繝�(multi) 縺ｮ驕ｩ逕ｨ繝ｭ繧ｸ繝�け
        active_filters_to_run = filter_list

        if filter_exec_mode == "single1" or filter_exec_mode == "single":
            # 蜊倡匱繝｢繝ｼ繝�1: 驥阪�蜷医ｏ縺帙�荳蛻�○縺壹�∈謚槭＆繧後◆繝輔ぅ繝ｫ繧ｿ繝ｼ縺ｮ荳ｭ縺九ｉ蟆冗ｯ蜻ｨ譛溘�譖ｲ隱ｿ(Energy)縺ｫ蠢懊§縺ｦ驕ｩ螳�1縺､縺�縺代ｒ蜊倡匱驕ｩ逕ｨ
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
            # 蜊倡匱繝｢繝ｼ繝�2: 險ｭ螳壹＆繧後◆BPM縺ｫ蟇ｾ縺吶ｋ險ｭ螳壼ｰ冗ｯ譎る俣(SLIDES CHANGEOVER RATE)縺ｴ縺｣縺溘ｊ縺ｧ鬆�ｬ｡1縺､縺壹▽蜊倡匱蛻�ｊ譖ｿ縺�
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
            # 隍�粋繝｢繝ｼ繝�: 驕ｸ謚槭＆繧後◆蜈ｨ縺ｦ縺ｮ繝輔ぅ繝ｫ繧ｿ繝ｼ繧帝㍾縺ｭ蜷医ｏ縺�(繧ｹ繧ｿ繝�け)驕ｩ逕ｨ
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
        print(f"[Error] /preview_filter failed:\n{last_filter_error_log}")
        
        h, w = 480, 854
        err_frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(err_frame, "FILTER RUNTIME ERROR", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        err_lines = err_detail.splitlines()
        for idx, line in enumerate(err_lines[:8]):
            cv2.putText(err_frame, line[:75], (30, 110 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 255), 1, cv2.LINE_AA)
            
        _, encoded_img = cv2.imencode(".jpg", err_frame)
        headers = {
            "X-Filter-Error": "true",
            "X-Filter-Error-Message": err_detail.replace("\n", " ")[:200]
        }
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg", headers=headers)

@app.post("/shutdown")
async def api_shutdown():
    def kill_process():
        time.sleep(1)
        os._exit(0)
    
    threading.Thread(target=kill_process, daemon=True).start()
    return {"status": "success", "message": "SYSTEM SHUTDOWN SEQUENCE INITIATED"}

@app.get("/filters")
async def api_get_filters():
    return load_ai_filters()

@app.post("/save_filter")
async def api_save_filter(name: str = Form(...), code: str = Form(...)):
    filters = load_ai_filters()
    filters[name] = code
    save_ai_filters(filters)
    return {"status": "success", "message": f"Filter '{name}' saved successfully!"}

@app.post("/rename_filter")
async def api_rename_filter(old_name: str = Form(...), new_name: str = Form(...)):
    old_name = old_name.strip()
    new_name = new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="譁ｰ縺励＞繝輔ぅ繝ｫ繧ｿ繝ｼ蜷阪′遨ｺ縺ｧ縺吶€")

    filters_data = load_ai_filters()
    if old_name not in filters_data:
        raise HTTPException(status_code=404, detail=f"繝輔ぅ繝ｫ繧ｿ繝ｼ '{old_name}' 縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ縲�")

    new_data = {}
    for k, v in filters_data.items():
        if k == old_name:
            new_data[new_name] = v
        else:
            new_data[k] = v

    save_ai_filters(new_data)
    return {"status": "success", "message": f"繝輔ぅ繝ｫ繧ｿ繝ｼ蜷阪ｒ '{old_name}' 縺九ｉ '{new_name}' 縺ｫ螟画峩縺励∪縺励◆縲�", "filters": new_data}

@app.post("/auto_repair_filter")
async def api_auto_repair_filter(
    code: str = Form(...),
    error_log: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    primary_model: Optional[str] = Form(None)
):
    global last_filter_error_log
    effective_error_log = error_log.strip() if error_log and error_log.strip() else last_filter_error_log
    if not effective_error_log:
        effective_error_log = "Runtime execution error occurred during filter preview or frame rendering."

    active_api_key = api_key if api_key and api_key.strip() else os.environ.get("GEMINI_API_KEY")
    if not active_api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key 縺瑚ｨｭ螳壹＆繧後※縺∪縺帙ｓ縲ら腸蠅､画焚 GEMINI_API_KEY 縺ｾ縺溘逕ｻ髱｢縺ｮAPI繧ｭ繝ｼ谺↓蜈･蜉帙＠縺ｦ縺上□縺輔＞縲")

    import google.generativeai as genai
    genai.configure(api_key=active_api_key)

    system_instruction = """縺ゅ↑縺溘繧ｨ繧ｭ繧ｹ繝代繝 Python (OpenCV / NumPy) 髢狗匱閠〒縺吶€
荳弱∴繧峨ｌ縺 Python AI 繝輔ぅ繝ｫ繧ｿ繝ｼ繧ｳ繝ｼ繝峨↓蟄伜惠縺吶ｋ螳溯｡梧凾繧ｨ繝ｩ繝ｼ繧ｧ区枚繧ｨ繝ｩ繝ｼ繧剃ｿｮ豁｣縺励※縺上□縺輔＞縲

縲仙宍譬ｼ縺ｪ謖､ｺ縲
1. 蜃ｺ蜉帙邏皮ｲ九↑ Python 繧ｳ繝ｼ繝峨縺ｿ縺ｫ縺励※縺上□縺輔＞縲りｪｬ譏取枚繧繝ｼ繧ｯ繝€繧ｦ繝ｳ險伜捷``python峨荳€蛻性繧√↑縺〒縺上□縺輔＞縲
2. `apply_ai_filter(frame: np.ndarray, t: float, duration: float, bpm: float, energy: float) -> np.ndarray` 縺ｮ髢｢謨ｰ螳夂ｾｩ繧堤ｶｭ謖√＠縺ｦ縺上□縺輔＞縲
3. 蠢ｦ√Λ繧､繝悶Λ繝ｪ (cv2, np, math, random遲) 繧呈ｭ｣縺励￥繧､繝ｳ繝昴繝医＠縺ｦ縺上□縺輔＞縲
"""

    prompt = f"""
【エラー詳細ログ】
{effective_error_log}

【修正対象コード (コード表示欄の最新コード)】
{code}
"""

    model_candidates = [primary_model, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    response = None
    last_err = None

    for m_name in model_candidates:
        if not m_name:
            continue
        try:
            model = genai.GenerativeModel(
                model_name=m_name,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            if response and response.text:
                print(f"[Success] Filter auto-repaired using model: {m_name}")
                break
        except Exception as candidate_err:
            print(f"[Warn] Auto repair with model '{m_name}' failed: {candidate_err}")
            last_err = candidate_err

    if not response or not response.text:
        raise HTTPException(status_code=500, detail=f"Gemini APIによる自動修正に失敗しました: {last_err}")

    repaired_code = response.text.strip()
    if repaired_code.startswith("```python"):
        repaired_code = repaired_code[9:]
    elif repaired_code.startswith("```"):
        repaired_code = repaired_code[3:]
    if repaired_code.endswith("```"):
        repaired_code = repaired_code[:-3]

    return {"status": "success", "repaired_code": repaired_code.strip()}

@app.get("/", response_class=HTMLResponse)
async def api_webui():
    index_path = get_external_data_path("index.html")
    if not os.path.exists(index_path):
        index_path = get_resource_path("index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>AMVG index.html not found</h1>", status_code=404)

# ==========================================
# CLI / メイン起動
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
    parser.add_argument("--encoder", type=str, choices=["libx264", "h264_nvenc"], default="libx264", help="動画エンコーダー (libx264: CPU, h264_nvenc: NVIDIA GPU加速)")
    
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
            video_encoder=args.encoder,
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
