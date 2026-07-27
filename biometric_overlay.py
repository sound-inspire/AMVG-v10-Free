# -*- coding: utf-8 -*-
"""
A.M.V.G v2 - biometric_overlay.py
生体データのノイズ・オーバードライブモジュール
OpenCV と NumPy を用いて、床反力ベクトル、心拍変動波形、16進メモリダンプなどの理学療法風ノイズをプログラマティックに上書き合成する
"""

import cv2
import math
import numpy as np
import random

# 16進ダンプやシステムログ用の静的データ
SYSTEM_LOGS = [
    "[BIOMETRIC] INITIATING EEG DETECTOR... OK",
    "[SIGNAL] SIGNAL STRENGTH: 98.4%",
    "[COP] SHIFTING PRESSURE TO METATARSAL HEAD",
    "[GRF] VERTICAL FORCE EXCEEDS BODY_WEIGHT (1.2G)",
    "[HRV] HEART RATE VARIABILITY DECREASED",
    "[ALERT] PLASTICITY THRESHOLD EXCEEDED",
    "[SYSTEM] NEURAL CORE LOAD: 87.2%",
    "[WARN] MEMORY OVERFLOW IN COMPARTMENT 0x07",
    "[BIOMETRIC] SYMPATHETIC NERVOUS SYSTEM: OVERDRIVE",
    "[GRF] FOREFOOT CONTACT DETECTED // TRANSITION",
    "[EEG] ALPHA WAVE SUPPRESSION ACTIVE",
    "[SIGNAL] EMG AMPLITUDE RANGE EXCEEDED IN TIBIALIS ANTERIOR"
]

def generate_ecg_point(t: float, bpm: float) -> float:
    """
    時間 t (秒) における心電図 (ECG) の振幅をシミュレートする (P-Q-R-S-T 波)
    """
    period = 60.0 / bpm
    cycle_t = t % period
    
    val = 0.0
    
    # 拍動のベースラインノイズ
    val += math.sin(t * 50.0) * 0.02
    
    # P波 (小さな山)
    p_pos = 0.15 * period
    p_width = 0.05 * period
    if abs(cycle_t - p_pos) < p_width:
        val += 0.15 * math.cos((cycle_t - p_pos) / p_width * math.pi / 2) ** 2
        
    # QRS波 (鋭い谷 -> 高い山 -> 谷)
    qrs_pos = 0.25 * period
    q_pos = qrs_pos - 0.02 * period
    r_pos = qrs_pos
    s_pos = qrs_pos + 0.02 * period
    
    if cycle_t < qrs_pos + 0.04 * period and cycle_t > qrs_pos - 0.04 * period:
        # R波 (スパイク)
        r_width = 0.015 * period
        if abs(cycle_t - r_pos) < r_width:
            val += 1.0 * math.cos((cycle_t - r_pos) / r_width * math.pi / 2) ** 2
        # Q波 (小さな下向き)
        q_width = 0.01 * period
        if abs(cycle_t - q_pos) < q_width:
            val -= 0.2 * math.cos((cycle_t - q_pos) / q_width * math.pi / 2) ** 2
        # S波 (やや深い下向き)
        s_width = 0.015 * period
        if abs(cycle_t - s_pos) < s_width:
            val -= 0.35 * math.cos((cycle_t - s_pos) / s_width * math.pi / 2) ** 2
            
    # T波 (やや大きめの緩やかな山)
    t_pos = 0.45 * period
    t_width = 0.08 * period
    if abs(cycle_t - t_pos) < t_width:
        val += 0.3 * math.cos((cycle_t - t_pos) / t_width * math.pi / 2) ** 2
        
    return val

def apply_biometric_overlay(
    frame_np: np.ndarray,
    t: float,
    total_duration: float,
    aspect_ratio: str = "16:9",
    enable_ecg: bool = True,
    enable_grf: bool = True,
    enable_hexdump: bool = True,
    bpm: float = 120.0,
    bpm_offset: float = 0.0
) -> np.ndarray:
    """
    フレーム（RGB NumPy配列）に、床反力ベクトル、心拍変動波形、16進システムログ等の生体データノイズを合成する
    """
    img = frame_np.copy()
    h, w, c = img.shape
    scale = h / 720.0
    
    color_cyan = (0, 243, 255)
    color_magenta = (255, 0, 127)
    color_green = (57, 255, 20)
    color_white = (255, 255, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4 * scale
    thickness = max(1, int(1 * scale))

    # ==========================================
    # 1. 心拍変動 (HRV / ECG) オシロスコープ波形描画
    # ==========================================
    if enable_ecg:
        # 設定された BPM をベースにしつつ、わずかに自然な揺らぎを加える
        active_bpm = bpm + 2.0 * math.sin(t * 0.2)
        period = 60.0 / active_bpm
        
        ecg_x = int(w * 0.05)
        ecg_y = int(h * 0.12)
        ecg_w = int(w * 0.35)
        ecg_h = int(h * 0.08)
        
        # 背景枠
        cv2.rectangle(img, (ecg_x, ecg_y - ecg_h), (ecg_x + ecg_w, ecg_y), (15, 15, 25), -1)
        for gx in range(ecg_x, ecg_x + ecg_w, int(20 * scale)):
            cv2.line(img, (gx, ecg_y - ecg_h), (gx, ecg_y), (35, 35, 55), 1)
        for gy in range(ecg_y - ecg_h, ecg_y, int(15 * scale)):
            cv2.line(img, (ecg_x, gy), (ecg_x + ecg_w, gy), (35, 35, 55), 1)
            
        points = []
        step_px = 2
        n_points = ecg_w // step_px
        for i in range(n_points):
            pt_t_raw = t - (n_points - i) * 0.008
            pt_t = pt_t_raw - bpm_offset + 0.25 * period
            amp = generate_ecg_point(pt_t, active_bpm)
            px = ecg_x + i * step_px
            py = int(ecg_y - (ecg_h / 2) - amp * (ecg_h * 0.4))
            py = max(ecg_y - ecg_h + 2, min(ecg_y - 2, py))
            points.append((px, py))
            
        if len(points) > 1:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, color_green, thickness=max(1, int(1.5 * scale)), lineType=cv2.LINE_AA)
            
        cv2.putText(img, f"HR: {bpm:.1f} BPM  HRV: 48.2ms", (ecg_x + 5, ecg_y - ecg_h + 12), font, font_scale, color_green, thickness)
        cv2.putText(img, "SYS.ECG // MONITOR ACTIVE", (ecg_x + 5, ecg_y - 5), font, font_scale * 0.8, (120, 255, 120), thickness)

    # ==========================================
    # 2. 床反力ベクトル (GRF) / COP 描画
    # ==========================================
    if enable_grf:
        grf_cx = int(w * 0.8)
        grf_cy = int(h * 0.78)
        grf_r = int(w * 0.08)
        
        cv2.circle(img, (grf_cx, grf_cy), grf_r, (40, 40, 65), 1)
        cv2.line(img, (grf_cx - grf_r, grf_cy), (grf_cx + grf_r, grf_cy), (40, 40, 65), 1)
        cv2.line(img, (grf_cx, grf_cy - grf_r), (grf_cx, grf_cy + grf_r), (40, 40, 65), 1)

        cop_angle = t * 4.0
        cop_dx = math.sin(cop_angle) * (grf_r * 0.5)
        cop_dy = math.cos(cop_angle * 0.5) * (grf_r * 0.4)
        cop_x = int(grf_cx + cop_dx)
        cop_y = int(grf_cy + cop_dy)
        
        cv2.circle(img, (cop_x, cop_y), int(4 * scale), color_magenta, -1)
        
        force_z = 600.0 + 180.0 * math.sin(cop_angle * 2.0)
        force_x = cop_dx * 3.0
        force_y = cop_dy * 2.0
        
        arrow_len = int((force_z / 1000.0) * (grf_r * 1.2))
        arrow_dx = int(-force_x * 0.2 * scale)
        arrow_dy = int(-arrow_len)
        
        arrow_end_x = cop_x + arrow_dx
        arrow_end_y = cop_y + arrow_dy
        
        cv2.arrowedLine(img, (cop_x, cop_y), (arrow_end_x, arrow_end_y), color_cyan, thickness=max(1, int(2.5 * scale)), tipLength=0.2)
        
        txt_y = grf_cy - grf_r - 10
        cv2.putText(img, f"GRF VERTICAL (Fz): {force_z:.1f} N", (grf_cx - grf_r, txt_y), font, font_scale, color_cyan, thickness)
        cv2.putText(img, f"GRF SHEAR (Fx/Fy): ({force_x:.1f}, {force_y:.1f}) N", (grf_cx - grf_r, txt_y + 15), font, font_scale * 0.9, color_cyan, thickness)
        cv2.putText(img, f"COP (X/Y): ({cop_dx:.2f}, {cop_dy:.2f})", (grf_cx - grf_r, txt_y + 30), font, font_scale * 0.9, color_magenta, thickness)
        cv2.ellipse(img, (grf_cx, grf_cy), (int(grf_r * 0.3), int(grf_r * 0.7)), 0, 0, 360, (60, 60, 90), 1)

    # ==========================================
    # 3. 16進メモリダンプ・システムログのスクロール描画
    # ==========================================
    if enable_hexdump:
        log_x = int(w * 0.05)
        log_y_start = int(h * 0.8)
        n_logs = 6
        log_idx_base = int(t * 1.5)
        
        for i in range(n_logs):
            log_idx = (log_idx_base + i) % len(SYSTEM_LOGS)
            log_text = SYSTEM_LOGS[log_idx]
            
            addr = 0x7FFA0B00 + (log_idx_base * 0x10) + (i * 0x40)
            val1 = (addr & 0xFF) ^ 0xAA
            val2 = ((addr >> 8) & 0xFF) ^ 0x55
            dump_prefix = f"0x{addr:08X}: {val1:02X} {val2:02X} E8 90"
            full_line = f"{dump_prefix} | {log_text}"
            
            draw_y = log_y_start + i * int(20 * scale)
            if draw_y < h - 10:
                if "WARN" in log_text or "ALERT" in log_text:
                    line_color = color_magenta
                elif "SYSTEM" in log_text:
                    line_color = color_cyan
                else:
                    line_color = (0, 180, 255)
                    
                cv2.putText(img, full_line, (log_x, draw_y), font, font_scale * 0.9, line_color, thickness)

    # ==========================================
    # 4. 前面デバッグUI (ターゲット照準、カメラブラケット)
    # ==========================================
    # ECG/GRF/HexDumpが1つ以上有効な場合にデバッグブラケットを描画する
    if enable_ecg or enable_grf or enable_hexdump:
        margin = int(15 * scale)
        corner_len = int(25 * scale)
        # 左上
        cv2.line(img, (margin, margin), (margin + corner_len, margin), color_cyan, thickness)
        cv2.line(img, (margin, margin), (margin, margin + corner_len), color_cyan, thickness)
        # 右上
        cv2.line(img, (w - margin, margin), (w - margin - corner_len, margin), color_cyan, thickness)
        cv2.line(img, (w - margin, margin), (w - margin, margin + corner_len), color_cyan, thickness)
        # 左下
        cv2.line(img, (margin, h - margin), (margin + corner_len, h - margin), color_cyan, thickness)
        cv2.line(img, (margin, h - margin), (margin, h - margin - corner_len), color_cyan, thickness)
        # 右下
        cv2.line(img, (w - margin, h - margin), (w - margin - corner_len, h - margin), color_cyan, thickness)
        cv2.line(img, (w - margin, h - margin), (w - margin, h - margin - corner_len), color_cyan, thickness)

        # 照準クロス
        cx, cy = w // 2, h // 2
        cross_sz = int(8 * scale)
        cv2.line(img, (cx - cross_sz, cy), (cx + cross_sz, cy), color_cyan, 1)
        cv2.line(img, (cx, cy - cross_sz), (cx, cy + cross_sz), color_cyan, 1)

        # ヘッダー情報
        cv2.putText(img, "SYS.MODE: OVERDRIVE_MULTIPLE", (int(w * 0.65), int(h * 0.06)), font, font_scale, color_magenta, thickness)
        progress_pct = (t / total_duration) * 100
        cv2.putText(img, f"PROG: {progress_pct:.2f}% (t={t:.2f}s/{total_duration:.1f}s)", (int(w * 0.65), int(h * 0.09)), font, font_scale, color_white, thickness)

    return img
