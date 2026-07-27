# -*- coding: utf-8 -*-
"""
A.M.V.G v2 - pixel_melter.py
視覚の脱水・融解モジュール
輝度閾値に基づく縦・横方向のピクセルソートと、曲の後半に向けて指数関数的にグリッチ強度が高まる融解アルゴリズムを実装する
"""

import numpy as np
from PIL import Image, ImageChops

def get_melt_intensity(t: float, total_duration: float, max_melt: float = 1.0) -> float:
    """
    時間 t に応じた指数関数的な融解強度 (0.0 ～ 1.0) を計算する。
    曲の後半(特にアウトロ)に向けて急激に強度が上昇する。
    """
    progress = t / total_duration
    # 指数関数 (progress の4乗)
    intensity = (progress ** 4.0) * max_melt
    return float(intensity)

def pixel_sort_vertical(img_np: np.ndarray, intensity: float) -> np.ndarray:
    """
    輝度ベースの縦方向（下方向）ピクセルソート
    """
    if intensity < 0.05:
        return img_np
        
    h, w, c = img_np.shape
    out = img_np.copy()
    
    # 強度に応じて処理する列数を決定 (最大で全列の 35%)
    num_columns = int(w * 0.35 * intensity)
    if num_columns <= 0:
        return out
        
    # ランダムな列を選択
    cols = np.random.choice(w, num_columns, replace=False)
    
    # 輝度閾値 (0..255)
    # 強度が高くなるにつれて閾値を下げ、より広範囲を融解させる
    threshold = 130.0 - 50.0 * intensity
    
    for x in cols:
        col_pixels = out[:, x, :3]
        # 簡易的にR+G+Bの平均で輝度を計算
        brightness = np.sum(col_pixels, axis=1) / 3.0
        
        # 閾値を超える区間（マスク）
        mask = brightness > threshold
        
        # 連続するTrueの区間（セグメント）を検出
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        if mask[0]:
            starts = np.insert(starts, 0, 0)
        ends = np.where(diff == -1)[0] + 1
        if mask[-1]:
            ends = np.append(ends, h)
            
        # 各区間内を輝度順にソートする
        for start, end in zip(starts, ends):
            if end - start > 4:
                segment = col_pixels[start:end]
                seg_brightness = np.sum(segment, axis=1)
                # ソートインデックス取得
                sort_idx = np.argsort(seg_brightness)
                out[start:end, x, :3] = segment[sort_idx]
                
    return out

def pixel_sort_horizontal(img_np: np.ndarray, intensity: float) -> np.ndarray:
    """
    輝度ベースの横方向ピクセルソート
    """
    if intensity < 0.05:
        return img_np
        
    h, w, c = img_np.shape
    out = img_np.copy()
    
    # 強度に応じて処理する行数を決定 (最大で全行の 30%)
    num_rows = int(h * 0.30 * intensity)
    if num_rows <= 0:
        return out
        
    # ランダムな行を選択
    rows = np.random.choice(h, num_rows, replace=False)
    threshold = 120.0 - 40.0 * intensity
    
    for y in rows:
        row_pixels = out[y, :, :3]
        brightness = np.sum(row_pixels, axis=1) / 3.0
        
        mask = brightness > threshold
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        if mask[0]:
            starts = np.insert(starts, 0, 0)
        ends = np.where(diff == -1)[0] + 1
        if mask[-1]:
            ends = np.append(ends, w)
            
        for start, end in zip(starts, ends):
            if end - start > 4:
                segment = row_pixels[start:end]
                seg_brightness = np.sum(segment, axis=1)
                sort_idx = np.argsort(seg_brightness)
                out[y, start:end, :3] = segment[sort_idx]
                
    return out

def apply_chromatic_destruction(img_rgba: Image.Image, intensity: float) -> Image.Image:
    """
    後半のオーバードライブ時に、激しく色収差（R/G/Bのズレ）を発生させる
    """
    if intensity < 0.2:
        return img_rgba
    r, g, b, a = img_rgba.split()
    
    # ズレの幅を強度の指数関数的に大きくする
    shift_max = int(5 + 40 * (intensity ** 2.0))
    shift_r = (np.random.randint(-shift_max, shift_max + 1), np.random.randint(-shift_max, shift_max + 1))
    shift_g = (np.random.randint(-shift_max, shift_max + 1), np.random.randint(-shift_max, shift_max + 1))
    shift_b = (np.random.randint(-shift_max, shift_max + 1), np.random.randint(-shift_max, shift_max + 1))
    
    r_shifted = ImageChops.offset(r, shift_r[0], shift_r[1])
    g_shifted = ImageChops.offset(g, shift_g[0], shift_g[1])
    b_shifted = ImageChops.offset(b, shift_b[0], shift_b[1])
    
    return Image.merge("RGBA", (r_shifted, g_shifted, b_shifted, a))

def melt_frame(
    frame_np: np.ndarray,
    t: float,
    total_duration: float,
    max_melt: float = 1.0,
    glitch_freq: float = 1.0
) -> np.ndarray:
    """
    現在の時間 t における指数強度のピクセルソートおよび融解エフェクトを適用する
    """
    intensity = get_melt_intensity(t, total_duration, max_melt)
    
    # 強度が極めて低いうちは何もしない
    if intensity < 0.05:
        return frame_np
        
    # NumPy 配列に対する縦・横方向のピクセルソート適用
    out_np = frame_np.copy()
    
    # 進行度に応じて縦と横両方を組み合わせる
    # 縦方向 (重力融解)
    out_np = pixel_sort_vertical(out_np, intensity)
    
    # 後半(強度 0.4以上)は横方向の遠心融解も重ねる
    if intensity > 0.4:
        out_np = pixel_sort_horizontal(out_np, intensity * 0.8)
        
    # 色の破壊的色収差 (PILを使用)
    if intensity > 0.3:
        img = Image.fromarray(out_np).convert('RGBA')
        img = apply_chromatic_destruction(img, intensity)
        out_np = np.array(img.convert('RGB'))
        
    # 後半になるほどランダムにホワイトアウトやブラックアウトのグリッチノイズを乗せる
    if intensity > 0.7 and np.random.rand() < (0.15 * glitch_freq):
        # 強烈な光フラッシュグリッチ
        out_np = np.clip(out_np.astype(float) * 1.5, 0, 255).astype(np.uint8)
    elif intensity > 0.85 and np.random.rand() < (0.1 * glitch_freq):
        # 映像途切れブラックアウト
        out_np = (out_np.astype(float) * 0.05).astype(np.uint8)
        
    return out_np
