# -*- coding: utf-8 -*-
"""
A.M.V.G V10 Free - metadata_generator.py
Gemini API 歌詞解析 ＆ 音声マルチモーダル同期タイミング決定モジュール (Phase 1)
"""

import os
import re
import json
import time
from moviepy import AudioFileClip

# Google GenAI SDK (google.generativeai) のインポート
import google.generativeai as genai

# Geminiモデル優先順位定義 (音声入力対応のため、基本的には3.5-flash / 2.5-flashなどを使用)
MODELS = [
    "gemini-2.5-flash",        # 音声マルチモーダルの認識精度・処理速度に優れる
    "gemini-3.5-flash",        # Primary候補
    "gemini-3-flash-preview",  # Secondary 1
    "gemini-2.5-flash-lite",   # Secondary 2
    "gemini-3.1-flash"         # Secondary 3
]

def is_section_marker(text: str) -> bool:
    """テキストが [Intro] や ［サビ］ などのセクションマーカー記号であるかを判定する"""
    t = text.strip()
    if (t.startswith('[') and t.endswith(']')) or (t.startswith('［') and t.endswith('］')):
        return True
    # 括弧で始まって括弧で終わる一般的なメタデータ表記
    if re.match(r'^[\(\[［（].*[\)\]］）]$', t):
        return True
    return False

def parse_srt(srt_content: str) -> tuple[list, float]:
    """SRT形式の歌詞ファイルをパースしてテキストとタイムコードのリストを返す (ミリ秒精度)
    戻り値: (parsed_lyrics, intro_duration)
    """
    if not srt_content:
        return [], 0.0

    # BOM削除と改行の統一
    clean_content = srt_content.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n').strip()
    blocks = re.split(r'\n\s*\n', clean_content)
    parsed_lyrics = []
    intro_duration = 0.0
    first_lyric_time = None
    
    time_pattern = re.compile(
        r'(?:(\d+):)?(\d{1,2}):(\d{1,2})[,\.](\d{1,3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})[,\.](\d{1,3})'
    )
    
    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
            
        time_line_idx = -1
        match = None
        for idx, line in enumerate(lines):
            m = time_pattern.search(line)
            if m:
                match = m
                time_line_idx = idx
                break
                
        if match and time_line_idx != -1:
            s_h_str, s_m_str, s_s_str, s_ms_str, e_h_str, e_m_str, e_s_str, e_ms_str = match.groups()
            
            def to_sec(h_str, m_str, s_str, ms_str):
                h = int(h_str) if h_str else 0
                m = int(m_str) if m_str else 0
                s = int(s_str) if s_str else 0
                ms_len = len(ms_str) if ms_str else 3
                ms = int(ms_str) if ms_str else 0
                return h * 3600.0 + m * 60.0 + s + (ms / (10 ** ms_len))
                
            start_sec = to_sec(s_h_str, s_m_str, s_s_str, s_ms_str)
            end_sec = to_sec(e_h_str, e_m_str, e_s_str, e_ms_str)
            
            text_lines = lines[time_line_idx + 1:]
            text = " ".join(text_lines).strip()
            if not text:
                continue
                
            # MM:SS.ms 形式文字列と float 秒数の両方で参照可能なフォーマット
            start_min = int(start_sec // 60)
            start_rem_sec = start_sec % 60
            start_formatted = f"{start_min:02d}:{start_rem_sec:05.2f}"
            
            end_min = int(end_sec // 60)
            end_rem_sec = end_sec % 60
            end_formatted = f"{end_min:02d}:{end_rem_sec:05.2f}"
            
            is_marker = is_section_marker(text)
            parsed_lyrics.append({
                "start": start_formatted,
                "end": end_formatted,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "text": text,
                "is_marker": is_marker
            })
            
            if not is_marker and first_lyric_time is None:
                first_lyric_time = start_sec
                
    if first_lyric_time is not None:
        intro_duration = first_lyric_time
        
    return parsed_lyrics, intro_duration

def parse_txt_to_lines(txt_content: str, total_duration: float) -> tuple[list, float]:
    """通常のテキスト歌詞を行ごとのリストに変換し、セクション記号と歌詞行を識別する。
    さらに、[Intro: 12.5] などの記述から前奏秒数を検出し、仮配分を補正する。
    """
    lines = [line.strip() for line in txt_content.strip().split('\n') if line.strip()]
    parsed_lyrics = []
    
    # [Intro: 12.5] や ［前奏: 15］ などの表記から前奏秒数を抽出 (デフォルト 0.0)
    intro_duration = 0.0
    for line in lines:
        match = re.search(r'[\[［](?:Intro|前奏|インスト|前奏曲)\s*[:：]\s*(\d+(?:\.\d+)?)[\]］]', line, re.IGNORECASE)
        if match:
            intro_duration = float(match.group(1))
            break

    n = len(lines)
    if n == 0:
        return [], 0.0
        
    # 前奏時間以降の残りの時間で均等仮配分を行う
    usable_duration = max(1.0, total_duration - intro_duration)
    step = usable_duration / n
    
    for idx, line in enumerate(lines):
        s_sec = intro_duration + (idx * step)
        e_sec = total_duration if idx == n - 1 else intro_duration + ((idx + 1) * step)
        
        s_m = int(s_sec // 60)
        s_s = int(s_sec % 60)
        s_ms = int((s_sec % 1) * 100)
        
        e_m = int(e_sec // 60)
        e_s = int(e_sec % 60)
        e_ms = int((e_sec % 1) * 100)
        
        start_str = f"{s_m:02d}:{s_s:02d}.{s_ms:02d}"
        end_str = f"{e_m:02d}:{e_s:02d}.{e_ms:02d}"
        
        parsed_lyrics.append({
            "start": start_str,
            "end": end_str,
            "text": line,
            "is_marker": is_section_marker(line)
        })
    return parsed_lyrics, intro_duration

def get_audio_duration_seconds(audio_path: str) -> float:
    """音声ファイルの長さを秒単位で取得する"""
    try:
        with AudioFileClip(audio_path) as audio:
            return float(audio.duration)
    except Exception as e:
        print(f"[Warning] Failed to get duration using MoviePy: {e}")
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return float(librosa.get_duration(y=y, sr=sr))
        except Exception as e2:
            print(f"[Error] Failed to get duration using librosa: {e2}")
            return 30.0

def generate_metadata_json(
    audio_path: str,
    lyrics_content: str,
    api_key: str,
    is_srt: bool = False,
    primary_model: str = None,
    bpm: float = 120.0,
    bpm_offset: float = 0.0
) -> dict:
    """
    Gemini API のマルチモーダル音声理解機能を利用し、音声ファイル（歌声）から
    歌詞の実際の歌唱タイミング（start, end）をミリ秒精度で自動検出する。
    """
    # APIキー設定
    genai.configure(api_key=api_key)
    
    total_duration = get_audio_duration_seconds(audio_path)
    total_min = int(total_duration // 60)
    total_sec = int(total_duration % 60)
    total_time_str = f"{total_min:02d}:{total_sec:02d}"
    
    # 歌詞の構造化パース ＆ 前奏検知
    if is_srt or "-->" in lyrics_content:
        parsed_lyrics, intro_duration = parse_srt(lyrics_content)
        lyrics_desc = "SRT format with high-precision timestamps (already timed)"
    else:
        parsed_lyrics, intro_duration = parse_txt_to_lines(lyrics_content, total_duration)
        lyrics_desc = "Plain text with section tags [Intro]/［サビ］ (need to recognize vocal timing)"

    response_text = None
    uploaded_file = None
    try:
        # 音声ファイルをGemini APIにアップロード
        print(f"[Gemini] Uploading audio file to API server: {os.path.basename(audio_path)} ...")
        uploaded_file = genai.upload_file(path=audio_path)
        
        # アップロード完了を待機
        while uploaded_file.state.name == "PROCESSING":
            print("[Gemini] Waiting for audio processing...")
            time.sleep(1.0)
            uploaded_file = genai.get_file(uploaded_file.name)
            
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError("Audio file upload processing failed on Google API server.")
            
        print(f"[Success] Audio uploaded successfully. Remote file URI: {uploaded_file.uri}")

        # プロンプトの構築 (前奏時間の制約を明示化、およびBPM情報を注入)
        user_prompt = f"""
あなたは先進的なマルチモーダル音声同期・MV演出メタデータ生成システムです。
入力された音声ファイル（楽曲）を聴き取り、以下の歌詞データに基づいて同期メタデータJSONを設計してください。

[INPUT DATA]
- 音声の総再生時間: {total_time_str} (約 {total_duration:.2f} 秒)
- 歌詞フォーマット: {lyrics_desc}
- 前奏（歌唱なしインスト）区間: {intro_duration:.2f} 秒
- 楽曲のBPM (テンポ): {bpm}
- 1拍目のオフセット秒数: {bpm_offset} 秒
- 1拍の長さ: {60.0 / bpm:.3f} 秒
- 歌詞データ構造: {json.dumps(parsed_lyrics, ensure_ascii=False, indent=2)}

[INSTRUCTIONS]
1. 添付された音声ファイルを注意深く聴き取り、各歌詞行が「実際に歌われ始めた開始時間（start）」と「歌い終わった時間（end）」を正確に判定して、タイムコード（MM:SS.hh 形式）を割り当ててください。
2. 【極めて重要】各歌詞行の start, end タイミングは、可能な限り上記のBPMビートテンポ（拍の境界線やその1/2、1/4のタイミング）に美しく同期・一致するように割り当ててください。
3. 曲の最初の {intro_duration:.2f} 秒間は「前奏（歌唱なしのインストゥルメンタル区間）」です。したがって、最初の歌詞（is_marker: false の行）の start 時間は、絶対に {intro_duration:.2f} 秒より前（例: 00:00）に設定してはなりません。実際の歌声をしっかり聴き取り、最初の発声が開始される正確なタイムコード（例: 前奏が12.5秒なら 00:12.50 以降）から歌い出しの start を設定してください。
3. `is_marker: true`（例: [Intro], ［サビ］等）のセクションヘッダー行は、歌唱タイミングの検出対象外です。これらは「映像エフェクト切り替えのヒント」としてのみ使用してください。
4. 映像表現を最大限引き立てるため、各行ごとに「英語の画像生成ビジュアルプロンプト（visual_prompt）」を生成してください。プロンプトはサイバーパンク、レトロフューチャー、ネオン調を意識したものにしてください。
5. 各セクションに適用するエフェクト「effect」（"pixel_sort", "glitch_noise", "crt_scanline", "chromatic_aberration" のいずれか、または組み合わせ）を設定してください。特にセクションヘッダー行でエフェクトをドラマチックに切り替えてください。
6. 出力は以下のJSONスキーマに厳密に準拠したフォーマットにしてください。

[JSON SCHEMA]
{{
  "total_duration": {total_duration:.2f},
  "sections": [
    {{
      "start": "MM:SS.hh",
      "end": "MM:SS.hh",
      "lyric": "歌詞テキスト（またはセクション名）",
      "visual_prompt": "An abstract cybernetic neon background, hyperrealistic...",
      "effect": "pixel_sort"
    }}
  ]
}}
"""

        # 優先順位リストの構築
        model_list = MODELS.copy()
        if primary_model:
            if primary_model in model_list:
                model_list.remove(primary_model)
            model_list.insert(0, primary_model)
            
        # 優先順位に従ってAPI呼び出しをループ
        for model_name in model_list:
            try:
                print(f"[Gemini] Attempting audio multimodal analysis using model: {model_name}...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    contents=[uploaded_file, user_prompt],
                    generation_config={
                        "response_mime_type": "application/json"
                    }
                )
                response_text = response.text
                print(f"[Success] Gemini API Call succeeded with model: {model_name}")
                break
            except Exception as e:
                print(f"[Warning] Model {model_name} failed: {e}")
                continue

        if not response_text:
            raise RuntimeError("All specified Gemini models failed to process multimodal audio.")

        # JSONパース
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        
        result_json = json.loads(clean_text)
        ai_sections = result_json.get("sections", [])
        
        # ポストプロセスマージによるタイミングとセクション記号（字幕非表示化）の強制完全同期
        final_sections = []
        first_lyric_processed = False
        
        # テキストを基準としたマージ処理 (インデックスずれ・セクションスキップによるバグを防止)
        for idx, orig in enumerate(parsed_lyrics):
            orig_text_clean = orig["text"].strip().lower()
            
            # 完全一致または類似テキストを持つ要素を検索
            best_match_sec = None
            for ai_sec in ai_sections:
                ai_lyric = ai_sec.get("lyric", "").strip().lower()
                if ai_lyric == orig_text_clean or (ai_lyric and (ai_lyric in orig_text_clean or orig_text_clean in ai_lyric)):
                    best_match_sec = ai_sec
                    break
            
            # フォールバックとしてインデックスでの一致
            if not best_match_sec and idx < len(ai_sections):
                best_match_sec = ai_sections[idx]
                
            start_t = None
            end_t = None
            if best_match_sec:
                if is_srt or "-->" in lyrics_content:
                    start_t = orig["start"]
                    end_t = orig["end"]
                else:
                    start_t = best_match_sec.get("start")
                    end_t = best_match_sec.get("end")
                    
            start_t = start_t or orig["start"]
            end_t = end_t or orig["end"]
            
            # 前奏時間の強制クリップ（最初の歌唱歌詞行に対してのみ適用）
            if not orig["is_marker"] and not first_lyric_processed:
                start_sec = 0.0
                try:
                    parts = start_t.split(':')
                    if len(parts) == 2:
                        start_sec = float(parts[0]) * 60.0 + float(parts[1])
                    else:
                        start_sec = float(start_t)
                except:
                    pass
                    
                if start_sec < intro_duration:
                    s_m = int(intro_duration // 60)
                    s_s = int(intro_duration % 60)
                    s_ms = int((intro_duration % 1) * 100)
                    start_t = f"{s_m:02d}:{s_s:02d}.{s_ms:02d}"
                    
                first_lyric_processed = True
            
            final_sections.append({
                "start": start_t,
                "end": end_t,
                "lyric": orig["text"],
                "visual_prompt": (best_match_sec.get("visual_prompt") if best_match_sec else None) or "An abstract cybernetic neon background, hyperrealistic",
                "effect": (best_match_sec.get("effect") if best_match_sec else None) or "pixel_sort"
            })
            
        return {
            "total_duration": total_duration,
            "sections": final_sections
        }

    except Exception as e:
        print(f"[Error] Failed to parse/merge Gemini response: {e}")
        if response_text:
            print(f"Raw response from Gemini:\n{response_text}")
        raise e

    finally:
        # 音声オブジェクトの確実なクリーンアップ
        if uploaded_file:
            try:
                print(f"[Gemini] Cleaning up remote audio file: {uploaded_file.name} ...")
                genai.delete_file(uploaded_file.name)
                print("[Success] Remote file deleted.")
            except Exception as e:
                print(f"[Warning] Failed to delete remote file: {e}")


def generate_orchestration_timeline(
    audio_path: str,
    lyrics_content: str,
    api_key: str,
    bpm: float = 120.0,
    available_filters: list = None,
    primary_model: str = None
) -> dict:
    """
    Gemini APIを利用して楽曲の進行を解析し、セクションごとのエフェクト強度と適用フィルターを指定した
    「演出タイムラインJSON」を生成する。
    """
    if not available_filters:
        available_filters = ["none"]
    
    total_duration = get_audio_duration_seconds(audio_path)
    
    # APIキーが未設定の場合は即時フォールバック
    if not api_key or not api_key.strip():
        print("[Orchestration] No API key. Falling back to rule-based timeline.")
        return generate_fallback_timeline(total_duration, bpm, available_filters)
        
    genai.configure(api_key=api_key)
    
    uploaded_file = None
    response_text = None
    
    try:
        print(f"[Gemini Orchestration] Uploading audio: {os.path.basename(audio_path)} ...")
        uploaded_file = genai.upload_file(path=audio_path)
        
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1.0)
            uploaded_file = genai.get_file(uploaded_file.name)
            
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError("Audio upload failed on API server.")
            
        user_prompt = f"""
あなたは先進的なMV演出監督AIです。添付された楽曲を聴き取り、曲の構成（イントロ、メロ、サビ、アウトロ等）を自動判定した上で、映像演出パラメータのタイムラインを設計してください。

[INPUT DATA]
- 曲の総再生時間: {total_duration:.2f} 秒
- 楽曲BPM: {bpm}
- 使用可能なカスタムビジュアルフィルター名の一覧: {json.dumps(available_filters, ensure_ascii=False)}
- 歌詞情報 (参考):
{lyrics_content}

[INSTRUCTIONS]
1. 曲の展開を解析し、適切な時間（秒単位）で区切った「セクション (sections)」の配列を作成してください。
2. セクションごとに、曲調に合わせて以下のパラメータを割り当ててください：
   - "section_name": セクションの名称 (例: "Intro", "Verse A", "Chorus(サビ)", "Outro" 等)
   - "active_filter": 提供されたカスタムフィルター一覧からそのセクションに最もマッチするものを1つ選択。なければ "none"。(完全一致で大文字小文字を区別して記述)
   - "melt_intensity": ピクセルソートの適用度 (0.0 から 1.0)
   - "glitch_intensity": グリッチノイズの適用度 (0.0 から 1.0)
   - "biometric_opacity": 生体データの輝度/不透明度 (0.0 から 1.0)
   - "lyric_effect_style": その区間に最適な歌詞表示スタイル ("GLITCH", "KINETIC_BOUNCE", "CHROME_RGB", "NEON_GLOW", "SIMPLE" の中から選択)
3. 出力は以下のJSONスキーマに厳密に準拠させてください。Markdownの ```json 等の囲みは不要です。

[JSON SCHEMA]
{{
  "sections": [
    {{
      "start": 0.0,
      "end": 12.5,
      "section_name": "Intro",
      "active_filter": "none",
      "melt_intensity": 0.1,
      "glitch_intensity": 0.05,
      "biometric_opacity": 0.2,
      "lyric_effect_style": "NEON_GLOW"
    }}
  ]
}}
"""
        model_list = MODELS.copy()
        if primary_model:
            if primary_model in model_list:
                model_list.remove(primary_model)
            model_list.insert(0, primary_model)
            
        for model_name in model_list:
            try:
                print(f"[Gemini] Analyzing music using model: {model_name}...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    contents=[uploaded_file, user_prompt],
                    generation_config={
                        "response_mime_type": "application/json"
                    }
                )
                response_text = response.text
                break
            except Exception as e:
                print(f"[Warning] Model {model_name} failed: {e}")
                continue

        if not response_text:
            raise RuntimeError("All models failed.")
            
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        
        return json.loads(clean_text)
        
    except Exception as e:
        print(f"[Warning] Gemini timeline orchestration failed, fallback to rule-based: {e}")
        return generate_fallback_timeline(total_duration, bpm, available_filters)
        
    finally:
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass

def generate_fallback_timeline(total_duration: float, bpm: float, available_filters: list) -> dict:
    """
    Gemini API が利用できない場合の、ルールベース（簡易的）な演出タイムライン生成
    """
    sections = []
    
    intro_end = total_duration * 0.2
    sections.append({
        "start": 0.0,
        "end": intro_end,
        "section_name": "Intro (Fallback)",
        "active_filter": "none",
        "melt_intensity": 0.15,
        "glitch_intensity": 0.1,
        "biometric_opacity": 0.3,
        "lyric_effect_style": "NEON_GLOW"
    })
    
    main_end = total_duration * 0.8
    active_filter = available_filters[0] if len(available_filters) > 0 and available_filters[0] != "none" else "none"
    sections.append({
        "start": intro_end,
        "end": main_end,
        "section_name": "Main Chorus (Fallback)",
        "active_filter": active_filter,
        "melt_intensity": 0.7,
        "glitch_intensity": 0.5,
        "biometric_opacity": 0.85,
        "lyric_effect_style": "KINETIC_BOUNCE"
    })
    
    sections.append({
        "start": main_end,
        "end": total_duration,
        "section_name": "Outro (Fallback)",
        "active_filter": "none",
        "melt_intensity": 0.1,
        "glitch_intensity": 0.05,
        "biometric_opacity": 0.2,
        "lyric_effect_style": "SIMPLE"
    })
    
    return {"sections": sections}
