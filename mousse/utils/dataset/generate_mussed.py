import os
import json
import random
import glob
import csv
from pathlib import Path
from pydub import AudioSegment

# Audio constants
RANDOM_SEED = 42
TARGET_DBFS = -23.0
GAIN_VARIATION_DB = 3.0
MIN_SPK_DUR_SEC = 1.5
MIN_MIX_DUR_SEC = 3.0
MAX_MIX_DUR_SEC = 15.0
TARGET_SR = 16000
OVERLAP_OPTIONS = [0.0, 0.25, 0.50, 0.75, 1.00]

# Dataset roots
libri_dir = "/data/LibriTTS-R/LibriTTS_R/"
kor_train_dir = "/data/korean-speech/014.multispeaker_audio_data/01.data/1.Training/"
kor_val_dir = "/data/korean-speech/014.multispeaker_audio_data/01.data/2.Validation/"
output_base_dir = "/data/MuSSeD-EnKo/"

# Paths derived from the dataset roots.
output_train = os.path.join(output_base_dir, "train")
output_dev = os.path.join(output_base_dir, "dev")
output_test = os.path.join(output_base_dir, "test")
libri_meta_path = os.path.join(libri_dir, "SPEAKERS.txt")

KOR_PATH_MAP = {
     "train": ("/label/VL1/1.VL/1.VL/", "/raw/VS1/1.VS/1.VS/"),
    "dev": ("/label/VL5/1.VL5/1.VL5/", "/raw/VS5/1.VS5/1.VS5/"),
    "test": ("/label/TL10/1.TL/1.TL/", "/raw/TS10/1.TS10/1.TS10/"),
}

KOR_LABEL_ROOT = {
    "train": os.path.join(kor_val_dir, "label/VL1"),
    "dev": os.path.join(kor_val_dir, "label/VL5"),
    "test":  os.path.join(kor_train_dir, "label/TL10"),
}

KOR_KEY_MAP = {
    "TransLabelText": "transcription", "OrgLabelText": "org_label_text",
    "RefinedLabelText": "refined_label_text", "ModifyLabelText": "modify_label_text",
    "TextCategory": "text_category", "SpeakerName": "speaker_name",
    "Gender": "gender", "Age": "age",
    "Region": "region", "Dialect": "dialect",
    "Emotion": "emotion", "Sensitivity": "sensitivity",
    "RecordingEnviron": "recording_environ", "NoiseEnviron": "noise_environ",
    "RecordingDevice": "recording_device", "SamplingRate": "sampling_rate",
    "NumberOfBit": "number_of_bit", "ByteOrder": "byte_order",
    "EncodingLaw": "encoding_law", "NumberOfChannel": "number_of_channel",
    "SignalToNoiseRatio": "snr", "Language": "language",
    "DataCategory": "data_category", "Distributor": "distributor",
    "QualityStatus": "quality_status", "SpeechStart": "speech_start",
    "SpeechEnd": "speech_end",
}

def load_libri_speakers(meta_path):
    spk_metadata = {}
    if not os.path.exists(meta_path):
        print(f"Warning: SPEAKERS.txt not found at {meta_path}")
        return spk_metadata
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(';') or not line.strip():
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                spk_metadata[parts[0]] = {
                    "gender": "Female" if parts[1].upper() == 'F' else "Male",
                    "name": parts[4],
                }
    return spk_metadata

def parse_libri_tsv_data(libri_dir):
    tsv_lookup = {}
    for tsv_path in glob.glob(os.path.join(libri_dir, "**", "*.book.tsv"), recursive=True):
        try:
            with open(tsv_path, 'r', encoding='utf-8') as f:
                for row in csv.reader(f, delimiter='\t'):
                    if len(row) >= 6:
                        utt_id = row[0].strip()
                        tsv_lookup[utt_id] = {
                            "aligned": "true" if row[3].lower() == 'true' else "false",
                            "speech_start": float(row[4]),
                            "speech_end": float(row[5]),
                        }
        except Exception:
            continue
    return tsv_lookup

def index_libri_dataset(libri_dir, spk_metadata_map):
    utterances = []
    tsv_lookup = parse_libri_tsv_data(libri_dir)

    for norm_path in glob.glob(os.path.join(libri_dir, "**", "*.normalized.txt"), recursive=True):
        orig_path = norm_path.replace(".normalized.txt", ".original.txt")
        wav_path  = norm_path.replace(".normalized.txt", ".wav")
        if not os.path.exists(wav_path) or not os.path.exists(orig_path):
            continue

        parts  = Path(norm_path).parts
        spk_id = parts[-3]
        utt_id = Path(norm_path).stem.replace(".normalized", "")

        try:
            base_idx = parts.index("LibriTTS-R")
            original_file_id = "/" + os.path.join(*parts[base_idx + 2:]).replace(".normalized.txt", "")
        except ValueError:
            original_file_id = utt_id

        with open(norm_path, 'r', encoding='utf-8') as f:
            transcription = f.read().strip()
        with open(orig_path, 'r', encoding='utf-8') as f:
            org_label_text = f.read().strip()

        tsv  = tsv_lookup.get(utt_id, {"aligned": "false", "speech_start": 0.0, "speech_end": None})
        info = spk_metadata_map.get(spk_id, {"gender": "Unknown", "name": "Unknown"})

        utterances.append({
            "speaker_id": spk_id,
            "speaker_name": info["name"],
            "gender": info["gender"],
            "language": "ENG",
            "audio_path": wav_path,
            "transcription": transcription,
            "org_label_text": org_label_text,
            "aligned": tsv["aligned"],
            "speech_start": tsv["speech_start"],
            "speech_end": tsv["speech_end"],
            "original_file_id": original_file_id,
            "file_stem": f"EN_{utt_id}",
        })

    return utterances

def index_korean_dataset(setting):
    label_sub, raw_sub = KOR_PATH_MAP[setting]
    label_root = KOR_LABEL_ROOT[setting]

    utterances = []
    for json_path in glob.glob(os.path.join(label_root, "**", "*.json"), recursive=True):
        try:
            wav_path = json_path.replace(label_sub, raw_sub).replace(".json", ".wav")
            if not os.path.exists(wav_path):
                print(f"  Audio missing: {wav_path}")
                continue

            parts = Path(json_path).parts
            file_stem = Path(json_path).stem
            parent_folder = parts[-2]
            spk_id = parent_folder.split("_")[0]

            try:
                if "1.Training" in parts:
                    base_idx = parts.index("1.Training")
                elif "2.Validation" in parts:
                    base_idx = parts.index("2.Validation")
                else:
                    raise ValueError
                original_file_id = os.path.join(*parts[base_idx:]).replace(".json", "")
            except ValueError:
                original_file_id = file_stem

            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            flat = {}
            for cat in ["기본정보", "음성정보", "전사정보", "화자정보", "환경정보", "파일정보", "기타정보"]:
                if cat in meta:
                    for k, v in meta[cat].items():
                        flat[KOR_KEY_MAP.get(k, k)] = v

            flat["speaker_id"] = spk_id
            flat["speaker_name"] = flat.get("speaker_name", "Unknown")
            flat["audio_path"] = wav_path
            flat["original_file_id"] = original_file_id
            flat["language"] = "KOR"
            flat["file_stem"] = f"KO_{file_stem}"

            utterances.append(flat)
        except Exception:
            continue

    return utterances

def _speech_duration(utt):
    try:
        sp_s = utt.get("speech_start")
        sp_e = utt.get("speech_end")
        if sp_s is None or sp_e is None or sp_e == "N/A":
            return None
        return float(sp_e) - float(sp_s)
    except (ValueError, TypeError):
        return None

def _filter_pool(pool, min_dur=MIN_SPK_DUR_SEC):
    result = []
    for u in pool:
        dur = _speech_duration(u)
        if dur is None or dur >= min_dur:
            result.append(u)
    return result

def _load_prepare(utt):
    audio = (AudioSegment.from_wav(utt["audio_path"])
             .set_frame_rate(TARGET_SR)
             .set_channels(1)
             .set_sample_width(2))

    sp_start = utt.get("speech_start")
    sp_end = utt.get("speech_end")
    try:
        start_ms = max(0, int(float(sp_start) * 1000)) if sp_start is not None else 0
        end_ms = (min(len(audio), int(float(sp_end) * 1000))
                    if (sp_end is not None and sp_end != "N/A") else len(audio))
        if end_ms > start_ms:
            audio = audio[start_ms:end_ms]
    except (ValueError, TypeError):
        pass

    if len(audio) == 0:
        raise ValueError(f"Empty audio after trimming: {utt['audio_path']}")

    if audio.dBFS > -60.0:
        audio = audio.apply_gain(TARGET_DBFS - audio.dBFS)

    return audio

def mix_speakers(selected_utterances, output_wav_path):
    overlap_ratio = random.choice(OVERLAP_OPTIONS)

    audios = [_load_prepare(u) for u in selected_utterances]
    audios = [a.apply_gain(random.uniform(-GAIN_VARIATION_DB, GAIN_VARIATION_DB))
              for a in audios]

    mixed_audio = audios[0]
    spk_timings = [(0, len(audios[0]))]
    current_start_ms, current_end_ms = 0, len(audios[0])

    for audio in audios[1:]:
        len_next = len(audio)
        len_curr = current_end_ms - current_start_ms
        overlap_ms = int(min(len_curr, len_next) * overlap_ratio)

        next_start_ms = max(0, len(mixed_audio) - overlap_ms)
        next_end_ms = next_start_ms + len_next

        if next_end_ms > len(mixed_audio):
            mixed_audio = mixed_audio + AudioSegment.silent(
                duration=next_end_ms - len(mixed_audio))

        mixed_audio = mixed_audio.overlay(audio, position=next_start_ms)
        spk_timings.append((next_start_ms, next_end_ms))
        current_start_ms, current_end_ms = next_start_ms, next_end_ms

    mixed_audio.export(output_wav_path, format="wav")

    speakers_meta = []
    for idx, utt in enumerate(selected_utterances):
        start_sec = round(spk_timings[idx][0] / 1000.0, 3)
        end_sec = round(spk_timings[idx][1] / 1000.0, 3)
        dur_sec = round((spk_timings[idx][1] - spk_timings[idx][0]) / 1000.0, 3)

        # Original source boundaries before trimming
        raw_sp_s = utt.get("speech_start")
        raw_sp_e = utt.get("speech_end")
        try:
            org_sp_start = float(raw_sp_s) if raw_sp_s is not None else 0.0
        except (ValueError, TypeError):
            org_sp_start = 0.0
        try:
            org_sp_end = (float(raw_sp_e)
                          if (raw_sp_e is not None and raw_sp_e != "N/A")
                          else org_sp_start + dur_sec)
        except (ValueError, TypeError):
            org_sp_end = org_sp_start + dur_sec

        # Speech boundaries in the generated mixture
        spk_entry = {
            "speaker_id": utt["speaker_id"],
            "speaker_name": utt.get("speaker_name", "Unknown"),
            "gender": utt.get("gender", "Unknown"),
            "language": utt.get("language", "ENG"),
            "length_sec": dur_sec,
            "start_time_sec": start_sec,
            "end_time_sec": end_sec,
            "transcription": utt.get("transcription", ""),
            "org_label_text": utt.get("org_label_text", ""),
            "speech_start": start_sec,
            "speech_end": end_sec,
            "org_speech_start": round(org_sp_start, 3),
            "org_speech_end": round(org_sp_end,   3),
            "aligned": utt.get("aligned", "false"),
            "original_file_id": utt.get("original_file_id", ""),
        }

        if utt.get("language") == "KOR":
            skip = {"audio_path", "transcription", "org_label_text", "aligned",
                    "speech_start", "speech_end", "speaker_id", "speaker_name",
                    "gender", "language", "file_stem"}
            for k, v in utt.items():
                if k not in skip:
                    spk_entry[k] = v

        speakers_meta.append(spk_entry)

    return round(overlap_ratio, 2), round(len(mixed_audio) / 1000.0, 3), speakers_meta

def _lang_combos(num_spk):
    return [["ENG"] * e + ["KOR"] * (num_spk - e) for e in range(num_spk + 1)]

def _group_by_speaker(pool):
    groups = {}
    for u in pool:
        groups.setdefault(u["speaker_id"], []).append(u)
    return groups

def _sample_speaker_first(groups, n):
    chosen_speakers = random.sample(list(groups.keys()), n)
    return [random.choice(groups[spk]) for spk in chosen_speakers]

def main():
    random.seed(RANDOM_SEED)

    print("Indexing LibriTTS-R speaker metadata...")
    spk_gender_map = load_libri_speakers(libri_meta_path)

    print("Indexing English recordings (train / dev / test)...")
    en_train_pool = index_libri_dataset(os.path.join(libri_dir, "train-clean-100"), spk_gender_map)
    en_dev_pool = index_libri_dataset(os.path.join(libri_dir, "dev-clean"),       spk_gender_map)
    en_test_pool  = index_libri_dataset(os.path.join(libri_dir, "test-clean"),      spk_gender_map)

    print("Indexing Korean recordings (train: VS1 / dev: VS5 / test: TS10)...")
    ko_train_pool = index_korean_dataset("train")  # VS1 — 81 speakers
    ko_dev_pool   = index_korean_dataset("dev")    # VS5 — 61 speakers
    ko_test_pool  = index_korean_dataset("test")   # TS10 — 50 speakers

    print("Filtering pools by minimum speech duration...")
    en_train_pool = _filter_pool(en_train_pool)
    en_dev_pool = _filter_pool(en_dev_pool)
    en_test_pool  = _filter_pool(en_test_pool)
    ko_train_pool = _filter_pool(ko_train_pool)
    ko_dev_pool = _filter_pool(ko_dev_pool)
    ko_test_pool  = _filter_pool(ko_test_pool)

    print(f"Pool sizes (train / dev / test)")
    print(f"  EN : {len(en_train_pool):>6} / {len(en_dev_pool):>5} / {len(en_test_pool):>5}")
    print(f"  KO : {len(ko_train_pool):>6} / {len(ko_dev_pool):>5} / {len(ko_test_pool):>5}")

    if not all([en_train_pool, en_dev_pool, en_test_pool,
                ko_train_pool, ko_dev_pool, ko_test_pool]):
        print("CRITICAL: One or more pools are empty. Check paths above.")
        return

    # Equal speaker sampling across utterance counts
    en_train_grp = _group_by_speaker(en_train_pool)
    en_dev_grp   = _group_by_speaker(en_dev_pool)
    en_test_grp  = _group_by_speaker(en_test_pool)
    ko_train_grp = _group_by_speaker(ko_train_pool)
    ko_dev_grp   = _group_by_speaker(ko_dev_pool)
    ko_test_grp  = _group_by_speaker(ko_test_pool)

    print(f"Unique speakers (train / dev / test)")
    print(f"  EN : {len(en_train_grp):>6} / {len(en_dev_grp):>5} / {len(en_test_grp):>5}")
    print(f"  KO : {len(ko_train_grp):>6} / {len(ko_dev_grp):>5} / {len(ko_test_grp):>5}")

    split_cfg = {
        "train": (output_train, en_train_grp, ko_train_grp, 10000),
        "dev":   (output_dev,   en_dev_grp,   ko_dev_grp,    1300),
        "test":  (output_test,  en_test_grp,  ko_test_grp,   1300),
    }

    for num_spk in [2, 3, 4]:
        folder_name = f"{num_spk}spk"
        combos = _lang_combos(num_spk)

        for split_type, (target_base, en_grp, ko_grp, target_n) in split_cfg.items():
            raw_dir  = os.path.join(target_base, folder_name, "raw")
            json_dir = os.path.join(target_base, folder_name, "json")
            os.makedirs(raw_dir,  exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)

            print(f"\n[{split_type.upper()} | {folder_name}] Generating {target_n} mixtures "
                  f"(seed={RANDOM_SEED})...")

            generated = 0
            attempts = 0
            max_attempts = target_n * 100

            while generated < target_n and attempts < max_attempts:
                attempts += 1

                combo = random.choice(combos)
                n_eng = combo.count("ENG")
                n_kor = combo.count("KOR")

                try:
                    selected = _sample_speaker_first(en_grp, n_eng) + _sample_speaker_first(ko_grp, n_kor)
                except ValueError:
                    continue

                # Prevent same speaker appearing twice in one mixture
                if len({u["speaker_id"] for u in selected}) != len(selected):
                    continue

                random.shuffle(selected)

                mix_id = "_".join(u["file_stem"] for u in selected)
                output_wav  = os.path.join(raw_dir,  f"{mix_id}.wav")
                output_json = os.path.join(json_dir, f"{mix_id}.json")

                try:
                    overlap_ratio, total_dur, speakers_meta = mix_speakers(selected, output_wav)

                    if not (MIN_MIX_DUR_SEC <= total_dur <= MAX_MIX_DUR_SEC):
                        if os.path.exists(output_wav):
                            os.remove(output_wav)
                        continue

                    metadata = {
                        "mixture_id": mix_id,
                        "speaker_count": num_spk,
                        "overlapped_ratio": overlap_ratio,
                        "total_length_sec": total_dur,
                        "speakers": speakers_meta,
                    }
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=4)

                    generated += 1
                    if generated % 500 == 0 or generated == target_n:
                        print(f"  {generated:>5}/{target_n}  (attempts so far: {attempts})")

                except Exception:
                    if os.path.exists(output_wav):
                        os.remove(output_wav)
                    continue

            if generated < target_n:
                print(f"  WARNING: Only generated {generated}/{target_n} "
                      f"(pool too small or too many duration rejections)")

    print(f"\nDone. Output: {output_base_dir}")

if __name__ == "__main__":
    main()
