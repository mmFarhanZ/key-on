import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import os
import re
import json
import random

# ==========================================
# 1. ARSITEKTUR MODEL
# ==========================================
class ChordTransformerLSTMHybrid(nn.Module):
    def __init__(self, vocab_size, pad_id=0, d_model=256, n_head=4, n_layer=6, d_ff=512,
                 lstm_hidden=128, lstm_layers=2, dropout=0.15, max_seq_len=128, class_weights=None):
        super().__init__()
        self.pad_id  = pad_id
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.trans_norm  = nn.LayerNorm(d_model)

        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)
        self.dropout   = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, vocab_size, bias=False)

    def _make_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids, labels=None):
        B, T   = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(positions))

        pad_mask    = (input_ids == self.pad_id)
        causal_mask = self._make_causal_mask(T, device)

        trans_out = self.transformer(src=x, mask=causal_mask, is_causal=True)
        trans_out = self.trans_norm(trans_out)

        valid_mask = (~pad_mask).unsqueeze(-1).to(trans_out.dtype)
        trans_out  = trans_out * valid_mask

        lstm_out, _ = self.lstm(trans_out)
        lstm_out    = self.lstm_norm(lstm_out)
        lstm_out    = self.dropout(lstm_out)

        logits = self.head(lstm_out[:, -1, :])
        return {"logits": logits}

# ==========================================
# 2. LOAD MODEL DAN DATA PENDUKUNG
# ==========================================
@st.cache_resource
def load_assets():
    with open("vocab_final_terbaru.pkl", "rb") as f:
        vd = pickle.load(f)

    DEVICE = torch.device("cpu")
    model = ChordTransformerLSTMHybrid(
        vocab_size=vd["VOCAB_SIZE"], pad_id=vd["PAD_ID"], d_model=256, n_head=4,
        n_layer=6, d_ff=512, lstm_hidden=128, lstm_layers=2, dropout=0.15, max_seq_len=40
    )

    ckpt = torch.load("chord_transformer_decoder_V3.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    return model, vd, DEVICE

# ==========================================
# 3. KONFIGURASI HARMONI
# ==========================================
VALID_ROMAN = {
    "I","ii","iii","IV","V","vi","vii°",
    "I7","ii7","IV7","V7","vi7","Imaj7","IVmaj7","Vmaj7",
    "i","ii°","bIII","III","iv","v","bVI","VI","bVII","VII",
    "i7","iv7","v7","bVII7",
}

MAJOR_ONLY = {
    "I","ii","iii","IV","V","vi","vii°",
    "I7","ii7","IV7","V7","vi7","Imaj7","IVmaj7","Vmaj7",
}
MINOR_ONLY = {
    "i","ii°","bIII","III","iv","v","bVI","VI","bVII","VII",
    "i7","iv7","v7","bVII7",
}

CADENCES = {
    ("V","I"), ("V7","I"), ("IV","I"), ("vii°","I"),
    ("VII","i"), ("iv","i"), ("v","i"), ("bVII","i"),
    ("I","V"), ("i","V"),
}

COMMON_PROGRESSIONS = [
    ["I","V","vi","IV"], ["I","IV","V","I"],
    ["I","vi","IV","V"], ["I","ii","V","I"],
    ["i","VII","VI","VII"], ["i","iv","v","i"],
    ["i","VI","III","VII"], ["i","iv","VII","III"],
    ["ii","V","I"], ["ii7","V7","Imaj7"],
]

# ==========================================
# 4. KONVERSI ROMAN → ABSOLUTE
# ==========================================
NOTE_TO_INT = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,
    "E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,
    "Ab":8,"A":9,"A#":10,"Bb":10,"B":11
}
NOTE_CHROMATIC_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
NOTE_CHROMATIC_FLAT  = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
FLAT_KEYS = {"F","Bb","Eb","Ab","Db","Gb","Dm","Gm","Cm","Fm","Bbm","Ebm","Am"}

KEY_SPECIFIC_SPELLINGS = {
    "Gb": {11: "Cb"},                    
    "Cb": {11: "Cb", 6: "Gb"},          
    "F#": {1: "C#", 6: "F#", 8: "G#"}, 
    "C#": {1: "C#", 3: "D#", 6: "F#", 8: "G#", 10: "A#"},
}

ROMAN_INTERVAL_MAJOR = {
    "I":0,"ii":2,"iii":4,"IV":5,"V":7,"vi":9,"vii°":11,
    "I7":0,"IV7":5,"V7":7,"ii7":2,"vi7":9,
    "Imaj7":0,"IVmaj7":5,"Vmaj7":7,
}
ROMAN_INTERVAL_MINOR = {
    "i":0,"ii°":2,"bIII":3,"III":3,"iv":5,"v":7,
    "bVI":8,"VI":8,"bVII":10,"VII":10,"vii°":11,
    "V":7,"IV":5,
    "i7":0,"iv7":5,"v7":7,"bVII7":10,
}
DEGREE_QUALITY_MAJOR = {
    "I":"","ii":"m","iii":"m","IV":"","V":"","vi":"m","vii°":"dim",
    "I7":"7","IV7":"7","V7":"7","ii7":"m7","vi7":"m7",
    "Imaj7":"maj7","IVmaj7":"maj7","Vmaj7":"maj7",
}
DEGREE_QUALITY_MINOR = {
    "i":"m","ii°":"dim","bIII":"","III":"","iv":"m","v":"m",
    "bVI":"","VI":"","bVII":"","VII":"","vii°":"dim",
    "V":"","IV":"",
    "i7":"m7","iv7":"m7","v7":"m7","bVII7":"7",
}
MINOR_KEYS = {
    "Am":"A","Em":"E","Dm":"D","Bm":"B","Gm":"G","Cm":"C","Fm":"F",
    "C#m":"C#","F#m":"F#","G#m":"G#","A#m":"A#","D#m":"D#","Bbm":"Bb",
}

def is_minor_key(key_root):
    if key_root in MINOR_KEYS: return True
    if len(key_root) >= 2 and key_root.endswith("m"):
        return key_root[:-1] in NOTE_TO_INT
    return False

def get_root_note(key_root):
    if key_root in MINOR_KEYS: return MINOR_KEYS[key_root]
    if is_minor_key(key_root): return key_root[:-1]
    return key_root

def get_mode(key_root):
    return "minor" if is_minor_key(key_root) else "major"

def roman_to_absolute(roman_seq, key_root="C"):
    if is_minor_key(key_root):
        root_note    = get_root_note(key_root)
        interval_map = ROMAN_INTERVAL_MINOR
        quality_map  = DEGREE_QUALITY_MINOR
    else:
        root_note    = key_root
        interval_map = ROMAN_INTERVAL_MAJOR
        quality_map  = DEGREE_QUALITY_MAJOR

    if root_note not in NOTE_TO_INT: return roman_seq

    key_midi  = NOTE_TO_INT[root_note]
    chromatic = NOTE_CHROMATIC_FLAT if key_root in FLAT_KEYS else NOTE_CHROMATIC_SHARP
    key_spellings = KEY_SPECIFIC_SPELLINGS.get(key_root, {})
    result    = []
    
    for r in roman_seq:
        interval = interval_map.get(r)
        if interval is None:
            interval = ROMAN_INTERVAL_MAJOR.get(r) or ROMAN_INTERVAL_MINOR.get(r)
            quality  = DEGREE_QUALITY_MAJOR.get(r) or DEGREE_QUALITY_MINOR.get(r, "")
        else:
            quality = quality_map.get(r, "")
            
        if interval is None:
            result.append(r); continue
            
        midi = (key_midi + interval) % 12
        note_name = key_spellings.get(midi, chromatic[midi])
        result.append(note_name + quality)
    return result

# ==========================================
# 5. HARMONIC SCORING
# ==========================================
def harmonic_score(roman_seq):
    clean = [t for t in roman_seq if t in VALID_ROMAN]
    if not clean: return 0.0
    score = 0.0
    for prog in COMMON_PROGRESSIONS:
        n = len(prog)
        for i in range(len(clean) - n + 1):
            if clean[i:i+n] == prog: score += 8.0
    for i in range(len(clean) - 1):
        if (clean[i], clean[i+1]) in CADENCES: score += 5.0
    for i in range(len(clean) - 2):
        if clean[i] in ("ii","ii7") and clean[i+1] in ("V","V7") and clean[i+2] in ("I","Imaj7"):
            score += 10.0
        if clean[i] == "ii°" and clean[i+1] in ("V","V7") and clean[i+2] == "i":
            score += 10.0
    if len(clean) > 0:
        score += (len(set(clean)) / len(clean)) * 5.0
    invalid = [t for t in roman_seq if t not in VALID_ROMAN and not t.startswith("<")]
    score -= len(invalid) * 4.0
    return max(0.0, score)

def _normalize(vals, eps=1e-6):
    mn, mx = min(vals), max(vals)
    if mx - mn < eps:
        return [1.0] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]

# ==========================================
# 6. GENERASI CHORD
# ==========================================
SEQ_LEN = 32

@torch.no_grad()
def generate_roman_progression(model, vd, device, mood, tempo, key="C",
                                genre="<GENRE_POP>", max_length=16,
                                temperature=1.1, top_k=10, top_p=0.9,
                                repetition_penalty=2.0, rerank_alpha=0.5,
                                n_candidates=7, prev_last_roman=None):
    token_to_id = vd["token_to_id"]
    id_to_token = vd["id_to_token"]
    PAD_ID      = vd["PAD_ID"]

    start_tok = token_to_id.get("<START>", 0)
    genre_tok = token_to_id.get(genre, token_to_id.get("<GENRE_POP>", 0))
    mood_tok  = token_to_id.get(f"<MOOD_{mood}>", 0)
    tempo_tok = token_to_id.get(f"<TEMPO_{tempo}>", 0)
    seed      = [start_tok, genre_tok, mood_tok, tempo_tok]

    forbidden = {token_to_id.get(t, -1) for t in token_to_id if t.startswith("<")}
    forbidden.add(PAD_ID)

    mode = get_mode(key)
    if mode == "major":
        for tok_str in MINOR_ONLY:
            if tok_str in token_to_id: forbidden.add(token_to_id[tok_str])
    elif mode == "minor":
        for tok_str in MAJOR_ONLY - {"V", "V7", "IV", "IV7"}:
            if tok_str in token_to_id: forbidden.add(token_to_id[tok_str])

    for tok_str, tok_id in token_to_id.items():
        if not tok_str.startswith("<") and tok_str not in VALID_ROMAN:
            forbidden.add(tok_id)

    candidates = []

    for _ in range(n_candidates):
        gen          = seed.copy()
        log_prob_sum = 0.0
        token_count  = 0

        for _ in range(max_length):
            window = gen[-SEQ_LEN:]
            if len(window) < SEQ_LEN:
                window = [PAD_ID] * (SEQ_LEN - len(window)) + window

            ids = torch.tensor([window], dtype=torch.long).to(device)
            out = model(ids)
            lgt = out["logits"][0].clone() / temperature

            for fid in forbidden:
                if 0 <= fid < len(lgt): lgt[fid] = float("-inf")

            gen_tokens = gen[len(seed):]
            if prev_last_roman is not None and len(gen_tokens) == 0:
                prev_id = token_to_id.get(prev_last_roman)
                if prev_id is not None and 0 <= prev_id < len(lgt):
                    lgt[prev_id] = float("-inf")

            if gen_tokens:
                tc = {}
                for t in gen_tokens: tc[t] = tc.get(t, 0) + 1
                for t, count in tc.items():
                    if 0 <= t < len(lgt):
                        lgt[t] = lgt[t] / (repetition_penalty ** count)

                if len(gen_tokens) >= 3:
                    for i in range(len(gen_tokens) - 2):
                        if gen_tokens[i] == gen_tokens[-2] and gen_tokens[i+1] == gen_tokens[-1]:
                            nxt_blocked = gen_tokens[i+2] if i+2 < len(gen_tokens) else None
                            if nxt_blocked is not None and 0 <= nxt_blocked < len(lgt):
                                lgt[nxt_blocked] = float("-inf")

                if len(gen_tokens) >= 2 and gen_tokens[-1] == gen_tokens[-2]:
                    if 0 <= gen_tokens[-1] < len(lgt):
                        lgt[gen_tokens[-1]] = float("-inf")
                
                if len(gen_tokens) >= 1:
                    last_tok = gen_tokens[-1]
                    if tc.get(last_tok, 0) >= 2 and 0 <= last_tok < len(lgt):
                        lgt[last_tok] = float("-inf")

            if top_k > 0:
                th = torch.topk(lgt, min(top_k, lgt.size(-1))).values[-1]
                lgt[lgt < th] = float("-inf")

            if top_p < 1.0:
                s_lgt, s_idx = torch.sort(lgt, descending=True)
                cum = torch.cumsum(F.softmax(s_lgt, dim=-1), dim=-1)
                s_lgt[cum - F.softmax(s_lgt, dim=-1) > top_p] = float("-inf")
                lgt.scatter_(0, s_idx, s_lgt)

            log_probs = F.log_softmax(lgt, dim=-1)
            nxt = torch.multinomial(F.softmax(lgt, dim=-1), 1).item()

            if nxt == token_to_id.get("<END>", -1): break
            if 0 <= nxt < len(log_probs):
                lp = log_probs[nxt].item()
                if not (lp == float("-inf") or lp != lp):
                    log_prob_sum += lp; token_count += 1
            gen.append(nxt)

        roman = [
            id_to_token.get(i, "?") for i in gen[len(seed):]
            if id_to_token.get(i, "?") in VALID_ROMAN
        ]
        avg_log_prob = log_prob_sum / max(token_count, 1)
        candidates.append((roman, avg_log_prob))

    h_scores  = [harmonic_score(r) for r, _ in candidates]
    lp_scores = [lp               for _, lp in candidates]
    h_norm    = _normalize(h_scores)
    lp_norm   = _normalize(lp_scores)
    
    soft_scores = [
        rerank_alpha * lp_norm[i] + (1 - rerank_alpha) * h_norm[i]
        for i in range(len(candidates))
    ]
    best_idx   = max(range(len(candidates)), key=lambda i: soft_scores[i])
    return candidates[best_idx][0]

# ==========================================
# 7. MIDI GENERATOR  (voice-led + humanized)
# ==========================================
NOTE_MIDI = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
    "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71
}

# ── 7a. Chord parsing ─────────────────────
def parse_chord_to_intervals(chord_name):
    """Return (root_pitch_class 0-11, interval_list) for a chord name."""
    root_match = re.match(r'[A-G][#b]?', chord_name)
    if not root_match:
        return None, [0, 4, 7]
    root    = root_match.group()
    root_pc = NOTE_MIDI.get(root, 60) % 12
    suffix  = chord_name[len(root):]
    lower   = suffix.lower()

    if 'dim7' in lower:
        intervals = [0, 3, 6, 9]
    elif 'dim' in lower:
        intervals = [0, 3, 6]
    elif 'aug' in lower:
        intervals = [0, 4, 8]
    elif 'maj7' in suffix:
        intervals = [0, 4, 7, 11]
    elif 'm7' in suffix and 'maj7' not in suffix:
        intervals = [0, 3, 7, 10]
    elif 'm' in suffix and 'maj' not in suffix and 'dim' not in lower:
        intervals = [0, 3, 7]
    elif '7' in suffix and 'maj7' not in suffix:
        intervals = [0, 4, 7, 10]
    else:
        intervals = [0, 4, 7]

    return root_pc, intervals


# ── 7b. Voice leading ─────────────────────
def _chord_tones_in_range(root_pc, intervals, lo, hi):
    """All chord-tone MIDI notes within [lo, hi]."""
    tones = set()
    for interval in intervals:
        pc = (root_pc + interval) % 12
        for base in range(0, 120, 12):
            n = base + pc
            if lo <= n <= hi:
                tones.add(n)
    return sorted(tones)

def build_voicing(root_pc, intervals, prev_notes=None):
    bass_tones  = _chord_tones_in_range(root_pc, [0],       36, 52)
    tenor_tones = _chord_tones_in_range(root_pc, intervals, 48, 64)
    alto_tones  = _chord_tones_in_range(root_pc, intervals, 55, 72)
    sop_tones   = _chord_tones_in_range(root_pc, intervals, 60, 79)

    for lst, lo, hi in [(bass_tones, 36, 60), (tenor_tones, 48, 72),
                        (alto_tones, 48, 76), (sop_tones, 55, 80)]:
        if not lst:
            lst += _chord_tones_in_range(root_pc, intervals, lo, hi) or [root_pc + 60]

    def nearest(tones, target):
        return min(tones, key=lambda n: abs(n - target))

    if not prev_notes:
        bass = nearest(bass_tones,  40)
        ten  = nearest(tenor_tones, 55)
        alt  = nearest(alto_tones,  60)
        sop  = nearest(sop_tones,   64)
    else:
        bass = nearest(bass_tones,  prev_notes[0])
        ten  = nearest(tenor_tones, prev_notes[1])
        alt  = nearest(alto_tones,  prev_notes[2])
        sop  = nearest(sop_tones,   prev_notes[3])

    voices = sorted([bass, ten, alt, sop])
    voices[0] = nearest(bass_tones, voices[0])
    return voices


def voice_lead_progression(chords_abs):
    """
    Return a list of voiced chord dicts:
      { 'chord': str, 'notes': [b,t,a,s], 'velocities': [int,...] }
    Mendukung token [GAP] sebagai jeda antar bagan.
    """
    result     = []
    prev_notes = None

    for chord_name in chords_abs:
        if chord_name == "[GAP]":
            result.append({'chord': '[GAP]', 'notes': [], 'velocities': []})
            continue

        root_pc, intervals = parse_chord_to_intervals(chord_name)
        if root_pc is None:
            continue

        notes = build_voicing(root_pc, intervals, prev_notes)

        # Velocity humanization: bass lebih keras, inner voices lebih pelan
        vel_base = [82, 62, 65, 70]
        vels = [max(40, min(110, b + random.randint(-8, 8))) for b in vel_base]

        result.append({'chord': chord_name, 'notes': notes, 'velocities': vels})
        prev_notes = notes

    return result


# ── 7c. MIDI file writer ──────────────────
def create_midi(chords_abs, tempo_label, filename="output.mid"):
    """
    Tulis MIDI dari flat list chord (bisa mengandung token [GAP] antar bagan).
    Fitur:
      - Voice leading 4 suara (SATB)
      - Chord roll / arpeggiation (bass → soprano, 18 ms jarak)
      - Velocity humanization per note
      - GAP = jeda satu chord-duration antar bagan
    """
    mid   = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    bpm_map = {
        "SLOW": 60, "MEDIUM_SLOW": 76, "MEDIUM": 96,
        "FAST": 120, "VERY_FAST": 150,
    }
    bpm             = bpm_map.get(tempo_label, 96)
    tempo_us        = int(60_000_000 / bpm)
    ticks_per_chord = 2 * 480   # 2 ketukan per chord

    track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))
    track.append(Message('program_change', program=0, time=0))  # Grand Piano

    arp_ticks  = 18   # ~18 ms roll antar suara
    rest_ticks = 0    # akumulator jeda dari [GAP]

    voiced = voice_lead_progression(chords_abs)

    for v in voiced:
        notes = v['notes']
        vels  = v['velocities']
        n     = len(notes)

        # Jika GAP → akumulasi jeda, lanjut
        if not notes or v['chord'] == '[GAP]':
            rest_ticks += ticks_per_chord
            continue

        # note_on dengan chord roll (bass paling dulu)
        for i, (note, vel) in enumerate(zip(notes, vels)):
            t = arp_ticks if i > 0 else rest_ticks
            track.append(Message('note_on', note=note, velocity=vel, time=t))

        rest_ticks = 0  # reset akumulator

        # note_off: tahan durasi penuh, lepas bersama
        hold = ticks_per_chord - arp_ticks * (n - 1) - 16
        for i, note in enumerate(notes):
            track.append(Message('note_off', note=note, velocity=0,
                                 time=hold if i == 0 else 0))

    mid.save(filename)
    return filename


# ==========================================
# 8. PIANO AUDIO PLAYER  (Web Audio API)
# ==========================================
CHORD_DURATION_SEC = {
    "SLOW": 2.0, "MEDIUM_SLOW": 1.5, "MEDIUM": 1.0,
    "FAST": 0.75, "VERY_FAST": 0.5,
}

def build_chord_events(song_data, tempo_label):
    """
    Konversi song_data ke list event untuk JS piano player.
    Menambahkan event [GAP] (jeda + silence) di akhir setiap bagan.

    Tipe event:
      chord : { chord, section, notes, velocities, duration }
      gap   : { chord:'[GAP]', section, notes:[], velocities:[], duration }
    """
    duration = CHORD_DURATION_SEC.get(tempo_label, 1.0)

    # Flatten: chord per bagan + [GAP] di ujung setiap bagan
    flat = []
    for entry in song_data:
        for chord_name in entry["abs"]:
            flat.append((entry["section"], chord_name))
        flat.append((entry["section"], "[GAP]"))

    if not flat:
        return []

    voiced = voice_lead_progression([c for _, c in flat])

    events = []
    for (section, chord_name), v in zip(flat, voiced):
        if chord_name == "[GAP]":
            events.append({
                "label":      f"{section} (Pause)",
                "chord":      "⏸",
                "section":    section,
                "notes":      [],
                "velocities": [],
                "duration":   duration,
            })
        else:
            events.append({
                "label":      f"{section}: {chord_name}",
                "chord":      chord_name,
                "section":    section,
                "notes":      v["notes"],
                "velocities": v["velocities"],
                "duration":   duration,
            })
    return events


def generate_piano_player_html(song_data, tempo_label):
    """Kembalikan HTML self-contained berisi Web Audio piano player."""
    events    = build_chord_events(song_data, tempo_label)
    events_js = json.dumps(events)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: transparent;
    font-family: 'Outfit', 'Segoe UI', sans-serif;
    color: #1C1B18;
    padding: 4px 0;
  }}
  .player {{
    background: #FFFFFF;
    border: 1.5px solid #E2E0DA;
    border-radius: 10px;
    padding: 14px 16px;
    user-select: none;
  }}
  .player-title {{
    font-size: 10px;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #A8A39B;
    margin-bottom: 10px;
  }}
  .progress-wrap {{
    width: 100%;
    height: 3px;
    background: #E2E0DA;
    border-radius: 2px;
    margin-bottom: 10px;
    cursor: pointer;
  }}
  .progress-bar {{
    height: 100%;
    width: 0%;
    background: #4A7CF7;
    border-radius: 2px;
    transition: width 0.1s linear;
  }}
  .controls {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .btn {{
    background: #EFF0EC;
    border: 1.5px solid #E2E0DA;
    border-radius: 6px;
    color: #1C1B18;
    cursor: pointer;
    font-size: 13px;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s, border-color 0.15s;
    flex-shrink: 0;
  }}
  .btn:hover {{ background: #E2E0DA; }}
  .btn.active {{ background: rgba(74,124,247,0.10); border-color: #4A7CF7; color: #4A7CF7; }}
  .now-playing {{ flex: 1; min-width: 0; }}
  .np-chord {{
    font-size: 14px;
    font-weight: 600;
    color: #1C1B18;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .np-section {{
    font-size: 10px;
    color: #A8A39B;
    margin-top: 1px;
  }}
  .time-display {{
    font-size: 10px;
    color: #A8A39B;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }}
  .timeline {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 10px;
    align-items: center;
  }}
  .section-sep {{
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #C9C5BC;
    padding: 0 2px;
    white-space: nowrap;
  }}
  .chord-pill {{
    background: #EFF0EC;
    border: 1px solid #E2E0DA;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 11px;
    color: #6A665E;
    cursor: pointer;
    transition: all 0.12s;
    white-space: nowrap;
    font-family: 'Space Mono', monospace;
  }}
  .chord-pill:hover {{ border-color: #C9C5BC; color: #1C1B18; }}
  .chord-pill.playing {{
    background: rgba(74,124,247,0.08);
    border-color: #4A7CF7;
    color: #4A7CF7;
    font-weight: 600;
  }}
  .gap-pill {{
    width: 6px;
    height: 6px;
    background: #E2E0DA;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
  }}
  .gap-pill.playing {{ background: #4A7CF7; }}
</style>
</head>
<body>
<div class="player">
  <div class="player-title">🎹 &nbsp; Piano Preview</div>
  <div class="progress-wrap" id="progressWrap">
    <div class="progress-bar" id="progressBar"></div>
  </div>
  <div class="controls">
    <button class="btn" id="btnPlay">▶</button>
    <button class="btn" id="btnStop">■</button>
    <div class="now-playing">
      <div class="np-chord" id="npChord">Press ▶ to play</div>
      <div class="np-section" id="npSection">—</div>
    </div>
    <div class="time-display" id="timeDisplay">0:00 / 0:00</div>
  </div>
  <div class="timeline" id="timeline"></div>
</div>

<script>
(function() {{
  const events = {events_js};
  const totalDuration = events.reduce((s, e) => s + e.duration, 0);

  // ── Bangun pill timeline ──────────────────
  const timeline = document.getElementById('timeline');
  let lastSection = null;
  events.forEach((ev, i) => {{
    if (ev.chord === '⏸') {{
      const dot = document.createElement('span');
      dot.className = 'gap-pill';
      dot.dataset.index = i;
      dot.addEventListener('click', () => seekTo(i));
      timeline.appendChild(dot);
    }} else {{
      if (ev.section !== lastSection) {{
        if (lastSection !== null) {{
          const sep = document.createElement('span');
          sep.className = 'section-sep';
          sep.textContent = ev.section;
          timeline.appendChild(sep);
        }}
        lastSection = ev.section;
      }}
      const pill = document.createElement('div');
      pill.className = 'chord-pill';
      pill.textContent = ev.chord;
      pill.dataset.index = i;
      pill.addEventListener('click', () => seekTo(i));
      timeline.appendChild(pill);
    }}
  }});

  function getAllPills() {{
    return document.querySelectorAll('.chord-pill, .gap-pill');
  }}

  // ── Audio context ─────────────────────────
  let audioCtx = null;
  function getCtx() {{
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    return audioCtx;
  }}

  // ── Piano synthesis ───────────────────────
  function playNote(ctx, freq, startTime, duration, velocity) {{
    const partials = [
      [1, 1.00, 'triangle'], [2, 0.45, 'sine'], [3, 0.18, 'sine'],
      [4, 0.08, 'sine'],     [5, 0.04, 'sine'], [6, 0.015,'sine'],
    ];
    const detuneCents = (Math.random() - 0.5) * 6;
    partials.forEach(([h, amp, type]) => {{
      const osc = ctx.createOscillator();
      const g   = ctx.createGain();
      osc.connect(g); g.connect(ctx.destination);
      osc.type            = type;
      osc.frequency.value = freq * h * Math.pow(2, detuneCents / 1200);
      const peak    = (velocity / 127) * amp * 0.20;
      const sustain = Math.max(peak * 0.30, 0.0001);
      const t0 = startTime, tA = t0 + 0.007;
      const tD = t0 + Math.min(0.10, duration * 0.12);
      const tR = t0 + duration - 0.06;
      g.gain.setValueAtTime(0, t0);
      g.gain.linearRampToValueAtTime(peak, tA);
      g.gain.exponentialRampToValueAtTime(sustain, tD);
      g.gain.setValueAtTime(sustain, Math.max(tR, tD + 0.01));
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + duration + 0.08);
      osc.start(t0); osc.stop(t0 + duration + 0.15);
    }});
  }}

  function playChord(ctx, notes, velocities, startTime, duration) {{
    if (!notes || notes.length === 0) return; // GAP → diam
    const arpDelay = 0.018;
    notes.forEach((midi, i) => {{
      const vel  = (velocities && velocities[i]) ? velocities[i] : 70;
      const freq = 440 * Math.pow(2, (midi - 69) / 12);
      const t    = startTime + i * arpDelay;
      const dur  = duration + (i === 0 ? 0.15 : 0);
      playNote(ctx, freq, t, dur, vel);
    }});
  }}

  // ── State ─────────────────────────────────
  let playing = false, startedAt = 0, startIndex = 0, rafId = null;

  function totalTimeUpTo(idx) {{
    let t = 0;
    for (let i = 0; i < idx; i++) t += events[i].duration;
    return t;
  }}
  function indexAtTime(elapsed) {{
    let t = 0;
    for (let i = 0; i < events.length; i++) {{
      if (elapsed < t + events[i].duration) return i;
      t += events[i].duration;
    }}
    return events.length;
  }}
  function scheduleAll(ctx, fromIdx, startAudio) {{
    let offset = 0;
    for (let i = fromIdx; i < events.length; i++) {{
      const ev = events[i];
      playChord(ctx, ev.notes, ev.velocities, startAudio + offset, ev.duration);
      offset += ev.duration;
    }}
  }}
  function formatTime(sec) {{
    return Math.floor(sec/60) + ':' + String(Math.floor(sec%60)).padStart(2,'0');
  }}
  function updateUI(elapsed) {{
    const pct = Math.min(elapsed / totalDuration * 100, 100);
    document.getElementById('progressBar').style.width = pct + '%';
    document.getElementById('timeDisplay').textContent =
      formatTime(elapsed) + ' / ' + formatTime(totalDuration);
    const idx = Math.min(indexAtTime(elapsed), events.length - 1);
    if (idx >= 0 && idx < events.length) {{
      const ev = events[idx];
      document.getElementById('npChord').textContent   = ev.chord === '⏸' ? '— Pause —' : ev.chord;
      document.getElementById('npSection').textContent = ev.section;
    }}
    const pills = getAllPills();
    pills.forEach((p, i) => p.classList.toggle('playing', i === idx && playing));
  }}
  function animLoop() {{
    if (!playing) return;
    const ctx     = getCtx();
    const elapsed = (ctx.currentTime - startedAt) + totalTimeUpTo(startIndex);
    if (elapsed >= totalDuration) {{ stopPlayback(); updateUI(totalDuration); return; }}
    updateUI(elapsed);
    rafId = requestAnimationFrame(animLoop);
  }}
  function startPlayback(fromIdx) {{
    const ctx = getCtx();
    if (ctx.state === 'suspended') ctx.resume();
    startIndex = fromIdx; startedAt = ctx.currentTime;
    scheduleAll(ctx, fromIdx, startedAt);
    playing = true;
    document.getElementById('btnPlay').textContent = '⏸';
    document.getElementById('btnPlay').classList.add('active');
    rafId = requestAnimationFrame(animLoop);
  }}
  function pausePlayback() {{
    if (!playing) return;
    playing = false;
    cancelAnimationFrame(rafId);
    document.getElementById('btnPlay').textContent = '▶';
    document.getElementById('btnPlay').classList.remove('active');
    getCtx().suspend();
  }}
  function stopPlayback() {{
    playing = false;
    cancelAnimationFrame(rafId);
    startIndex = 0;
    if (audioCtx) {{ audioCtx.close(); audioCtx = null; }}
    document.getElementById('btnPlay').textContent = '▶';
    document.getElementById('btnPlay').classList.remove('active');
    document.getElementById('npChord').textContent   = 'Press ▶ to play';
    document.getElementById('npSection').textContent = '—';
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('timeDisplay').textContent = '0:00 / ' + formatTime(totalDuration);
    getAllPills().forEach(p => p.classList.remove('playing'));
  }}
  function seekTo(idx) {{
    const wasPlaying = playing;
    if (playing) {{ playing = false; cancelAnimationFrame(rafId); if (audioCtx) {{ audioCtx.close(); audioCtx = null; }} }}
    updateUI(totalTimeUpTo(idx));
    if (wasPlaying) startPlayback(idx);
    else {{ document.getElementById('btnPlay').textContent = '▶'; document.getElementById('btnPlay').classList.remove('active'); }}
  }}

  document.getElementById('btnPlay').addEventListener('click', () => {{
    if (!playing) {{
      const pct = parseFloat(document.getElementById('progressBar').style.width) / 100 || 0;
      startPlayback(indexAtTime(pct * totalDuration));
    }} else pausePlayback();
  }});
  document.getElementById('btnStop').addEventListener('click', stopPlayback);
  document.getElementById('progressWrap').addEventListener('click', function(e) {{
    const pct = (e.clientX - this.getBoundingClientRect().left) / this.offsetWidth;
    seekTo(indexAtTime(pct * totalDuration));
  }});

  document.getElementById('timeDisplay').textContent = '0:00 / ' + formatTime(totalDuration);
}})();
</script>
</body>
</html>"""
    return html


# ==========================================
# 9. UI STREAMLIT
# ==========================================

# ── Label Maps ───────────────────────────────
MOOD_LABELS = {
    "HAPPY":       "HAPPY",
    "SAD":         "SAD",
    "DARK":        "DARK",
    "PEACEFUL":    "PEACEFUL",
    "MELANCHOLIC": "MELANCHOLIC",
    "NEUTRAL":     "NEUTRAL",
}
TEMPO_LABELS_DISPLAY = {
    "SLOW":        "Slow",
    "MEDIUM_SLOW": "Medium Slow",
    "MEDIUM":      "Medium",
    "FAST":        "Fast",
    "VERY_FAST":   "Very Fast",
}
SECTION_ACCENT = {
    "Intro":      "#3B82C4",
    "Verse":      "#3A9A6E",
    "Pre-Chorus": "#D4961A",
    "Chorus":     "#D45252",
    "Bridge":     "#7B5CC4",
    "Outro":      "#6B7280",
}
CREATIVITY_PRESETS = {
    1: {"label": "Conventional",  "desc": "Common & predictable chords",         "temperature": 0.8,  "top_k": 5,  "top_p": 0.85, "repetition_penalty": 1.5},
    2: {"label": "Balanced",      "desc": "natural variation",      "temperature": 1.1,  "top_k": 10, "top_p": 0.90, "repetition_penalty": 2.0},
    3: {"label": "Experimental", "desc": "Unexpected and more creative chords", "temperature": 1.4,  "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.5},
}
SONG_STRUCTURES = {
    "Minimalist": ["Verse", "Chorus"],
    "Standard":   ["Intro", "Verse", "Chorus", "Verse", "Chorus", "Outro"],
    "Full":   ["Intro", "Verse", "Pre-Chorus", "Chorus", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Chorus", "Outro"],
    "Custom":    [],
}
SECTION_LENGTH = {
    "Intro": 4, "Verse": 8, "Pre-Chorus": 4,
    "Chorus": 8, "Bridge": 4, "Outro": 4,
}

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="Key-on",
    page_icon="🎼",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Outfit:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ── Variables ── */
:root {
    --bg:         #F7F6F3;
    --surface:    #FFFFFF;
    --surface-2:  #EFF0EC;
    --border:     #E2E0DA;
    --border-2:   #C9C5BC;
    --text:       #1C1B18;
    --text-2:     #6A665E;
    --text-3:     #A8A39B;
    --accent:     #4A7CF7;
    --accent-dim: rgba(74,124,247,0.10);
    --r:          8px;
}

/* ── Base ── */
.stApp {
    background-color: var(--bg) !important;
    font-family: 'Outfit', sans-serif !important;
}
.main .block-container {
    max-width: 740px !important;
    padding: 2rem 2rem 5rem !important;
}
p, li, label { color: var(--text) !important; }
#MainMenu, footer, header { visibility: hidden !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 2px; }

/* ── App Header ── */
.app-header {
    text-align: center;
    padding: 2.5rem 0 2.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.app-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 0.35rem;
}
.app-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.8rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 0.4rem !important;
    padding: 0 !important;
    line-height: 1 !important;
}
.app-subtitle {
    font-size: 0.9rem;
    color: var(--text-3);
    font-weight: 300;
    margin: 0;
}

/* ── Step Labels ── */
.step-label {
    display: flex;
    align-items: center;
    gap: 9px;
    margin: 1.8rem 0 0.7rem;
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px; height: 20px;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    border-radius: 50%;
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--accent);
    flex-shrink: 0;
}
.step-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

/* ── Radio → Pill Buttons ── */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] [role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 100px !important;
    padding: 5px 16px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    color: var(--text-2) !important;
    cursor: pointer !important;
    transition: border-color 0.15s, color 0.15s !important;
    white-space: nowrap !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label:hover {
    border-color: var(--border-2) !important;
    color: var(--text) !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label input[type="radio"],
div[data-testid="stRadio"] [role="radiogroup"] label > div:first-of-type {
    display: none !important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--text-2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 4px !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--r) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
}
/* Dropdown list */
div[data-baseweb="popover"] ul {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: var(--r) !important;
}
div[data-baseweb="popover"] li {
    color: var(--text-2) !important;
    font-family: 'Outfit', sans-serif !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li[aria-selected="true"] {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
}

/* ── Select Slider ── */
div[data-testid="stSlider"] label,
div[data-testid="stSelectSlider"] label { display: none !important; }

/* ── Multiselect ── */
div[data-testid="stMultiSelect"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--text-2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="stMultiSelect"] > div > div {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-family: 'Outfit', sans-serif !important;
}
span[data-baseweb="tag"] {
    background: var(--accent-dim) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Outfit', sans-serif !important;
    border-radius: 100px !important;
}

/* ── Generate Button ── */
div[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.7rem 1.5rem !important;
    transition: opacity 0.15s, transform 0.12s !important;
}

div[data-testid="stButton"] > button p,
div[data-testid="stButton"] > button span,
div[data-testid="stButton"] > button div {
    color: #FFFFFF !important;
}

div[data-testid="stButton"] > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
    opacity: 1 !important;
}

/* ── Download Button ── */
div[data-testid="stDownloadButton"] > button {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    transition: border-color 0.15s !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Alert / Warning ── */
div[data-testid="stAlert"] {
    background: var(--surface) !important;
    border-radius: var(--r) !important;
    border-color: var(--border-2) !important;
}

/* ── Result Header ── */
.result-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin: 2.5rem 0 0.6rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.result-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--text);
    margin: 0;
}
.result-meta {
    font-size: 0.78rem;
    color: var(--text-3);
    font-family: 'Outfit', sans-serif;
}

/* ── Config Bar ── */
.config-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 0.8rem 1.2rem;
    display: flex;
    gap: 0;
    flex-wrap: wrap;
    margin-bottom: 1.25rem;
}
.config-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 0 1.2rem 0 0;
    margin-right: 1.2rem;
    border-right: 1px solid var(--border);
}
.config-item:last-child { border-right: none; padding-right: 0; margin-right: 0; }
.config-key {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-3);
    font-family: 'Outfit', sans-serif;
}
.config-val {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    font-family: 'Outfit', sans-serif;
}

/* ── Section Cards ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid;
    border-radius: var(--r);
    padding: 1.1rem 1.4rem 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.15s;
}
.section-card:hover {
    border-color: var(--border-2);
}
.section-name {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-family: 'Outfit', sans-serif;
    margin-bottom: 0.65rem;
}
.chord-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
    margin-bottom: 0.55rem;
}
.chord-name {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 11px;
    letter-spacing: -0.01em;
}
.chord-sep {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-3);
}
.roman-row {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-3);
    letter-spacing: 0.02em;
    line-height: 1.7;
}

/* ── Creativity Labels Row ── */
.creativity-labels {
    display: flex;
    justify-content: space-between;
    margin-top: -0.4rem;
    margin-bottom: 0.25rem;
    padding: 0 2px;
}
.creativity-lbl {
    font-size: 0.72rem;
    color: var(--text-3);
    font-family: 'Outfit', sans-serif;
}
.creativity-lbl.active {
    color: var(--accent);
    font-weight: 600;
}
</style>""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────
model, vd, device = load_assets()

# ── Header ───────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-logo">♮</div>
    <h1 class="app-title">Key-on</h1>
    <p class="app-subtitle">Create a chord progression for your song · Hybrid Transformer-LSTM</p>
</div>
""", unsafe_allow_html=True)

# ── STEP 1 — Suasana Lagu ────────────────────
st.markdown("""<div class="step-label">
    <span class="step-num">1</span>
    <span class="step-title">Mood of the song</span>
</div>""", unsafe_allow_html=True)

mood_input = st.radio(
    "mood",
    list(MOOD_LABELS.keys()),
    horizontal=True,
    label_visibility="collapsed",
    format_func=lambda x: MOOD_LABELS[x],
)

# ── STEP 2 — Nada Dasar & Tempo ─────────
st.markdown("""<div class="step-label">
    <span class="step-num">2</span>
    <span class="step-title">Root note &amp; Tempo</span>
</div>""", unsafe_allow_html=True)

col_key, col_tempo = st.columns(2)
with col_key:
    key_input = st.selectbox(
        "Root Note",
        ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb",
         "Am", "Em", "Dm", "Bm", "Gm", "Cm", "Fm", "C#m", "F#m", "G#m"],
    )
with col_tempo:
    tempo_input = st.select_slider(
        "Tempo",
        options=list(TEMPO_LABELS_DISPLAY.keys()),
        value="MEDIUM",
        format_func=lambda x: TEMPO_LABELS_DISPLAY[x],
    )

# ── STEP 3 — Struktur & Kreativitas ─────────
st.markdown("""<div class="step-label">
    <span class="step-num">3</span>
    <span class="step-title">Structure &amp; Creativity</span>
</div>""", unsafe_allow_html=True)

col_struct, col_creative = st.columns(2)
with col_struct:
    structure_choice = st.selectbox(
        "Song Structures",
        list(SONG_STRUCTURES.keys()),
    )
with col_creative:
    creativity_level = st.select_slider(
        "Creativity Level",
        options=[1, 2, 3],
        value=2,
        format_func=lambda x: CREATIVITY_PRESETS[x]["label"],
    )

st.markdown(f"""<div class="creativity-labels">
    <span class="creativity-lbl {'active' if creativity_level == 1 else ''}">Conventional</span>
    <span class="creativity-lbl {'active' if creativity_level == 2 else ''}">Balanced</span>
    <span class="creativity-lbl {'active' if creativity_level == 3 else ''}">Experimental</span>
</div>""", unsafe_allow_html=True)

if structure_choice == "Custom":
    all_sections = ["Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Outro"]
    selected_sections = st.multiselect(
        "Select Song Sections",
        all_sections,
        default=["Verse", "Chorus"],
    )
    song_structure = selected_sections
else:
    song_structure = SONG_STRUCTURES[structure_choice]

# ── Generate Button ───────────────────────────
st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)
generate_clicked = st.button("🎼  Generate Chord Progression", use_container_width=True)

# ── Session State ─────────────────────────────
if "song_data" not in st.session_state:
    st.session_state.song_data   = None
    st.session_state.midi_path   = None
    st.session_state.last_config = None

if generate_clicked:
    preset = CREATIVITY_PRESETS[creativity_level]

    if not song_structure:
        st.warning("Please select at least one song section first.")
        st.stop()

    with st.spinner("Generating chord progression..."):
        song_data       = []
        all_abs         = []
        seen_sections   = {}
        prev_last_roman = None

        for section in song_structure:
            length   = SECTION_LENGTH.get(section, 8)
            count    = seen_sections.get(section, 0)
            temp_adj = preset["temperature"] + (count * 0.15)
            seen_sections[section] = count + 1

            roman = []
            for attempt in range(3):
                roman = generate_roman_progression(
                    model, vd, device, mood_input, tempo_input,
                    key=key_input,
                    max_length=length,
                    temperature=temp_adj + (attempt * 0.2),
                    top_k=preset["top_k"],
                    top_p=preset["top_p"],
                    repetition_penalty=preset["repetition_penalty"],
                    rerank_alpha=0.4,
                    n_candidates=5,
                    prev_last_roman=prev_last_roman,
                )
                if len(roman) >= 2:
                    break

            if roman:
                prev_last_roman = roman[-1]

            abs_c = roman_to_absolute(roman, key_root=key_input)
            song_data.append({"section": section, "roman": roman, "abs": abs_c})
            all_abs.extend(abs_c)
            all_abs.append("[GAP]")  # jeda antar bagan di MIDI

        midi_file = create_midi(all_abs, tempo_input, "generated_chord.mid")

    st.session_state.song_data   = song_data
    st.session_state.midi_path   = midi_file
    st.session_state.last_config = {
        "mood":       MOOD_LABELS.get(mood_input, mood_input),
        "key":        key_input,
        "tempo":      TEMPO_LABELS_DISPLAY.get(tempo_input, tempo_input),
        "tempo_raw":  tempo_input,
        "structure":  structure_choice,
        "creativity": CREATIVITY_PRESETS[creativity_level]["label"],
    }

# ── Results ───────────────────────────────────
if st.session_state.song_data:
    cfg            = st.session_state.last_config
    total_sections = len(st.session_state.song_data)
    total_chords   = sum(len(e["abs"]) for e in st.session_state.song_data)

    st.markdown(f"""
<div class="result-header">
    <div class="result-title">Generated Results</div>
    <div class="result-meta">{total_sections} section{'s' if total_sections != 1 else ''} &nbsp;·&nbsp; {total_chords} chord{'s' if total_chords != 1 else ''}</div>
</div>
<div class="config-bar">
    <div class="config-item">
        <span class="config-key">Mood</span>
        <span class="config-val">{cfg["mood"]}</span>
    </div>
    <div class="config-item">
        <span class="config-key">Root Note</span>
        <span class="config-val">{cfg["key"]}</span>
    </div>
    <div class="config-item">
        <span class="config-key">Tempo</span>
        <span class="config-val">{cfg["tempo"]}</span>
    </div>
    <div class="config-item">
        <span class="config-key">Structure</span>
        <span class="config-val">{cfg["structure"]}</span>
    </div>
    <div class="config-item">
        <span class="config-key">Creativity</span>
        <span class="config-val">{cfg["creativity"]}</span>
    </div>
</div>
""", unsafe_allow_html=True)

    for entry in st.session_state.song_data:
        section = entry["section"]
        color   = SECTION_ACCENT.get(section, "#7A7570")
        abs_c   = entry["abs"]
        roman_c = entry["roman"]

        if abs_c:
            chords_html = (" "
                + '<span class="chord-sep">›</span> '.join(
                    f'<span class="chord-name">{c}</span>' for c in abs_c
                )
            )
        else:
            chords_html = '<span style="color:var(--text-3)">—</span>'

        roman_str = "  ›  ".join(roman_c) if roman_c else "—"

        st.markdown(f"""
<div class="section-card" style="border-left-color:{color}">
    <div class="section-name" style="color:{color}">{section}</div>
    <div class="chord-row">{chords_html}</div>
    <div class="roman-row">{roman_str}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

    # ── Piano Audio Player ────────────────────
    player_html = generate_piano_player_html(
        st.session_state.song_data,
        st.session_state.last_config.get("tempo_raw", "MEDIUM"),
    )
    components.html(player_html, height=195, scrolling=False)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    if st.session_state.midi_path and os.path.exists(st.session_state.midi_path):
        with open(st.session_state.midi_path, "rb") as f:
            st.download_button(
                label="⬇  Download MIDI",
                data=f,
                file_name="Key-on_output.mid",
                mime="audio/midi",
                use_container_width=True,
            )