import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import mido
from mido import Message, MidiFile, MidiTrack
import os
import re

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
                                repetition_penalty=2.0, rerank_alpha=0.4,
                                n_candidates=5, prev_last_roman=None):
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
# 7. MIDI GENERATOR
# ==========================================
NOTE_MIDI = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
    "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71
}

def create_midi(chords_abs, tempo_label, filename="output.mid"):
    mid   = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    tempo_ticks = {
        "SLOW": 960, "MEDIUM_SLOW": 720, "MEDIUM": 480,
        "FAST": 360, "VERY_FAST": 240,
    }
    duration = tempo_ticks.get(tempo_label, 480)

    for chord_name in chords_abs:
        root_match = re.match(r'[A-G][#b]?', chord_name)
        if not root_match: continue
        root      = root_match.group()
        root_midi = NOTE_MIDI.get(root, 60)

        intervals = [0, 4, 7]
        lower     = chord_name.lower()
        if "dim" in lower:
            intervals = [0, 3, 6]
        elif "aug" in lower:
            intervals = [0, 4, 8]
        elif "m7" in chord_name and "maj7" not in chord_name and "dim" not in lower:
            intervals = [0, 3, 7]
        elif "m" in chord_name and "maj" not in chord_name and "dim" not in lower:
            intervals = [0, 3, 7]

        notes = [root_midi + i for i in intervals]

        if "maj7" in chord_name:
            notes.append(root_midi + 11)
        elif "m7" in chord_name and "maj7" not in chord_name:
            notes.append(root_midi + 10)
        elif "7" in chord_name and "maj7" not in chord_name:
            notes.append(root_midi + 10)

        for note in notes:
            track.append(Message("note_on",  note=note, velocity=70, time=0))
        for idx, note in enumerate(notes):
            track.append(Message("note_off", note=note, velocity=70, time=duration if idx == 0 else 0))

    mid.save(filename)
    return filename

# ==========================================
# 8. ANTARMUKA STREAMLIT
# ==========================================
st.set_page_config(page_title="Rekomendasi Chord", page_icon="🎹")
st.title("🎹 Sistem Rekomendasi Chord Progression")
st.markdown("Menggunakan Hybrid Transformer-LSTM berdasarkan Roman Numeral Modeling.")

model, vd, device = load_assets()

col1, col2, col3 = st.columns(3)
with col1:
    mood_input = st.selectbox(
        "Pilih Mood",
        ["HAPPY", "SAD", "DARK", "PEACEFUL", "MELANCHOLIC", "NEUTRAL"]
    )
with col2:
    key_input = st.selectbox(
        "Pilih Key",
        ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb",
         "Am", "Em", "Dm", "Bm", "Gm", "Cm", "Fm", "C#m", "F#m", "G#m"]
    )
with col3:
    tempo_input = st.selectbox(
        "Pilih Tempo",
        ["SLOW", "MEDIUM_SLOW", "MEDIUM", "FAST", "VERY_FAST"]
    )

CREATIVITY_PRESETS = {
    1: {"label": "🎯 Konvensional",  "desc": "Chord umum & mudah ditebak",         "temperature": 0.8,  "top_k": 5,  "top_p": 0.85, "repetition_penalty": 1.5},
    2: {"label": "🎵 Seimbang",      "desc": "Variasi natural, tetap musikal",      "temperature": 1.1,  "top_k": 10, "top_p": 0.90, "repetition_penalty": 2.0},
    3: {"label": "🎨 Eksperimental", "desc": "Chord tidak terduga & lebih kreatif", "temperature": 1.4,  "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.5},
}

SONG_STRUCTURES = {
    "Minimalis":  ["Verse", "Chorus"],
    "Standar":    ["Intro", "Verse", "Chorus", "Verse", "Chorus", "Outro"],
    "Lengkap":    ["Intro", "Verse", "Pre-Chorus", "Chorus", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Chorus", "Outro"],
    "Custom":     [],
}

SECTION_LENGTH = {
    "Intro": 4, "Verse": 8, "Pre-Chorus": 4,
    "Chorus": 8, "Bridge": 4, "Outro": 4,
}

SECTION_COLOR = {
    "Intro": "🟦", "Verse": "🟩", "Pre-Chorus": "🟨",
    "Chorus": "🟥", "Bridge": "🟪", "Outro": "⬜",
}

col_struct, col_creative = st.columns([1, 1])
with col_struct:
    structure_choice = st.selectbox("🎼 Struktur Lagu", list(SONG_STRUCTURES.keys()))

if structure_choice == "Custom":
    all_sections = ["Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Outro"]
    selected_sections = st.multiselect(
        "Pilih dan urutkan seksi lagu:",
        all_sections,
        default=["Verse", "Chorus"]
    )
    song_structure = selected_sections
else:
    song_structure = SONG_STRUCTURES[structure_choice]

creativity_level = st.select_slider(
    "🎚️ Tingkat Kreativitas",
    options=[1, 2, 3],
    value=2,
    format_func=lambda x: CREATIVITY_PRESETS[x]["label"]
)
st.caption(CREATIVITY_PRESETS[creativity_level]["desc"])

# UI Element untuk pemilihan Kandidat telah dihilangkan agar lebih sederhana dan otomatis.

if st.button("Generate Chord Progression", use_container_width=True):
    preset = CREATIVITY_PRESETS[creativity_level]

    if not song_structure:
        st.warning("Pilih minimal satu seksi lagu.")
        st.stop()

    with st.spinner("Model sedang berpikir..."):
        song_data   = []
        all_abs     = []

        seen_sections   = {}
        prev_last_roman = None

        for section in song_structure:
            length = SECTION_LENGTH.get(section, 8)
            count  = seen_sections.get(section, 0)

            temp_adj = preset["temperature"] + (count * 0.15)
            seen_sections[section] = count + 1

            MIN_CHORDS = 2
            roman = []
            
            # n_candidates dikunci secara default ke angka 5 untuk menjamin kualitas terbaik
            optimal_candidates = 5 
            
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
                    n_candidates=optimal_candidates,
                    prev_last_roman=prev_last_roman
                )
                if len(roman) >= MIN_CHORDS:
                    break

            if roman:
                prev_last_roman = roman[-1]

            abs_c = roman_to_absolute(roman, key_root=key_input)
            song_data.append({"section": section, "roman": roman, "abs": abs_c})
            all_abs.extend(abs_c)

        midi_file = create_midi(all_abs, tempo_input, "generated_chord.mid")

    st.success("Berhasil di-generate!")
    st.subheader("🎼 Hasil Generasi")

    for entry in song_data:
        section   = entry["section"]
        icon      = SECTION_COLOR.get(section, "🎵")
        roman_str = " → ".join(entry["roman"]) if entry["roman"] else "-"
        abs_str   = " → ".join(entry["abs"])   if entry["abs"]   else "-"

        with st.expander(f"{icon} **{section}**", expanded=True):
            st.markdown(f"**Roman Numeral:** `{roman_str}`")
            st.markdown(f"**Absolute Chords (Key: {key_input}):** `{abs_str}`")

    st.divider()
    with open(midi_file, "rb") as file:
        st.download_button(
            label="⬇️ Download Full Song MIDI",
            data=file,
            file_name="full_song.mid",
            mime="audio/midi",
            use_container_width=True
        )