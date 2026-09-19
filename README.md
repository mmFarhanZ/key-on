# Key-on 🎼

Key-on is a Streamlit web app that generates chord progressions for a song.
Pick a mood, key, tempo, and song structure, and a Hybrid Transformer-LSTM model
produces a progression for every section (Intro, Verse, Chorus, ...). You can
preview the result with the built-in piano player and download it as a MIDI file.

## Features

- **Mood selection:** Happy, Sad, Dark, Peaceful, Melancholic, Neutral
- **Key and tempo:** 14 major keys, 10 minor keys, and 5 tempo levels
- **Song structures:** Minimalist, Standard, Full, or a Custom set of sections
- **Creativity level:** Conventional, Balanced, or Experimental sampling presets
- **Output:** chord names plus the Roman numeral analysis of every section
- **Piano preview:** in-browser playback with the Web Audio API (no plugins needed)
- **MIDI export:** four-voice voice leading with light humanization

## How to use

1. **Mood of the song:** choose the emotional color of the piece.
2. **Root note & Tempo:** choose a major key (`C`, `G`, `Bb`, ...) or a minor key (`Am`, `F#m`, ...), then a tempo from *Slow* to *Very Fast*.
3. **Structure & Creativity:** choose a song structure and how adventurous the chords should be.
   With *Custom*, pick any combination of sections yourself.
4. Click **Generate Chord Progression**.
5. Read the result cards, press ▶ on the piano player to listen, and click **Download MIDI** to save the file.

### Song structures

| Structure | Sections |
|---|---|
| Minimalist | Verse, Chorus |
| Standard | Intro, Verse, Chorus, Verse, Chorus, Outro |
| Full | Intro, Verse, Pre-Chorus, Chorus, Verse, Pre-Chorus, Chorus, Bridge, Chorus, Outro |
| Custom | Any combination of the six sections |

Each section has a fixed length: Verse and Chorus have 8 chords, and Intro, Pre-Chorus, Bridge, and Outro have 4.

### Creativity levels

| Level | Sampling behavior |
|---|---|
| Conventional | Common and predictable chords (temperature 0.8, top-k 5) |
| Balanced | Natural variation (temperature 1.1, top-k 10) |
| Experimental | Unexpected, more adventurous chords (temperature 1.4, top-k 20) |

## Example output

> The examples below show what a result looks like. Chord names are converted from
> the Roman numerals with the app's own key-mapping logic, but the progressions
> themselves are illustrative. Because the model samples its output, every run
> gives a different result.

### Example 1: a bright song in a major key

**Settings:** Mood `Happy` · Root Note `G` · Tempo `Medium` · Structure `Minimalist` · Creativity `Balanced`

```
Generated Results
2 sections · 16 chords
Mood: Happy | Root Note: G | Tempo: Medium | Structure: Minimalist | Creativity: Balanced

┃ VERSE
┃ G › Em › C › D › G › Bm › C › D
┃ I › vi › IV › V › I › iii › IV › V

┃ CHORUS
┃ C › D › Bm › Em › C › G › D › G
┃ IV › V › iii › vi › IV › I › V › I

[ ▶ Piano player ]
[ ⬇ Download MIDI ]
```

### Example 2: a somber song in a minor key

**Settings:** Mood `Sad` · Root Note `Am` · Tempo `Medium Slow` · Structure `Minimalist` · Creativity `Balanced`

```
Generated Results
2 sections · 16 chords
Mood: Sad | Root Note: Am | Tempo: Medium Slow | Structure: Minimalist | Creativity: Balanced

┃ VERSE
┃ Am › F › C › G › Am › Dm › G › Am
┃ i › bVI › bIII › bVII › i › iv › bVII › i

┃ CHORUS
┃ F › G › Am › Dm › F › G › E › Am
┃ bVI › bVII › i › iv › bVI › bVII › V › i

[ ▶ Piano player ]
[ ⬇ Download MIDI ]
```

Reading the cards: the first line of each section is the chord progression in the
chosen key, and the second line is the same progression as Roman numerals,
which stays the same no matter which key you pick. The downloaded file is named
`Key-on_output.mid` and can be opened in any DAW or notation program.

## How it works

1. A causal Transformer encoder (6 layers, 4 heads, d_model 256) reads a
   context window of up to 32 tokens: start, genre, mood, tempo, then the chords so far.
2. A 2-layer LSTM (hidden size 128) on top predicts the next Roman-numeral chord.
3. Several candidates are sampled (top-k / top-p, repetition penalty) and then
   re-ranked by a mix of model likelihood and a rule-based harmonic score
   (common progressions, cadences, ii-V-I).
4. Roman numerals are converted to absolute chords in the chosen key, then
   voice-led and written to MIDI.

## Project structure

```
.
├── appV2.py                        # Streamlit app (entry point)
├── chord_vocab.pkl         # Vocabulary and token mappings
├── chord_transformer_decoder_V3.pt # Trained model checkpoint
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # Theme and server settings
├── .gitignore
└── README.md
```

## Installation

Requires Python 3.10 to 3.12.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
streamlit run appV2.py
```

The app opens at http://localhost:8501. The `.pkl` and `.pt` files must sit in the
same folder as `appV2.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for the `.pkl` or `.pt` file | Put both files next to `appV2.py` and check that the names match exactly. |
| `Weights only load failed` (`UnpicklingError`) | Newer PyTorch versions default to `weights_only=True`. If your checkpoint holds extra objects, change the call in `load_assets()` to `torch.load(..., weights_only=False)`. Only do this for files you trust. |
| `Import "torch" could not be resolved` in VS Code | Select the interpreter that has the packages installed (**Python: Select Interpreter**). |
| Text hard to read in dark mode | Keep `.streamlit/config.toml`; it locks the light theme the custom styling is designed for. |
