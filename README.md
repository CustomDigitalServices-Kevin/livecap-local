# LiveCap Local

Sous-titrage et traduction temps reel (FR <-> EN) des reunions Teams, 100%
local. Remplace Windows Live Captions par un pipeline maison : capture audio
systeme + micro -> transcription Kyutai `stt-1b-en_fr` (moshi) -> traduction
`qwen3:4b-instruct` via Ollama localhost -> overlay PyQt6.

Voir `CONTRACT.md` pour les interfaces entre modules, et le PRD
(`.taskmaster/docs/prd.txt`) pour le perimetre complet.

## Prerequis

- Windows, GPU NVIDIA compatible CUDA 12.8 (valide sur RTX 5080, Blackwell sm_120)
- Python 3.12
- [uv](https://docs.astral.sh/uv/) installe (`pip install uv` ou standalone)
- [Ollama](https://ollama.com) installe et le modele `qwen3:4b-instruct` telecharge
  (`ollama pull qwen3:4b-instruct`)

## Installation

Le venv se cree avec `uv`. Torch doit venir de l'index PyTorch cu128 (l'index
PyPI par defaut ne fournit pas les wheels CUDA) :

```bash
uv venv --python 3.12 .venv

# 1. Torch cu128
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cu128 torch

# 2. Le reste des dependances, depuis PyPI
uv pip install --python .venv moshi soundfile PyAudioWPatch PyQt6 pyyaml pytest pytest-qt

# 3. PIEGE CONFIRME : l'etape 2 (moshi) redescend torch vers un build CPU
#    (2.9.1) depuis PyPI par defaut, car moshi ne contraint pas l'index.
#    Reinstaller torch cu128 en DERNIER pour ecraser ce downgrade :
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
```

Versions validees au POC (`C:\dev\Antigravity\tools\livecap-poc`, 2026-07-18) et
reconfirmees a l'installation de ce projet : `torch==2.11.0+cu128`, `moshi==0.2.13`.

`uv pip check` signale une incompatibilite `moshi requires torch<2.10` : c'est
un plafond de metadonnees de moshi trop strict, sans impact fonctionnel — le
POC a valide `torch 2.11.0+cu128` + `moshi 0.2.13` en conditions reelles
(RTX 5080, transcript EN/FR correct). Le meme warning est present dans le venv
du POC. A surveiller si moshi publie une nouvelle version, sinon ignorer.

### Verification post-install

```bash
.venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available()"
.venv/Scripts/python.exe -c "import moshi, PyQt6, pyaudiowpatch, yaml"
uv pip check --python .venv
```

## Configuration

Editer `config.yaml` a la racine du projet (schema documente dans
`CONTRACT.md`) :

- `target_lang` : `fr` ou `en` — langue vers laquelle traduire (la langue
  source est detectee implicitement par le modele ASR bilingue)
- `devices.loopback` / `devices.mic` : `auto`, un nom de peripherique, ou
  `none` pour `mic` (desactive la capture micro)
- `overlay.position`, `overlay.font_size`, `overlay.click_through`,
  `overlay.history` : apparence et comportement de l'overlay
- `log_transcript` : si `true`, ecrit le transcript horodate dans
  `transcripts/` (dossier ignore par git)

`livecap/config.py` valide ce fichier au demarrage et leve une erreur precise
si une cle est manquante ou invalide.

## Lancement

*Sera complete en tache 6* : `launch.cmd` activera le venv, posera
`NO_TORCH_COMPILE=1`, verifiera qu'Ollama tourne, et lancera `main.py`.

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```

## Limites connues

*Sera complete en tache 6* (limites mesurees sur le pipeline complet :
latence bout en bout, qualite de la synchronisation micro/loopback,
comportement hors ligne d'Ollama, etc.)
