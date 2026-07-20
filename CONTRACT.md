# CONTRACT.md — interfaces entre modules LiveCap Local

Ce fichier est la reference que les taches 2 a 6 doivent respecter. Les
dataclasses et fabriques de queues sont implementees (pas juste documentees)
dans `livecap/contract.py` : importer depuis ce module, ne pas redefinir ces
formes ailleurs.

## Pipeline

```
capture.py --AudioChunk--> audio_q --> asr.py + segmenter.py --TranscriptSegment--> segment_q --> translator.py --TranslatedSegment--> ui_q --> overlay.py
```

`main.py` (tache 6) instancie les 3 queues via `livecap.contract.make_audio_queue`,
`make_segment_queue`, `make_ui_queue`, lance un thread par etage (capture, asr+segmenter,
translator) et fait tourner l'overlay PyQt6 sur le thread principal (Qt l'exige).

## Dataclasses (`livecap/contract.py`)

### `AudioChunk`

| Champ | Type | Description |
|---|---|---|
| `samples` | `NDArray[np.float32]` | audio mono, 16 kHz, shape `(n_samples,)` |
| `timestamp` | `float` | `time.monotonic()` au debut du bloc |

Produit par `capture.py`. Consomme par `asr.py`.

### `TranscriptSegment`

| Champ | Type | Description |
|---|---|---|
| `text` | `str` | texte transcrit, langue source, deja `strip()` |
| `lang` | `str` | `"fr"` ou `"en"` (detection implicite du modele bilingue) |
| `t_start` | `float` | `time.monotonic()` debut du segment |
| `t_end` | `float` | `time.monotonic()` au point de stabilisation |
| `final` | `bool` | toujours `True` sur `segment_q` — les partiels instables restent internes a `asr.py`/`segmenter.py` |

Regle de segmentation (tache 3) : un segment est stabilise sur ponctuation
terminale (`.`, `?`, `!`) OU silence > 700 ms. Produit par `segmenter.py`.
Consomme par `translator.py`.

### `TranslatedSegment`

| Champ | Type | Description |
|---|---|---|
| `source` | `TranscriptSegment` | segment source complet |
| `text` | `str` | traduction dans `target_lang`. `""` si la traduction a echoue/timeout (fallback VO seule, ne jamais bloquer l'affichage) |
| `latency_ms` | `float` | latence de traduction mesuree, `0.0` si `text == ""` |

Produit par `translator.py`. Consomme par `overlay.py`.

## Queues (`livecap/contract.py`)

Toutes des `queue.Queue` standard (thread-safe, bloquantes avec timeout cote
consommateur). Bornees (`maxsize`) pour eviter un emballement memoire si un
etage ralentit ; le producteur doit gerer `queue.Full` (drop le plus ancien ou
attendre selon le module, a documenter dans chaque module).

| Fabrique | Type d'element | Producteur | Consommateur | `maxsize` par defaut |
|---|---|---|---|---|
| `make_audio_queue()` | `AudioChunk` | `capture.py` | `asr.py` | 64 |
| `make_segment_queue()` | `TranscriptSegment` | `segmenter.py` | `translator.py` | 32 |
| `make_ui_queue()` | `TranslatedSegment` | `translator.py` | `overlay.py` | 32 |

## Schema `config.yaml` (`livecap/config.py`)

```yaml
target_lang: fr | en      # langue cible de la traduction

devices:
  loopback: auto | "<nom>"          # peripherique WASAPI loopback
  mic: auto | none | "<nom>"        # peripherique micro, "none" = desactive

overlay:
  position: top-left | top-center | top-right | bottom-left | bottom-center | bottom-right
  font_size: <int > 0>
  click_through: <bool>             # WS_EX_TRANSPARENT si true
  history: <int >= 0>               # segments precedents affiches

log_transcript: <bool>              # ecrit transcripts/ si true
```

`livecap.config.load_config(path)` lit et valide ce fichier, leve
`ConfigError` (sous-classe de `ValueError`) avec un message precis sur la
premiere cle manquante ou valeur invalide.

## Modules attendus par tache

| Fichier | Tache | Role |
|---|---|---|
| `livecap/capture.py` | 2 | Capture WASAPI loopback + micro, resampling 16 kHz mono, mixage, push `AudioChunk` |
| `livecap/asr.py` | 3 | Kyutai `stt-1b-en_fr` via moshi (streaming), consomme `audio_q` |
| `livecap/segmenter.py` | 3 | Stabilise les tokens ASR en `TranscriptSegment`, push `segment_q` |
| `livecap/translator.py` | 4 | Client Ollama `qwen3:4b-instruct`, consomme `segment_q`, push `ui_q` |
| `livecap/overlay.py` | 5 | Fenetre PyQt6 borderless always-on-top, consomme `ui_q` |
| `main.py` | 6 | Assemble le pipeline, threads, arret propre, logging transcript, mode `--input-wav` pour l'E2E |

## Gotchas connus (a respecter, cf RAG `live-captioning-translation-stack-2026.md`)

- `NO_TORCH_COMPILE=1` **obligatoire** pour moshi sous Windows (Triton absent) :
  sinon le process s'arrete en silence apres le chargement, exit 0, aucun
  transcript. A poser dans le code (`asr.py` ou `main.py`), pas seulement dans
  le launcher, pour que les tests/imports directs soient aussi proteges.
- Ollama doit tourner en 127.0.0.1:11434 avant `translator.py` (adresse
  IPv4 litterale obligatoire : "localhost" resout en ::1 d'abord sous Windows
  alors qu'Ollama n'ecoute qu'en IPv4, ce qui coute ~2 s par appel) ; gerer
  absence/arret avec message clair (tache 4).
- `.vscode/settings.json` doit exclure `**/.venv/**` du watcher et de la
  recherche (tempete rg constatee le 2026-07-18 sur ce workspace).
