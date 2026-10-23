# LiveCap Local

Sous-titrage et traduction temps reel (FR <-> EN) des reunions Teams, 100%
local. Remplace Windows Live Captions par un pipeline maison : capture audio
systeme + micro -> transcription Kyutai `stt-1b-en_fr` (moshi) -> traduction
`qwen3:4b-instruct` via Ollama sur 127.0.0.1:11434 -> overlay PyQt6.

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

```bash
launch.cmd
```

`launch.cmd` : pose `NO_TORCH_COMPILE=1` (defense en profondeur, deja pose par
`livecap/asr.py` et `main.py`), verifie qu'Ollama repond sur `127.0.0.1:11434`
(le demarre en arriere-plan sinon, best-effort — `main.py` refait de toute
facon la verification/relance autoritative via
`livecap.translator.ensure_ollama_running`), puis lance
`.venv\Scripts\python.exe main.py` avec les arguments passes tels quels.

Usage direct de `main.py` (`launch.cmd` transmet les memes arguments) :

```bash
# Mode normal : capture WASAPI (loopback + micro), overlay PyQt6
launch.cmd
launch.cmd --config config.yaml --device cuda

# Rejoue un fichier wav au lieu de capturer (court-circuite capture.py) :
# decoupe en blocs de 80 ms, asr.flush() en fin de fichier, quitte proprement.
launch.cmd --input-wav tests\fixtures\test_fr.wav

# --no-overlay : headless, exige --input-wav (pas de mode live sans overlay
# pour l'instant). Utilise par l'E2E scripte (tests/test_e2e.py).
launch.cmd --input-wav tests\fixtures\test_fr.wav --no-overlay
```

### Panneau de reglages (roue dentee)

La roue dentee, a gauche de la croix, ouvre un panneau modifiable en cours
de reunion. Chaque changement s'applique immediatement et est enregistre
dans `settings.local.yaml`, fusionne par-dessus `config.yaml` au demarrage
suivant. Supprimer ce fichier remet les valeurs de `config.yaml`.

| Reglage | Effet |
|---------|-------|
| Modele | `qwen3:4b-instruct` (~126 ms) ou `qwen3:14b-q4_K_M` (~351 ms avec l'ASR charge) |
| Contexte | 0 a 5 phrases precedentes envoyees au modele |
| Traduire vers | francais ou anglais |
| Taille du texte | 10 a 72 px |

**Choix du modele.** Mesure du 2026-07-20 sur RTX 5080 (16,3 Go) : le 14B
traduit mieux les pronoms ("le lui transmettre" la ou le 4B produit du
flou), mais une fois le checkpoint Kyutai charge il ne tient plus
entierement en VRAM. Ollama rapporte alors 11%/89% CPU/GPU et la VRAM
atteint 15,3/16,3 Go. La latence reste sous le budget de 500 ms mais il n'y
a plus de marge pour un autre consommateur GPU. Le 4B reste le defaut.

**Contexte.** Les phrases precedentes sont rejouees comme de vrais tours de
dialogue. Mesure sur 5 cas : cela corrige la polysemie ("board summary" ->
"resume du conseil" au lieu de "du tableau") sur les deux modeles, et ne
change rien au reste. Ce n'est donc pas un gain general. Cout : environ
+70 tokens de prompt pour +1 a +10 ms.

### Raccourci Bureau

Un raccourci `LiveCap Local` pointe vers `launch.cmd` (icone
`assets/livecap.ico`, fenetre console demarree reduite). Le recreer, ou le
recreer apres un deplacement du projet :

```powershell
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut((Join-Path $ws.SpecialFolders('Desktop') 'LiveCap Local.lnk'))
$s.TargetPath = 'C:\dev\Antigravity	ools\livecap-local\launch.cmd'
$s.WorkingDirectory = 'C:\dev\Antigravity	ools\livecap-local'
$s.IconLocation = 'C:\dev\Antigravity	ools\livecap-localssets\livecap.ico'
$s.WindowStyle = 7   # 7 = reduit (https://learn.microsoft.com/en-us/previous-versions/windows/internet-explorer/ie-developer/windows-scripting/w88k7fw2(v=vs.84))
$s.Save()
```

L'icone se regenere avec `.venv\Scripts\python.exe tools/make_icon.py`.

Il n'y a pas d'executable autonome : l'application tourne dans son venv et
`launch.cmd` en est le point d'entree. La console reduite reste ouverte
pendant toute la session et la fermer arrete l'outil, exactement comme le
bouton `X` de l'overlay.

Arret :
- **Mode normal (overlay)** : bouton `X` en haut a droite du panneau,
  raccourci `Ctrl+Alt+Q`, ou Ctrl+C dans la console — les deux declenchent la meme sequence d'arret
  (`Pipeline.shutdown()` : chaque etage `stop()` + `join(timeout=5s)`, fichier
  transcript ferme). Ctrl+C sous Qt exige un idiome specifique (un `QTimer`
  sans effet qui force periodiquement l'interpreteur a repasser en bytecode
  Python, sinon le signal n'est jamais livre tant que la boucle d'evenements
  Qt tourne) : voir le docstring de `main.py` et les sources citees
  (machinekoder.com, mailing list PyQt de Riverbank).
- **Mode `--input-wav --no-overlay`** : se termine seul des que le fichier est
  entierement rejoue, traduit et journalise (aucune interaction requise).

### Controles de l'overlay

| Action | Geste / raccourci |
|--------|-------------------|
| Deplacer le panneau | clic gauche + glisser (hors bouton `X` et poignee) |
| Redimensionner | glisser la poignee en bas a droite (min 240 x 60 px) |
| Faire defiler l'historique | molette ; revenir en bas rearme le suivi du direct |
| Reinitialiser position et taille | `Ctrl+Alt+R` |
| Pause / reprise | `Ctrl+Alt+P` (voir Limites connues) |
| Fermer (arrete tout le pipeline) | bouton `X`, ou `Ctrl+Alt+Q` |

Le panneau grandit avec la transcription jusqu'a 45% de la hauteur d'ecran,
puis cesse de grandir et devient une zone defilante : `overlay.history` borne
la profondeur remontable (400 segments par defaut).

Ces trois interactions souris (deplacer, redimensionner, faire defiler)
exigent `overlay.click_through: false`. Avec `click_through: true` la fenetre
laisse passer les clics et ne recoit plus aucun evenement souris : seuls les
raccourcis clavier restent disponibles.

L'overlay est une fenetre `Qt.Tool` sans cadre : par construction elle
n'apparait pas dans la barre des taches Windows, d'ou le bouton `X` et le
raccourci `Ctrl+Alt+Q` comme seuls moyens de fermeture a la souris/clavier.

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```

139 tests (2026-07-18), tous verts, GPU inclus (`pytest -m gpu` pour les
selectionner seuls). Necessitent une RTX 5080 ou equivalent + Ollama demarre
avec `qwen3:4b-instruct` disponible pour les tests marques `@pytest.mark.gpu`
(`tests/test_asr_integration.py`, `tests/test_e2e.py`) — skip documente
(`OLLAMA_UNAVAILABLE_REASON`) si Ollama ne repond pas, sans faire echouer la
suite.

## Architecture

```
capture.py (WASAPI loopback+mic, 16 kHz mono)
   │  AudioChunk
   ▼
audio_q ──▶ AsrSegmenterThread (main.py)
                │  KyutaiStreamingAsr (asr.py) + Segmenter (segmenter.py)
                │  TranscriptSegment (final uniquement, cf CONTRACT.md)
                ▼
            segment_q ──▶ TranslatorThread (translator.py, Ollama qwen3:4b-instruct)
                                │  TranslatedSegment
                                ▼
                            ui_q ──▶ UiFanoutThread (main.py)
                                        │  log transcript (TranscriptLogger)
                                        ▼
                                    overlay_ui_q ──▶ LiveOverlay (main.py, sur Overlay de overlay.py)
```

`main.py` assemble ce pipeline (tache 6) : chaque etage tourne sur son propre
thread (sauf l'overlay, sur le thread principal — Qt l'exige), `--input-wav`
remplace `capture.py` par `WavFeederThread` en amont de `audio_q`, le reste du
pipeline est identique en mode live et en mode fichier. Le texte VO progressif
(callback `on_partial` de `Segmenter`, jamais construit en `TranscriptSegment`
per CONTRACT.md) transite par une queue interne dediee (`partial_q`), drainee
par un second `QTimer` de `LiveOverlay` — jamais par `ui_q`.

## Limites connues

- **Anti-echo v1** : attenuation du micro basee sur le niveau (RMS du
  loopback), pas d'annulation d'echo acoustique (AEC) reelle. Documente dans
  `livecap/capture.py`. Suffisant pour reduire (pas eliminer) la
  retranscription de l'audio des haut-parleurs capte par le micro.
- **Raccourci pause non global** : `Ctrl+Alt+P` (overlay) utilise un
  `QShortcut` avec `Qt.ShortcutContext.ApplicationShortcut` : il fonctionne
  tant que LiveCap Local a le focus, mais PAS quand une autre application
  (Teams) est au premier plan. Un vrai raccourci systeme demanderait
  `RegisterHotKey`/`WM_HOTKEY` (documente peu fiable sur Qt 6.2-6.5, cf issue
  Qt forum citee dans `livecap/overlay.py`) ou une dependance tierce
  (keyboard/pynput/pywin32), hors perimetre v1 (aucune nouvelle dependance).
- **FR/EN uniquement** : modele Kyutai `stt-1b-en_fr` bilingue, detection de
  langue heuristique (stopwords + ratio de caracteres accentues, cf
  `livecap/segmenter.py`) — pas de vraie identification de langue, adequat
  pour ce perimetre uniquement.
- **Resampling capture** : interpolation lineaire (`numpy.interp`), pas de
  polyphase (scipy hors dependances) — legere alteration des hautes
  frequences, sans impact mesure sur la qualite de transcription.
- **Mode live headless** (`--no-overlay` sans `--input-wav`) volontairement
  non supporte pour l'instant : `--no-overlay` exige `--input-wav` (message
  d'erreur explicite dans `main.py`), la capture WASAPI sans affichage
  n'ayant pas de cas d'usage identifie dans le PRD.
- **Diarisation** : hors perimetre v1 (PRD), pas de distinction des locuteurs.

## Resultats mesures (2026-07-18, RTX 5080, torch 2.11.0+cu128)

- ASR streaming (`tests/test_asr_integration.py`) : 16-18 ms/step mesures au
  POC, plafond de sanite teste a < 200 ms/step.
- Traduction Ollama `qwen3:4b-instruct` : latence mediane < 500 ms (critere
  PRD, `tests/test_translator.py::test_translate_text_live_ollama_latency`).
- Latence bout en bout (1er segment final -> 1ere traduction, mode
  `--input-wav`, `tests/test_e2e.py`) : mesuree systematiquement 140-1780 ms
  selon l'etat de chauffe d'Ollama (premier appel = chargement modele en
  VRAM), toujours trouvee tres en dessous du seuil PRD de 2,5 s.
- Chargement modele Kyutai `stt-1b-en_fr` (cold start) : ~2,4-2,9 s. Bug
  trouve et corrige pendant l'integration (tache 6) : un producteur audio
  (capture ou `--input-wav`) demarrant avant la fin de ce chargement remplit
  la queue bornee `audio_q` et perd des chunks — corrige par un
  `threading.Event` (`AsrSegmenterThread.ready`) que `main.py` attend avant de
  demarrer le producteur (`ASR_READY_TIMEOUT_S = 60 s`).
- Verification reseau (methode : sondage `netstat -ano` toutes les 200 ms
  pendant un run `--input-wav --no-overlay` complet, filtre sur le PID du
  process `.venv` du pipeline) : 0/20 echantillons avec une connexion etablie
  depuis le process LiveCap Local (loopback Ollama trop bref pour ce pas
  d'echantillonnage, et surtout aucune connexion externe). Un process
  compagnon utilisant le Python systeme (hors `.venv`) apparait pour
  n'importe quel script Python sur cette machine (verifie avec un script sans
  aucun import) et etablit une connexion HTTPS externe : infrastructure de
  l'environnement de session, sans rapport avec le code de ce projet.

## Licence

GNU General Public License v3.0 (voir `LICENSE`).

Ce choix est impose par une dependance, pas par preference : `PyQt6` est
distribue par Riverbank Computing sous GPL v3 ou licence commerciale, et
explicitement PAS sous LGPL
(https://www.riverbankcomputing.com/software/pyqt/). Toute application qui
l'embarque et qui est distribuee doit donc etre GPL v3, ou disposer d'une
licence commerciale Riverbank.

Les autres dependances sont permissives et ne contraignent rien :

| Dependance | Licence |
|------------|---------|
| moshi, pyyaml | MIT |
| torch, soundfile | BSD-3-Clause |
| PyAudioWPatch | Apache-2.0 |
| PyQt6 | **GPL v3 / commerciale** |

Les modeles telecharges a l'execution (Kyutai `stt-1b-en_fr` via Hugging
Face, `qwen3` via Ollama) ont leurs propres licences, independantes de
celle de ce code.

## À propos

Conçu et maintenu par [Kevin Tomas](https://www.custom-digital-services.com/?utm_source=github&utm_medium=readme&utm_campaign=livecap-local), consultant freelance Power Platform et IA Microsoft (Custom Digital Services).
Autres outils gratuits : [www.custom-digital-services.com/outils](https://www.custom-digital-services.com/outils/?utm_source=github&utm_medium=readme&utm_campaign=livecap-local)
