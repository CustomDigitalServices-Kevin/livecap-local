"""LiveCap Local — sous-titrage et traduction temps reel 100% local.

Pipeline (voir CONTRACT.md pour le detail des interfaces) :
    capture -> audio_q -> asr/segmenter -> segment_q -> translator -> ui_q -> overlay
"""

__version__ = "0.1.0"
