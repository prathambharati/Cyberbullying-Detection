"""Cyberbullying detection: a text classifier, a moderated feed and webcam login tools."""

import os

# TensorFlow is chatty by default. Hide its info logs and turn off the oneDNN
# custom ops, which also makes results repeatable from run to run.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

__version__ = "2.0.0"
