# Medical Script Detector 

Transforming messy medical prescriptions into structured digital data using
Vision Transformers (TrOCR) and Domain-Specific NLP.

MedScript-OCR is an end-to-end AI pipeline designed to solve the age-old problem
of "doctor's handwriting." Unlike generic OCR tools, this system is optimized
for the medical domain, utilizing a hybrid approach of Computer Vision (CV) for
image preparation, Deep Learning for transcription, and Natural Language
Processing (NLP) for medical error correction.

## Key Features

  - Intelligent Word Segmentation: Utilizes OpenCV contour detection to isolate
    individual handwritten words from a prescription line, allowing the model to
    process complex multi-word entries.
  - Vision Transformer (TrOCR) Backbone: Employs Microsoft’s TrOCR (small), a
    state-of-the-art encoder-decoder model that treats handwriting recognition
    as a sequence-to-sequence task.
  - Medical NLP Autocorrect: A custom post-inference layer that uses Fuzzy
    String Matching (Levenshtein Distance) against an FDA-aligned drug database
    to correct OCR misspellings (e.g., "Amoxioillin" ➔ "Amoxicillin").
  - Shorthand Translation: Automatically expands Latin medical "Sig Codes"
    (e.g., bid, po, prn) into plain English instructions.
  - Safety-First Design: Implements probabilistic Confidence Scoring for every
    prediction, flagging "Low Confidence" results for human-in-the-loop
    verification.

## The Pipeline

The system processes data through five distinct stages:

1.  Preprocessing: OpenCV-based Grayscale conversion and Adaptive Thresholding
    to isolate ink from paper noise.
2.  Segmentation: Morphological dilation and contour sorting to identify and
    order individual words.
3.  Inference: The Vision Transformer (ViT) encoder extracts visual features,
    while the GPT-2 decoder generates the character sequence.
4.  NLP Correction: RapidFuzz-based cross-referencing against a medical
    knowledge base.
5.  Sig Translation: Final conversion of medical abbreviations to human-readable
    dosage instructions.

🛠️ Tech Stack

  - Deep Learning: PyTorch, HuggingFace Transformers
  - Computer Vision: OpenCV, PIL
  - NLP: RapidFuzz (Levenshtein Distance)
  - Dashboard: Streamlit

## Installation & Setup

1. Clone the Repo

git clone https://github.com/tanishaacodes/medical-nlp.git
cd medscript-nlp

2. Environment Setup

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch torchvision transformers opencv-python rapidfuzz streamlit Pillow

3. Run the App

streamlit run app.py




Thank you

