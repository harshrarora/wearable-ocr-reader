# Wearable AI Reading Cap

A computer vision-based assistive system that reads printed text aloud using a camera and OCR. The goal is to help visually impaired users read books, labels, or printed documents using a wearable device.

The system is currently tested on a laptop using webcam input. Target deployment is a Raspberry Pi–based wearable device.


## Overview

The system captures frames from a camera, detects text regions, extracts text using OCR, and reads it aloud. A stability check ensures OCR runs only when the camera frame is steady, and a state machine prevents repeated reading of the same text when the camera remains pointed at the same page.

The architecture is modular and divided into three layers:
- Perception (vision processing)
- Intelligence (OCR and text processing)
- Interaction (guidance and speech output)

## Features

- Real-time text detection from camera frames
- OCR using EasyOCR
- Motion stability detection to avoid blurred reads
- Text cleaning before speech output
- Text-to-speech reading using pyttsx3
- State machine to prevent repeated reading of the same text
- Optional document boundary detection


## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CAMERA INPUT                         │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  PERCEPTION LAYER       │
        ├─────────────────────────┤
        │ • Stability Detection   │
        │ • Text Detection        │
        │ • Document Detection    │
        │ • Finger Tracking       │                         
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  INTELLIGENCE LAYER     │
        ├─────────────────────────┤
        │ • Intent Resolution     │
        │ • OCR Engine            │
        │ • OCR Fusion            │
        │ • Text Cleaning         │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  INTERACTION LAYER      │
        ├─────────────────────────┤
        │ • Spatial Guidance      │
        │ • TTS Manager           │
        │ • State Machine         │
        └─────────────────────────┘
```

### Module Breakdown

**Perception:**
- `stability.py` - Optical flow-based motion detection
- `text_detector.py` - EasyOCR-based text region detection
- `document_detector.py` - Page boundary detection via contour analysis
- `finger_tracker.py` - Experimental MediaPipe-based hand tracking module

**Intelligence:**
- `intent_resolver.py` - Decision logic for system modes
- `ocr_engine.py` - Image preprocessing + OCR execution
- `ocr_fusion.py` - Temporal fusion across multiple frames
- `text_cleaner.py` - Post-processing for clean TTS output

**Interaction:**
- `guidance.py` - Spatial positioning audio cues
- `tts_manager.py` - Threaded text-to-speech
- `state_machine.py` - System state management


## Project Structure

```
wearable-ocr-reader/
├── camera/
│   └── camera_manager.py      
├── perception/
│   ├── stability.py          
│   ├── text_detector.py      
│   ├── document_detector.py   
│   └── finger_tracker.py  #experimental module  
├── intelligence/
│   ├── intent_resolver.py    
│   ├── ocr_engine.py          
│   ├── ocr_fusion.py          
│   └── text_cleaner.py        
├── interaction/
│   ├── guidance.py            
│   ├── tts_manager.py         
│   └── state_machine.py       
├── utils/
│   └── logger.py              
├── config.py                  
├── main.py                   
└── requirements.txt           
```

## Configuration

All tunable parameters are in `config.py`:

```python
# Key settings you might want to adjust:
CAMERA_INDEX = 0               # Change if using external webcam
OCR_LANGUAGES = ["en"]         # Add more: ["en", "hi", "es"]
TTS_RATE = 160                 # Speech speed (words per minute)
STABILITY_THRESHOLD = 2.0      # Lower = stricter stability requirement
```

## Technologies Used

- Python
- OpenCV
- EasyOCR
- NumPy
- pyttsx3

## Installation

1. Clone the repository
```bash
git clone https://github.com/harshrarora/wearable-ocr-reader.git
cd wearable-ocr-reader
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the system
```bash
python main.py
```

**Note:** On first run, EasyOCR will download language models (~100MB).

## Current Status

- Text detection and OCR working
- Stability detection working
- Text-to-speech working
- Spatial guidance system under testing
- Finger tracking module included but not fully tested
- Raspberry Pi deployment planned