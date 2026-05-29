# Trained Model Weights

This folder stores the trained model weights produced by the Colab training pipeline.
The Streamlit app loads them on startup. **All three files are required** for the AI
generator to run — without them the app will fall back to the rule-based packer.

## Files to place here

| File                       | Approx size | Source                                                      |
| -------------------------- | ----------- | ----------------------------------------------------------- |
| `graph_vae.pth`            | ~2 MB       | Cell B output — Graph VAE trained on 3283 Indian RAG graphs |
| `geometry_predictor.pth`   | ~2 MB       | Cell D1 output — GCN regressor that predicts room boxes     |
| `best.pt` (optional)       | ~6 MB       | Cell A — YOLOv8 room detector (for future room-detection UI)|

Download these from Google Drive:
```
/MyDrive/yolo_runs/graph_vae.pth
/MyDrive/yolo_runs/geometry_predictor.pth
/MyDrive/yolo_runs/room_detector_v2/weights/best.pt   # optional
```

## Pipeline numbers (from training)

- **YOLO**: 1.2k images, 11 classes, mAP50 ≈ 54%
- **Graph VAE**: 3283 training graphs, 200 epochs, final loss 2.15
- **Geometry Predictor**: 500 epochs, final loss 0.38
- **End-to-end validation**: 100/100 valid plans (rejection sampling, ~23% acceptance, ~4 attempts per delivered plan)
