# The Visual Storyteller

An image captioning system that takes an image as input and produces a natural language description as output. The model bridges two distinct modalities: it interprets visual information and expresses that understanding through generated text.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Image Captioning Model                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Image     │    │   Encoder   │    │      Decoder        │ │
│  │  (224x224)  │───▶│  (ResNet50) │───▶│      (LSTM)         │ │
│  │             │    │             │    │                     │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│                            │                     │              │
│                            ▼                     ▼              │
│                     [256-dim features]    [Generated Caption]   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Model Components

| Component | Description |
|-----------|-------------|
| **Encoder** | Pre-trained ResNet50 CNN (frozen) for visual feature extraction |
| **Decoder** | 2-layer LSTM for autoregressive sequence generation |
| **Embedding Size** | 256 dimensions |
| **Hidden Size** | 256 dimensions |
| **Dropout** | 0.6 (for regularization) |

## Project Structure

```
The-Visual-Storyteller/
├── notebooks/
│   ├── data_and_training.ipynb   # Training (run on Colab with GPU)
│   ├── inference.ipynb           # Inference (run locally)
│   └── test_images/              # Place your images here for captioning
├── requirements.txt
└── README.md
```

## Quick Start

### Step 1: Train on Google Colab (GPU Required)

1. Upload `notebooks/data_and_training.ipynb` to [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **GPU**
3. Run all cells (dataset downloads automatically from Hugging Face)
4. Download the generated files from `models/` folder:
   - `best_model.pth` - Best model weights
   - `final_model.pth` - Final model with config
   - `vocab.pkl` - Vocabulary mappings
   - `val_images.pkl` - Validation image list

### Step 2: Run Inference Locally (CPU)

1. Place downloaded model files in `notebooks/models/`
2. Add your images to `notebooks/test_images/`
3. Run `inference.ipynb`:

```bash
cd notebooks
jupyter notebook inference.ipynb
```

## Usage

### Caption Your Own Images

1. Place images in `notebooks/test_images/` folder
2. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`
3. Run the inference notebook - captions will be generated automatically

### Programmatic Usage

```python
def generate_caption(image_path: str, model: any) -> str:
    """
    Takes a path to an image and returns a generated caption string.

    Args:
        image_path: Path to the image file
        model: The trained ImageCaptioningModel

    Returns:
        str: Generated caption for the image
    """
```

Example:
```python
caption = generate_caption('test_images/dog.jpg', model)
print(caption)  # "a dog running through the grass"
```

## Training Details

### Dataset
- **Flickr8k** from Hugging Face (`jxie/flickr8k`)
- 8,000 images with 5 captions each (40,000 total captions)
- 90/10 train/validation split

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Size | 256 |
| Hidden Size | 512 |
| LSTM Layers | 2 |
| Dropout | 0.6 |
| Batch Size | 32 |
| Learning Rate | 3e-4 |
| Max Epochs | 10 |
| Early Stopping | 3 epochs patience |

### Data Augmentation
- Random crop (256 → 224)
- Random horizontal flip (50%)
- Color jitter (brightness, contrast, saturation)

### Regularization Techniques
- Dropout on embeddings and LSTM output
- Early stopping based on validation loss
- Gradient clipping (max norm = 5.0)
- Learning rate scheduling (step decay)

## Model Performance

### Evaluation Metrics
The model is evaluated using word overlap between generated captions and ground truth captions.

### Expected Results
- Average word overlap: ~30-40%
- High quality captions (>50% overlap): ~20-30% of images
- The model performs best on:
  - Clear subjects (people, dogs, outdoor scenes)
  - Simple compositions with single main subject
  - Images similar to training distribution

### Limitations
- May confuse similar objects (e.g., dog vs. cat)
- Generic captions for complex scenes
- Limited vocabulary (words appearing <5 times are unknown)
- No attention mechanism (uses simple encoder-decoder)

## Sample Results

| Image Description | Generated Caption |
|-------------------|-------------------|
| Dog running on beach | "a brown dog is running on the beach" |
| Child playing | "a young boy is playing in the grass" |
| Mountain landscape | "a man is standing on a mountain" |

## Requirements

- Python 3.8+
- PyTorch, torchvision
- datasets (Hugging Face)
- numpy, pandas, Pillow, matplotlib, tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```

## File Descriptions

| File | Description |
|------|-------------|
| `data_and_training.ipynb` | Complete training pipeline - downloads data, builds vocabulary, trains model |
| `inference.ipynb` | Loads trained model, generates captions for test images, evaluates performance |
| `requirements.txt` | Python package dependencies |

## Future Improvements

- [ ] Add attention mechanism (show-attend-tell)
- [ ] Implement beam search decoding
- [ ] Train on larger datasets (COCO, Flickr30k)
- [ ] Add BLEU/METEOR evaluation metrics
- [ ] Support for batch inference
- [ ] Web interface for easy captioning

## References

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) (Vinyals et al., 2015)
- [Flickr8k Dataset](https://forms.illinois.edu/sec/1713398)
- [ResNet Paper](https://arxiv.org/abs/1512.03385) (He et al., 2015)
