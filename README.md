# The Visual Storyteller

An image captioning system that takes an image as input and produces a natural language description as output. The model bridges two distinct modalities: it interprets visual information and expresses that understanding through generated text.

## Architecture

- **Encoder**: Pre-trained ResNet50 CNN for visual feature extraction
- **Decoder**: 2-layer LSTM for sequence generation
- **Embedding Size**: 256
- **Hidden Size**: 512

## Project Structure

```
The-Visual-Storyteller/
├── data/
│   ├── Images/          # 8,000 training images
│   └── captions.txt     # 5 captions per image
├── models/              # Saved model artifacts
├── notebooks/
│   ├── data_and_training.ipynb   # Training pipeline
│   └── inference.ipynb           # Inference and analysis
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place the dataset in the `data/` directory:
- `data/Images/` - containing all image files
- `data/captions.txt` - CSV file with image-caption pairs

## Usage

### Training

1. Open `notebooks/data_and_training.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Load and preprocess the data
   - Build the vocabulary
   - Train the CNN-LSTM model
   - Save model artifacts to `models/`

### Inference

1. Open `notebooks/inference.ipynb`
2. Run all cells to load the trained model
3. Use the `generate_caption` function:

```python
def generate_caption(image_path: str, model: any) -> str:
    """
    Takes a path to an image and returns a generated caption string.
    """
```

Example:
```python
caption = generate_caption('path/to/image.jpg', model)
print(caption)  # "a dog running in the grass"
```

## Model Artifacts

After training, the following files are saved in `models/`:
- `best_model.pth` - Best model checkpoint (lowest validation loss)
- `final_model.pth` - Final model with configuration
- `vocab.pkl` - Vocabulary mappings
- `val_images.pkl` - Validation image list
- `training_curves.png` - Loss curves visualization

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- Pillow
- matplotlib
- tqdm
