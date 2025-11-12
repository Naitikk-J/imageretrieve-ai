# CBIR System - Python Backend Implementation

This document contains the complete Python backend code for the Content-Based Image Retrieval system.

## File Structure

```
backend/
├── model/
│   ├── autoencoder.py      # CAE model definition
│   └── train.py            # Training script
├── index.py                # FAISS indexing script
├── api.py                  # FastAPI server
├── requirements.txt        # Python dependencies
└── config.py              # Configuration
```

## 1. Requirements (requirements.txt)

```txt
tensorflow>=2.13.0
torch>=2.0.0
torchvision>=0.15.0
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
pillow>=10.0.0
numpy>=1.24.0
faiss-cpu>=1.7.4
pydantic>=2.4.0
python-dotenv>=1.0.0
```

## 2. Configuration (config.py)

```python
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset" / "images"
MODEL_DIR = BASE_DIR / "models"
INDEX_DIR = BASE_DIR / "indexes"

# Model parameters
IMAGE_SIZE = (224, 224)
FEATURE_DIM = 512
BATCH_SIZE = 32
EPOCHS = 50

# FAISS parameters
TOP_K = 10  # Number of similar images to return

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
STATIC_DIR = DATASET_DIR

# Create directories
MODEL_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
```

## 3. Convolutional Autoencoder Model (model/autoencoder.py)

### TensorFlow/Keras Implementation

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class ConvolutionalAutoencoder:
    """
    Convolutional Autoencoder for feature extraction.
    
    Architecture:
    - Encoder: Extracts a compressed feature representation
    - Decoder: Reconstructs the original image
    
    For CBIR, we only use the Encoder part to generate feature vectors.
    """
    
    def __init__(self, input_shape=(224, 224, 3), feature_dim=512):
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self._build_model()
    
    def _build_model(self):
        """Build the complete autoencoder architecture."""
        
        # ===== ENCODER =====
        encoder_input = keras.Input(shape=self.input_shape, name='encoder_input')
        
        # Block 1: 224x224 -> 112x112
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Block 2: 112x112 -> 56x56
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Block 3: 56x56 -> 28x28
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Block 4: 28x28 -> 14x14
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Flatten and compress to feature vector
        x = layers.Flatten()(x)
        encoder_output = layers.Dense(self.feature_dim, activation='relu', name='features')(x)
        
        self.encoder = keras.Model(encoder_input, encoder_output, name='encoder')
        
        # ===== DECODER =====
        decoder_input = keras.Input(shape=(self.feature_dim,), name='decoder_input')
        
        # Expand to spatial dimensions (14x14x512)
        x = layers.Dense(14 * 14 * 512, activation='relu')(decoder_input)
        x = layers.Reshape((14, 14, 512))(x)
        
        # Block 1: 14x14 -> 28x28
        x = layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # Block 2: 28x28 -> 56x56
        x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # Block 3: 56x56 -> 112x112
        x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # Block 4: 112x112 -> 224x224
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # Output layer: reconstruct original image
        decoder_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        self.decoder = keras.Model(decoder_input, decoder_output, name='decoder')
        
        # ===== COMPLETE AUTOENCODER =====
        autoencoder_input = keras.Input(shape=self.input_shape)
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        self.autoencoder = keras.Model(autoencoder_input, decoded, name='autoencoder')
        
        # Compile
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, x_train, x_val=None, epochs=50, batch_size=32):
        """Train the autoencoder."""
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'models/autoencoder_best.h5',
                save_best_only=True,
                monitor='val_loss' if x_val is not None else 'loss'
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if x_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if x_val is not None else 'loss',
                factor=0.5,
                patience=5
            )
        ]
        
        history = self.autoencoder.fit(
            x_train, x_train,  # Input = Output for autoencoder
            validation_data=(x_val, x_val) if x_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def extract_features(self, images):
        """Extract feature vectors using the encoder."""
        return self.encoder.predict(images)
    
    def save_encoder(self, path):
        """Save only the encoder model."""
        self.encoder.save(path)
    
    def load_encoder(self, path):
        """Load a saved encoder model."""
        self.encoder = keras.models.load_model(path)
    
    def save(self, path):
        """Save the complete autoencoder."""
        self.autoencoder.save(path)
    
    def load(self, path):
        """Load a saved autoencoder."""
        self.autoencoder = keras.models.load_model(path)
        # Rebuild encoder and decoder from loaded model
        self.encoder = keras.Model(
            self.autoencoder.input,
            self.autoencoder.get_layer('features').output
        )
```

### PyTorch Implementation (Alternative)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchConvAutoencoder(nn.Module):
    """PyTorch implementation of Convolutional Autoencoder."""
    
    def __init__(self, feature_dim=512):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 112x112x64 -> 56x56x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 56x56x128 -> 28x28x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 28x28x256 -> 14x14x512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            nn.Linear(14 * 14 * 512, feature_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 14 * 14 * 512),
            nn.ReLU(),
            nn.Unflatten(1, (512, 14, 14)),
            
            # 14x14x512 -> 28x28x512
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            # 28x28x512 -> 56x56x256
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            # 56x56x256 -> 112x112x128
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            # 112x112x128 -> 224x224x64
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def extract_features(self, x):
        return self.encoder(x)
```

## 4. FAISS Indexing Script (index.py)

```python
import numpy as np
import faiss
from pathlib import Path
from PIL import Image
import tensorflow as tf
from config import *
import json

def load_encoder(model_path):
    """Load the trained encoder model."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """Load and preprocess an image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array

def build_faiss_index(dataset_dir, encoder_model_path, output_index_path, output_mapping_path):
    """
    Build FAISS index from a dataset of images.
    
    Args:
        dataset_dir: Directory containing images
        encoder_model_path: Path to trained encoder model
        output_index_path: Where to save the FAISS index
        output_mapping_path: Where to save the filename mapping
    """
    print("Loading encoder model...")
    encoder = load_encoder(encoder_model_path)
    
    print(f"Processing images from {dataset_dir}...")
    image_paths = list(Path(dataset_dir).glob("*.jpg")) + \
                  list(Path(dataset_dir).glob("*.jpeg")) + \
                  list(Path(dataset_dir).glob("*.png"))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {dataset_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    # Extract features for all images
    features_list = []
    filenames = []
    
    for idx, img_path in enumerate(image_paths):
        try:
            # Preprocess image
            img = preprocess_image(img_path)
            img_batch = np.expand_dims(img, axis=0)
            
            # Extract features
            features = encoder.predict(img_batch, verbose=0)
            features_list.append(features[0])
            filenames.append(img_path.name)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_paths)} images")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Convert to numpy array
    features_array = np.array(features_list).astype('float32')
    print(f"Feature matrix shape: {features_array.shape}")
    
    # Normalize features (important for cosine similarity)
    faiss.normalize_L2(features_array)
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = features_array.shape[1]
    
    # Using IndexFlatIP for inner product (cosine similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to index
    index.add(features_array)
    
    print(f"Index built with {index.ntotal} vectors")
    
    # Save index
    faiss.write_index(index, str(output_index_path))
    print(f"Index saved to {output_index_path}")
    
    # Save filename mapping
    mapping = {i: filename for i, filename in enumerate(filenames)}
    with open(output_mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Filename mapping saved to {output_mapping_path}")
    
    return index, mapping

if __name__ == "__main__":
    # Paths
    encoder_path = MODEL_DIR / "encoder.h5"
    index_path = INDEX_DIR / "image_features.index"
    mapping_path = INDEX_DIR / "filename_mapping.json"
    
    # Build index
    build_faiss_index(
        dataset_dir=DATASET_DIR,
        encoder_model_path=encoder_path,
        output_index_path=index_path,
        output_mapping_path=mapping_path
    )
    
    print("\nIndexing complete!")
    print(f"To search, run: python api.py")
```

## 5. FastAPI Backend (api.py)

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import numpy as np
import faiss
import json
from PIL import Image
import io
import tensorflow as tf
from pathlib import Path
from config import *

app = FastAPI(title="CBIR API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static images
app.mount("/static/images", StaticFiles(directory=str(STATIC_DIR)), name="images")

# Global variables for model and index
encoder = None
index = None
filename_mapping = None

def load_models():
    """Load encoder model, FAISS index, and filename mapping."""
    global encoder, index, filename_mapping
    
    encoder_path = MODEL_DIR / "encoder.h5"
    index_path = INDEX_DIR / "image_features.index"
    mapping_path = INDEX_DIR / "filename_mapping.json"
    
    print("Loading encoder model...")
    encoder = tf.keras.models.load_model(str(encoder_path))
    
    print("Loading FAISS index...")
    index = faiss.read_index(str(index_path))
    
    print("Loading filename mapping...")
    with open(mapping_path, 'r') as f:
        filename_mapping = json.load(f)
    
    print(f"Loaded index with {index.ntotal} images")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    try:
        load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure you've run index.py first!")

def preprocess_image(image_bytes):
    """Preprocess uploaded image."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype('float32')

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "CBIR API is running",
        "indexed_images": index.ntotal if index else 0
    }

@app.post("/search")
async def search_similar_images(file: UploadFile = File(...)):
    """
    Search for similar images.
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON with list of similar images and their similarity scores
    """
    if encoder is None or index is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run index.py first."
        )
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)
        
        # Extract features
        query_features = encoder.predict(img_array, verbose=0)
        query_features = query_features.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_features)
        
        # Search in FAISS index
        similarities, indices = index.search(query_features, TOP_K)
        
        # Prepare results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            filename = filename_mapping[str(idx)]
            results.append({
                "filename": filename,
                "similarity": float(sim),
                "url": f"/static/images/{filename}"
            })
        
        return JSONResponse(content={"results": results})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "total_indexed_images": index.ntotal if index else 0,
        "feature_dimension": FEATURE_DIM,
        "top_k_results": TOP_K,
        "model_loaded": encoder is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
```

## Usage Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
mkdir -p dataset/images
# Add your images to dataset/images/
```

### 3. Train the Model (Optional)

Create `model/train.py`:

```python
import numpy as np
from pathlib import Path
from PIL import Image
from autoencoder import ConvolutionalAutoencoder
from config import *

def load_dataset(dataset_dir, max_images=None):
    """Load and preprocess dataset."""
    image_paths = list(Path(dataset_dir).glob("*.jpg")) + \
                  list(Path(dataset_dir).glob("*.jpeg")) + \
                  list(Path(dataset_dir).glob("*.png"))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        images.append(img_array)
    
    return np.array(images)

if __name__ == "__main__":
    print("Loading dataset...")
    images = load_dataset(DATASET_DIR)
    print(f"Loaded {len(images)} images")
    
    # Split train/val
    split = int(0.9 * len(images))
    x_train = images[:split]
    x_val = images[split:]
    
    print("Building model...")
    cae = ConvolutionalAutoencoder(
        input_shape=(*IMAGE_SIZE, 3),
        feature_dim=FEATURE_DIM
    )
    
    print("Training...")
    cae.train(x_train, x_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print("Saving model...")
    cae.save_encoder(MODEL_DIR / "encoder.h5")
    
    print("Training complete!")
```

Run: `python model/train.py`

### 4. Build FAISS Index

```bash
python index.py
```

### 5. Start API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### 6. Connect Frontend

Update the frontend code to use your backend URL:

```typescript
const response = await fetch('http://localhost:8000/search', {
  method: 'POST',
  body: formData,
});
```

## API Endpoints

- `GET /` - Health check
- `POST /search` - Upload image and get similar images
- `GET /stats` - Get system statistics
- `GET /static/images/{filename}` - Serve image files

## Notes

- For production, use a pre-trained encoder or train on a large dataset
- Consider using GPU for faster indexing and search
- Add authentication and rate limiting for production deployment
- Use environment variables for sensitive configuration
