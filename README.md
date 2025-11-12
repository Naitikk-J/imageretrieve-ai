# CBIR System - Content-Based Image Retrieval

A modern, AI-powered Content-Based Image Retrieval system using Convolutional Autoencoders and FAISS vector search.

## 🎯 Features

- **Deep Learning**: Convolutional Autoencoder for feature extraction
- **Fast Search**: FAISS vector database for efficient similarity search
- **Modern UI**: Beautiful React interface with drag-and-drop upload
- **Real-time Results**: Instant visual similarity matching
- **Scalable**: Handles large image datasets efficiently

## 🏗️ Architecture

### Frontend (React + TypeScript)
- Modern, minimalist UI with drag-and-drop image upload
- Responsive grid layout for search results
- Real-time loading states and animations
- Built with Vite, React, TailwindCSS, and shadcn/ui

### Backend (Python + FastAPI)
- FastAPI server for API endpoints
- TensorFlow/Keras Convolutional Autoencoder
- FAISS vector database for similarity search
- Static file serving for result images

### AI Model
- **Architecture**: Convolutional Autoencoder (CAE)
- **Feature Dimension**: 512D vectors
- **Similarity Metric**: Cosine similarity
- **Indexing**: FAISS (Facebook AI Similarity Search)

## 📁 Project Structure

```
cbir-system/
├── frontend/                 # React frontend (this directory)
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUpload.tsx
│   │   │   ├── ResultsGrid.tsx
│   │   │   └── ui/
│   │   ├── pages/
│   │   │   └── Index.tsx
│   │   └── index.css
│   └── package.json
│
└── backend/                  # Python backend (see PYTHON_BACKEND.md)
    ├── model/
    │   ├── autoencoder.py   # CAE model definition
    │   └── train.py         # Training script
    ├── index.py             # FAISS indexing script
    ├── api.py               # FastAPI server
    ├── config.py            # Configuration
    ├── requirements.txt     # Python dependencies
    └── dataset/
        └── images/          # Your image dataset
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ (for frontend)
- Python 3.9+ (for backend)
- pip and npm/yarn

### Frontend Setup

1. **Install dependencies**:
```bash
npm install
```

2. **Start development server**:
```bash
npm run dev
```

The frontend will be available at `http://localhost:8080`

### Backend Setup

**Complete Python backend code is provided in `PYTHON_BACKEND.md`**

1. **Create backend directory**:
```bash
mkdir backend
cd backend
```

2. **Copy the code from PYTHON_BACKEND.md** and create:
   - `requirements.txt`
   - `config.py`
   - `model/autoencoder.py`
   - `model/train.py`
   - `index.py`
   - `api.py`

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

4. **Prepare your dataset**:
```bash
mkdir -p dataset/images
# Copy your images to dataset/images/
```

5. **Train the model** (optional - can use pre-trained):
```bash
python model/train.py
```

6. **Build FAISS index**:
```bash
python index.py
```

This will:
- Load all images from `dataset/images/`
- Extract features using the encoder
- Build a FAISS index for fast similarity search
- Save `image_features.index` and `filename_mapping.json`

7. **Start the API server**:
```bash
python api.py
```

The backend API will be available at `http://localhost:8000`

### Connecting Frontend to Backend

1. Open `src/pages/Index.tsx`
2. Uncomment the API call section:

```typescript
const formData = new FormData();
formData.append('image', file);
const response = await fetch('http://localhost:8000/search', {
  method: 'POST',
  body: formData,
});
const data = await response.json();
setResults(data.results);
```

3. Comment out the mock data section

## 🎨 Design System

The UI uses a modern, tech-forward design:

- **Colors**: Deep blue/cyan gradient theme
- **Typography**: Clean, modern fonts
- **Animations**: Smooth transitions and loading states
- **Layout**: Card-based, responsive grid
- **Components**: shadcn/ui components with custom variants

## 🔧 Configuration

### Backend Configuration (`backend/config.py`)

```python
IMAGE_SIZE = (224, 224)    # Input image size
FEATURE_DIM = 512          # Feature vector dimension
TOP_K = 10                 # Number of results to return
API_PORT = 8000           # Backend port
```

### Frontend Configuration

Update the API URL in `src/pages/Index.tsx` to match your backend:

```typescript
const BACKEND_URL = 'http://localhost:8000';
```

## 📊 How It Works

1. **Image Upload**: User uploads a query image via drag-and-drop or file browser
2. **Preprocessing**: Image is resized and normalized
3. **Feature Extraction**: Encoder extracts a 512D feature vector
4. **Similarity Search**: FAISS finds the top-K most similar vectors
5. **Results Display**: Similar images are displayed in a responsive grid

### FAISS Indexing Process

```
Dataset Images → Encoder → Feature Vectors → FAISS Index
     ↓              ↓            ↓               ↓
  1000 images   512D vectors  Normalized   Fast Search
```

### Search Process

```
Query Image → Encoder → Feature Vector → FAISS Search → Top-K Results
     ↓           ↓            ↓              ↓              ↓
  Uploaded   512D vector  Normalized   Cosine Sim.   Ranked Images
```

## 🧪 Testing

### Test the Backend

```bash
# Health check
curl http://localhost:8000/

# Get stats
curl http://localhost:8000/stats

# Search (using curl)
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/search
```

### Test the Frontend

1. Start the dev server: `npm run dev`
2. Upload a test image
3. Verify results are displayed correctly

## 🚀 Deployment

### Frontend

Deploy to Lovable (automatic):
```bash
# Just push to your repository
# Lovable automatically builds and deploys
```

Or deploy to other platforms:
- Vercel: `vercel deploy`
- Netlify: `netlify deploy`
- Custom server: `npm run build` and serve the `dist/` folder

### Backend

Deploy Python backend to:
- **Railway**: `railway up`
- **Render**: Connect your repo
- **AWS EC2**: Traditional server deployment
- **Docker**: Use the provided Dockerfile (create one)

Update the frontend API URL to your deployed backend URL.

## 🔐 Security Notes

- Add CORS configuration for production
- Use environment variables for sensitive data
- Add rate limiting to prevent abuse
- Implement authentication if needed
- Validate uploaded files (size, type)

## 📈 Performance Optimization

- **GPU Acceleration**: Use CUDA for faster feature extraction
- **Batch Processing**: Process multiple images simultaneously
- **Caching**: Cache frequently searched images
- **CDN**: Use CDN for static image serving
- **Index Optimization**: Use FAISS IVF indices for larger datasets

## 🤝 Contributing

This is a template project. Feel free to:
- Enhance the UI/UX
- Add more search filters
- Implement user authentication
- Add image upload to dataset
- Improve the AI model
- Add advanced FAISS indices

## 📝 License

MIT License - feel free to use for any purpose

## 🆘 Troubleshooting

### "Models not loaded" error
- Run `python index.py` first to build the FAISS index
- Ensure `encoder.h5` exists in `backend/models/`

### CORS errors
- Check that CORS is enabled in `api.py`
- Verify the frontend is calling the correct backend URL

### Slow search
- Ensure FAISS index is properly built
- Consider using GPU acceleration
- Use IVF indices for very large datasets

### Out of memory
- Reduce batch size in training
- Use smaller feature dimensions
- Process images in smaller batches during indexing

## 📚 Resources

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [TensorFlow/Keras Docs](https://www.tensorflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

## ✨ Future Enhancements

- [ ] Multi-modal search (text + image)
- [ ] User authentication and saved searches
- [ ] Image upload to dataset via UI
- [ ] Advanced filtering options
- [ ] Batch image search
- [ ] Model fine-tuning interface
- [ ] Performance analytics dashboard

---

Built with ❤️ using React, TensorFlow, and FAISS
