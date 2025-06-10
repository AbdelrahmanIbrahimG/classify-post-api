# Memory-optimized version for Render deployment
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import io
import os
import tempfile
import time
import gc
from typing import Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Travel Venue Classifier API",
    description="AI-powered venue classification from images and text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassificationRequest(BaseModel):
    image_url: HttpUrl
    caption: str
    confidence_threshold: Optional[float] = 0.5

class ClassificationResponse(BaseModel):
    success: bool
    attributes: dict
    metadata: dict
    error: Optional[str] = None

class TravelClassifierAPI:
    def __init__(self):
        logger.info("Initializing Travel Classifier API...")
        
        # Force CPU usage and optimize memory
        torch.set_num_threads(1)  # Reduce CPU usage
        
        # Initialize models lazily (only when needed)
        self.text_classifier = None
        self.vision_processor = None
        self.vision_model = None
        
        # Simplified feature labels (reduced memory footprint)
        self.feature_labels = {
            'attributes_GoodForKids': [
                "family-friendly with children, kids playing, families dining",
                "kid-friendly venue with playground, children's menu, toys"
            ],
            'attributes_Ambience_romantic': [
                "romantic atmosphere with candles, dim lighting, couples dining",
                "intimate date night venue with wine, roses, romantic music"
            ],
            'attributes_Ambience_trendy': [
                "trendy modern place with stylish decor, hip young crowd",
                "contemporary venue with modern furniture, fashionable clientele"
            ],
            'attributes_Ambience_casual': [
                "casual relaxed atmosphere with comfortable seating, laid-back vibe",
                "informal setting with people in casual clothes, relaxed dining"
            ],
            'attributes_Ambience_classy': [
                "upscale elegant venue with fine dining, sophisticated clientele",
                "high-end establishment with luxury furnishing, formal service"
            ],
            'Bars_Night': [
                "bar with alcohol bottles, cocktails, beer taps, nightlife",
                "drinking venue with people holding drinks, bar atmosphere"
            ],
            'Cafes': [
                "coffee shop with espresso machine, coffee cups, pastries",
                "cafe with people drinking coffee, laptop users, casual seating"
            ],
            'Restaurants_Cuisines': [
                "restaurant with dining tables, people eating meals, food service",
                "dining establishment with customers eating, food plates"
            ]
        }
        
        logger.info("✅ Travel Classifier API initialized (models will load on first use)")
    
    def _ensure_text_classifier(self):
        """Lazy load text classifier"""
        if self.text_classifier is None:
            logger.info("Loading text classifier...")
            self.text_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",  # Correct model name
                device=-1  # Force CPU
            )
            gc.collect()  # Clean up memory
    
    def _ensure_vision_models(self):
        """Lazy load vision models"""
        if self.vision_processor is None or self.vision_model is None:
            logger.info("Loading vision models...")
            self.vision_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.vision_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            gc.collect()  # Clean up memory
    
    def download_image(self, image_url: str) -> Image.Image:
        """Download image from URL"""
        try:
            response = requests.get(str(image_url), timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            # Resize large images to save memory
            max_size = 800
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    
    def get_simple_description(self, image: Image.Image) -> str:
        """Simple object detection using basic image analysis"""
        # Without YOLO, we'll use a simple placeholder
        # In a production environment, you might use a lighter object detection model
        return "image contains various objects and people"
    
    def classify_venue(self, image: Image.Image, caption: str, confidence_threshold: float = 0.5) -> dict:
        """Main classification function with memory optimization"""
        
        try:
            # Ensure models are loaded
            self._ensure_vision_models()
            self._ensure_text_classifier()
            
            # Step 1: Get BLIP description with memory optimization
            inputs = self.vision_processor(image, return_tensors="pt")
            
            with torch.no_grad():  # Disable gradient computation to save memory
                out = self.vision_model.generate(**inputs, max_length=30)  # Shorter output
            
            blip_description = self.vision_processor.decode(out[0], skip_special_tokens=True)
            
            # Clean up intermediate tensors
            del inputs, out
            gc.collect()
            
            # Step 2: Simple object description (without YOLO to save memory)
            simple_description = self.get_simple_description(image)
            
            # Step 3: Combine text sources
            full_text = f"{caption}. Scene: {blip_description}. Objects: {simple_description}"
            
            # Step 4: Classify with memory management
            results = {}
            
            for feature, labels in self.feature_labels.items():
                # Process one feature at a time to manage memory
                all_labels = labels + ["unrelated venue"]
                
                result = self.text_classifier(full_text, all_labels)
                
                # Get best positive score
                positive_scores = [result['scores'][i] for i, label in enumerate(result['labels']) if label in labels]
                max_score = max(positive_scores) if positive_scores else 0.0
                
                results[feature] = 1 if max_score > confidence_threshold else 0
                
                # Clean up after each classification
                del result
                gc.collect()
            
            return {
                'attributes': results,
                'metadata': {
                    'blip_description': blip_description,
                    'simple_description': simple_description,
                    'combined_text': full_text[:200] + "...",  # Truncate for memory
                    'confidence_threshold': confidence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Force cleanup on error
            gc.collect()
            raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Initialize the classifier
classifier = TravelClassifierAPI()

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Travel Venue Classifier API",
        "status": "running",
        "version": "1.0.0 (Memory Optimized)",
        "endpoints": {
            "classify_url": "/classify/url",
            "classify_upload": "/classify/upload",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": "lazy", "memory_optimized": True}

@app.post("/classify/url", response_model=ClassificationResponse)
async def classify_from_url(request: ClassificationRequest):
    """Classify venue from image URL and caption"""
    
    try:
        logger.info(f"Processing URL classification: {request.image_url}")
        
        # Download and resize image
        image = classifier.download_image(request.image_url)
        
        # Classify
        result = classifier.classify_venue(
            image, 
            request.caption, 
            request.confidence_threshold
        )
        
        # Force cleanup
        del image
        gc.collect()
        
        return ClassificationResponse(
            success=True,
            attributes=result['attributes'],
            metadata=result['metadata']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {e}")
        gc.collect()  # Cleanup on error
        return ClassificationResponse(
            success=False,
            attributes={},
            metadata={},
            error=str(e)
        )

@app.post("/classify/upload", response_model=ClassificationResponse)
async def classify_from_upload(
    file: UploadFile = File(...),
    caption: str = Form(...),
    confidence_threshold: float = Form(0.5)
):
    """Classify venue from uploaded image file and caption"""
    
    try:
        logger.info(f"Processing upload classification: {file.filename}")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image with size limit
        image_data = await file.read()
        
        # Limit file size to prevent memory issues
        if len(image_data) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="Image file too large (max 5MB)")
        
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Resize if too large
        max_size = 800
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Classify
        result = classifier.classify_venue(image, caption, confidence_threshold)
        
        # Cleanup
        del image_data, image
        gc.collect()
        
        return ClassificationResponse(
            success=True,
            attributes=result['attributes'],
            metadata=result['metadata']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload classification error: {e}")
        gc.collect()
        return ClassificationResponse(
            success=False,
            attributes={},
            metadata={},
            error=str(e)
        )

@app.get("/features")
async def get_available_features():
    """Get list of all available classification features"""
    return {
        "features": list(classifier.feature_labels.keys()),
        "total_features": len(classifier.feature_labels),
        "note": "Memory optimized version - YOLO object detection disabled"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )