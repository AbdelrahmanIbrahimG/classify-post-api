# Install: pip install fastapi uvicorn python-multipart requests

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from ultralytics import YOLO
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

# Add CORS middleware for web access
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
        
        # Zero-shot text classifier
        self.text_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Vision model for image understanding
        self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # YOLO for object detection
        logger.info("Loading YOLO model...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Enhanced feature labels
        self.feature_labels = {
            'attributes_GoodForKids': [
                "family-friendly establishment with children welcome, kids playing, families dining together, high chairs visible",
                "kid-friendly venue with family activities, playground equipment, children's menu, toys and games",
                "place where parents bring children, kids running around, family atmosphere with strollers and baby items",
                "venue with families eating together, children laughing, parents with kids, child-friendly environment"
            ],
            'attributes_Ambience_romantic': [
                "romantic atmosphere perfect for couples with candles, dim lighting, intimate seating for two",
                "intimate dining experience for date nights with wine glasses, roses, romantic music",
                "romantic setting with mood lighting, couples holding hands, anniversary dinner atmosphere",
                "date night venue with romantic decor, candlelit tables, couples dining intimately together"
            ],
            'attributes_Ambience_trendy': [
                "trendy modern place with stylish contemporary decor, Instagram-worthy design, hip young crowd",
                "hip contemporary venue with modern furniture, trendy lighting, fashionable clientele",
                "stylish modern establishment with cutting-edge design, trendy atmosphere, social media worthy",
                "fashionable venue with modern aesthetic, trendy decorations, hipster crowd, contemporary style"
            ],
            'attributes_Ambience_casual': [
                "casual relaxed atmosphere with comfortable seating, people in casual clothes, laid-back vibe",
                "informal comfortable setting with relaxed customers, casual dining, everyday atmosphere",
                "laid-back environment with people wearing jeans, casual conversations, comfortable chairs",
                "relaxed venue with informal atmosphere, people eating casually, comfortable and easygoing"
            ],
            'attributes_Ambience_classy': [
                "upscale elegant venue with fine dining, formal atmosphere, sophisticated clientele, expensive decor",
                "high-end sophisticated establishment with luxury furnishing, formal service, elegant design",
                "classy refined venue with expensive wine, formal dining, sophisticated atmosphere, luxury setting",
                "elegant upscale restaurant with fine china, crystal glasses, formal waitstaff, luxury ambiance"
            ],
            'attributes_Ambience_intimate': [
                "intimate cozy setting with small tables, quiet conversations, private booths, soft lighting",
                "small quiet venue with few people, whispered conversations, cozy corners, personal space",
                "cozy intimate atmosphere with private seating, quiet environment, personal conversations",
                "secluded quiet place with intimate seating, low voices, private dining, personal atmosphere"
            ],
            'attributes_Ambience_touristy': [
                "popular tourist destination with tour groups, tourists taking photos, guidebooks visible, crowds",
                "famous tourist location with visitors, cameras, tourist buses, international crowd",
                "tourist hotspot with sightseers, vacation atmosphere, tourist attractions, photo opportunities",
                "well-known tourist spot with travelers, vacation photos, tourist signs, famous landmark"
            ],
            'Bars_Night': [
                "bar or nightlife venue with alcohol bottles, bartender mixing drinks, cocktails, beer taps",
                "cocktail lounge or pub with people drinking alcohol, bar stools, liquor display, nighttime crowd",
                "nightlife establishment with drinks being served, people holding beers, bar atmosphere, alcohol service",
                "drinking venue with cocktail glasses, wine bottles, people socializing with drinks, bar counter"
            ],
            'Beauty_Health_Care': [
                "spa wellness center with massage tables, treatment rooms, relaxation area, beauty equipment",
                "beauty salon or health facility with styling chairs, mirrors, beauty treatments, wellness services",
                "health and beauty establishment with spa treatments, facial services, wellness therapy, relaxation",
                "wellness center with massage therapy, beauty treatments, health services, relaxation environment"
            ],
            'Cafes': [
                "coffee shop or cafe with espresso machine, coffee cups, barista, pastries, casual seating",
                "coffee house with pastries, laptop users, coffee beans, casual atmosphere, study environment",
                "cafe establishment with coffee service, people drinking coffee, cafe tables, relaxed atmosphere",
                "coffee place with espresso drinks, people working on laptops, coffee aroma, casual dining"
            ],
            'GYM': [
                "fitness center or gym with exercise equipment, people working out, weights, cardio machines",
                "workout facility with gym equipment, people exercising, fitness machines, athletic activities",
                "fitness gym with people lifting weights, exercise bikes, workout area, athletic equipment",
                "athletic center with fitness training, people exercising, gym machines, workout environment"
            ],
            'Restaurants_Cuisines': [
                "restaurant serving food with dining tables, people eating meals, kitchen, food service, waitstaff",
                "dining establishment with customers eating, food plates, restaurant atmosphere, meal service",
                "food restaurant with people dining, plates of food, restaurant tables, culinary service",
                "eating establishment with food being served, diners enjoying meals, restaurant setting, cuisine"
            ],
            'Shops': [
                "retail store or shop with products for sale, shopping bags, customers browsing, merchandise displays",
                "boutique or shopping establishment with items to buy, retail displays, shopping atmosphere",
                "store with merchandise, people shopping, retail products, commercial environment, sales",
                "shopping venue with products on shelves, customers purchasing, retail atmosphere, store displays"
            ]
        }
        
        logger.info("✅ Travel Classifier API ready!")
    
    def download_image(self, image_url: str) -> Image.Image:
        """Download image from URL"""
        try:
            response = requests.get(str(image_url), timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    
    def get_yolo_description(self, image: Image.Image) -> str:
        """Generate natural language description from YOLO detections - FIXED version"""
        
        temp_file_path = None
        
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                temp_file_path = tmp_file.name
                image.save(temp_file_path)
            
            # Run YOLO detection
            results = self.yolo_model(temp_file_path, verbose=False)
            
            detected_objects = []
            
            # Extract high-confidence objects
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.yolo_model.names[class_id]
                        
                        if confidence > 0.5:
                            detected_objects.append(class_name)
            
            # Force garbage collection to release file handles
            del results
            gc.collect()
            
            # Wait a bit for Windows to release the file
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            detected_objects = []
        
        finally:
            # Clean up temp file with retry mechanism
            if temp_file_path and os.path.exists(temp_file_path):
                max_attempts = 5
                for attempt in range(max_attempts):
                    try:
                        os.unlink(temp_file_path)
                        break
                    except PermissionError:
                        if attempt < max_attempts - 1:
                            time.sleep(0.5)  # Wait and retry
                        else:
                            logger.warning(f"Could not delete temp file {temp_file_path}")
        
        # Convert objects to natural description
        if not detected_objects:
            return "no specific objects detected"
        
        # Group similar objects
        object_counts = {}
        for obj in detected_objects:
            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        # Create natural language description
        descriptions = []
        for obj, count in object_counts.items():
            if count == 1:
                descriptions.append(f"a {obj}")
            else:
                descriptions.append(f"{count} {obj}s")
        
        # Join naturally
        if len(descriptions) == 1:
            return f"contains {descriptions[0]}"
        elif len(descriptions) == 2:
            return f"contains {descriptions[0]} and {descriptions[1]}"
        else:
            return f"contains {', '.join(descriptions[:-1])}, and {descriptions[-1]}"
    
    def classify_venue(self, image: Image.Image, caption: str, confidence_threshold: float = 0.5) -> dict:
        """Main classification function"""
        
        try:
            # Step 1: Get BLIP description
            inputs = self.vision_processor(image, return_tensors="pt")
            out = self.vision_model.generate(**inputs, max_length=50)
            blip_description = self.vision_processor.decode(out[0], skip_special_tokens=True)
            
            # Step 2: Get YOLO description (now with proper file handling)
            yolo_description = self.get_yolo_description(image)
            
            # Step 3: Combine all text sources
            full_text = f"{caption}. Scene description: {blip_description}. Objects: {yolo_description}"
            
            # Step 4: Classify using enhanced text
            results = {}
            
            for feature, labels in self.feature_labels.items():
                # Add negative labels for better discrimination
                all_labels = labels + ["unrelated venue", "different type of place"]
                
                result = self.text_classifier(full_text, all_labels)
                
                # Get best positive score
                positive_scores = [result['scores'][i] for i, label in enumerate(result['labels']) if label in labels]
                max_score = max(positive_scores) if positive_scores else 0.0
                
                results[feature] = 1 if max_score > confidence_threshold else 0
            
            return {
                'attributes': results,
                'metadata': {
                    'blip_description': blip_description,
                    'yolo_description': yolo_description,
                    'combined_text': full_text,
                    'confidence_threshold': confidence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Initialize the classifier
classifier = TravelClassifierAPI()

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Travel Venue Classifier API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "classify_url": "/classify/url",
            "classify_upload": "/classify/upload",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": True}

@app.post("/classify/url", response_model=ClassificationResponse)
async def classify_from_url(request: ClassificationRequest):
    """Classify venue from image URL and caption"""
    
    try:
        logger.info(f"Processing URL classification: {request.image_url}")
        
        # Download image
        image = classifier.download_image(request.image_url)
        
        # Classify
        result = classifier.classify_venue(
            image, 
            request.caption, 
            request.confidence_threshold
        )
        
        return ClassificationResponse(
            success=True,
            attributes=result['attributes'],
            metadata=result['metadata']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {e}")
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
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Classify
        result = classifier.classify_venue(image, caption, confidence_threshold)
        
        return ClassificationResponse(
            success=True,
            attributes=result['attributes'],
            metadata=result['metadata']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload classification error: {e}")
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
        "feature_descriptions": {
            feature: labels[0] for feature, labels in classifier.feature_labels.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )