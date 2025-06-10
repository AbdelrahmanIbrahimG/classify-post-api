# Travel Venue Classifier API

AI-powered venue classification from images and text using YOLO + BLIP + BART.

## Features

- 13 venue attributes classification
- Image URL or file upload support
- Real-time object detection
- REST API with Swagger docs

## API Endpoints

- POST `/classify/url` - Classify from image URL
- POST `/classify/upload` - Classify from file upload
- GET `/health` - Health check
- GET `/docs` - API documentation

## Example Usage

```bash
curl -X POST "https://your-app.onrender.com/classify/url" \
-H "Content-Type: application/json" \
-d '{
    "image_url": "https://example.com/restaurant.jpg",
    "caption": "Amazing romantic dinner!",
    "confidence_threshold": 0.5
}'
```
