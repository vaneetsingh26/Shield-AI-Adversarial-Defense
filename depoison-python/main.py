from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import base64
from detector import PoisonDetector 

app = FastAPI(title="Image Depoisoning API")

print("Initializing ML Model...")
try:
    detector = PoisonDetector("saved_model/autoencoder_best.keras")
    detector.load_threshold("saved_model/threshold.npy")
    print("✅ Model loaded and ready!")
except Exception as e:
    print(f"⚠️ Warning: Could not load model. Error: {e}")

def array_to_base64(img_array):
    img_uint8 = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        image = image.resize((32, 32))
        img_array = np.array(image).astype('float32') / 255.0

        results = detector.detect(img_array)
        result = results[0]

        img_batch = np.expand_dims(img_array, axis=0)
        reconstructed_array = detector.autoencoder.predict(img_batch, verbose=0)[0]
        heatmap_array, _ = detector.get_error_heatmap(img_array)

        reconstructed_b64 = array_to_base64(reconstructed_array)
        heatmap_b64 = array_to_base64(heatmap_array)

        # THE FIX IS HERE: We added float() around the confidence and error numbers
        return JSONResponse(content={
            "success": True,
            "label": result["label"],
            "confidence": float(result["confidence"]), 
            "error": float(result["error"]),           
            "reconstructed_image": f"data:image/png;base64,{reconstructed_b64}",
            "heatmap_image": f"data:image/png;base64,{heatmap_b64}"
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)