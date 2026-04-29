const ImageScan = require('../models/ImageScan');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

exports.uploadAndAnalyze = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ success: false, message: 'No image provided' });
        }

        const imagePath = req.file.path;
        const currentScan = await ImageScan.create({
            originalImagePath: imagePath,
            status: 'PENDING'
        });

        const formData = new FormData();
        formData.append('file', fs.createReadStream(imagePath));

        const pythonApiUrl = process.env.PYTHON_API_URL || 'http://127.0.0.1:8000/analyze';

        const pythonResponse = await axios.post(pythonApiUrl, formData, {
            headers: { ...formData.getHeaders() }
        });

        const { label, confidence, error, reconstructed_image, heatmap_image } = pythonResponse.data;

        // 1. Save ONLY the math to MongoDB (respecting your current schema)
        currentScan.status = 'COMPLETED';
        currentScan.analysisResult = {
            label: label,
            confidence: confidence,
            reconstructionError: error
        };
        await currentScan.save();

        // 2. Convert the MongoDB document to a normal object and explicitly attach the images
        const finalResponseData = currentScan.toObject();
        finalResponseData.analysisResult.reconstructed_image = reconstructed_image;
        finalResponseData.analysisResult.heatmap_image = heatmap_image;

        // 3. Send the complete package back to React
        return res.status(200).json({
            success: true,
            data: finalResponseData
        });

    } catch (error) {
        console.error("Backend Bridge Error:", error.message);
        return res.status(500).json({
            success: false,
            message: 'Failed to communicate with Python ML Service.'
        });
    }
};