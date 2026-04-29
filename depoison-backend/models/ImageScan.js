const mongoose = require('mongoose');

const imageScanSchema = new mongoose.Schema({
    originalImagePath: { 
        type: String, 
        required: true 
    },
    status: {
        type: String,
        enum: ['PENDING', 'COMPLETED', 'FAILED'],
        default: 'PENDING'
    },
    analysisResult: {
        label: { type: String, enum: ['CLEAN', 'POISONED', null], default: null },
        confidence: { type: Number, default: null },
        reconstructionError: { type: Number, default: null }
    }
}, { timestamps: true });

// THIS IS THE CRUCIAL LINE:
module.exports = mongoose.model('ImageScan', imageScanSchema);