const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const { uploadAndAnalyze } = require('../controllers/analyzeController');

// Configure Multer Storage
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/'); // Make sure the 'uploads' folder exists!
    },
    filename: (req, file, cb) => {
        // Rename file to prevent overwriting: timestamp + original extension
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname));
    }
});

// Initialize Multer
const upload = multer({ storage: storage });

// The POST Route
// 'image' is the key name we will use in Postman/Frontend to send the file
router.post('/', upload.single('image'), uploadAndAnalyze);

module.exports = router;