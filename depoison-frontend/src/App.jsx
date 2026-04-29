import { useState, useRef } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post('http://localhost:5000/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      // Set the full object (including the new base64 images) into state
      setResult(response.data.data.analysisResult);
    } catch (err) {
      console.error(err);
      setError('Connection failed. Ensure both Node and Python servers are online.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen text-gray-100 flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8">

      <div className="text-center max-w-3xl w-full mb-12">
        <h1 className="text-5xl font-extrabold tracking-tight mb-4 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">
          Shield AI Matrix
        </h1>
        <p className="text-gray-400 text-lg">
          Upload any image to detect invisible FGSM mathematical adversarial poisoning.
        </p>
      </div>

      <div className="max-w-6xl w-full space-y-8">

        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-2xl flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex-1 w-full">
            <input type="file" accept="image/*" onChange={handleFileChange} ref={fileInputRef} className="hidden" />
            <div onClick={() => fileInputRef.current.click()} className="border-2 border-dashed border-gray-700 hover:border-cyan-500/50 bg-gray-950/50 rounded-xl p-6 text-center cursor-pointer transition-all duration-200 group">
              <p className="text-gray-400 group-hover:text-cyan-400 font-medium">
                {selectedFile ? selectedFile.name : "Click to select or drag and drop an image"}
              </p>
            </div>
          </div>
          <button onClick={handleAnalyze} disabled={!selectedFile || loading} className={`px-8 py-4 rounded-xl font-bold text-lg shadow-lg transition-all duration-300 w-full sm:w-auto ${(!selectedFile || loading) ? 'bg-gray-800 text-gray-500 cursor-not-allowed' : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-cyan-500/25 hover:shadow-cyan-500/50 hover:-translate-y-1'}`}>
            {loading ? 'Processing Tensor...' : 'Analyze Image'}
          </button>
        </div>

        {error && <div className="bg-red-900/30 border border-red-500/50 text-red-400 p-4 rounded-xl text-center font-medium">{error}</div>}

        {/* 3-COLUMN VISUAL DASHBOARD */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

          {/* Column 1: Input */}
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-2xl flex flex-col items-center">
            <h3 className="text-lg font-semibold text-gray-300 mb-4 border-b border-gray-800 pb-2 w-full text-center">1. Target Input</h3>
            <div className="w-48 h-48 bg-gray-950 rounded-xl flex items-center justify-center overflow-hidden border border-gray-800">
              {previewUrl ? <img src={previewUrl} alt="Input" className="w-full h-full object-cover" /> : <span className="text-gray-600 text-sm italic">Awaiting Target</span>}
            </div>
          </div>

          {/* Column 2: Reconstructed */}
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-2xl flex flex-col items-center">
            <h3 className="text-lg font-semibold text-gray-300 mb-4 border-b border-gray-800 pb-2 w-full text-center">2. AI Reconstruction</h3>
            <div className="w-48 h-48 bg-gray-950 rounded-xl flex items-center justify-center overflow-hidden border border-gray-800">
              {result?.reconstructed_image ? <img src={result.reconstructed_image} alt="Reconstructed" className="w-full h-full object-cover rendering-pixelated" /> : <span className="text-gray-600 text-sm italic animate-pulse">Awaiting Render</span>}
            </div>
          </div>

          {/* Column 3: Heatmap */}
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-2xl flex flex-col items-center">
            <h3 className="text-lg font-semibold text-gray-300 mb-4 border-b border-gray-800 pb-2 w-full text-center">3. Anomaly Heatmap</h3>
            <div className="w-48 h-48 bg-gray-950 rounded-xl flex items-center justify-center overflow-hidden border border-gray-800">
              {result?.heatmap_image ? <img src={result.heatmap_image} alt="Heatmap" className="w-full h-full object-cover rendering-pixelated" /> : <span className="text-gray-600 text-sm italic animate-pulse">Awaiting Render</span>}
            </div>
          </div>

        </div>

        {/* RESULTS BAR */}
        {result && (
          <div className={`mt-8 p-6 rounded-2xl border flex items-center justify-between shadow-2xl ${result.label === 'POISONED' ? 'bg-red-950/20 border-red-500/30' : 'bg-green-950/20 border-green-500/30'}`}>
            <div>
              <span className="text-sm uppercase tracking-widest text-gray-400 font-semibold block mb-1">Final Verdict</span>
              <h2 className={`text-4xl font-black tracking-tight ${result.label === 'POISONED' ? 'text-red-500' : 'text-green-500'}`}>
                {result.label} DETECTED
              </h2>
            </div>
            <div className="text-right space-y-2">
              <p className="text-gray-400">Confidence: <span className="text-gray-100 font-mono text-xl ml-2">{result.confidence}%</span></p>
              <p className="text-gray-400">MSE Score: <span className="text-gray-100 font-mono text-xl ml-2">{result.reconstructionError.toFixed(5)}</span></p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;