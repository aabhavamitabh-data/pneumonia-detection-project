"use client";

import { useState } from "react";

const API_URL = "https://pneumonia-detector-api-195849682317.asia-south1.run.app/predict";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

 const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files[0];
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

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Server error. Please try again.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Something went wrong. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-lg p-8">
        <h1 className="text-2xl font-bold text-gray-800 mb-2 text-center">
          Chest X-Ray Analyzer
        </h1>
        <p className="text-sm text-gray-500 text-center mb-6">
          Upload a chest X-ray image to check for Covid, Normal, or Viral Pneumonia patterns.
        </p>

        {/* Upload box */}
        <label className="block border-2 border-dashed border-gray-300 rounded-xl p-6 text-center cursor-pointer hover:border-blue-400 transition mb-4">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
          {previewUrl ? (
            <img
              src={previewUrl}
              alt="Preview"
              className="mx-auto max-h-48 rounded-lg object-contain"
            />
          ) : (
            <span className="text-gray-500">Click to choose an image</span>
          )}
        </label>

        {/* Analyze button */}
        <button
          onClick={handleAnalyze}
          disabled={!selectedFile || loading}
          className="w-full bg-blue-600 text-white font-semibold py-3 rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        {/* Error message */}
        {error && (
          <p className="text-red-500 text-sm text-center mt-4">{error}</p>
        )}

        {/* Result display */}
        {result && (
          <div className="mt-6 border-t pt-6">
            <p className="text-center text-lg font-bold text-gray-800 mb-1">
              {result.prediction}
            </p>
            <p className="text-center text-sm text-gray-500 mb-4">
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </p>

            {result.message && (
              <p className="text-center text-sm text-amber-600 mb-4">
                {result.message}
              </p>
            )}

            {result.all_probabilities && (
              <div className="space-y-2">
                {Object.entries(result.all_probabilities).map(([label, prob]) => (
                  <div key={label}>
                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                      <span>{label}</span>
                      <span>{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: `${prob * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <p className="text-xs text-gray-400 text-center mt-6">
          Educational demo only. Not a substitute for professional medical diagnosis.
        </p>
      </div>
    </div>
  );
}
