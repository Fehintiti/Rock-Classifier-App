import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [isCorrect, setIsCorrect] = useState('yes');
  const [correctRockGroup, setCorrectRockGroup] = useState('');
  const [actualRock, setActualRock] = useState('');
  const [customRock, setCustomRock] = useState('');
  const [certainty, setCertainty] = useState('');

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setFeedbackSubmitted(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setFeedbackSubmitted(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleUploadClick = (e) => {
    e.preventDefault();
    document.getElementById('fileInput').click();
  };

  const classifyRock = async () => {
    if (!selectedFile) return;

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      // TEMPORARY: Use localhost for testing
      // After AWS deployment, change to: https://YOUR-AWS-URL/predict
      const response = await fetch('http://3.239.114.61:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Classification failed');
      }

      const data = await response.json();
      setResult(data);

    } catch (error) {
      console.error('Error:', error);
      alert('Error classifying rock. Make sure the backend is running on http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async () => {
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model_prediction_type', result?.l1_class || '');
      formData.append('model_prediction_name', result?.l2_predictions[0]?.rock_type || '');
      formData.append('user_correction_type', correctRockGroup);
      formData.append('user_correction_name', actualRock === 'other' ? customRock : actualRock);
      formData.append('certainty', certainty);

      // TEMPORARY: Use localhost for testing
      // After AWS deployment, change to: https://YOUR-AWS-URL/feedback
      const response = await fetch('http://3.239.114.61:8000/feedback', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Feedback submission failed');
      }

      const data = await response.json();
      console.log('Feedback saved:', data);
      setFeedbackSubmitted(true);

    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Error submitting feedback. Make sure the backend is running!');
    }
  };

  const commonRockTypes = [
    'Granite', 'Basalt', 'Limestone', 'Sandstone', 'Marble',
    'Slate', 'Quartzite', 'Gneiss', 'Schist', 'Shale',
    'Diorite', 'Gabbro', 'Andesite', 'Rhyolite', 'Pumice',
    'Obsidian', 'Dolomite', 'Conglomerate', 'Breccia', 'Siltstone',
    'Phyllite', 'Migmatite', 'Granodiorite', 'Peridotite', 'Amphibolite'
  ];

  return (
    <div className="App">
      <nav className="navbar">
        <div className="nav-content">
          <div className="logo">🪨 Rock Classifier AI</div>
          <div className="nav-links">
            <a href="https://github.com/Fehintiti/Rock-Classifier-App" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a href="https://www.linkedin.com/in/fehintiti" target="_blank" rel="noopener noreferrer">LinkedIn</a>
          </div>
        </div>
      </nav>

      <div className="hero">
        <h1>Identify Rocks from Photographs</h1>
        <p>Upload a rock image and get instant AI-powered classification</p>
        <div className="stats">
          <div className="stat-badge">
            <div className="stat-number">2,734</div>
            <div className="stat-label">Training Images</div>
          </div>
          <div className="stat-badge">
            <div className="stat-number">41</div>
            <div className="stat-label">Rock Types</div>
          </div>
        </div>
      </div>

      <div className="disclaimer-banner">
        <div className="disclaimer-content">
          <strong>⚠️ Research & Educational Project</strong>
          <p>
            This is a personal research project exploring AI applications in geological classification.
            Built to demonstrate how deep learning can assist field geologists with preliminary rock identification.
            <br /><br />
            <strong>Not for professional use:</strong> This is a decision support tool, not a replacement for expert identification.
            Always confirm predictions with field tests (hardness, acid reaction, hand lens examination).
          </p>
        </div>
      </div>

      <div className="container">
        <h2>Upload Your Rock Image</h2>

        <div
          className="upload-area"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          {preview ? (
            <img src={preview} alt="Preview" className="preview-image" />
          ) : (
            <>
              <div className="upload-icon">☁️</div>
              <div className="upload-text">Drag & drop your image here</div>
              <div className="upload-subtext">or click the button below</div>
              <div className="upload-formats">Supports: JPG, PNG, JPEG</div>
            </>
          )}
        </div>

        {!selectedFile && (
          <label htmlFor="fileInput" className="upload-btn-label">
            <input
              id="fileInput"
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
            <button
              className="classify-btn"
              onClick={handleUploadClick}
              type="button"
            >
              Choose File
            </button>
          </label>
        )}

        {selectedFile && !result && (
          <button className="classify-btn" onClick={classifyRock} disabled={loading}>
            {loading ? 'Analyzing...' : 'Classify Rock'}
          </button>
        )}

        {result && (
          <div className="results">
            <h2>Classification Results</h2>

            <div className="result-cards">
              <div className="result-card">
                <div className="card-label">Rock Group</div>
                <div className="rock-type">{result.l1_class?.toUpperCase() || 'PROCESSING'}</div>
                <div className="card-label">Confidence</div>
                <div className={`confidence ${result.l1_confidence >= 0.7 ? 'high' : result.l1_confidence >= 0.5 ? 'medium' : 'low'}`}>
                  {result.l1_confidence >= 0.7 ? '✅' : result.l1_confidence >= 0.5 ? '⚠️' : '❌'} {(result.l1_confidence * 100).toFixed(1)}%
                </div>
                <div className="reliability">
                  {result.l1_confidence >= 0.7 ? 'High Confidence' : result.l1_confidence >= 0.5 ? 'Medium Confidence' : 'Low Confidence'}
                </div>
              </div>

              <div className="result-card">
                <div className="card-label">Specific Rock Type (Top 3)</div>
                {result.l2_predictions?.map((pred, idx) => (
                  <div key={idx} className={`prediction-item ${idx === 0 ? 'primary' : ''}`}>
                    <div className="prediction-name">
                      {idx + 1}. {pred.rock_type.charAt(0).toUpperCase() + pred.rock_type.slice(1)}
                    </div>
                    <div className="progress-bar">
                      <div
                        className={`progress-fill ${idx === 0 ? 'primary' : 'secondary'}`}
                        style={{ width: `${pred.confidence * 100}%` }}
                      ></div>
                    </div>
                    <div className="prediction-confidence">{(pred.confidence * 100).toFixed(1)}% confidence</div>
                  </div>
                )) || <p>Processing results...</p>}
              </div>
            </div>

            <div className="feedback-section">
              <h3>Help Improve the Model</h3>
              <p>Your feedback helps make the classifier better for everyone.</p>

              <div className="feedback-form">
                <div className="feedback-question">
                  <label>Was the prediction correct?</label>
                  <div className="radio-group">
                    <label className="radio-label">
                      <input
                        type="radio"
                        value="yes"
                        checked={isCorrect === 'yes'}
                        onChange={(e) => setIsCorrect(e.target.value)}
                      />
                      Yes, correct
                    </label>
                    <label className="radio-label">
                      <input
                        type="radio"
                        value="no"
                        checked={isCorrect === 'no'}
                        onChange={(e) => setIsCorrect(e.target.value)}
                      />
                      No, incorrect
                    </label>
                  </div>
                </div>

                {isCorrect === 'no' && (
                  <>
                    <div className="feedback-question">
                      <label>What's the correct rock group?</label>
                      <select
                        value={correctRockGroup}
                        onChange={(e) => setCorrectRockGroup(e.target.value)}
                        className="rock-select"
                      >
                        <option value="">Select rock group...</option>
                        <option value="igneous">Igneous</option>
                        <option value="sedimentary">Sedimentary</option>
                        <option value="metamorphic">Metamorphic</option>
                      </select>
                    </div>

                    <div className="feedback-question">
                      <label>What's the actual rock type?</label>
                      <select
                        value={actualRock}
                        onChange={(e) => {
                          setActualRock(e.target.value);
                          if (e.target.value !== 'other') setCustomRock('');
                        }}
                        className="rock-select"
                      >
                        <option value="">Select rock type...</option>
                        {commonRockTypes.map((rock, idx) => (
                          <option key={idx} value={rock.toLowerCase()}>{rock}</option>
                        ))}
                        <option value="other">Other (type below)</option>
                      </select>

                      {actualRock === 'other' && (
                        <input
                          type="text"
                          placeholder="Type the rock name here..."
                          value={customRock}
                          onChange={(e) => setCustomRock(e.target.value)}
                          className="custom-rock-input"
                        />
                      )}
                    </div>
                    <div className="feedback-question">
                      <label>How certain are you about this correction?</label>
                      <div className="radio-group">
                        <label className="radio-label">
                          <input
                            type="radio"
                            name="certainty"
                            value="very_certain"
                            checked={certainty === 'very_certain'}
                            onChange={(e) => setCertainty(e.target.value)}
                          />
                          Very certain (I'm confident/expert)
                        </label>
                        <label className="radio-label">
                          <input
                            type="radio"
                            name="certainty"
                            value="somewhat_certain"
                            checked={certainty === 'somewhat_certain'}
                            onChange={(e) => setCertainty(e.target.value)}
                          />
                          Somewhat certain (I think this is right)
                        </label>
                        <label className="radio-label">
                          <input
                            type="radio"
                            name="certainty"
                            value="not_sure"
                            checked={certainty === 'not_sure'}
                            onChange={(e) => setCertainty(e.target.value)}
                          />
                          Not sure (Just guessing)
                        </label>
                      </div>
                    </div>
                  </>
                )}

                <button
                  className="submit-feedback-btn"
                  onClick={submitFeedback}
                  disabled={feedbackSubmitted || (isCorrect === 'no' && (!correctRockGroup || !actualRock || !certainty)) || (actualRock === 'other' && !customRock)}
                >
                  {feedbackSubmitted ? '✓ Feedback Submitted' : 'Submit Feedback'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      <footer className="footer">
        <p><strong>AI Rock Classifier</strong> | Built by Tomisin Okunlola</p>
        <p>Model trained on 2,734 field rock images • ConvNeXt-Tiny architecture • 77% rock group accuracy</p>
        <div className="footer-links">
          <a href="https://github.com/Fehintiti/Rock-Classifier-App" target="_blank" rel="noopener noreferrer">GitHub</a>
          <span>|</span>
          <a href="https://www.linkedin.com/in/fehintiti" target="_blank" rel="noopener noreferrer">LinkedIn</a>
        </div>
      </footer>
    </div>
  );
}

export default App;