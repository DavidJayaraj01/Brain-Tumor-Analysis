import { type PredictionResponse } from '../services/api';
import ImageOverlay from './ImageOverlay';

interface ResultPanelProps {
  result: PredictionResponse | null;
}

export default function ResultPanel({ result }: ResultPanelProps) {
  if (!result) {
    return (
      <div className="result-panel">
        <div className="empty-state">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="12" y1="18" x2="12" y2="12" />
            <line x1="9" y1="15" x2="15" y2="15" />
          </svg>
          <p>No results yet</p>
          <span>Upload an MRI scan to get started</span>
        </div>
      </div>
    );
  }

  const getTrustClass = (score: number) => {
    if (score >= 0.8) return 'high';
    if (score >= 0.6) return 'medium';
    return 'low';
  };

  return (
    <div className="result-panel">
      <h2>Analysis Results</h2>

      <div className="result-grid">
        <div className="result-card primary">
          <h3>Tumor Classification</h3>
          <div className="result-value">{result.predicted_class}</div>
          <div className="confidence-bar">
            <div 
              className="confidence-fill" 
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
          <div className="result-meta">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </div>
        </div>

        <div className="result-card">
          <h3>Tumor Grade</h3>
          <div className="result-value">{result.predicted_grade}</div>
          <div className="grade-distribution">
            {Object.entries(result.grading).map(([grade, prob]) => (
              <div key={grade} className="grade-item">
                <span>{grade}</span>
                <span>{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>

        <div className="result-card">
          <h3>Uncertainty</h3>
          <div className="uncertainty-value">
            {(result.uncertainty.normalized_entropy * 100).toFixed(1)}%
          </div>
          <div className="uncertainty-details">
            <div>Entropy: {result.uncertainty.entropy.toFixed(3)}</div>
          </div>
        </div>

        <div className={`result-card trust-${getTrustClass(result.trustworthiness_score)}`}>
          <h3>Trustworthiness</h3>
          <div className="trust-score">
            {(result.trustworthiness_score * 100).toFixed(0)}%
          </div>
          <div className="trust-indicator">
            {result.trustworthiness_score >= 0.8 ? '✓ High Confidence' :
             result.trustworthiness_score >= 0.6 ? '⚠ Medium Confidence' :
             '⚠ Low Confidence'}
          </div>
        </div>
      </div>

      <div className="classification-details">
        <h3>Classification Probabilities</h3>
        <div className="prob-list">
          {Object.entries(result.classification)
            .sort(([, a], [, b]) => b - a)
            .map(([className, prob]) => (
              <div key={className} className="prob-item">
                <span className="prob-label">{className}</span>
                <div className="prob-bar-container">
                  <div 
                    className="prob-bar" 
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
                <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
        </div>
      </div>

      <ImageOverlay
        heatmapUrl={result.visualizations.heatmap}
        segmentationUrl={result.visualizations.segmentation}
      />
    </div>
  );
}
