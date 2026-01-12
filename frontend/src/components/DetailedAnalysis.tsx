import { type AgentAnalysis } from '../services/api';

interface DetailedAnalysisProps {
  analysis: AgentAnalysis;
}

export default function DetailedAnalysis({ analysis }: DetailedAnalysisProps) {
  if (!analysis) return null;

  const { differential_diagnosis, tumor_features, recommended_tests, quality_assurance } = analysis;

  return (
    <div className="detailed-analysis">
      <h2>🔬 Detailed AI Analysis</h2>

      {/* Differential Diagnosis */}
      <div className="analysis-section">
        <h3>Differential Diagnosis</h3>
        {differential_diagnosis?.map((dx: any, idx: number) => (
          <div key={idx} className="diagnosis-card">
            <div className="dx-header">
              <span className="dx-rank">#{idx + 1}</span>
              <h4>{dx.type}</h4>
              <span className="dx-probability">{(dx.probability * 100).toFixed(1)}%</span>
            </div>
            <p className="dx-reasoning"><strong>Reasoning:</strong> {dx.reasoning}</p>
            {dx.key_features && (
              <div className="dx-features">
                <strong>Supporting Features:</strong>
                <ul>
                  {dx.key_features.map((feature: string, fidx: number) => (
                    <li key={fidx}>{feature}</li>
                  ))}
                </ul>
              </div>
            )}
            {dx.next_steps && (
              <div className="dx-next-steps">
                <strong>Recommended Next Steps:</strong>
                <ul>
                  {dx.next_steps.map((step: string, sidx: number) => (
                    <li key={sidx}>{step}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Tumor Characteristics */}
      {tumor_features && (
        <div className="analysis-section">
          <h3>Tumor Characteristics</h3>
          <div className="tumor-details">
            <div className="detail-row">
              <span className="label">Location:</span>
              <span className="value">{tumor_features.location}</span>
            </div>
            <div className="detail-row">
              <span className="label">Size:</span>
              <span className="value">
                {tumor_features.size?.length_mm} × {tumor_features.size?.width_mm} × {tumor_features.size?.height_mm} mm 
                ({tumor_features.size?.volume_ml} mL)
              </span>
            </div>
            <div className="detail-row">
              <span className="label">Borders:</span>
              <span className="value">{tumor_features.borders}</span>
            </div>
            <div className="detail-row">
              <span className="label">Enhancement:</span>
              <span className="value">{tumor_features.enhancement_pattern}</span>
            </div>
            <div className="detail-row">
              <span className="label">Necrosis:</span>
              <span className="value">{tumor_features.necrosis ? 'Present' : 'Absent'}</span>
            </div>
            <div className="detail-row">
              <span className="label">Edema:</span>
              <span className="value">{tumor_features.edema}</span>
            </div>
            <div className="detail-row">
              <span className="label">Mass Effect:</span>
              <span className="value">{tumor_features.mass_effect}</span>
            </div>
          </div>
        </div>
      )}

      {/* Recommended Tests */}
      {recommended_tests && recommended_tests.length > 0 && (
        <div className="analysis-section">
          <h3>Recommended Additional Tests</h3>
          <div className="tests-list">
            {recommended_tests.map((test: any, idx: number) => (
              <div key={idx} className={`test-card priority-${test.priority?.toLowerCase()}`}>
                <div className="test-header">
                  <span className="test-name">{test.test}</span>
                  <span className={`priority-badge ${test.priority?.toLowerCase()}`}>
                    {test.priority}
                  </span>
                </div>
                <p className="test-rationale">{test.rationale}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quality Assurance */}
      {quality_assurance && (
        <div className="analysis-section">
          <h3>Quality Assurance</h3>
          <div className={`qa-status status-${quality_assurance.validation_status?.toLowerCase()}`}>
            <strong>Validation Status:</strong> {quality_assurance.validation_status}
          </div>
          <div className="qa-metrics">
            <div className="qa-metric">
              <span className="metric-label">Quality Score:</span>
              <span className="metric-value">{(quality_assurance.quality_score * 100).toFixed(1)}%</span>
            </div>
            <div className="qa-metric">
              <span className="metric-label">Confidence Level:</span>
              <span className={`metric-value confidence-${quality_assurance.confidence_level?.toLowerCase()}`}>
                {quality_assurance.confidence_level}
              </span>
            </div>
          </div>
          <div className="qa-recommendation">
            <strong>Recommendation:</strong>
            <p>{quality_assurance.recommendation}</p>
          </div>
          {quality_assurance.quality_flags && quality_assurance.quality_flags.length > 0 && (
            <div className="qa-flags">
              {quality_assurance.quality_flags.map((flag: any, idx: number) => (
                <div key={idx} className={`flag flag-${flag.type?.toLowerCase()}`}>
                  <span className="flag-icon">
                    {flag.type === 'WARNING' ? '⚠️' : flag.type === 'INFO' ? 'ℹ️' : '✓'}
                  </span>
                  <div className="flag-content">
                    <strong>{flag.category}:</strong> {flag.message}
                    {flag.recommendation && <p className="flag-rec">{flag.recommendation}</p>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
