import { useState } from 'react';
import UploadPanel from './components/UploadPanel';
import ResultPanel from './components/ResultPanel';
import DetailedAnalysis from './components/DetailedAnalysis';
import ChatInterface from './components/ChatInterface';
import MedicalReport from './components/MedicalReport';
import { predictImage, type PredictionResponse } from './services/api';
import './styles/app.css';

function App() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'detailed' | 'report' | 'chat'>('overview');

  const handleUpload = async (file: File, patientData: any) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const prediction = await predictImage(
        file,
        patientData.age,
        patientData.sex,
        patientData.symptoms
      );
      setResult(prediction);
      setActiveTab('overview');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🧠 MediMind Brain Tumor Intelligence System</h1>
        <p>Multi-Agent AI Platform for Comprehensive Tumor Analysis & Diagnosis</p>
      </header>

      <main className="app-main">
        <div className="app-container">
          <div className="upload-section">
            <UploadPanel onUpload={handleUpload} isLoading={isLoading} />
            
            {error && (
              <div className="error-message">
                <strong>Error:</strong> {error}
              </div>
            )}

            {result?.analysis_metadata && (
              <div className="analysis-metadata">
                <h3>Analysis Info</h3>
                <p><strong>Execution Time:</strong> {result.analysis_metadata.total_execution_time.toFixed(2)}s</p>
                <p><strong>Agents Used:</strong> {result.analysis_metadata.agents_executed}</p>
                <p><strong>Version:</strong> {result.analysis_metadata.system_version}</p>
              </div>
            )}
          </div>

          <div className="results-section">
            {result && (
              <>
                <div className="results-tabs">
                  <button
                    className={activeTab === 'overview' ? 'active' : ''}
                    onClick={() => setActiveTab('overview')}
                  >
                    📊 Overview
                  </button>
                  <button
                    className={activeTab === 'detailed' ? 'active' : ''}
                    onClick={() => setActiveTab('detailed')}
                  >
                    🔬 Detailed Analysis
                  </button>
                  <button
                    className={activeTab === 'report' ? 'active' : ''}
                    onClick={() => setActiveTab('report')}
                  >
                    📋 Medical Report
                  </button>
                  <button
                    className={activeTab === 'chat' ? 'active' : ''}
                    onClick={() => setActiveTab('chat')}
                  >
                    💬 Ask AI
                  </button>
                </div>

                <div className="tab-content">
                  {activeTab === 'overview' && <ResultPanel result={result} />}
                  
                  {activeTab === 'detailed' && result.multi_agent_analysis && (
                    <DetailedAnalysis analysis={result.multi_agent_analysis} />
                  )}
                  
                  {activeTab === 'report' && result.multi_agent_analysis?.medical_report && (
                    <MedicalReport report={result.multi_agent_analysis.medical_report} />
                  )}
                  
                  {activeTab === 'chat' && (
                    <ChatInterface caseContext={JSON.stringify(result.multi_agent_analysis || {})} />
                  )}
                </div>
              </>
            )}

            {!result && !isLoading && (
              <div className="empty-state">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="12" y1="18" x2="12" y2="12" />
                  <line x1="9" y1="15" x2="15" y2="15" />
                </svg>
                <p>No results yet</p>
                <span>Upload an MRI scan to get started with multi-agent AI analysis</span>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Powered by 5-Agent AI System: Vision • Knowledge • Patient Context • QA • Reporting | For research purposes only</p>
      </footer>
    </div>
  );
}

export default App;
