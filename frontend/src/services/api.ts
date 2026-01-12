export interface AgentAnalysis {
  tumor_features: any;
  differential_diagnosis: any[];
  literature_references: any[];
  recommended_tests: any[];
  treatment_guidelines: any;
  patient_context: any;
  quality_assurance: any;
  medical_report: any;
}

export interface PredictionResponse {
  classification: Record<string, number>;
  predicted_class: string;
  confidence: number;
  grading: Record<string, number>;
  predicted_grade: string;
  uncertainty: {
    entropy: number;
    normalized_entropy: number;
    confidence: number;
  };
  trustworthiness_score: number;
  visualizations: {
    heatmap: string;
    segmentation: string;
  };
  multi_agent_analysis?: AgentAnalysis;
  analysis_metadata?: any;
}

const API_BASE_URL = 'http://localhost:8000';

export async function predictImage(
  file: File,
  patientAge?: number,
  patientSex?: string,
  symptoms?: string
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (patientAge) formData.append('patient_age', patientAge.toString());
  if (patientSex) formData.append('patient_sex', patientSex);
  if (symptoms) formData.append('symptoms', symptoms);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Prediction failed');
  }

  return await response.json();
}

export async function chatWithAI(question: string, caseContext: string): Promise<any> {
  const formData = new FormData();
  formData.append('question', question);
  formData.append('case_context', caseContext);

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Chat request failed');
  }

  return await response.json();
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return await response.json();
}

export async function getAgentHealth(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/agent/health`);
  return await response.json();
}
