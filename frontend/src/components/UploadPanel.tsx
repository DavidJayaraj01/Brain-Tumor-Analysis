import { useState, useRef } from 'react';

interface UploadPanelProps {
  onUpload: (file: File, patientData: {age?: number, sex?: string, symptoms?: string}) => void;
  isLoading: boolean;
}

export default function UploadPanel({ onUpload, isLoading }: UploadPanelProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [patientAge, setPatientAge] = useState<string>('');
  const [patientSex, setPatientSex] = useState<string>('');
  const [symptoms, setSymptoms] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = () => {
    if (selectedFile) {
      const patientData = {
        age: patientAge ? parseInt(patientAge) : undefined,
        sex: patientSex || undefined,
        symptoms: symptoms || undefined
      };
      onUpload(selectedFile, patientData);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setPatientAge('');
    setPatientSex('');
    setSymptoms('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="upload-panel">
      <h2>Upload MRI Scan</h2>
      
      <div className="upload-area">
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <button onClick={handleReset} className="reset-btn" disabled={isLoading}>
              ✕ Clear
            </button>
          </div>
        ) : (
          <label htmlFor="file-upload" className="upload-label">
            <div className="upload-placeholder">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <p>Click to upload or drag and drop</p>
              <span>PNG, JPG, JPEG (max. 10MB)</span>
            </div>
          </label>
        )}
        
        <input
          ref={fileInputRef}
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          disabled={isLoading}
        />
      </div>

      {/* Patient Information */}
      <div className="patient-info-form">
        <h3>Patient Information (Optional)</h3>
        
        <div className="form-group">
          <label htmlFor="patient-age">Age</label>
          <input
            id="patient-age"
            type="number"
            value={patientAge}
            onChange={(e) => setPatientAge(e.target.value)}
            placeholder="e.g., 52"
            min="0"
            max="120"
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="patient-sex">Sex</label>
          <select
            id="patient-sex"
            value={patientSex}
            onChange={(e) => setPatientSex(e.target.value)}
            disabled={isLoading}
          >
            <option value="">Select</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="symptoms">Symptoms</label>
          <textarea
            id="symptoms"
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            placeholder="e.g., headaches, seizures, vision changes (comma separated)"
            rows={3}
            disabled={isLoading}
          />
        </div>
      </div>

      <button
        onClick={handleSubmit}
        disabled={!selectedFile || isLoading}
        className="submit-btn"
      >
        {isLoading ? 'Analyzing...' : 'Analyze Scan'}
      </button>
    </div>
  );
}
