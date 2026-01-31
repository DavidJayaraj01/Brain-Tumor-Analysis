import { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/Card';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Label } from './ui/Label';
import { Textarea } from './ui/Textarea';
import { Upload, X, Loader2, ImageIcon } from 'lucide-react';

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
  const [isDragOver, setIsDragOver] = useState(false);
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

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
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
    <Card className="bg-card border-border">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-xl">
          <Upload className="h-5 w-5 text-primary" />
          Upload MRI Scan
        </CardTitle>
        <CardDescription>
          Upload brain MRI for AI-powered tumor analysis
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Upload Area */}
        <div
          className={`
            relative rounded-lg border-2 border-dashed transition-all duration-200 cursor-pointer
            ${isDragOver ? 'border-primary bg-primary/5' : 'border-muted-foreground/25 hover:border-primary/50'}
            ${preview ? 'p-0' : 'p-8'}
          `}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !preview && fileInputRef.current?.click()}
        >
          {preview ? (
            <div className="relative group">
              <img 
                src={preview} 
                alt="Preview" 
                className="w-full h-48 object-cover rounded-lg" 
              />
              <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleReset();
                  }}
                  disabled={isLoading}
                >
                  <X className="h-4 w-4 mr-1" />
                  Remove
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="h-12 w-12 rounded-lg bg-muted flex items-center justify-center">
                <ImageIcon className="h-6 w-6 text-muted-foreground" />
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Drop your image here or click to browse
                </p>
                <p className="text-xs text-muted-foreground">
                  PNG, JPG, JPEG (max. 10MB)
                </p>
              </div>
            </div>
          )}
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isLoading}
          />
        </div>

        {/* Patient Information */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-foreground">Patient Information (Optional)</h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="patient-age" className="text-xs">Age</Label>
              <Input
                id="patient-age"
                type="number"
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                placeholder="e.g., 52"
                min="0"
                max="120"
                disabled={isLoading}
                className="bg-muted/50"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="patient-sex" className="text-xs">Sex</Label>
              <select
                id="patient-sex"
                value={patientSex}
                onChange={(e) => setPatientSex(e.target.value)}
                disabled={isLoading}
                className="flex h-10 w-full rounded-md border border-input bg-muted/50 px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <option value="">Select</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="symptoms" className="text-xs">Symptoms</Label>
            <Textarea
              id="symptoms"
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
              placeholder="e.g., headaches, seizures, vision changes"
              rows={2}
              disabled={isLoading}
              className="bg-muted/50 resize-none"
            />
          </div>
        </div>

        {/* Submit Button */}
        <Button
          onClick={handleSubmit}
          disabled={!selectedFile || isLoading}
          className="w-full"
          size="lg"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analyzing...
            </>
          ) : (
            'Analyze Scan'
          )}
        </Button>
      </CardContent>
    </Card>
  );
}
