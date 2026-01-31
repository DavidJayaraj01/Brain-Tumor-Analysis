import { useState } from 'react';
import UploadPanel from './components/UploadPanel';
import ResultPanel from './components/ResultPanel';
import DetailedAnalysis from './components/DetailedAnalysis';
import ChatInterface from './components/ChatInterface';
import MedicalReport from './components/MedicalReport';
import { predictImage, type PredictionResponse } from './services/api';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/Tabs';
import { Alert, AlertDescription } from './components/ui/Alert';
import { Card } from './components/ui/Card';
import { Brain, AlertCircle, CheckCircle2 } from 'lucide-react';
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
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <header className="border-b border-border bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 shadow-lg">
        <div className="container mx-auto px-4 py-8 md:py-12">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-primary/20 rounded-lg">
              <Brain className="h-8 w-8 text-primary" />
            </div>
            <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              MediMind BTIS
            </h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Multi-Agent AI Platform for Comprehensive Tumor Analysis & Diagnosis
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <div className="sticky top-8 space-y-4">
              <UploadPanel onUpload={handleUpload} isLoading={isLoading} />
              
              {error && (
                <Alert className="border-destructive/50 bg-destructive/10">
                  <AlertCircle className="h-4 w-4 text-destructive" />
                  <AlertDescription className="text-destructive">
                    {error}
                  </AlertDescription>
                </Alert>
              )}

              {result?.analysis_metadata && (
                <Card className="bg-muted/50 border-primary/20">
                  <div className="p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-primary" />
                      <h3 className="font-semibold text-sm">Analysis Complete</h3>
                    </div>
                    <div className="space-y-2 text-xs text-muted-foreground">
                      <div className="flex justify-between">
                        <span>Execution Time:</span>
                        <span className="font-mono font-semibold text-foreground">
                          {result.analysis_metadata.total_execution_time.toFixed(2)}s
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Agents Used:</span>
                        <span className="font-mono font-semibold text-foreground">
                          {result.analysis_metadata.agents_executed}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>System Version:</span>
                        <span className="font-mono font-semibold text-foreground">
                          {result.analysis_metadata.system_version}
                        </span>
                      </div>
                    </div>
                  </div>
                </Card>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            {result && (
              <Tabs value={activeTab} onValueChange={(value: any) => setActiveTab(value)} className="w-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="detailed">Analysis</TabsTrigger>
                  <TabsTrigger value="report">Report</TabsTrigger>
                  <TabsTrigger value="chat">AI Chat</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="mt-6">
                  <ResultPanel result={result} />
                </TabsContent>
                
                <TabsContent value="detailed" className="mt-6">
                  {result.multi_agent_analysis && (
                    <DetailedAnalysis analysis={result.multi_agent_analysis} />
                  )}
                </TabsContent>

                <TabsContent value="report" className="mt-6">
                  {result.multi_agent_analysis?.medical_report && (
                    <MedicalReport report={result.multi_agent_analysis.medical_report} />
                  )}
                </TabsContent>

                <TabsContent value="chat" className="mt-6">
                  <ChatInterface caseContext={JSON.stringify(result.multi_agent_analysis || {})} />
                </TabsContent>
              </Tabs>
            )}

            {!result && !isLoading && (
              <Card className="flex flex-col items-center justify-center py-16 border-dashed border-2 border-muted">
                <div className="text-center space-y-4">
                  <div className="inline-flex items-center justify-center h-16 w-16 bg-muted rounded-lg">
                    <Brain className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">No Analysis Yet</h3>
                    <p className="text-sm text-muted-foreground">
                      Upload an MRI scan to begin multi-agent AI analysis
                    </p>
                  </div>
                </div>
              </Card>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-muted/30 py-6 mt-12">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>
            Powered by 5-Agent AI System: Vision • Knowledge • Patient Context • QA • Reporting | 
            <span className="block mt-2">For research and educational purposes only</span>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
