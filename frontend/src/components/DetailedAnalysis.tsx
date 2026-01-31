import { type AgentAnalysis } from '../services/api';
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card';
import { Badge } from './ui/Badge';
import { Brain, Activity, Beaker, CheckCircle, AlertCircle } from 'lucide-react';

interface DetailedAnalysisProps {
  analysis: AgentAnalysis;
}

export default function DetailedAnalysis({ analysis }: DetailedAnalysisProps) {
  if (!analysis) return null;

  const { differential_diagnosis, tumor_features, recommended_tests, quality_assurance } = analysis;

  return (
    <div className="space-y-6">
      {/* Differential Diagnosis */}
      {differential_diagnosis && differential_diagnosis.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Differential Diagnosis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {differential_diagnosis.map((dx: any, idx: number) => (
              <Card key={idx} className="border-l-4 border-l-primary">
                <CardContent className="pt-6 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Badge variant="default">#{idx + 1}</Badge>
                      <h4 className="text-lg font-semibold">{dx.type}</h4>
                    </div>
                    <Badge variant="secondary" className="text-lg">
                      {(dx.probability * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  
                  {dx.reasoning && (
                    <div className="bg-muted/50 rounded-md p-3">
                      <p className="text-sm"><strong>Reasoning:</strong> {dx.reasoning}</p>
                    </div>
                  )}
                  
                  {dx.key_features && dx.key_features.length > 0 && (
                    <div>
                      <p className="text-sm font-semibold mb-2">Supporting Features:</p>
                      <ul className="space-y-1">
                        {dx.key_features.map((feature: string, fidx: number) => (
                          <li key={fidx} className="text-sm flex items-start gap-2">
                            <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                            <span>{feature}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {dx.next_steps && dx.next_steps.length > 0 && (
                    <div>
                      <p className="text-sm font-semibold mb-2">Recommended Next Steps:</p>
                      <ul className="space-y-1">
                        {dx.next_steps.map((step: string, sidx: number) => (
                          <li key={sidx} className="text-sm flex items-start gap-2">
                            <Activity className="h-4 w-4 text-accent mt-0.5 flex-shrink-0" />
                            <span>{step}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Tumor Characteristics */}
      {tumor_features && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Tumor Characteristics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {tumor_features.location && (
              <div className="flex justify-between items-center border-b border-border pb-2">
                <span className="text-sm font-medium text-muted-foreground">Location</span>
                <span className="text-sm font-semibold">{tumor_features.location}</span>
              </div>
            )}
            {tumor_features.size && (
              <div className="flex justify-between items-center border-b border-border pb-2">
                <span className="text-sm font-medium text-muted-foreground">Size</span>
                <span className="text-sm font-semibold">
                  {typeof tumor_features.size === 'string' 
                    ? tumor_features.size
                    : tumor_features.size.estimated_volume_ml 
                      ? `${tumor_features.size.estimated_volume_ml.toFixed(1)} mL (${tumor_features.size.category || 'moderate'})`
                      : 'Not specified'}
                </span>
              </div>
            )}
            {tumor_features.enhancement_pattern && (
              <div className="flex justify-between items-center border-b border-border pb-2">
                <span className="text-sm font-medium text-muted-foreground">Enhancement Pattern</span>
                <span className="text-sm font-semibold">{tumor_features.enhancement_pattern}</span>
              </div>
            )}
            {tumor_features.margins && (
              <div className="flex justify-between items-center border-b border-border pb-2">
                <span className="text-sm font-medium text-muted-foreground">Margins</span>
                <span className="text-sm font-semibold">{tumor_features.margins}</span>
              </div>
            )}
            {tumor_features.associated_findings && tumor_features.associated_findings.length > 0 && (
              <div>
                <p className="text-sm font-medium text-muted-foreground mb-2">Associated Findings</p>
                <ul className="space-y-1">
                  {tumor_features.associated_findings.map((finding: string, idx: number) => (
                    <li key={idx} className="text-sm flex items-start gap-2">
                      <span className="text-primary">•</span>
                      <span>{finding}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Recommended Tests */}
      {recommended_tests && recommended_tests.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Beaker className="h-5 w-5" />
              Recommended Tests
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {recommended_tests.map((test: any, idx: number) => (
              <div key={idx} className="border border-border rounded-md p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <h5 className="font-semibold">{test.test_name || test.name}</h5>
                  <Badge variant={test.priority === 'high' ? 'destructive' : test.priority === 'medium' ? 'default' : 'secondary'}>
                    {test.priority || 'Normal'} Priority
                  </Badge>
                </div>
                {test.reason && (
                  <p className="text-sm text-muted-foreground">{test.reason}</p>
                )}
                {test.expected_findings && (
                  <div className="text-sm">
                    <span className="font-medium">Expected Findings: </span>
                    <span className="text-muted-foreground">{test.expected_findings}</span>
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Quality Assurance */}
      {quality_assurance && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {quality_assurance.passed ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <AlertCircle className="h-5 w-5 text-red-500" />
              )}
              Quality Assurance
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Status</span>
              <Badge variant={quality_assurance.passed ? 'default' : 'destructive'}>
                {quality_assurance.passed ? 'Passed' : 'Failed'}
              </Badge>
            </div>
            {quality_assurance.confidence_score !== undefined && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Confidence Score</span>
                <span className="text-sm font-semibold">{(quality_assurance.confidence_score * 100).toFixed(1)}%</span>
              </div>
            )}
            {quality_assurance.warnings && quality_assurance.warnings.length > 0 && (
              <div>
                <p className="text-sm font-medium mb-2">Warnings</p>
                <ul className="space-y-1">
                  {quality_assurance.warnings.map((warning: string, idx: number) => (
                    <li key={idx} className="text-sm flex items-start gap-2 text-yellow-600 dark:text-yellow-500">
                      <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                      <span>{warning}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {quality_assurance.recommendations && quality_assurance.recommendations.length > 0 && (
              <div>
                <p className="text-sm font-medium mb-2">Recommendations</p>
                <ul className="space-y-1">
                  {quality_assurance.recommendations.map((rec: string, idx: number) => (
                    <li key={idx} className="text-sm flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
