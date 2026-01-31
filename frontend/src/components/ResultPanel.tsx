import { type PredictionResponse } from '../services/api';
import ImageOverlay from './ImageOverlay';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/Card';
import { Badge } from './ui/Badge';
import { Progress } from './ui/Progress';
import { Brain, Activity, AlertTriangle, Shield, TrendingUp } from 'lucide-react';

interface ResultPanelProps {
  result: PredictionResponse | null;
}

export default function ResultPanel({ result }: ResultPanelProps) {
  if (!result) {
    return (
      <Card className="flex flex-col items-center justify-center py-16 border-dashed border-2 border-muted">
        <div className="text-center space-y-4">
          <div className="inline-flex items-center justify-center h-16 w-16 bg-muted rounded-lg">
            <Brain className="h-8 w-8 text-muted-foreground" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">No Results Yet</h3>
            <p className="text-sm text-muted-foreground">
              Upload an MRI scan to get started
            </p>
          </div>
        </div>
      </Card>
    );
  }

  const getTrustClass = (score: number) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getTrustBadge = (score: number) => {
    if (score >= 0.8) return 'default';
    if (score >= 0.6) return 'secondary';
    return 'destructive';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Activity className="h-6 w-6 text-primary" />
          Analysis Results
        </h2>
      </div>

      {/* Primary Results Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Tumor Classification */}
        <Card className="border-l-4 border-l-primary bg-gradient-to-br from-primary/5 to-transparent">
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Tumor Classification
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-primary mb-3">
              {result.predicted_class}
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Confidence</span>
                <span className="font-semibold">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <Progress value={result.confidence * 100} className="h-2" />
            </div>
          </CardContent>
        </Card>

        {/* Tumor Grade */}
        <Card className="border-l-4 border-l-accent">
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Tumor Grade
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-3">
              {result.predicted_grade}
            </div>
            <div className="space-y-1">
              {Object.entries(result.grading).map(([grade, prob]) => (
                <div key={grade} className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">{grade}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-accent rounded-full" 
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="text-xs font-mono w-12 text-right">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Uncertainty */}
        <Card className="border-l-4 border-l-yellow-500/50">
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Uncertainty Analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-3">
              {(result.uncertainty.normalized_entropy * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">
              <span>Entropy: </span>
              <span className="font-mono">{result.uncertainty.entropy.toFixed(4)}</span>
            </div>
          </CardContent>
        </Card>

        {/* Trustworthiness */}
        <Card className={`border-l-4 ${result.trustworthiness_score >= 0.8 ? 'border-l-green-500' : result.trustworthiness_score >= 0.6 ? 'border-l-yellow-500' : 'border-l-red-500'}`}>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Trustworthiness
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className={`text-3xl font-bold mb-3 ${getTrustClass(result.trustworthiness_score)}`}>
              {(result.trustworthiness_score * 100).toFixed(0)}%
            </div>
            <Badge variant={getTrustBadge(result.trustworthiness_score)}>
              {result.trustworthiness_score >= 0.8 ? '✓ High Confidence' :
               result.trustworthiness_score >= 0.6 ? '⚠ Medium Confidence' :
               '⚠ Low Confidence'}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Classification Probabilities */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Classification Probabilities</CardTitle>
          <CardDescription>Probability distribution across all tumor types</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {Object.entries(result.classification)
              .sort(([, a], [, b]) => b - a)
              .map(([className, prob]) => (
                <div key={className} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{className}</span>
                    <span className="text-muted-foreground font-mono">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 ${
                        prob === Math.max(...Object.values(result.classification)) 
                          ? 'bg-primary' 
                          : 'bg-primary/40'
                      }`}
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                </div>
              ))}
          </div>
        </CardContent>
      </Card>

      {/* Visualizations */}
      <ImageOverlay
        heatmapUrl={result.visualizations.heatmap}
        segmentationUrl={result.visualizations.segmentation}
      />
    </div>
  );
}
