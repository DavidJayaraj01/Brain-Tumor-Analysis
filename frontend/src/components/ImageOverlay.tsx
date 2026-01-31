import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/Tabs';
import { Eye } from 'lucide-react';

interface ImageOverlayProps {
  heatmapUrl: string;
  segmentationUrl: string;
}

export default function ImageOverlay({ heatmapUrl, segmentationUrl }: ImageOverlayProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Eye className="h-5 w-5" />
          Visualizations
        </CardTitle>
        <CardDescription>AI-generated attention and segmentation maps</CardDescription>
      </CardHeader>
      
      <CardContent>
        <Tabs defaultValue="heatmap" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="heatmap">Grad-CAM Heatmap</TabsTrigger>
            <TabsTrigger value="segmentation">Segmentation Mask</TabsTrigger>
          </TabsList>
          
          <TabsContent value="heatmap" className="mt-4">
            <div className="space-y-3">
              <div className="rounded-lg border border-border overflow-hidden bg-black">
                <img 
                  src={heatmapUrl} 
                  alt="Grad-CAM Heatmap" 
                  className="w-full h-auto"
                />
              </div>
              <p className="text-sm text-muted-foreground text-center">
                Regions of interest highlighted by the model's attention mechanism
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="segmentation" className="mt-4">
            <div className="space-y-3">
              <div className="rounded-lg border border-border overflow-hidden bg-black">
                <img 
                  src={segmentationUrl} 
                  alt="Segmentation Mask" 
                  className="w-full h-auto"
                />
              </div>
              <p className="text-sm text-muted-foreground text-center">
                Predicted tumor segmentation boundary
              </p>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
