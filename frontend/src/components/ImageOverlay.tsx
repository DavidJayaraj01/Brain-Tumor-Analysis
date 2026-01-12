import { useState } from 'react';

interface ImageOverlayProps {
  heatmapUrl: string;
  segmentationUrl: string;
}

export default function ImageOverlay({ heatmapUrl, segmentationUrl }: ImageOverlayProps) {
  const [activeView, setActiveView] = useState<'heatmap' | 'segmentation'>('heatmap');

  return (
    <div className="image-overlay-container">
      <h3>Visualizations</h3>
      
      <div className="view-tabs">
        <button
          className={activeView === 'heatmap' ? 'active' : ''}
          onClick={() => setActiveView('heatmap')}
        >
          Grad-CAM Heatmap
        </button>
        <button
          className={activeView === 'segmentation' ? 'active' : ''}
          onClick={() => setActiveView('segmentation')}
        >
          Segmentation Mask
        </button>
      </div>

      <div className="overlay-display">
        {activeView === 'heatmap' && (
          <div className="overlay-image">
            <img src={heatmapUrl} alt="Grad-CAM Heatmap" />
            <p className="overlay-caption">
              Regions of interest highlighted by the model's attention mechanism
            </p>
          </div>
        )}
        
        {activeView === 'segmentation' && (
          <div className="overlay-image">
            <img src={segmentationUrl} alt="Segmentation Mask" />
            <p className="overlay-caption">
              Predicted tumor segmentation boundary
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
