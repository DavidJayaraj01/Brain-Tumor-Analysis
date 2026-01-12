import torch
import timm

# Test EfficientNetV2 feature dimensions
model = timm.create_model(
    'tf_efficientnetv2_s',
    pretrained=False,
    features_only=True,
    out_indices=(1, 2, 3, 4)
)

# Test with dummy input
x = torch.randn(1, 3, 224, 224)
features = model(x)

print("EfficientNetV2-S Feature Dimensions:")
for i, feat in enumerate(features):
    print(f"  Scale {i+1}: {feat.shape}")
print(f"\nFeature channels: {model.feature_info.channels()}")
