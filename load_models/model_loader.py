from torchvision import models as tv_models
import timm
import torch.nn as nn
from timm.data.transforms_factory import create_transform
from timm.layers import ClassifierHead
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.transforms import augment_then_model_transform

def get_model_with_head(
    model_name: str,
    num_classes: int,
    source: str = "torchvision",  # or 'timm'
    tv_weights: str = "DEFAULT",
    freeze: bool = True,
    keep_imagenet_head=False
):
    if source == "torchvision":
        model_name = model_name.lower()
        model_fn = getattr(tv_models, model_name)
        if isinstance(tv_weights, str):
            weights_enum = tv_models.get_model_weights(model_fn)
            if tv_weights.upper() == "DEFAULT":
                weights = weights_enum.DEFAULT
            else:
                try:
                    weights = getattr(weights_enum, tv_weights)
                except AttributeError:
                    raise ValueError(f"Invalid weight name '{tv_weights}' for model '{model_name}'. "
                                     f"Available: {[w.name for w in weights_enum]}")
        else:
            weights = None
        model = model_fn(weights=weights)
        model_transform = weights.transforms() if weights is not None else None
    elif source == "timm":
        model = timm.create_model(model_name, pretrained=True)
        model_transform = create_transform(**timm.data.resolve_data_config({}, model=model))
    else:
        raise ValueError(f"Currently only support source = 'torchvision' or 'timm', received invalid source {source}")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    def build_new_head(in_features, num_classes):
        def enable_grad(m):
            for p in m.parameters():
                p.requires_grad = True
            return m
        return enable_grad(nn.Linear(in_features, num_classes))

    def replace_fc(m, attr_name):
        layer = getattr(m, attr_name)
        def inject_head(in_features):
            return build_new_head(in_features, num_classes)
        if isinstance(layer, nn.Sequential):
            for i in reversed(range(len(layer))):
                if isinstance(layer[i], nn.Linear):
                    in_features = layer[i].in_features
                    layer[i] = build_new_head(in_features, num_classes)
                    return
            print(f"Warning: No nn.Linear found in Sequential '{attr_name}'. Head not replaced.")

        elif isinstance(layer, nn.Linear):
            in_features = layer.in_features
            new_head = build_new_head(in_features, num_classes)
            setattr(m, attr_name, new_head)
            
        elif isinstance(layer, ClassifierHead):
            if isinstance(layer.fc, nn.Linear):
                in_features = layer.fc.in_features
                layer.fc = inject_head(in_features)
            else:
                print(f"Warning: ClassifierHead.fc is not a Linear layer")

        else:
            print(f"Warning: Attribute '{attr_name}' is not a Linear or Sequential layer. Got {type(layer)}")

    classifier_attr = None
    if hasattr(model, "classifier"):
        classifier_attr = "classifier"
    elif hasattr(model, "fc"):
        classifier_attr = "fc"
    elif hasattr(model, "head"):
        classifier_attr = "head"
        
    if ((not keep_imagenet_head) and classifier_attr):
        replace_fc(model, classifier_attr)
        
    # Add forward_features method to the model
    def add_forward_features_method(model, classifier_attr):
        import types
        if hasattr(model, 'forward_features') and callable(getattr(model, 'forward_features')):
            print(f"(timm) Model already has forward_features method, skipping addition, adding pooling.")
            def forward_features_pooled(self, x):
                x = self.forward_features(x)
                if x.dim() == 4:  # (batch, channels, H, W)
                    x = x.mean(dim=[-2, -1]) # Pooling for timm model
                if x.dim() > 2:
                    x = x.flatten(1)
                return x
            model.forward_features_pooled = types.MethodType(forward_features_pooled, model)
            return
        
        """Add a forward_features method that extracts features before the final classifier."""
        if classifier_attr is None:
            print("Warning: Could not find classifier layer. forward_features will not be added.")
            return
        # Get reference to the classifier layer
        classifier_parent = model
        classifier_name = classifier_attr
        # Create the forward_features method
        def forward_features_pooled(self, x):
            # Save original classifier
            original = getattr(classifier_parent, classifier_name)
            try:
                setattr(classifier_parent, classifier_name, nn.Identity())
                features = self(x)
                if features.dim() > 2:
                    features = features.flatten(1)
            finally:
                # Restore it even if there's an error
                setattr(classifier_parent, classifier_name, original)
            return features
        # Bind the method to the model instance
        model.forward_features_pooled = types.MethodType(forward_features_pooled, model)
    
    add_forward_features_method(model, classifier_attr)
    
    transform=augment_then_model_transform(model_transform)
    return model, transform
