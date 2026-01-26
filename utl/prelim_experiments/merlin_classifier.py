import torch
import torch.nn as nn
import torch.nn.functional as F


class MerlinClassifier(nn.Module):
    def __init__(self, base_model, projection_dim=128, num_classes=2):
        super(MerlinClassifier, self).__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
            
        self.projection1 = nn.Sequential(
            nn.Linear(2048, 256),
            #nn.LayerNorm(256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
            )
        self.projection2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            #nn.ReLU(),
            nn.GELU(),
            nn.Dropout(0.3)
            )
        self.projection3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            #nn.ReLU(),
            nn.GELU(),
            nn.Dropout(0.3)
            )
        self.classifier = CosineClassifier(256, num_classes, 10.0)
        #self.classifier = nn.Linear(128, num_classes)
        #self.classifier = nn.Linear(2048, num_classes)
    
    def _init_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.weight.data *= 0.05
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.projection1.apply(init_weights)
        if hasattr(self.classifier, 'weight'):
            with torch.no_grad():
                nn.init.orthogonal_(self.classifier.weight)
                self.classifier.weight[:] = F.normalize(self.classifier.weight, dim=1)
            
    
    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        
    def forward(self, x, return_norms=False):
        with torch.no_grad():
            x = self.base_model(x)
            
        x = x.squeeze(0)
        x = self.projection1(x)
        #x = self.projection2(x)
        #x = self.projection3(x)
        norms = x.norm(dim=-1, keepdim=True)
        x = x/norms.clamp(min=1e-6, max=10.0)
        if return_norms:
            norms = x.norm(dim=-1)
        
        
        x = F.normalize(x, dim=-1)
        x = self.classifier(x)
        if return_norms:
            return x, norms
        return x
