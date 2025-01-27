from utils.tools import *
from utils.contrastive import SupConLoss
import logging

class BertForModel(nn.Module):
    def __init__(self,model_name, num_labels, device=None):
        super(BertForModel, self).__init__()
        self.logger = logging.getLogger('pretrain')  # Use existing logger from pretrain
        self.num_labels = num_labels
        self.model_name = model_name
        self.device = device
        # 1. BERT backbone:
        # Load BERT backbone
        self.logger.info(f"Loading BERT backbone from {self.model_name}")
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name, local_files_only=True, trust_remote_code=True)
        # 2. Classifier: Used for classification tasks
        self.logger.info("BERT backbone loaded successfully")
        
        # Initialize classifier
        self.classifier = nn.Linear(768, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.backbone.to(self.device)
        self.classifier.to(self.device)
        self.logger.info(f"Model components moved to device: {self.device}")

    def forward(self, X):
        # BERT backbone handles input 1. Generate embeddings
        outputs = self.backbone(**X, output_hidden_states=True)
        # get [CLS] token
        CLSEmbedding = outputs.hidden_states[-1][:,0]
        # Classifier # 2. Prediction Label
        logits = self.classifier(CLSEmbedding)
        return {"logits": logits}

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output
    
    def save_backbone(self, save_path):
        # Saves the BERT backbone weights to a specified path for reuse or further fine-tuning.
        try: 
            self.logger.info(f"Saving BERT backbone to {save_path}")
            self.backbone.save_pretrained(save_path)
            self.logger.info("BERT backbone saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving BERT backbone: {str(e)}")
            raise e


class CLBert(nn.Module):
    def __init__(self,model_name, device, feat_dim=128):
        super(CLBert, self).__init__()
        self.logger = logging.getLogger('HNSCL')  # Use existing logger from HNSCL
        
        self.model_name = model_name
        self.device = device
        self.feat_dim = feat_dim
        
        # Load BERT backbone
        self.logger.info(f"Loading BERT backbone for HNSCL from {self.model_name}")
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name, local_files_only=True, trust_remote_code=True)
        hidden_size = self.backbone.config.hidden_size
        self.logger.info("BERT backbone loaded successfully")
        
        # Initialize projection head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.feat_dim)
        )
        self.backbone.to(device)
        self.head.to(device)
        self.logger.info(f"Model components moved to device: {device}")
        
    def forward(self, X, output_hidden_states=False, output_attentions=False, output_logits=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True, output_attentions=True)
        cls_embed = outputs.hidden_states[-1][:,0]
        # maps the embeddings into a lower-dimensional feature space
        features = F.normalize(self.head(cls_embed), dim=1)
        output_dir = {"features": features}
        if output_hidden_states:
            output_dir["hidden_states"] = cls_embed
        if output_attentions:
            output_dir["attentions"] = outputs.attentions
        return output_dir

    def loss_cl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=0.07):
        """compute contrastive loss"""
        loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        output = loss(embds, labels=label, mask=mask)
        return output
    
    def save_backbone(self, save_path):
        """Saves the BERT backbone weights"""
        try:
            self.logger.info(f"Saving BERT backbone to {save_path}")
            self.backbone.save_pretrained(save_path)
            self.logger.info("BERT backbone saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving BERT backbone: {str(e)}")
            raise e
