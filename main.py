import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class ThreeStreamDataset(Dataset):
    def __init__(self, img_csv, clin_csv, label_csv):
        super().__init__()
        
        df_img = pd.read_csv(img_csv)
        df_clin = pd.read_csv(clin_csv)
        df_label = pd.read_csv(label_csv)

        self.img_feat_cols = [c for c in df_img.columns if c != 'patient_id']
        self.clin_feat_cols = [c for c in df_clin.columns if c != 'patient_id']
        label_col_name = df_label.columns[1] 

        temp_df = pd.merge(df_img, df_clin, on='patient_id', how='inner')
        self.data = pd.merge(temp_df, df_label, on='patient_id', how='inner')
        
        img_data = self.data[self.img_feat_cols]
        clin_data = self.data[self.clin_feat_cols]
        
        if img_data.isnull().values.any():
            print("NaN to 0")
            img_data = img_data.fillna(0)
            
        if clin_data.isnull().values.any():
            print("NaN to 0")
            clin_data = clin_data.fillna(0)

        img_data = img_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        clin_data = clin_data.apply(pd.to_numeric, errors='coerce').fillna(0)

        self.img_features = img_data.values.astype(np.float32)
        raw_clinical = clin_data.values.astype(np.float32)
        self.labels = self.data[label_col_name].values.astype(np.int64)

        if not np.isfinite(self.img_features).all():
            self.img_features = np.nan_to_num(self.img_features)
        if not np.isfinite(raw_clinical).all():
            raw_clinical = np.nan_to_num(raw_clinical)

        actual_clin_dim = raw_clinical.shape[1]
        print(f"Dataset feature num: {actual_clin_dim}")
        
        self.scaler = StandardScaler()
        try:
            self.clinical_data = self.scaler.fit_transform(raw_clinical)
        except ValueError as e:
            print("wrong")
            self.clinical_data = raw_clinical 
            
        if np.isnan(self.clinical_data).any():
            self.clinical_data = np.nan_to_num(self.clinical_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_feat = torch.tensor(self.img_features[idx])
        clin_data = torch.tensor(self.clinical_data[idx])
        label = torch.tensor(self.labels[idx])
        return img_feat, clin_data, label

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, 
                                               dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_input, key_value_input):
        """
        query_input: 作为 Q (例如图像特征) [Batch, Dim]
        key_value_input: 作为 K, V (例如临床特征) [Batch, Dim]
        """

        q = query_input.unsqueeze(1)
        k = v = key_value_input.unsqueeze(1)

        attn_output, _ = self.attention(query=q, key=k, value=v)

        out = query_input + attn_output.squeeze(1)
        out = self.norm(out)
        
        return out

class SwinMlpFusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(36, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        self.img_encoder = nn.Sequential(
             nn.Linear(1024, 1024),
             nn.LayerNorm(1024),
             nn.ReLU()
        )

        feature_dim = 1024
        
        self.cross_attn_img_query = CrossAttentionBlock(dim=feature_dim)
        self.cross_attn_clin_query = CrossAttentionBlock(dim=feature_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim), # 2048 -> 1024
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, img_feat, clin_data):

        F2 = self.clinical_encoder(clin_data)
        F1 = self.img_encoder(img_feat) 

        left_out = self.cross_attn_img_query(query_input=F1, key_value_input=F2)

        right_out = self.cross_attn_clin_query(query_input=F2, key_value_input=F1)
        concat_feat = torch.cat([left_out, right_out], dim=1)
        F_fused = self.fusion_mlp(concat_feat)
        logits = self.classifier(F_fused)
        
        return logits
if __name__ == "__main__":

    IMG_CSV = '1024d_features.csv'   
    CLIN_CSV = 'DCM prognosis.csv'
    LABEL_CSV = 'maces.csv'
    
    BATCH_SIZE = 16
    LR = 0.001
    EPOCHS = 50
    K_FOLDS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {DEVICE}")

    full_dataset = ThreeStreamDataset(IMG_CSV, CLIN_CSV, LABEL_CSV)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    plt.figure(figsize=(10, 8))

    for fold, (train_ids, val_ids) in enumerate(skf.split(np.zeros(len(full_dataset)), full_dataset.labels)):
        print(f"\n================ Fold {fold + 1} / {K_FOLDS} ================")
        
        train_sub = Subset(full_dataset, train_ids)
        val_sub = Subset(full_dataset, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
        
        model = SwinMlpFusionModel(num_classes=2).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        
        for epoch in range(EPOCHS):
            model.train()
            for img, clin, labels in train_loader:
                img, clin, labels = img.to(DEVICE), clin.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(img, clin)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        print(f"Fold {fold+1} evaluating...")
        model.eval()
        
        y_true_fold = []
        y_scores_fold = []
        
        with torch.no_grad():
            for img, clin, labels in val_loader:
                img, clin, labels = img.to(DEVICE), clin.to(DEVICE), labels.to(DEVICE)
                outputs = model(img, clin)
                probs = torch.softmax(outputs, dim=1)

                preds = probs[:, 1].cpu().numpy()
                targets = labels.cpu().numpy()
                
                y_true_fold.extend(targets)
                y_scores_fold.extend(preds)
        
        y_true_fold = np.array(y_true_fold)
        y_scores_fold = np.array(y_scores_fold)
        
        fpr, tpr, thresholds = roc_curve(y_true_fold, y_scores_fold)
        roc_auc = auc(fpr, tpr)
        
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0 
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label=f'ROC Fold {fold+1} (AUC = {roc_auc:.3f})')
        
        print(f"--> Fold {fold + 1} AUC: {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})',
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=f'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'ROC Curve - Cross Validation (Mean AUC={mean_auc:.3f})', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    save_path = 'roc_curve_5folds.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nROC curve was saved to : {save_path}")
    plt.show()
