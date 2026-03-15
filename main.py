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

# ==========================================
# 1. 数据集类 (保持你的数据清洗逻辑不变)
# ==========================================
class ThreeStreamDataset(Dataset):
    def __init__(self, img_csv, clin_csv, label_csv):
        super().__init__()
        
        # 1. 读取 CSV
        df_img = pd.read_csv(img_csv)
        df_clin = pd.read_csv(clin_csv)
        df_label = pd.read_csv(label_csv)

        # 2. 确定特征列
        self.img_feat_cols = [c for c in df_img.columns if c != 'patient_id']
        self.clin_feat_cols = [c for c in df_clin.columns if c != 'patient_id']
        label_col_name = df_label.columns[1] 

        # 3. 合并数据
        temp_df = pd.merge(df_img, df_clin, on='patient_id', how='inner')
        self.data = pd.merge(temp_df, df_label, on='patient_id', how='inner')

        # ==================================================
        # 数据清洗步骤
        # ==================================================
        
        img_data = self.data[self.img_feat_cols]
        clin_data = self.data[self.clin_feat_cols]
        
        # 填充 NaN
        if img_data.isnull().values.any():
            print("⚠️ 警告: 图像特征中发现缺失值(NaN)，已自动填充为0")
            img_data = img_data.fillna(0)
            
        if clin_data.isnull().values.any():
            print("⚠️ 警告: 临床数据中发现缺失值(NaN)，已自动填充为0")
            clin_data = clin_data.fillna(0)
            
        # 转换为数值类型
        img_data = img_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        clin_data = clin_data.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 转换为 Numpy
        self.img_features = img_data.values.astype(np.float32)
        raw_clinical = clin_data.values.astype(np.float32)
        self.labels = self.data[label_col_name].values.astype(np.int64)

        # 检查 Inf
        if not np.isfinite(self.img_features).all():
            self.img_features = np.nan_to_num(self.img_features)
        if not np.isfinite(raw_clinical).all():
            raw_clinical = np.nan_to_num(raw_clinical)

        # ==================================================

        # 维度检查 (请确保这里和你实际CSV列数一致，假设是36)
        actual_clin_dim = raw_clinical.shape[1]
        print(f"Dataset 初始化: 检测到临床特征数量: {actual_clin_dim}")
        
        # 归一化
        self.scaler = StandardScaler()
        try:
            self.clinical_data = self.scaler.fit_transform(raw_clinical)
        except ValueError as e:
            print("❌ 归一化出错，使用原始数据继续")
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

# ==========================================
# 2. 模型定义 (修改版)
# ==========================================

class CrossAttentionBlock(nn.Module):
    """
    对应图中单侧的结构：
    Input -> Linear Proj (in MHA) -> Cross Attention -> Residual Add
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        # MultiheadAttention 内部包含了图中的 "Linear proj" (生成 Q, K, V)
        # batch_first=True 使得输入格式为 (Batch, Seq_len, Dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, 
                                               dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim) #通常在Attention前后加Norm有助于收敛，图中虽未画但建议加上

    def forward(self, query_input, key_value_input):
        """
        query_input: 作为 Q (例如图像特征) [Batch, Dim]
        key_value_input: 作为 K, V (例如临床特征) [Batch, Dim]
        """
        # 1. 调整维度：从 [Batch, Dim] -> [Batch, 1, Dim] 以适应 Attention 接口
        q = query_input.unsqueeze(1)
        k = v = key_value_input.unsqueeze(1)

        # 2. Cross Attention
        # attn_output 形状: [Batch, 1, Dim]
        attn_output, _ = self.attention(query=q, key=k, value=v)

        # 3. Residual Add (图中圆圈加号)
        # 这里的 query_input 对应图中最左/最右下来的那条绿线
        out = query_input + attn_output.squeeze(1)
        
        # (可选) Norm
        out = self.norm(out)
        
        return out

class SwinMlpFusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # -------------------------------------------------------
        # 1. 特征编码层
        # -------------------------------------------------------
        # 临床数据编码 (36 -> 1024)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(36, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # 图像特征已经是 1024 维，通常建议也加一层映射以对齐特征空间，
        # 但如果直接用也可以。这里加一个简单的映射层保持对称性（可选）。
        self.img_encoder = nn.Sequential(
             nn.Linear(1024, 1024),
             nn.LayerNorm(1024),
             nn.ReLU()
        )

        # -------------------------------------------------------
        # 2. 交叉注意力融合模块 (对应图中间部分)
        # -------------------------------------------------------
        feature_dim = 1024
        
        # 左路：F1(图像) 查询 F2(临床)
        self.cross_attn_img_query = CrossAttentionBlock(dim=feature_dim)
        
        # 右路：F2(临床) 查询 F1(图像)
        self.cross_attn_clin_query = CrossAttentionBlock(dim=feature_dim)

        # -------------------------------------------------------
        # 3. 融合后的 MLP (对应图中底部的 Concat -> MLP)
        # -------------------------------------------------------
        # Concat 后维度变两倍 (1024 + 1024 = 2048)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim), # 2048 -> 1024
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # -------------------------------------------------------
        # 4. 分类器
        # -------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, img_feat, clin_data):
        # --- 1. 特征准备 ---
        # img_feat: [Batch, 1024]
        # clin_data: [Batch, 36]
        
        # 编码临床特征 -> [Batch, 1024]
        F2 = self.clinical_encoder(clin_data)
        
        # 编码图像特征 (可选，如果不想用 img_encoder 可以直接 F1 = img_feat)
        F1 = self.img_encoder(img_feat) 

        # --- 2. 交叉注意力 ---
        # 左边：图像为主，临床为辅
        # F1 做 Query, F2 做 Key/Value
        left_out = self.cross_attn_img_query(query_input=F1, key_value_input=F2)

        # 右边：临床为主，图像为辅
        # F2 做 Query, F1 做 Key/Value
        right_out = self.cross_attn_clin_query(query_input=F2, key_value_input=F1)

        # --- 3. Concat & MLP ---
        # 拼接 [Batch, 2048]
        concat_feat = torch.cat([left_out, right_out], dim=1)
        
        # 融合 [Batch, 1024]
        F_fused = self.fusion_mlp(concat_feat)

        # --- 4. 分类 ---
        logits = self.classifier(F_fused)
        
        return logits

# ... (前面的 Dataset 和 Model 代码保持不变) ...

# ==========================================
# 3. 主程序：五折交叉验证 + ROC 绘图
# ==========================================
if __name__ == "__main__":

    # --- 配置 ---
    IMG_CSV = '1024d_features.csv'   
    CLIN_CSV = 'DCM prognosis.csv'
    LABEL_CSV = 'maces.csv'
    
    BATCH_SIZE = 16
    LR = 0.001
    EPOCHS = 50
    K_FOLDS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"运行设备: {DEVICE}")

    full_dataset = ThreeStreamDataset(IMG_CSV, CLIN_CSV, LABEL_CSV)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # ---用于存储 ROC 绘图所需的数据 ---
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # 绘图初始化
    plt.figure(figsize=(10, 8))

    print(f"\n⚡ 开始 {K_FOLDS} 折交叉验证 ⚡")

    for fold, (train_ids, val_ids) in enumerate(skf.split(np.zeros(len(full_dataset)), full_dataset.labels)):
        print(f"\n================ Fold {fold + 1} / {K_FOLDS} ================")
        
        train_sub = Subset(full_dataset, train_ids)
        val_sub = Subset(full_dataset, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
        
        # 初始化模型 (记得使用上面修改后的 SwinMlpFusionModel)
        model = SwinMlpFusionModel(num_classes=2).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        
        # --- 训练 ---
        # 为了演示简洁，这里只保留训练逻辑，省略了详细的每个 epoch 打印
        # 实际使用时你可以保留之前的 print
        for epoch in range(EPOCHS):
            model.train()
            for img, clin, labels in train_loader:
                img, clin, labels = img.to(DEVICE), clin.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(img, clin)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # --- 验证 (关键步骤：获取概率) ---
        print(f"Fold {fold+1} 训练结束，正在进行评估...")
        model.eval()
        
        y_true_fold = []
        y_scores_fold = [] # 存储正类的概率
        
        with torch.no_grad():
            for img, clin, labels in val_loader:
                img, clin, labels = img.to(DEVICE), clin.to(DEVICE), labels.to(DEVICE)
                outputs = model(img, clin)
                
                # 1. 使用 Softmax 将 Logits 转为 概率
                probs = torch.softmax(outputs, dim=1)
                
                # 2. 获取 "类别1" (正类/患病/死亡等) 的概率
                # probs[:, 1] 代表取第2列
                preds = probs[:, 1].cpu().numpy()
                targets = labels.cpu().numpy()
                
                y_true_fold.extend(targets)
                y_scores_fold.extend(preds)
        
        # --- 计算当前 Fold 的 ROC/AUC ---
        y_true_fold = np.array(y_true_fold)
        y_scores_fold = np.array(y_scores_fold)
        
        fpr, tpr, thresholds = roc_curve(y_true_fold, y_scores_fold)
        roc_auc = auc(fpr, tpr)
        
        # --- 记录数据用于画图 ---
        aucs.append(roc_auc)
        # 使用插值法，让所有 fold 的曲线对齐到同一个 X 轴 (mean_fpr)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0 # 修正原点
        
        # 画当前 Fold 的曲线
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label=f'ROC Fold {fold+1} (AUC = {roc_auc:.3f})')
        
        print(f"--> Fold {fold + 1} AUC: {roc_auc:.4f}")

    # ==========================================
    # 4. 绘制平均曲线 (Mean ROC)
    # ==========================================
    
    # 画对角线 (随机猜测线)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # 计算平均 TPR 和 AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0 # 修正终点
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # 画平均曲线
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})',
             lw=2, alpha=.8)

    # 画标准差阴影区域 (展示模型波动范围)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=f'$\pm$ 1 std. dev.')

    # 图表美化
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'ROC Curve - Cross Validation (Mean AUC={mean_auc:.3f})', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 保存或显示
    save_path = 'roc_curve_5folds.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ ROC 曲线已保存至: {save_path}")
    plt.show()