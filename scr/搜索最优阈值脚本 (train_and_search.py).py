import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, precision_score
import pandas as pd
import os
import numpy as np

# 1. 配置
DATA_PATH = r'D:\shuju\protein_graphs.pt'
MODEL_PATH = 'best_active_site_model.pth'
INPUT_DIM = 25  # 匹配 25 维理化特征数据


# 2. 模型定义 (保持与 train_final.py 一致)
class ProteinGCN(nn.Module):
    def __init__(self, input_dim):
        super(ProteinGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x).view(-1)


# 3. 加载数据与模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = torch.load(DATA_PATH, weights_only=False)
# 取后 20% 作为测试集
test_loader = DataLoader(dataset[int(len(dataset) * 0.8):], batch_size=16)

model = ProteinGCN(input_dim=INPUT_DIM).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"✅ 成功加载模型: {MODEL_PATH}")
else:
    print(f"❌ 找不到模型文件，请先运行训练脚本")
    exit()

# 4. 获取预测概率
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        # 使用 sigmoid 将输出转为 0-1 之间的概率
        probs = torch.sigmoid(model(data))
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

# 5. 精细化阈值搜索 (核心修改部分)
results = []
# 生成从 0.01 到 0.99 的 99 个等间距阈值点
thresholds = np.linspace(0.01, 0.99, 99)

print("正在计算 99 个阈值下的性能指标...")

for t in thresholds:
    # 按照当前阈值进行分类
    preds = (np.array(all_probs) >= t).astype(int)

    # 计算指标
    mcc = matthews_corrcoef(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)

    results.append({
        "Threshold": round(t, 2),
        "MCC": mcc,
        "Precision": prec,
        "Recall": rec
    })

# 6. 保存完整结果
df = pd.DataFrame(results)
# 使用 utf-8-sig 编码确保 Excel 打开不乱码
df.to_csv("threshold_search_results_full.csv", index=False, encoding='utf-8-sig')

# 7. 输出最优结果提醒
best_mcc_row = df.loc[df['MCC'].idxmax()]
print("\n" + "=" * 40)
print(f"✅ 搜索完成！完整数据已保存至: threshold_search_results_full.csv")
print(f"🌟 表现最好的阈值: {best_mcc_row['Threshold']}")
print(f"📈 对应最高 MCC: {best_mcc_row['MCC']:.4f}")
print("=" * 40)