import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 关键配置 ---
DATA_PATH = r'D:\shuju\protein_graphs.pt'
MODEL_SAVE_PATH = 'best_active_site_model.pth'
INPUT_DIM = 25  # 匹配 3(坐标) + 20(One-hot) + 2(理化特征)
BATCH_SIZE = 4
LR = 0.001
EPOCHS = 300
POS_WEIGHT = 75.0  # 针对不平衡数据的加权


# --- 2. 模型定义 ---
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


# --- 3. 训练与绘图主逻辑 ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到数据文件: {DATA_PATH}")
        return

    dataset = torch.load(DATA_PATH)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = ProteinGCN(INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))

    # --- 新增：用于记录训练历史的列表 ---
    history = []
    best_mcc = -1

    print(f"开始训练，设备: {device}，总轮次: {EPOCHS}")

    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # 验证阶段
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                logits = model(data)
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                all_true.extend(data.y.cpu().numpy())
                all_pred.extend(pred.cpu().numpy())

        # 计算本轮指标
        mcc = matthews_corrcoef(all_true, all_pred)
        precision = precision_score(all_true, all_pred, zero_division=0)
        recall = recall_score(all_true, all_pred, zero_division=0)
        avg_loss = total_loss / len(train_loader)


        history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'mcc': mcc,
            'precision': precision,
            'recall': recall
        })

        # 打印日志
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | MCC: {mcc:.4f} | Precision: {precision:.4f}")

        # 保存最优模型
        if mcc > best_mcc:
            best_mcc = mcc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # --- 训练结束后的收尾工作 ---

    # 1. 保存 CSV 数据
    history_df = pd.DataFrame(history)
    history_df.to_csv("training_history.csv", index=False)
    print("\n✅ 训练历史已保存至: training_history.csv")

    # 2. 自动生成并保存论文图表
    plt.figure(figsize=(8, 5), dpi=300)

    # 绘制 Loss (左轴)
    ax1 = plt.gca()
    ax1.plot(history_df['epoch'], history_df['loss'], color='tab:red', label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 绘制 MCC (右轴)
    ax2 = ax1.twinx()
    ax2.plot(history_df['epoch'], history_df['mcc'], color='tab:blue', label='Validation MCC')
    ax2.set_ylabel('Validation MCC', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Training Process: Loss vs Validation MCC')
    plt.grid(axis='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("Training_Convergence_Plot.png")
    print("📈 训练收敛图已生成: Training_Convergence_Plot.png")
    plt.show()


if __name__ == "__main__":
    main()