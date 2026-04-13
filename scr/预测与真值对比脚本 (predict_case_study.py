import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from Bio.PDB import PDBParser
import pandas as pd
import os
import numpy as np

# --- 1. 理化字典与特征配置 ---
AA_PROPS = {
    'ALA': [0, -1], 'ARG': [1, 1], 'ASN': [0, 1], 'ASP': [-1, 1], 'CYS': [0, 0],
    'GLN': [0, 1], 'GLU': [-1, 1], 'GLY': [0, 0], 'HIS': [1, 1], 'ILE': [0, -1],
    'LEU': [0, -1], 'LYS': [1, 1], 'MET': [0, -1], 'PHE': [0, -1], 'PRO': [0, -1],
    'SER': [0, 1], 'THR': [0, 1], 'TRP': [0, -1], 'TYR': [0, 0], 'VAL': [0, -1]
}
AA_LIST = list(AA_PROPS.keys())
INPUT_DIM = 25
MODEL_PATH = 'best_active_site_model.pth'
TEST_PDB = r'D:\shuju\pdb_files\pdb1a16.ent'  # 请确保路径正确


# --- 2. 模型定义 ---
class ProteinGCN(nn.Module):
    def __init__(self, input_dim):
        super(ProteinGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x).view(-1)


# --- 3. 预测核心函数 ---
def predict_protein(pdb_path, model, device, threshold=0.51):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    nodes = []
    info = []

    for model_struct in structure:
        for chain in model_struct:
            for residue in chain:
                res_name = residue.get_resname()
                if res_name not in AA_PROPS: continue

                try:
                    ca_coord = residue['CA'].get_coord()
                    one_hot = [1 if res_name == aa else 0 for aa in AA_LIST]
                    props = AA_PROPS[res_name]
                    feat = list(ca_coord) + one_hot + props
                    nodes.append(feat)
                    info.append({
                        'chain': chain.id,
                        'res_name': res_name,
                        'res_num': residue.get_id()[1]
                    })
                except KeyError:
                    continue

    coords = np.array([n[:3] for n in nodes])
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
    edge_index = torch.tensor(np.argwhere(dist_matrix < 8.0).T, dtype=torch.long)
    x = torch.tensor(nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).cpu().numpy()

    df = pd.DataFrame(info)
    df['probability'] = probs
    # 判定逻辑：即使没过阈值，我们也记录下来
    df['is_active_site'] = (df['probability'] > threshold).astype(int)
    return df


# --- 4. 执行预测并生成分析报表 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProteinGCN(input_dim=INPUT_DIM).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print(f"✅ 成功加载模型: {MODEL_PATH}")
    else:
        print(f"❌ 错误：找不到模型文件")
        exit()

    # 使用最优阈值
    CURRENT_THRESHOLD = 0.51
    result_df = predict_protein(TEST_PDB, model, device, threshold=CURRENT_THRESHOLD)

    # --- 关键修改点：保存全量数据供绘图脚本使用 ---
    # 这个 CSV 文件包含了 1A16 所有的残基概率，你的绘图脚本读取它就能画出分布图了
    result_df.to_csv("1A16_case_study_full_results.csv", index=False)
    print(f"📂 全量预测数据已保存至: 1A16_case_study_full_results.csv (用于生成预测图)")

    print(f"\n--- 1A16 预测概率排名前 10 的残基 (供论文表格使用) ---")
    top_10 = result_df.sort_values(by='probability', ascending=False).head(10)
    print(top_10[['res_name', 'res_num', 'probability', 'is_active_site']].to_string(index=False))