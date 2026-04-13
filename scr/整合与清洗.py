import os
import torch
import pandas as pd
import re
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from tqdm import tqdm

# --- 1. 路径配置 ---
PDB_DIR = r'D:\shuju\pdb_files'
CSV_PATH = r'D:\shuju\literature_pdb_residues.csv'
OUTPUT_PATH = r'D:\shuju\protein_graphs.pt'

# 氨基酸理化属性：[电荷, 亲疏水性] -> 提升 MCC 的关键
AA_PROPS = {
    'ALA': [0, -1], 'ARG': [1, 1], 'ASN': [0, 1], 'ASP': [-1, 1], 'CYS': [0, 0],
    'GLN': [0, 1], 'GLU': [-1, 1], 'GLY': [0, 0], 'HIS': [1, 1], 'ILE': [0, -1],
    'LEU': [0, -1], 'LYS': [1, 1], 'MET': [0, -1], 'PHE': [0, -1], 'PRO': [0, -1],
    'SER': [0, 1], 'THR': [0, 1], 'TRP': [0, -1], 'TYR': [0, 0], 'VAL': [0, -1]
}
AA_LIST = list(AA_PROPS.keys())

# --- 2. 建立索引 ---
print("正在读取 M-CSA 数据并建立索引...")
df = pd.read_csv(CSV_PATH, encoding_errors='ignore')
df.columns = [c.lower().strip().replace('\ufeff', '') for c in df.columns]

active_sites_set = set()
csv_pdb_ids = set()
for _, row in df.iterrows():
    p_id = str(row['pdb id']).lower().strip()
    chn = str(row['chain id']).strip()
    r_num = int(float(row['residue number']))
    active_sites_set.add((p_id, chn, r_num))
    csv_pdb_ids.add(p_id)


# --- 3. 处理函数 ---
def process_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    fname = os.path.basename(pdb_path).lower()


    if 'pdb' in fname:
        match = re.search(r'pdb([a-z0-9]{4})', fname)
    else:
        match = re.search(r'([a-z0-9]{4})', fname)

    if not match: return None
    pdb_id = match.group(1)

    if pdb_id not in csv_pdb_ids: return None

    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except:
        return None

    nodes, coords, labels = [], [], []
    found_active = 0

    for model in structure:
        for chain in model:
            chain_id = chain.id.strip()
            for residue in chain:
                res_name = residue.get_resname().upper()
                if res_name in AA_PROPS and 'CA' in residue:
                    pos = residue['CA'].get_coord()
                    one_hot = [1 if res_name == aa else 0 for aa in AA_LIST]
                    physicochemical = AA_PROPS[res_name]  # 理化特征

                    # 组合 25 维特征
                    nodes.append(list(pos) + one_hot + physicochemical)
                    coords.append(pos)

                    res_num = residue.id[1]
                    if (pdb_id, chain_id, res_num) in active_sites_set:
                        labels.append(1)
                        found_active += 1
                    else:
                        labels.append(0)

    if nodes and found_active > 0:
        x = torch.tensor(nodes, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)
        pos_t = torch.tensor(coords, dtype=torch.float)

        dist = torch.cdist(pos_t, pos_t)
        edge_index = (dist < 6.0).nonzero(as_tuple=False).t()
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        return Data(x=x, edge_index=edge_index, y=y)
    return None


if __name__ == "__main__":
    all_graphs = []
    pdb_files = [f for f in os.listdir(PDB_DIR) if f.endswith(('.pdb', '.ent'))]
    print(f"开始处理 {len(pdb_files)} 个文件...")

    for f in tqdm(pdb_files):
        graph = process_pdb(os.path.join(PDB_DIR, f))
        if graph: all_graphs.append(graph)

    if all_graphs:
        torch.save(all_graphs, OUTPUT_PATH)
        print(f"\n✅ 成功转换 {len(all_graphs)} 个有效样本！特征维度: 25")
    else:
        print("\n❌ 匹配依然失败。请检查：")
        print(
            f"文件名提取出的第一个 ID 是否为: {re.search(r'pdb([a-z0-9]{4})', pdb_files[0].lower()).group(1) if 'pdb' in pdb_files[0].lower() else '提取失败'}")