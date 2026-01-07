# ==============================================================================
# Script 4: Academic Full Benchmark (Complete)
# ==============================================================================

import os, copy, math, time, random, warnings
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TAGConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from collections import Counter
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import rcParams
from torch.nn.utils.rnn import pad_sequence

# ---------------- Style Settings ----------------
plt.style.use('seaborn-v0_8-whitegrid')
try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
except:
    pass
plt.rcParams['figure.dpi'] = 300
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

def set_origin_style():
    """设置符合顶刊 Origin 风格的绘图参数"""
    plt.style.use('default')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 10
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.major.size'] = 6
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.size'] = 6
    rcParams['ytick.major.width'] = 1.2
    rcParams['xtick.top'] = True
    rcParams['ytick.right'] = True
    rcParams['axes.linewidth'] = 1.5
    rcParams['axes.grid'] = False
    rcParams['figure.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.05

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
set_seed(42)

# ============================================================
# 1. Data Preparation
# ============================================================

def build_line_graph(link_gdf):
    print("  - Building Line Graph...")
    df = link_gdf[['link_id', 'from_node', 'to_node']].copy()
    sources = list(df['link_id'])
    targets = list(df['link_id'])
    connections = pd.merge(df, df, left_on='to_node', right_on='from_node', suffixes=('_src', '_dst'))
    sources.extend(connections['link_id_src'].tolist())
    targets.extend(connections['link_id_dst'].tolist())
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return edge_index

def get_log_clamp(train_dl, device):
    logs = []
    for i, batch in enumerate(train_dl):
        logs.append(torch.log1p(batch['time_raw']).to(device))
        if i > 100: break
    all_logs = torch.cat(logs)
    clamp_max = torch.quantile(all_logs, 0.999).item()
    print(f"[Eval] Using dynamic log clamp max={clamp_max:.3f}")
    return clamp_max

def prepare_real_data(shapefile_path, matched_csv_path, weather_path):
    print(f"[Data] Loading Network & Trajectories...")
    link_gdf = gpd.read_file(shapefile_path, encoding='utf-8')
    link_gdf['link_id'] = np.arange(1, len(link_gdf) + 1)
    num_edges = link_gdf['link_id'].max() + 1
    link_gdf = link_gdf.sort_values('link_id')
    
    scaler = MinMaxScaler()
    edge_features_np = np.hstack([
        scaler.fit_transform(link_gdf[['length', 'speed_kph', 'lanes']].fillna(0).values),
        link_gdf['dir'].values.reshape(-1,1) if 'dir' in link_gdf.columns else np.zeros((len(link_gdf), 1))
    ])
    edge_features = torch.zeros((num_edges, edge_features_np.shape[1]), dtype=torch.float)
    edge_features.index_copy_(0,
        torch.tensor(link_gdf['link_id'].values, dtype=torch.long),
        torch.tensor(edge_features_np, dtype=torch.float))
    graph_data = {'edge_index': build_line_graph(link_gdf), 'edge_features': edge_features}
    
    full_df = pd.read_csv(matched_csv_path)
    full_df['time'] = pd.to_datetime(full_df['time'])
    full_df.sort_values(['trajectory_id', 'time'], inplace=True)
    link_dict = link_gdf.set_index('link_id')[['length', 'speed_kph']].to_dict('index')
    
    trajs, dyn_feats, times = [], [], []
    grouped = full_df.groupby('trajectory_id')
    for tid, df_traj in tqdm(grouped, total=len(grouped), desc="Processing", leave=False):
        link_seq = df_traj['link_id'].astype(int).tolist()
        t_raw = (df_traj['time'].iloc[-1] - df_traj['time'].iloc[0]).total_seconds()
        
        if t_raw < 60 or len(link_seq) < 5: continue
        fft_sum = sum([link_dict[lid]['length'] / (max(link_dict[lid]['speed_kph'], 10) / 3.6)
                       for lid in link_seq if lid in link_dict])
        if fft_sum <= 0: continue
        
        start_time = df_traj['time'].iloc[0]
        hour_norm = start_time.hour / 23.0
        day_norm = start_time.dayofweek / 6.0
        fft_log = np.log1p(fft_sum)
        h = start_time.hour
        is_peak = 1.0 if (7 <= h <= 9) or (17 <= h <= 19) else 0.0
        
        trajs.append(link_seq)
        times.append(t_raw)
        dyn_feats.append([hour_norm, day_norm, fft_log, is_peak])
        
    return trajs, dyn_feats, times, graph_data, num_edges

def stratified_subset(trajs, dyn_feats, times, subset_size=20000, seed=42):
    if len(trajs) <= subset_size: return trajs, dyn_feats, times
    np.random.seed(seed)
    indices = np.random.choice(len(trajs), subset_size, replace=False)
    print(f"[Data] Subsampling to {len(indices)} trajectories.")
    return [trajs[i] for i in indices], [dyn_feats[i] for i in indices], [times[i] for i in indices]

def analyze_traj_lengths(dataset, save_path=None):
    traj_lengths = dataset.lengths
    print("Length Stats:", np.min(traj_lengths), np.max(traj_lengths), np.mean(traj_lengths))

class FastTravelTimeDataset(Dataset):
    def __init__(self, trajectories, dynamic_features, travel_times, is_train=False):
        self.num_samples = len(travel_times)
        self.is_train = is_train
        self.traj_data = trajectories 
        self.dyn_data = torch.tensor(np.array(dynamic_features), dtype=torch.float)
        self.time_data = torch.tensor(np.array(travel_times), dtype=torch.float)
        self.lengths = np.array([len(t) for t in trajectories])
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        return {'traj': torch.tensor(self.traj_data[idx], dtype=torch.long),
                'dyn_feat': self.dyn_data[idx], 'len': self.lengths[idx],
                'time_log': torch.log1p(self.time_data[idx]), 'time_raw': self.time_data[idx]}
    
def dynamic_collate_fn(batch):
    trajs = pad_sequence([item['traj'] for item in batch], batch_first=True, padding_value=0)
    dyn_feats = torch.stack([item['dyn_feat'] for item in batch])
    lengths = torch.tensor([item['len'] for item in batch], dtype=torch.long)
    time_logs = torch.stack([item['time_log'] for item in batch])
    time_raws = torch.stack([item['time_raw'] for item in batch])
    return {'traj': trajs, 'dyn_feat': dyn_feats, 'len': lengths, 'time_log': time_logs, 'time_raw': time_raws}

class BySequenceLengthSampler(Sampler):
    def __init__(self, data_source, batch_size, bucket_boundaries):
        ind_n_len = []
        for i, p in enumerate(data_source): ind_n_len.append((i, p['len']))
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
    def __iter__(self):
        data_buckets = dict()
        for p in self.ind_n_len:
            pid = p[0]; plen = p[1]; found = False
            for boundary in self.bucket_boundaries:
                if plen <= boundary:
                    if boundary not in data_buckets: data_buckets[boundary] = []
                    data_buckets[boundary].append(pid); found = True; break
            if not found:
                if self.bucket_boundaries[-1] not in data_buckets: data_buckets[self.bucket_boundaries[-1]] = []
                data_buckets[self.bucket_boundaries[-1]].append(pid)
        for k in data_buckets.keys():
            data_buckets[k] = np.array(data_buckets[k]); np.random.shuffle(data_buckets[k])
        iter_list = []
        for k in data_buckets.keys():
            for i in range(0, len(data_buckets[k]), self.batch_size):
                iter_list.append(data_buckets[k][i:i + self.batch_size])
        np.random.shuffle(iter_list) 
        for batch in iter_list: yield batch
    def __len__(self): return (len(self.ind_n_len) + self.batch_size - 1) // self.batch_size

# ============================================================
# 3. Model Definitions
# ============================================================
# ---------------- Shared Modules ----------------
class GCNModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU() 
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.act(x)
        return x

class UniversalAttention(nn.Module):
    def __init__(self, query_dim, key_dim, heads=4, temperature=0.5, attn_dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=key_dim, num_heads=heads, dropout=attn_dropout, batch_first=True)
        self.proj_q = nn.Linear(query_dim, key_dim)
        self.ln = nn.LayerNorm(key_dim)
    def forward(self, query, keys, mask):
        q = self.proj_q(query)
        if q.dim() == 2: q = q.unsqueeze(1)
        key_mask = (mask == 0)
        out, w = self.attn(q, keys, keys, key_padding_mask=key_mask)
        return self.ln(out.squeeze(1)), w

class GGNNCell(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.Wz = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.Wr = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.Wh = nn.Linear(in_dim + hidden_dim, hidden_dim)
    def forward(self, x, h_agg):
        z = torch.sigmoid(self.Wz(torch.cat([x, h_agg], dim=1)))
        r = torch.sigmoid(self.Wr(torch.cat([x, h_agg], dim=1)))
        h_tilde = torch.tanh(self.Wh(torch.cat([x, r * h_agg], dim=1)))
        return (1 - z) * h_agg + z * h_tilde

# ---------------- 1. Sequence Baseline (LSTM/GRU) ----------------
class SequenceBaseline(nn.Module):
    def __init__(self, static_feat_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, dropout_rate=0.3, rnn_type='GRU', **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(static_feat_dim, hidden_dim) 
        self.rnn = getattr(nn, rnn_type)(hidden_dim + id_embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(nn.Linear(hidden_dim*2+dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, batch, graph_data):
        traj = batch['traj']
        feat = graph_data[1][traj]
        feat_proj = F.relu(self.input_proj(feat))
        emb = self.embedding(traj)
        x = torch.cat([emb, feat_proj], dim=-1)
        
        packed = nn.utils.rnn.pack_padded_sequence(x, batch['len'].cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple): hidden = hidden[0]
        state = self.dropout(self.ln(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return self.fc(torch.cat([state, batch['dyn_feat']], dim=1)).squeeze(-1)

# ---------------- 2. Attentive RNN ----------------
class AttentiveRNN(nn.Module):
    def __init__(self, static_feat_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, dropout_rate=0.4, rnn_type='GRU', **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(static_feat_dim, hidden_dim)
        
        self.rnn = getattr(nn, rnn_type)(hidden_dim + id_embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln_rnn = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn = UniversalAttention(hidden_dim * 2, hidden_dim * 2, heads=4, attn_dropout=dropout_rate)
        self.gate_fc = nn.Sequential(nn.Linear(hidden_dim*4+dyn_feat_dim, hidden_dim*2), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2 + dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, batch, graph_data, return_attn=False):
        traj = batch['traj']
        feat = graph_data[1][traj]
        feat_proj = F.relu(self.input_proj(feat))
        
        x = torch.cat([self.embedding(traj), feat_proj], dim=-1)
        
        packed = nn.utils.rnn.pack_padded_sequence(x, batch['len'].clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        enc, hidden = self.rnn(packed)
        enc_seq, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True, total_length=x.size(1))
        
        if isinstance(hidden, tuple): hidden = hidden[0]
        rnn_feat = self.dropout(self.ln_rnn(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        
        attn_ctx, attn_w = self.attn(rnn_feat, enc_seq, (traj!=0).float())
        z = self.gate_fc(torch.cat([rnn_feat, attn_ctx, batch['dyn_feat']], dim=1))
        fused = rnn_feat + z * attn_ctx
        
        out = self.fc(torch.cat([fused, batch['dyn_feat']], dim=1)).squeeze(-1)
        return (out, attn_w) if return_attn else out

# ---------------- 3. GAT + LSTM ----------------
class GAT_LSTM(nn.Module):
    def __init__(self, static_feat_dim, gcn_out_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, num_heads=4, dropout_rate=0.3, **kwargs):
        super().__init__()
        self.id_embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.gat_in_proj = nn.Linear(static_feat_dim, gcn_out_dim)
        self.gat = GATConv(gcn_out_dim, gcn_out_dim, heads=num_heads, dropout=dropout_rate)
        
        self.lstm_in_proj = nn.Linear(id_embed_dim + gcn_out_dim * num_heads + static_feat_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2 + dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, batch, graph_data, return_attn=False):
        traj = batch['traj']
        gcn_in = self.gat_in_proj(graph_data[1])
        gat_all = self.gat(gcn_in, graph_data[0])
        
        combined = torch.cat([self.id_embedding(traj), gat_all[traj], graph_data[1][traj]], dim=2)
        combined = F.relu(self.lstm_in_proj(combined))
        
        packed = nn.utils.rnn.pack_padded_sequence(combined, batch['len'].cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.lstm(packed)
        if isinstance(hidden, tuple): hidden = hidden[0]
        state = self.dropout(self.ln(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return self.fc(torch.cat([state, batch['dyn_feat']], dim=1)).squeeze(-1)

# ---------------- 4. GCN + GRU ----------------
class GCN_GRU(nn.Module):
    def __init__(self, static_feat_dim, gcn_out_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, dropout_rate=0.3, **kwargs):
        super().__init__()
        self.id_embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.gcn = GCNModule(static_feat_dim, gcn_out_dim)
        
        self.gru_proj = nn.Linear(id_embed_dim + gcn_out_dim + static_feat_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(nn.Linear(hidden_dim*2+dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, batch, graph_data, return_attn=False):
        traj = batch['traj']
        gcn_all = self.gcn(graph_data[1], graph_data[0])
        
        combined = torch.cat([self.id_embedding(traj), gcn_all[traj], graph_data[1][traj]], dim=2)
        combined = F.relu(self.gru_proj(combined))
        
        packed = nn.utils.rnn.pack_padded_sequence(combined, batch['len'].cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        if isinstance(hidden, tuple): hidden = hidden[0]
        state = self.dropout(self.ln(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return self.fc(torch.cat([state, batch['dyn_feat']], dim=1)).squeeze(-1)

# ---------------- 5. TAGCN + GRU ----------------
class TAGCN_GRU(nn.Module):
    def __init__(self, static_feat_dim, gcn_out_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, dropout_rate=0.3, **kwargs):
        super().__init__()
        self.id_embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.tagcn = TAGConv(static_feat_dim, gcn_out_dim, K=3)
        self.bn_gcn = nn.BatchNorm1d(gcn_out_dim)
        
        self.gru_proj = nn.Linear(id_embed_dim + gcn_out_dim + static_feat_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(nn.Linear(hidden_dim*2+dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, batch, graph_data, return_attn=False):
        traj = batch['traj']
        gcn_out = F.elu(self.bn_gcn(self.tagcn(graph_data[1], graph_data[0])))
        
        combined = torch.cat([self.id_embedding(traj), gcn_out[traj], graph_data[1][traj]], dim=2)
        combined = F.relu(self.gru_proj(combined))
        
        packed = nn.utils.rnn.pack_padded_sequence(combined, batch['len'].cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        if isinstance(hidden, tuple): hidden = hidden[0]
        state = self.dropout(self.ln(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return self.fc(torch.cat([state, batch['dyn_feat']], dim=1)).squeeze(-1)

# ---------------- 6. DeepTTE + LSTM ----------------
class DeepTTE_LSTM(nn.Module):
    def __init__(self, static_feat_dim, gcn_out_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, dropout_rate=0.3, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.geo_conv = nn.Conv1d(static_feat_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn_conv = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(id_embed_dim + hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(nn.Linear(hidden_dim*2+dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, batch, graph_data):
        traj = batch['traj']
        feat_seq = graph_data[1][traj].permute(0, 2, 1) 
        geo_feat = F.elu(self.bn_conv(self.geo_conv(feat_seq))).permute(0, 2, 1)
        x = torch.cat([self.embedding(traj), geo_feat], dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(x, batch['len'].cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        state = self.dropout(self.ln(torch.cat((h_n[-2], h_n[-1]), dim=1)))
        return self.fc(torch.cat([state, batch['dyn_feat']], dim=1)).squeeze(-1)

# ---------------- 7. TCN ----------------
class CausalTemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding='same', dilation=dilation)
        self.bn1, self.act1, self.drop1 = nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding='same', dilation=dilation)
        self.bn2, self.act2, self.drop2 = nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        y = self.drop2(self.act2(self.bn2(self.conv2(self.drop1(self.act1(self.bn1(self.conv1(x))))))))
        return F.relu(y + res)

class TCN_Model(nn.Module):
    def __init__(self, static_feat_dim, gcn_out_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, dropout_rate=0.3, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(static_feat_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        in_ch = hidden_dim + id_embed_dim
        for d in [1, 2, 4]:
            self.blocks.append(CausalTemporalBlock(in_ch, hidden_dim, 3, d, dropout_rate))
            in_ch = hidden_dim
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(nn.Linear(hidden_dim+dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, batch, graph_data=None):
        traj = batch['traj']
        feat = graph_data[1][traj]
        feat_p = F.relu(self.input_proj(feat))
        
        x = torch.cat([self.embedding(traj), feat_p], dim=-1).permute(0,2,1)
        for block in self.blocks: x = block(x)
        
        mask = (traj != 0).unsqueeze(1).float()
        pooled = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        return self.fc(torch.cat([self.ln(pooled), batch['dyn_feat']], dim=1)).squeeze(-1)

# ---------------- 8. ST-WaveNet ----------------
class GatedDilatedConv(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        self.f = nn.Conv1d(channels, channels, kernel_size, padding=dilation*(kernel_size-1), dilation=dilation)
        self.g = nn.Conv1d(channels, channels, kernel_size, padding=dilation*(kernel_size-1), dilation=dilation)
        self.bn_f, self.bn_g = nn.BatchNorm1d(channels), nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.pad = dilation*(kernel_size-1)
    def forward(self, x):
        y = self.drop(torch.tanh(self.bn_f(self.f(x))) * torch.sigmoid(self.bn_g(self.g(x))))
        return y[:, :, :-self.pad] if self.pad > 0 else y

class ST_WaveNet(nn.Module):
    def __init__(self, static_feat_dim, gcn_out_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, dropout_rate=0.3, **kwargs):
        super().__init__()
        self.gcn = GCNModule(static_feat_dim, gcn_out_dim)
        self.id_embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        channels = hidden_dim * 2
        self.start = nn.Sequential(nn.Conv1d(id_embed_dim + gcn_out_dim + static_feat_dim, channels, 1), nn.BatchNorm1d(channels), nn.ReLU())
        self.blocks = nn.ModuleList([GatedDilatedConv(channels, 3, d, dropout_rate) for d in [1, 2, 4, 8]])
        self.res_proj = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in range(4)])
        self.skip_proj = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in range(4)])
        self.ln = nn.LayerNorm(channels)
        self.fc = nn.Sequential(nn.Linear(channels+dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, batch, graph_data):
        traj = batch['traj']
        # [Fix] GCN on static features
        gcn_all = self.gcn(graph_data[1], graph_data[0])
        
        x = torch.cat([self.id_embedding(traj), gcn_all[traj], graph_data[1][traj]], dim=2).permute(0,2,1)
        x = self.start(x)
        skip_sum = 0
        for blk, r, s in zip(self.blocks, self.res_proj, self.skip_proj):
            h = blk(x); skip_sum = skip_sum + s(h); x = F.relu(r(h) + x)
        mask = (traj != 0).unsqueeze(1).float()
        pooled = self.ln((skip_sum * mask).sum(dim=2))
        return self.fc(torch.cat([pooled, batch['dyn_feat']], dim=1)).squeeze(-1)

# ---------------- 9. GraphTTE ----------------
class GraphTTE(nn.Module):
    def __init__(self, static_feat_dim, gcn_out_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, dropout_rate=0.3, **kwargs):
        super().__init__()
        self.gcn = GCNModule(static_feat_dim, gcn_out_dim) # [Fix] Static only
        self.id_embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(id_embed_dim + gcn_out_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln_lstm = nn.LayerNorm(hidden_dim * 2); self.dropout = nn.Dropout(dropout_rate)
        self.path_attn = UniversalAttention(hidden_dim*2, hidden_dim*2, heads=4, attn_dropout=dropout_rate)
        self.gate_fc = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim*2), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(hidden_dim*2+dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))
    def forward(self, batch, graph_data, return_attn=False):
        traj = batch['traj']
        gcn_all = self.gcn(graph_data[1], graph_data[0])
        
        x = torch.cat([self.id_embedding(traj), gcn_all[traj]], dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(x, batch['len'].cpu(), batch_first=True, enforce_sorted=False)
        enc, _ = self.lstm(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True, total_length=x.size(1))
        mask = (traj != 0).unsqueeze(-1).float()
        mean_enc = (self.ln_lstm(enc) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        attn_ctx, attn_w = self.path_attn(mean_enc, enc, mask.squeeze(-1))
        z = self.gate_fc(attn_ctx)
        out = self.fc(torch.cat([mean_enc + z * attn_ctx, batch['dyn_feat']], dim=1)).squeeze(-1)
        return (out, attn_w) if return_attn else out

# ---------------- 10. GSTA ----------------
class GSTA(nn.Module):
    def __init__(self, static_feat_dim, gcn_out_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, dropout_rate=0.3, rnn_type='GRU', **kwargs):
        super().__init__()
        self.gcn = GCNModule(static_feat_dim, gcn_out_dim)
        self.id_embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.rnn = getattr(nn, rnn_type)(id_embed_dim + gcn_out_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln_rnn = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn = UniversalAttention(hidden_dim*2+dyn_feat_dim, hidden_dim*2, heads=4, attn_dropout=dropout_rate)
        self.gate_fc = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim*2), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(hidden_dim*2+dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, batch, graph_data, return_attn=False):
        gcn_out = self.gcn(graph_data[1], graph_data[0])
        x_seq = torch.cat([self.id_embedding(batch['traj']), gcn_out[batch['traj']]], dim=2)
        
        packed = nn.utils.rnn.pack_padded_sequence(x_seq, batch['len'].clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        enc, hidden = self.rnn(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True, total_length=x_seq.size(1))
        
        if isinstance(hidden, tuple): hidden = hidden[0]
        rnn_last = torch.cat((hidden[-2], hidden[-1]), dim=1)
        rnn_feat = self.dropout(self.ln_rnn(rnn_last))
        
        attn_ctx, attn_w = self.attn(torch.cat([rnn_feat, batch['dyn_feat']], dim=1), enc, (batch['traj']!=0).float())
        z = self.gate_fc(attn_ctx)
        
        out = self.fc(torch.cat([rnn_feat + z * attn_ctx, batch['dyn_feat']], dim=1)).squeeze(-1)
        return (out, attn_w) if return_attn else out

# ---------------- 11. Adaptive GGNN (Baseline) ----------------
class AdaptiveGGNN_TTE(nn.Module):
    def __init__(self, static_feat_dim, gcn_out_dim, id_embed_dim, hidden_dim, dyn_feat_dim, num_edges, dropout_rate=0.3, steps=3, **kwargs):
        super().__init__()
        self.id_embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.edge_proj = nn.Linear(static_feat_dim, hidden_dim)
        self.cell = GGNNCell(static_feat_dim, hidden_dim)
        self.steps = steps

        self.gru = nn.GRU(hidden_dim + id_embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_dim * 2); self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(nn.Linear(hidden_dim*2+dyn_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))
    
    def forward(self, batch, graph_data):
        x0 = graph_data[1]
        h = torch.tanh(self.edge_proj(x0))
        for _ in range(self.steps):
            h_agg = torch.zeros_like(h)
            h_agg.index_add_(0, graph_data[0][1], h[graph_data[0][0]])
            h = self.cell(x0, h_agg)
        
        h_seq = h[batch['traj']]
        id_seq = self.id_embedding(batch['traj'])
        rnn_in = torch.cat([h_seq, id_seq], dim=-1)
        
        packed = nn.utils.rnn.pack_padded_sequence(rnn_in, batch['len'].cpu(), batch_first=True, enforce_sorted=False)
        _, h_last = self.gru(packed)
        state = self.dropout(self.ln(torch.cat((h_last[-2], h_last[-1]), dim=1)))
        return self.fc(torch.cat([state, batch['dyn_feat']], dim=1)).squeeze(-1)

# ---------------- 12. DAGRN ----------------
class DAGRN(nn.Module):
    def __init__(self, static_feat_dim, gcn_out_dim, id_embed_dim, hidden_dim,
                 dyn_feat_dim, num_edges, dropout_rate=0.5, rnn_type='GRU', 
                 gcn_iter=3, attention_heads=4, 
                 ablation_mode='full', 
                 **kwargs):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.dropout = nn.Dropout(dropout_rate)
        self.steps = gcn_iter
        self.id_embedding = nn.Embedding(num_edges, id_embed_dim, padding_idx=0)
        self.feature_proj = nn.Linear(static_feat_dim, hidden_dim)
        if ablation_mode != 'no_gcn':
            self.cell = GGNNCell(static_feat_dim, hidden_dim)
        self.film_gen = nn.Sequential(
            nn.Linear(dyn_feat_dim, hidden_dim), 
            nn.Dropout(dropout_rate), 
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh()
        )
        self.gru = nn.GRU(hidden_dim + id_embed_dim + dyn_feat_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln_rnn = nn.LayerNorm(hidden_dim * 2)
        if ablation_mode != 'no_attn':
            self.attn_score = nn.Sequential(
                nn.Linear(hidden_dim * 2 + dyn_feat_dim, hidden_dim), 
                nn.Tanh(), 
                nn.Linear(hidden_dim, 1)
            )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear((hidden_dim * 2) * 2 + dyn_feat_dim, hidden_dim * 2), 
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + dyn_feat_dim, hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout_rate), 
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.film_gen[2].weight, 0)
        nn.init.constant_(self.film_gen[2].bias, 0)
        
        for m in self.fusion_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.out_features == 1: nn.init.constant_(m.bias, 0)

    def forward(self, batch, graph_data, return_attn=False):
        # A. Graph
        x0 = graph_data[1]
        h = torch.tanh(self.feature_proj(x0))
        if self.ablation_mode != 'no_gcn':
            for _ in range(self.steps):
                h_agg = torch.zeros_like(h)
                h_agg.index_add_(0, graph_data[0][1], h[graph_data[0][0]])
                h = self.cell(x0, h_agg)
        x_seq = h[batch['traj']]

        if self.ablation_mode != 'no_film':
            film = self.film_gen(batch['dyn_feat'])
            g, b = torch.chunk(film, 2, dim=-1)
            x_seq = x_seq * (1 + g.unsqueeze(1)) + b.unsqueeze(1)

        B, T, _ = x_seq.size()
        rnn_in = torch.cat([x_seq, self.id_embedding(batch['traj']), batch['dyn_feat'].unsqueeze(1).expand(B,T,-1)], dim=-1)
        
        packed = nn.utils.rnn.pack_padded_sequence(rnn_in, batch['len'].cpu(), batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.gru(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        
        if isinstance(hidden, tuple): h_last = torch.cat((hidden[0][-2], hidden[0][-1]), dim=1)
        else: h_last = torch.cat((hidden[-2], hidden[-1]), dim=1)
        rnn_last = self.ln_rnn(h_last)

        attn_w = None
        if self.ablation_mode == 'no_attn':
            final_vec = rnn_last
        else:
            dyn_expanded = batch['dyn_feat'].unsqueeze(1).expand(B, T, -1)
            score_in = torch.cat([rnn_out, dyn_expanded], dim=-1)
            scores = self.attn_score(score_in).squeeze(-1)
            
            mask = (batch['traj'] != 0)
            scores = scores.masked_fill(~mask, -1e9)
            alpha = F.softmax(scores, dim=1).unsqueeze(-1)
            context = (rnn_out * alpha).sum(dim=1)
            attn_w = alpha.squeeze(-1)
            if self.ablation_mode == 'no_gate':
                final_vec = (context + rnn_last) / 2.0 
            else:
                gate_in = torch.cat([context, rnn_last, batch['dyn_feat']], dim=1)
                z = self.fusion_gate(gate_in)
                final_vec = z * context + (1 - z) * rnn_last

        out = self.fc(torch.cat([final_vec, batch['dyn_feat']], dim=1)).squeeze(-1)
        if return_attn: return out, attn_w
        return out
    
# ============================================================
# Engines
# ============================================================

def eval_metrics_log(y_true_raw, y_pred_log):
    y_pred_raw = np.expm1(y_pred_log)
    y_pred_raw = np.maximum(y_pred_raw, 0)
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mape = np.mean(np.abs((y_true_raw - y_pred_raw) / (y_true_raw + 1e-1))) * 100
    r2 = r2_score(y_true_raw, y_pred_raw)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

def calc_accuracy(pred_log, target_log, threshold=0.20):
    pred = torch.expm1(pred_log)
    target = torch.expm1(target_log)
    ape = torch.abs(target - pred) / (target + 1e-1)
    return (ape < threshold).float().mean().item() * 100

def train_engine(model, train_dl, test_dl, config):
    device = config['device']
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=5e-4)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False
    )
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    print(f"[{config.get('model_name')}] Training (Standard Strategy)...")
    
    for epoch in range(config['epochs']):
        model.train()
        t_loss, cnt = 0.0, 0
        pbar = tqdm(train_dl, desc=f"Ep {epoch+1}", leave=False)

        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            try:
                if config.get('needs_graph'):
                    if 'return_attn' in model.forward.__code__.co_varnames:
                        out = model(batch, config['graph_data'], return_attn=True)
                    else:
                        out = model(batch, config['graph_data'])
                else:
                    out = model(batch)
                pred = out[0] if isinstance(out, tuple) else out
            except TypeError:
                pred = model(batch, config['graph_data']) if config.get('needs_graph') else model(batch)
            loss = criterion(pred, batch['time_log'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            t_loss += loss.item()
            cnt += 1
            pbar.set_postfix({'L': f"{loss.item():.4f}"})
            
        model.eval()
        v_loss, v_acc = 0.0, 0.0
        with torch.no_grad():
            for batch in test_dl:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                
                if config.get('needs_graph'):
                     out = model(batch, config['graph_data'])
                else:
                     out = model(batch)
                pred = out[0] if isinstance(out, tuple) else out
                
                v_loss += criterion(pred, batch['time_log']).item()
                v_acc += calc_accuracy(pred, batch['time_log'])

        avg_t = t_loss / cnt
        avg_v = v_loss / len(test_dl)
        scheduler.step(avg_v)
        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)
        
        print(f"  Ep {epoch+1:02d} | Train: {avg_t:.4f} | Val: {avg_v:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        if avg_v < best_loss:
            best_loss = avg_v
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print("  [Early Stop]")
            break

    model.load_state_dict(best_model_wts)
    return history, best_loss

def run_evaluation(model, loader, config):
    model.eval()
    device = config['device']
    preds, trues, dyns, lens = [], [], [], []
    
    clamp_max = config.get('log_clamp_max', 10.0)
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pred = model(batch, config['graph_data']) if config.get('needs_graph') else model(batch)
            if isinstance(pred, tuple): pred = pred[0]
            
            pred = torch.clamp(pred, max=clamp_max)
            preds.append(pred.cpu().numpy())
            trues.append(batch['time_raw'].cpu().numpy())
            dyns.append(batch['dyn_feat'].cpu().numpy())
            lens.append(batch['len'].cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    dyns = np.concatenate(dyns)
    lens = np.concatenate(lens)
    metrics = eval_metrics_log(trues, preds)
    pred_raw = np.expm1(preds)
    mask_long = lens >= 20
    if mask_long.sum() > 0:
        metrics['Long_RMSE'] = np.sqrt(mean_squared_error(trues[mask_long], pred_raw[mask_long]))
    else:
        metrics['Long_RMSE'] = 0.0
    mask_peak = dyns[:, 3] > 0.5
    if mask_peak.sum() > 0:
        metrics['Peak_RMSE'] = np.sqrt(mean_squared_error(trues[mask_peak], pred_raw[mask_peak]))
    else:
        metrics['Peak_RMSE'] = 0.0
    fft_vals = np.expm1(dyns[:, 2]) 
    ci = trues / (fft_vals + 1.0)
    thresh = np.percentile(ci, 90)
    mask_cong = ci >= thresh
    
    if mask_cong.sum() > 0:
        metrics['Congested_RMSE'] = np.sqrt(mean_squared_error(trues[mask_cong], pred_raw[mask_cong]))
        metrics['Congested_MAPE'] = np.mean(np.abs((trues[mask_cong] - pred_raw[mask_cong]) / (trues[mask_cong] + 1e-1))) * 100
    else:
        metrics['Congested_RMSE'] = 0.0

    return metrics

# ============================================================
# Plotting Utilities
# ============================================================
def plot_academic_benchmark(results_df):
    if results_df.empty: return
    set_origin_style()
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    metrics = [m for m in metrics if m in results_df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        if metric == 'R2':
            df_sorted = results_df.sort_values(metric, ascending=False)
        else:            df_sorted = results_df.sort_values(metric, ascending=True)
            
        models = df_sorted.index.tolist()
        values = df_sorted[metric].values
        bar_colors = []
        for m in models:
            if 'DAGRN' in m:
                bar_colors.append('#D62728')
            elif 'DeepTTE' in m or 'WaveNet' in m:
                bar_colors.append('#5D6D7E')
            else:
                bar_colors.append('#D5DBDB')
        bars = ax.bar(models, values, color=bar_colors, edgecolor='black', 
                      linewidth=1.0, width=0.7, zorder=3)
        ax.set_title(metric, fontweight='bold', pad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.set_xticklabels(models, rotation=35, ha='right', fontsize=10)
        if metric == 'R2':
            ax.set_ylim(0, 1.1)
        else:
            ax.set_ylim(0, max(values) * 1.15)
        for idx, (bar, model_name) in enumerate(zip(bars, models)):
            height = bar.get_height()
            if 'DAGRN' in model_name or idx < 3:
                fontweight = 'bold' if 'DAGRN' in model_name else 'normal'
                fmt = '{:.4f}' if metric == 'R2' else '{:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        fmt.format(height),
                        ha='center', va='bottom', fontsize=9, 
                        fontweight=fontweight, color='black')
    plt.tight_layout()
    plt.savefig('Benchmark_Comparison_Sorted.pdf')
    plt.savefig('Benchmark_Comparison_Sorted.png')
    print("[Plot] Saved sorted benchmark plots.")
    plt.close()

def plot_training_curves(histories_dict):
    set_origin_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1 = axes[0]
    for name, h in histories_dict.items():
        if 'DAGRN' in name:
            ax1.plot(h['val_loss'], label=name, linewidth=2.5, color='#D62728')
        elif name == 'Adaptive-GGNN':
            ax1.plot(h['val_loss'], label=name, linewidth=2.0, color='blue')
        else:
            ax1.plot(h['val_loss'], label=name, alpha=0.3)
    ax1.set_title('Validation Loss (Log Space)', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('L1 Loss on Log(Time)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    if 'val_mae' in list(histories_dict.values())[0]:
        ax2 = axes[1]
        for name, h in histories_dict.items():
            if 'DAGRN' in name:
                ax2.plot(h['val_mae'], label=name, linewidth=2.5, color='#D62728')
            elif name == 'Adaptive-GGNN':
                ax2.plot(h['val_mae'], label=name, linewidth=2.0, color='blue')
            else:
                ax2.plot(h['val_mae'], label=name, alpha=0.3)
        ax2.set_title('Validation MAE (Seconds)', fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('MAE (s)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Training_Curves_Dual.png')
    plt.savefig('Training_Curves_Dual.pdf')
    print("[Plot] Saved dual training curves.")
    plt.close()

def plot_ablation_study(ablation_df):
    if ablation_df.empty: return
    set_origin_style()
    df = ablation_df.sort_values('RMSE', ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    models = [name.replace('DAGRN_', '').replace('_', ' ') for name in df.index]
    rmse = df['RMSE'].values
    patterns = ['/', '\\', 'x', 'o'] 
    patterns = patterns * (len(models) // len(patterns) + 1)
    bars = ax.bar(models, rmse, color='white', edgecolor='black', 
                  linewidth=1.5, width=0.6)
    for bar, pattern in zip(bars, patterns):
        bar.set_hatch(pattern)
    bars[-1].set_edgecolor('#D62728')
    bars[-1].set_linewidth(2.0)
    ax.set_ylabel('RMSE (seconds)', fontweight='bold', fontsize=14)
    ax.set_title('Ablation Study: Contribution of Components', 
                 fontweight='bold', pad=15, fontsize=16)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=12, fontweight='bold')
    ax.tick_params(direction='in', length=6, width=1.5)
    x_start = bars[0].get_x() + bars[0].get_width()/2
    y_start = bars[0].get_height()
    x_end = bars[-1].get_x() + bars[-1].get_width()/2
    y_end = bars[-1].get_height()
    improvement = (rmse[0] - rmse[-1]) / rmse[0] * 100
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start + max(rmse)*0.1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    mid_x = (x_start + x_end) / 2
    mid_y = (y_start + max(rmse)*0.1 + y_end) / 2
    ax.text(mid_x, mid_y + max(rmse)*0.05, f"Improvement: {improvement:.1f}%", 
            ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.ylim(0, max(rmse) * 1.25)
    plt.tight_layout()
    plt.savefig('Ablation_Study_Textured.png', dpi=300)
    plt.savefig('Ablation_Study_Textured.pdf')
    print("[Plot] Saved textured ablation study.")
    plt.close()

def plot_sensitivity_heatmap(results_list):
    if not results_list: return
    set_origin_style()
    try:
        df = pd.DataFrame(results_list)
        best_lr = df.groupby('lr')['RMSE'].mean().idxmin()
        print(f"[Plot] Best LR found for heatmap: {best_lr}")
        subset = df[df['lr'] == best_lr]
        pivot_table = subset.pivot(index="hidden_dim", columns="dropout_rate", values="RMSE")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="Blues_r",
                    linewidths=1.2, linecolor='black', ax=ax,
                    cbar_kws={'label': 'RMSE (s)'},
                    annot_kws={'size': 12, 'family': 'Times New Roman'})
        ax.set_title(f'Parameter Sensitivity (LR={best_lr})', fontweight='bold', pad=15, fontsize=14)
        ax.set_xlabel('Dropout Rate', fontweight='bold', fontsize=12)
        ax.set_ylabel('Hidden Dimension', fontweight='bold', fontsize=12)
        plt.xticks(fontname='Times New Roman', fontsize=11)
        plt.yticks(fontname='Times New Roman', fontsize=11)
        plt.tight_layout()
        plt.savefig('Sensitivity_Heatmap_Origin.pdf', format='pdf')
        plt.savefig('Sensitivity_Heatmap_Origin.png', format='png', dpi=300)
        print("[Plot] Saved sensitivity heatmap.")
        plt.close()
    except Exception as e:
        print(f"[Plot Error] Could not plot heatmap: {e}")

# ============================================================
# Main Execution Wrapper
# ============================================================

def run_training_pipeline(root_dir):
    # Patch GCN (from original code)
    if not hasattr(GCNModule, "_orig_forward_backup"):
        GCNModule._orig_forward_backup = GCNModule.forward
    def gcn_forward_2d(self, x, edge_index):
        x = self.conv1(x, edge_index); x = self.bn1(x); x = F.relu(x)
        x = self.conv2(x, edge_index); x = self.bn2(x); x = F.relu(x)
        return x
    GCNModule.forward = gcn_forward_2d

    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # --- MODIFIED: Paths relative to root_dir ---
    SHP_PATH = os.path.join(root_dir, "data/network_road/network_link_enriched.shp")
    if not os.path.exists(SHP_PATH):
        SHP_PATH = os.path.join(root_dir, "data/network_road/network_link_aligned.shp")
        
    CSV_PATH = os.path.join(root_dir, "outputs/mapmatch_output/all_matched_results_filtered.csv")
    WEATHER_PATH = os.path.join(root_dir, "data/weather_data/wuxi_weather_2020.csv")
    
    # Change working directory to outputs/figures for plots
    FIGURE_DIR = os.path.join(root_dir, "outputs/figures")
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # Remember original dir to switch back if needed, or just let it be
    original_cwd = os.getcwd()
    os.chdir(FIGURE_DIR) 
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Matched CSV not found at {CSV_PATH}")
        return

    trajs, dyn_feats, times, graph_data, num_edges = prepare_real_data(SHP_PATH, CSV_PATH, WEATHER_PATH)
    
    trajs, dyn_feats, times = stratified_subset(trajs, dyn_feats, times, subset_size = 2000)
    
    static_dim = graph_data['edge_features'].shape[1]
    dyn_dim = len(dyn_feats[0])
    
    print("\n[Data Split] Standard 70% Train, 10% Val, 20% Test")
    X_temp, X_test, dyn_temp, dyn_test, y_temp, y_test = train_test_split(
        trajs, dyn_feats, times, test_size=0.2, random_state=42, shuffle=True
    )
    X_train_global, X_val_global, dyn_train_global, dyn_val_global, y_train_global, y_val_global = train_test_split(
        X_temp, dyn_temp, y_temp, test_size=0.125, random_state=42, shuffle=True
    )
    print(f"  - Train samples: {len(X_train_global)}")
    print(f"  - Val samples:   {len(X_val_global)}")
    print(f"  - Test samples:  {len(X_test)}")

    # Initial Datasets (used for default DataLoaders)
    train_ds_initial = FastTravelTimeDataset(X_train_global, dyn_train_global, y_train_global, True)
    val_ds_initial = FastTravelTimeDataset(X_val_global, dyn_val_global, y_val_global, False)
    test_ds_initial = FastTravelTimeDataset(X_test, dyn_test, y_test, False)

    # Initial DataLoaders
    train_dl = DataLoader(train_ds_initial, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dynamic_collate_fn, num_workers=0)
    val_dl   = DataLoader(val_ds_initial, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_fn, num_workers=0)
    test_dl  = DataLoader(test_ds_initial, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_fn, num_workers=0)

    CLAMP_MAX = get_log_clamp(train_dl, device)
    graph_data_tuple = (graph_data['edge_index'].to(device), graph_data['edge_features'].to(device))
    common_cfg = {'lr': 1e-3, 'epochs': 50, 'patience': 20, 'device': device}
    adv = {'id_embed_dim': 32, 'hidden_dim': 64, 'dyn_feat_dim': dyn_dim,
           'num_edges': num_edges, 'dropout_rate': 0.3,
           'static_feat_dim': static_dim, 'gcn_out_dim': 32}

    # ---------------- Part A: Baseline Comparison ----------------
    print("\n" + "="*40 + "\n PART A: BASELINE COMPARISON \n" + "="*40)
    models_map = {
        'LSTM': {'model': SequenceBaseline(**adv, rnn_type='LSTM'), 'graph': True},
        'GRU': {'model': SequenceBaseline(**adv, rnn_type='GRU'), 'graph': True},
        'Attn-GRU': {'model': AttentiveRNN(**adv, rnn_type='GRU'), 'graph': True},
        'DeepTTE': {'model': DeepTTE_LSTM(**adv), 'graph': True},
        'TCN': {'model': TCN_Model(**adv), 'graph': True},
        'GAT-LSTM': {'model': GAT_LSTM(**adv), 'graph': True},
        'GCN-GRU': {'model': GCN_GRU(**adv), 'graph': True},
        'TAGCN-GRU': {'model': TAGCN_GRU(**adv), 'graph': True},
        'GraphTTE': {'model': GraphTTE(**adv), 'graph': True},
        'GSTA': {'model': GSTA(**adv), 'graph': True},
        'Adaptive-GGNN': {'model': AdaptiveGGNN_TTE(**adv, steps=3), 'graph': True},
        'DAGRN_GRU': {'model': DAGRN(**adv, rnn_type='GRU', gcn_iter=3), 'graph': True},
        'Graph-WaveNet': {'model': ST_WaveNet(**adv), 'graph': True},
    }
    
    results, histories = {}, {}    
    for name, cfg in models_map.items():
        print(f"> Training {name}...")
        try:
            model = cfg['model'].to(device)
            run_cfg = {**common_cfg, 'needs_graph': cfg['graph'], 'graph_data': graph_data_tuple, 'model_name': name, 'log_clamp_max': CLAMP_MAX}
            history, best_loss = train_engine(model, train_dl, val_dl, run_cfg)
            histories[name] = history
            results[name] = run_evaluation(model, test_dl, run_cfg)
            print(f"  -> RMSE: {results[name]['RMSE']:.4f}")
        except Exception as e:
            print(f"Failed {name}: {e}")
            import traceback; traceback.print_exc()

    if results:
        df_res = pd.DataFrame(results).T
        print("\n--- Leaderboard ---")
        print(df_res.sort_values('RMSE'))
        plot_academic_benchmark(df_res.sort_values('RMSE'))
        plot_training_curves(histories)

    # ---------------- Part B: Ablation Study ----------------
    dagrn_full_metrics = results.get('DAGRN_GRU', None)
    
    if dagrn_full_metrics:
        print("\n" + "="*40 + "\n PART B: RIGOROUS ABLATION STUDY \n" + "="*40)
        modes = ['no_gcn', 'no_film', 'no_attn', 'no_gate'] 
        results_abl = {'DAGRN_Full': dagrn_full_metrics}

        for mode in modes:
            name = f"DAGRN_{mode}"
            abl_model_params = copy.deepcopy(adv)
            abl_model_params.update({
                'rnn_type': 'GRU',
                'gcn_iter': 3,
                'ablation_mode': mode
            })
            
            try:
                model = DAGRN(**abl_model_params).to(device)
                cfg = {**common_cfg, 'model_name': name, 'needs_graph': True, 'graph_data': graph_data_tuple, 'log_clamp_max': CLAMP_MAX}
                
                train_engine(model, train_dl, val_dl, cfg)
                metrics = run_evaluation(model, test_dl, cfg)
                results_abl[name] = metrics
                print(f"  -> {name}: RMSE={metrics['RMSE']:.4f}")
            except Exception as e:
                print(f"  [Fail] {name}: {e}")
                import traceback; traceback.print_exc()

        df_abl = pd.DataFrame(results_abl).T
        print("\n--- Ablation Summary ---")
        print(df_abl.round(4))
        plot_ablation_study(df_abl)

    # ---------------- PART C: SENSITIVITY ANALYSIS ----------------
    print("\n" + "="*40 + "\n PART C: SENSITIVITY ANALYSIS \n" + "="*40)
    
    default_base = {
        'hidden_dim': 64, 'dropout_rate': 0.3, 'lr': 1e-3, 
        'gcn_iter': 2, 'attention_heads': 4, 'batch_size': 64
    }

    params_grid = []
    for v in [32, 64, 128]: params_grid.append({'hidden_dim': v})
    for v in [0.2, 0.5, 0.7]: params_grid.append({'dropout_rate': v})
    for v in [5e-4, 1e-3, 5e-3]: params_grid.append({'lr': v})
    for v in [1, 2, 3]: params_grid.append({'gcn_iter': v})
    for v in [2, 4, 8]: params_grid.append({'attention_heads': v})
    for v in [32, 64, 128]: params_grid.append({'batch_size': v})

    sens_results = []
    print(f"Total Sensitivity Runs: {len(params_grid)}")

    for i, cur_params in enumerate(params_grid):
        diff_key = list(cur_params.keys())[0]
        val = cur_params[diff_key]
        case_name = f"{diff_key}={val}"
            
        print(f"> [{i+1}/{len(params_grid)}] Testing {case_name} ...")
        
        model_params = copy.deepcopy(adv)
        model_params.update({
            'rnn_type': 'GRU',
            'gcn_iter': 3,
            'ablation_mode': 'full'
        })
        current_lr = common_cfg['lr']
        current_bs = BATCH_SIZE
        if diff_key == 'lr':
            current_lr = val
        elif diff_key == 'batch_size':
            current_bs = val
        else:
            model_params[diff_key] = val
        if current_bs != BATCH_SIZE:
             temp_train_ds = FastTravelTimeDataset(X_train_global, dyn_train_global, y_train_global, is_train=True)
             temp_val_ds = FastTravelTimeDataset(X_val_global, dyn_val_global, y_val_global, is_train=False)
             
             target_train_dl = DataLoader(temp_train_ds, batch_size=current_bs, shuffle=True, collate_fn=dynamic_collate_fn, num_workers=0)
             target_val_dl = DataLoader(temp_val_ds, batch_size=current_bs, shuffle=False, collate_fn=dynamic_collate_fn, num_workers=0)
        else:
             target_train_dl, target_val_dl = train_dl, val_dl

        try:
            model = DAGRN(**model_params).to(device)
            run_cfg = {
                **common_cfg, 
                'lr': current_lr, 
                'needs_graph': True, 
                'graph_data': graph_data_tuple, 
                'model_name': case_name, 
                'log_clamp_max': CLAMP_MAX
            }
            
            train_engine(model, target_train_dl, target_val_dl, run_cfg)
            met = run_evaluation(model, test_dl, run_cfg)
            
            record = cur_params.copy()
            record['RMSE'] = met['RMSE']
            record['MAE'] = met['MAE']
            record['Parameter Changed'] = case_name
            sens_results.append(record)
            print(f"  -> Result: RMSE={met['RMSE']:.4f}")
            
        except Exception as e:
            print(f"  [Fail] {case_name} failed: {e}")
            import traceback; traceback.print_exc()
            continue

    if sens_results:
        print("\n--- Sensitivity Analysis Summary ---")
        df_sens = pd.DataFrame(sens_results)
        cols = [c for c in ['Parameter Changed', 'RMSE', 'MAE'] if c in df_sens.columns]
        print(df_sens[cols])
        df_sens.to_csv('Sensitivity_Results.csv', index=False)
        try:
            plot_sensitivity_heatmap(sens_results)
        except Exception as e:
            print(f"[Plot Error] {e}")

    # Restore CWD
    os.chdir(original_cwd)