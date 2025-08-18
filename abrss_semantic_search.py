import faiss
from sentence_transformers import SentenceTransformer
import torch

corpus = [
    "NWT 威技 14吋日本DC變頻馬達節能電風扇 WPF-14P7",
    "NWT 威技 16吋DC變頻直流電風扇 WPF-16S7",
    "Panasonic 國際牌 14吋微電腦DC直流電風扇 F-S14KM",
    "HERAN 禾聯 16吋ECO溫控智能變頻DC風扇",
    "LAPO 製冷高速可折疊隨身風扇 LF-C01",
    "PHILIPS 飛利浦小炫風 手持風扇 ACR3222",
    "SHARP 夏普 森呼吸 自動除菌離子3D清淨循環扇 PK-18S02T-B",
    "SAMPO 聲寶 8吋循環扇 SK-TC08S",

    "MOREFINE 摩方 M1K 迷你電腦棒 (J4125/8G/128G/W11)",
    "MOREFINE 摩方 M11 掌上平板電腦 (N200/16G/256G SSD/W11)",
    "LENOVO 聯想 ThinkCentre Neo 50t Gen5 (i5/32G/1TB SSD/Win11 專業版)",
    "MOREFINE 摩方 M6S 迷你電腦 (N100/12G/256G/W11)",
    "MP520-20 8核 8GB/128GB NVMe SSD Debian Linux 微型電腦",
    "ECS 精英 LIVA Z2 (N6000/4G/128G/Win11 專業版)",
    "Nugens 43 吋 AIO 觸控電腦一體機 (Celeron J6412/8G/128GB SSD/W11P)",

    "Google Pixel 8 Pro (12G/128G) 6.7 吋 5G（福利品）",
    "Google Pixel 7 Pro (12G/512G) 6.7 吋（福利品）",
    "Google Pixel 6 (8G/128G) 6.4 吋（福利品）",
    "Google Pixel 6 (8G/256G) 6.4 吋（福利品）",
    "Google Pixel 7a (8G/128G) 6.1 吋（福利品）"
]

# To access the model, please visit https://huggingface.co/clw8998/ABRSS and submit a request for access

model = SentenceTransformer(
    model_name_or_path="clw8998/ABRSS",
    trust_remote_code=True,
    device="cuda",
    truncate_dim=768,
)

emb = model.encode(
    corpus,
    batch_size=32,
    convert_to_tensor=True,
    show_progress_bar=True
).to(torch.float32).cpu().numpy()

faiss.normalize_L2(emb)
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)

query = "安卓手機"
q_vec = model.encode(
    query,
    convert_to_tensor=True,
).to(torch.float32).cpu().numpy().reshape(1, -1)

faiss.normalize_L2(q_vec)
D, I = index.search(q_vec, 10)

print("\nTop-10 Results:")
for score, idx in zip(D[0], I[0]):
    print(f"{score:>6.4f} │ {corpus[idx]}")
