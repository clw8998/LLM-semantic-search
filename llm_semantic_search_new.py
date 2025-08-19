import faiss
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

ENABLE_TSNE = True

def visualize_with_tsne(
    emb, q_vec, labels,
    topk_indices=None,
    query_text=None,
    random_state: int = 42,
    save_path: str = "tsne.html",
):
    X = np.vstack([emb.astype(np.float32), q_vec.astype(np.float32)])
    n = X.shape[0]
    perplexity = max(5, min(30, (n - 1) // 3))
    Y = TSNE(
        n_components=2, init="pca", learning_rate="auto",
        perplexity=perplexity, random_state=random_state
    ).fit_transform(X)
    Yc, Yq = Y[:-1], Y[-1]

    dists = np.linalg.norm(Yc - Yq, axis=1)
    order = np.argsort(dists)
    ranks_by_index = np.empty(len(emb), dtype=int)
    ranks_by_index[order] = np.arange(1, len(emb) + 1)
    rank_strs = [str(r) for r in ranks_by_index]

    df = pd.DataFrame({"x": Yc[:, 0], "y": Yc[:, 1], "name": labels, "rank": ranks_by_index})

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        specs=[[{"type": "xy"}, {"type": "table"}]],
        horizontal_spacing=0.06,
        subplot_titles=("t-SNE result", "Items")
    )

    fig.add_trace(go.Scattergl(
        x=df["x"], y=df["y"],
        mode="markers",
        marker=dict(size=10, color="rgba(0,0,0,0)"),
        text=df["name"],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False
    ), row=1, col=1)

    k = 0 if topk_indices is None else int(np.asarray(topk_indices).size)
    if k > 0:
        topk_2d = order[:k]
        dft = df.iloc[topk_2d]
        fig.add_trace(go.Scattergl(
            x=dft["x"], y=dft["y"],
            mode="markers",
            marker=dict(size=16, color="rgba(0,0,0,0)",
                        line=dict(width=1.8, color="black")),
            text=dft["name"],
            hovertemplate="%{text}<extra>Top-K (2D)</extra>",
            showlegend=False
        ), row=1, col=1)

    fig.add_trace(go.Scattergl(
        x=[Yq[0]], y=[Yq[1]],
        mode="markers",
        marker=dict(symbol="star", size=18),
        hovertemplate=(query_text or "") + "<extra></extra>",
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["x"], y=df["y"],
        mode="text",
        text=rank_strs,
        textposition="middle center",
        textfont=dict(size=12),
        hoverinfo="skip",
        showlegend=False
    ), row=1, col=1)

    tbl_order = order.tolist()
    tbl_names = [labels[i] for i in tbl_order]
    tbl_ranks = [str(ranks_by_index[i]) for i in tbl_order]

    base_fill = ["white"] * len(tbl_order)
    fill_cols = [base_fill.copy(), base_fill.copy()]

    fig.add_trace(go.Table(
        header=dict(
            values=["Top-K", "商品名稱"],
            fill_color="#e9ecef",
            align=["center", "left"],
            font=dict(size=11, color="black")
        ),
        cells=dict(
            values=[tbl_ranks, tbl_names],
            fill_color=fill_cols,
            align=["center", "left"],
            height=24,
            font=dict(size=10)
        ),
        columnwidth=[1, 9],
        name="items"
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers+text",
        marker=dict(size=22, color="rgba(0,0,0,0)",
                    line=dict(width=2, color="#2b8a3e")),
        text=[""], textposition="middle center",
        hoverinfo="skip",
        showlegend=False
    ), row=1, col=1)

    fig.update_layout(showlegend=False, margin=dict(l=30, r=20, t=60, b=30), height=640)

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    html += f"""
        <script>
        (function(){{
        const gd = document.querySelector('div.js-plotly-plot');
        const hoverIdx=0, rankTextIdx= {2 if k==0 else 3}, tableIdx={3 if k==0 else 4}, selectIdx={4 if k==0 else 5};

        const tblOrder = {tbl_order};
        const rankOf   = {ranks_by_index.tolist()};

        function colorizeRowByRowIndex(row){{
            const n = gd.data[tableIdx].cells.values[0].length;
            const base = new Array(n).fill('white'), hi = '#fff3bf';
            if (row>=0 && row<n) base[row] = hi;
            Plotly.restyle(gd, {{'cells.fill.color': [ [base, base] ]}}, [tableIdx]);
        }}

        function highlightByOrigIndex(origIdx){{
            const xs = gd.data[hoverIdx].x, ys = gd.data[hoverIdx].y;
            const label = String(rankOf[origIdx] || "");
            Plotly.restyle(gd, {{x:[[xs[origIdx]]], y:[[ys[origIdx]]], text:[[label]]}}, [selectIdx]);

            const row = tblOrder.indexOf(origIdx);
            colorizeRowByRowIndex(row);
        }}

        gd.on('plotly_click', function(ev){{
            const p = ev.points && ev.points[0]; if(!p) return;

            if (p.curveNumber===hoverIdx || p.curveNumber===rankTextIdx){{
              const origIdx = p.pointIndex ?? p.pointNumber;
              highlightByOrigIndex(origIdx);
            }}

            if (p.curveNumber===tableIdx || (gd.data[p.curveNumber].type==='table')){{
              const row = p.pointNumber, origIdx = tblOrder[row];
              highlightByOrigIndex(origIdx);
            }}
        }});
        }})();
        </script>
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
    return fig

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

model = SentenceTransformer(
    model_name_or_path="Qwen/Qwen3-Embedding-8B",
    trust_remote_code=True,
    device="cuda",
    truncate_dim=768,
    model_kwargs={"torch_dtype": torch.bfloat16},
)

model.prompts = {
    "query": (
        "Instruct: Given an e-commerce search query, your goal is to maximize all retrieval metrics. "
        "Retrieve the most relevant product titles that match the query\n"
        "Query:"
    ),
    "passage": "Product Title:"
}

# model.prompts = {
#     "query": (
#         "Instruct: Given an e-commerce search query, "
#         "retrieve the most relevant product titles that match the query\n"
#         "Query:"
#     ),
#     "passage": "Product Title:"
# }

# model.prompts = {
#     "query": (
#         "指令：給定一個電子商務搜尋詞，召回最符合該搜尋詞的產品名稱\n"
#         "搜尋詞:"
#     ),
#     "passage": "產品名稱："
# }

# model.prompts = {
#     "query": "",
#     "passage": ""
# }

emb = model.encode(
    corpus,
    batch_size=32,
    convert_to_tensor=True,
    show_progress_bar=True,
    prompt_name="passage" if model.prompts and 'passage' in model.prompts else None,
).to(torch.float32).cpu().numpy()

faiss.normalize_L2(emb)
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)

query = "安卓手機"
q_vec = model.encode(
    query, 
    convert_to_tensor=True,
    prompt_name="query" if model.prompts and 'query' in model.prompts else None,
).to(torch.float32).cpu().numpy().reshape(1, -1)

faiss.normalize_L2(q_vec)
D, I = index.search(q_vec, 10)

print("\nTop-10 Results:")
for score, idx in zip(D[0], I[0]):
    print(f"{score:>6.4f} │ {corpus[idx]}")

if ENABLE_TSNE:
    visualize_with_tsne(
        emb=emb,
        q_vec=q_vec,
        labels=corpus,
        topk_indices=I[0],
        query_text=query,
        save_path="tsne.html",
    )

