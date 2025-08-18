### Step 1：安裝插件

在 **VS Code** 安裝 [Parquet Visualizer](https://marketplace.visualstudio.com/items?itemName=lucien-martijn.parquet-visualizer) 插件。

---

### Step 2：開啟檔案

* **Round 0**：使用 Parquet Visualizer 開啟 `./data/round0_qrels.parquet`
* **Round 1**：使用 Parquet Visualizer 開啟 `./data/round1_qrels.parquet`

---

### Step 3：執行 SQL 查詢

* **Round 0**

```sql
SELECT query, name, relevance, query_id
FROM data
WHERE query_id >= 0 AND query_id <= 4 AND relevance = 2;
```

* **Round 1**

```sql
SELECT query, name, relevance, query_id
FROM data
WHERE query_id >= 250 AND query_id <= 254 AND relevance = 2;
```