# Chapter 4: Real-World Data Representation Using Tensors

## 🎯 Value Proposition
This chapter bridges the gap between raw data (images, tables, text) and PyTorch tensors. You will learn how to load, reshape, normalize, and encode diverse data types into properly shaped tensors ready for model ingestion.

## 📂 File Structure
| File | Section | Concept |
| :--- | :--- | :--- |
| `01_image_loading.py` | 4.1 | Image loading, HWC→CHW permute, normalization |
| `02_volumetric_ct.py` | 4.2 | 3D Volumetric data, DICOM, channel dimension |
| `03_tabular_wine.py` | 4.3 | CSV loading, continuous vs. categorical, one-hot |
| `04_time_series_bikes.py` | 4.4 | Time series reshaping (N×C×L) |
| `05_text_jane_austen.py` | 4.5 | Text tokenization, character one-hot encoding |

## 🔑 Key Concepts
1.  **Image Layout:** PyTorch expects **Channels × Height × Width (CHW)**. Most libraries load **Height × Width × Channels (HWC)**. Use `.permute(2, 0, 1)`.
2.  **Normalization:** Neural networks train best when inputs are roughly **0 to 1** or **-1 to 1**. Use `(x - mean) / std` for standardization.
3.  **Tabular Data:**
    *   **Continuous:** Use directly as floats (e.g., Age, Temperature).
    *   **Ordinal:** Can be integers or one-hot (e.g., Education). One-hot avoids implying false mathematical distances.
    *   **Categorical:** Use one-hot encoding (e.g., City, Color).
4.  **Time Series:** Reshape flat data into **(Samples, Channels, Time Steps)** to preserve temporal structure.
5.  **Text:** Convert characters/words to indices, then to one-hot vectors or embeddings.

## 📝 Exercises (Section 4.7)
1.  **Image Channels:** Load several images. Calculate the mean of each RGB channel. Can you identify red vs. blue items based on channel averages alone?
2.  **Source Code Tokenization:** Treat a Python file as text. Build a word index. Compare the vocabulary size to the Jane Austen example. What information is lost in one-hot encoding?

## ⚠️ Data Requirements
To run these scripts fully, you need the sample data from the book's repository:
- `data/p1ch4/image-dog/`
- `data/p1ch4/volumetric-dicom/`
- `data/p1ch4/tabular-wine/winequality-white.csv`
- `data/p1ch4/bike-sharing-dataset/`
- `data/p1ch4/jane-austen/`