# Log Interaksi Sesi 05: Eksperimen & Visualisasi Grid Perbandingan

- **Tanggal Sesi:** 2026-11-26
- **Target Modul:** `src/visualize_grid.py` dan Evaluasi Eksperimen (`outputs/exp1_baseline`, `outputs/exp2_high_style`, `outputs/exp3_lbfgs`, `outputs/exp4_avgpool`)
- **Keterangan Sesi:** Antigravity IDE Lokal

---

## 1. TAHAP INISIASI

### Prompt Lengkap

Prompt inisiasi & perencanaan modul `src/visualize_grid.py`:
```text
Buat skrip baru di src/visualize_grid.py menggunakan Pillow atau Matplotlib untuk menyusun grid gambar 2 baris x 3 kolom yang menampilkan: (1) data/content/batik-content1.jpg berjudul 'Content Image', (2) data/style/batik-style1.jpg berjudul 'Style Image', (3) hasil akhir dari outputs/exp1_baseline, (4) hasil akhir dari outputs/exp2_high_style, (5) hasil akhir dari outputs/exp3_lbfgs, dan (6) hasil akhir dari outputs/exp4_avgpool. Simpan hasil kolase ke outputs/perbandingan_eksperimen.png. Jangan buat file dulu, paparkan rencananya di chat ini.
```

Prompt persetujuan penulisan kode:
```text
LANJUTKAN
```

Prompt dokumentasi hasil eksperimen & sesi:
```text
Baca seluruh file log CSV/hasil di folder @outputs/ (exp1_baseline, exp2_high_style, exp3_lbfgs, exp4_avgpool). Buat file baru tepat di path docs/llm_logs/sesi_05_eksperimen.md yang mendokumentasikan seluruh proses pembuatan src/visualize_grid.py, tabel perbandingan nilai Content Loss, Style Loss, dan Total Loss akhir dari keempat eksperimen, serta catatan kendala/debugging jika ada, mengikuti format template resmi proyek.
```

### Alasan Khusus
1. **Kepatuhan Aturan Perencanaan Wajib (`AGENTS.md`):** Sebelum membuat `src/visualize_grid.py`, pemeriksaan resolusi & mode warna keenam citra sumber, tata letak panel `2 x 3`, daftar konstanta, fungsi (`load_rgb_image`, `create_comparison_grid`, `parse_args`, `main`), dan alur CLI dipaparkan terlebih dahulu di chat lalu baru ditulis setelah persetujuan `LANJUTKAN`.
2. **Kombinasi Pillow (`.convert("RGB")`) dan Matplotlib Backend `Agg`:** `load_rgb_image` membuka citra via Pillow dan mengonversinya ke mode `"RGB"` agar aman terhadap berbagai format gambar, sedangkan rendering grid menggunakan `matplotlib.use("Agg")` (konsisten dengan [main.py](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/main.py)) agar proses ekspor PNG tidak memicu GUI dan tidak memblokir terminal.
3. **Menjaga Rasio Aspek Citra Sumber (`target_size=None` Default):** Hasil pemeriksaan file menunjukkan perbedaan dimensi asli antara `data/content/batik-content1.jpg` (`1280 x 960`, rasio `4:3`), `data/style/batik-style1.jpg` (`1776 x 1772`), dan keempat `result.png` eksperimen (`384 x 384`). Dengan membiarkan `target_size=None` sebagai default, `ax.imshow()` menjaga rasio aspek alami masing-masing gambar di dalam grid `2 x 3`, sembari tetap menyediakan flag opsional `--uniform-size WIDTH HEIGHT` (resampling `Image.Resampling.LANCZOS`) apabila diperlukan ukuran panel yang seragam.
4. **Konfigurasi Ekspor Resolusi Tinggi & Antarmuka CLI Modular:** `create_comparison_grid` memvalidasi jumlah panel (`1..6`) dan nilai `dpi > 0`, menonaktifkan sumbu tiap subplot (`ax.axis("off")`), menambahkan `suptitle` opsional, serta menyimpan kolase ke [outputs/perbandingan_eksperimen.png](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/perbandingan_eksperimen.png) dengan `dpi=200`, `bbox_inches="tight"`, dan `facecolor="white"`. Fungsi `parse_args` dan `main` memungkinkan skrip dijalankan langsung melalui `.venv\Scripts\python.exe -m src.visualize_grid`.

### Cuplikan Kode Krusial (`src/visualize_grid.py`)

```python
GRID_ROWS: int = 2
GRID_COLS: int = 3
DEFAULT_DPI: int = 200
DEFAULT_FIGSIZE: tuple[float, float] = (15.0, 10.0)
DEFAULT_OUTPUT_PATH: Path = Path("outputs/perbandingan_eksperimen.png")
DEFAULT_SUPTITLE: str = "Perbandingan Eksperimen Neural Style Transfer"

DEFAULT_PANELS: tuple[tuple[str, Path], ...] = (
    ("Content Image", Path("data/content/batik-content1.jpg")),
    ("Style Image", Path("data/style/batik-style1.jpg")),
    ("Exp 1: Baseline (Adam)", Path("outputs/exp1_baseline/result.png")),
    ("Exp 2: High Style", Path("outputs/exp2_high_style/result.png")),
    ("Exp 3: L-BFGS", Path("outputs/exp3_lbfgs/result.png")),
    ("Exp 4: AvgPool", Path("outputs/exp4_avgpool/result.png")),
)
```

```python
def load_rgb_image(
    image_path: Path,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Load an image as RGB, optionally resizing to `(width, height)`."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")

    if target_size is not None:
        width, height = target_size
        if width <= 0 or height <= 0:
            raise ValueError(f"target_size dimensions must be positive, got {target_size}.")
        rgb_img = rgb_img.resize((width, height), Image.Resampling.LANCZOS)

    return rgb_img
```

```python
def create_comparison_grid(
    panels: Sequence[tuple[str, Path]] = DEFAULT_PANELS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    suptitle: str | None = DEFAULT_SUPTITLE,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    target_size: tuple[int, int] | None = None,
) -> Path:
    """Render up to six labeled images into a 2x3 grid and save to `output_path`."""
    max_panels = GRID_ROWS * GRID_COLS
    if not panels:
        raise ValueError("panels must contain at least one (title, path) entry.")
    if len(panels) > max_panels:
        raise ValueError(f"panels cannot exceed {max_panels} items for a 2x3 grid, got {len(panels)}.")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}.")

    loaded_panels = [
        (title, load_rgb_image(Path(path), target_size=target_size))
        for title, path in panels
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=figsize, facecolor="white")
    try:
        for ax, panel in zip(axes.flat, loaded_panels):
            title, image = panel
            ax.imshow(image)
            ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
            ax.axis("off")

        for ax in list(axes.flat)[len(loaded_panels) :]:
            ax.axis("off")

        if suptitle:
            fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.98)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        else:
            fig.tight_layout()

        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)

    return output_path
```

### Tabel Perbandingan Nilai Loss Akhir Keempat Eksperimen

Seluruh nilai diambil langsung dari berkas `loss_history.csv` masing-masing folder eksperimen di `outputs/` (keempat eksperimen menghasilkan `result.png` berukuran `384 x 384` RGB, berinisialisasi dari citra content karena `content_loss = 0.0` pada `step = 1`, dan mencatat `tv_loss = 0.0` di seluruh langkah):

| Eksperimen | Total Step | Bobot Efektif ($\alpha$, $\beta$)\* | Content Loss Awal (Step 1) | Style Loss Awal (Step 1) | Total Loss Awal (Step 1) | Content Loss Akhir | Style Loss Akhir | Total Loss Akhir | File Log & Hasil |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp 1: Baseline (Adam)** (`exp1_baseline`) | 300 | $\alpha = 1.0$, $\beta = 10^4$ | `0.000000` | `0.106595` | `1065.946899` | **`80.141472`** (`80.14147186279297`) | **`0.007071`** (`0.007070737890899181`) | **`150.848846`** (`150.84884643554688`) | [loss_history.csv](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/exp1_baseline/loss_history.csv) / [result.png](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/exp1_baseline/result.png) |
| **Exp 2: High Style** (`exp2_high_style`) | 300 | $\alpha = 1.0$, $\beta = 10^6$ | `0.000000` | `0.106595` | `106594.687500` | **`298.947418`** (`298.9474182128906`) | **`0.002412`** (`0.002411947585642338`) | **`2710.895020`** (`2710.89501953125`) | [loss_history.csv](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/exp2_high_style/loss_history.csv) / [result.png](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/exp2_high_style/result.png) |
| **Exp 3: L-BFGS** (`exp3_lbfgs`) | 200 | $\alpha = 1.0$, $\beta = 10^5$ | `0.000000` | `0.106595` | `10659.468750` | **`201.573853`** (`201.5738525390625`) | **`0.001450`** (`0.001450486946851015`) | **`346.622559`** (`346.62255859375`) | [loss_history.csv](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/exp3_lbfgs/loss_history.csv) / [result.png](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/exp3_lbfgs/result.png) |
| **Exp 4: AvgPool** (`exp4_avgpool`) | 300 | $\alpha = 1.0$, $\beta = 10^5$ | `0.000000` | `0.000540` | `53.955727` | **`2.815227`** (`2.8152272701263428`) | **`0.000035`** (`3.4680480894166976e-05`) | **`6.283276`** (`6.283275604248047`) | [loss_history.csv](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/exp4_avgpool/loss_history.csv) / [result.png](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/exp4_avgpool/result.png) |

*\*Catatan: Nilai $\alpha = 1.0$ dan $\beta$ pada tabel diverifikasi secara numerik dari persamaan `total_loss = alpha * content_loss + beta * style_loss` pada Step 1 dan Step akhir di setiap `loss_history.csv`.*

#### Ringkasan Analisis Kuantitatif & Visual (`perbandingan_eksperimen.png`)
1. **Trade-off Bobot Gaya (`Exp 1` vs `Exp 2`):** Kenaikan bobot gaya dari $\beta = 10^4$ (`exp1_baseline`) ke $\beta = 10^6$ (`exp2_high_style`) menekan `style_loss` akhir lebih jauh (`0.002412` vs `0.007071`), namun menaikkan `content_loss` akhir dari `80.141472` menjadi `298.947418`. Secara visual pada [perbandingan_eksperimen.png](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/perbandingan_eksperimen.png), `Exp 2` menampilkan pola tekstur batik gaya yang lebih dominan dibanding `Exp 1`.
2. **Efisiensi Konvergensi Optimizer (`Exp 3: L-BFGS`):** Dengan 200 langkah dan $\beta = 10^5$, `exp3_lbfgs` mencapai `style_loss` akhir terendah di antara skenario berbasis `MaxPool2d` (`0.001450`), dengan sedikit osilasi pada beberapa langkah awal/tengah (misalnya step 3, 13, 131, 171, 192) sebelum turun stabil ke `total_loss = 346.622559`.
3. **Pengaruh `AvgPool2d` (`Exp 4`):** Penggantian `MaxPool2d` menjadi `AvgPool2d` membuat magnitudo aktivasi fitur dan Gram Matrix jauh lebih kecil sejak step awal (`style_loss` awal `0.000540` dibanding `0.106595` pada `MaxPool2d`), sehingga nilai absolut `content_loss` akhir (`2.815227`), `style_loss` akhir (`0.000035`), dan `total_loss` akhir (`6.283276`) tampak paling kecil. Secara relatif, `style_loss` turun ~93,57% dari step 1 hingga step 300 dan menghasilkan transfer warna serta pola batik yang halus.

---

## 2. TAHAP PENANGANAN EROR / DEBUGGING

### Status Eksekusi `src/visualize_grid.py`
Tidak terdapat eror maupun bug pada implementasi kode Python di [src/visualize_grid.py](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/src/visualize_grid.py). Namun, ditemukan satu kendala sintaks shell PowerShell saat menjalankan perintah verifikasi gabungan di terminal Windows.

### Pesan Eror Konsol Utuh (Kendala Shell Terminal)
```text
At line:1 char:169
+ ... } catch {}; .venv\Scripts\python.exe -m src.visualize_grid && .venv\S ...
+                                                                ~~
The token '&&' is not a valid statement separator in this version.
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : InvalidEndOfLine
```

### Analisis Akar Masalah & Solusinya
1. **Akar Masalah:**
   Terminal workspace menggunakan Windows PowerShell versi lawas (5.1) yang belum mendukung operator `&&` sebagai pemisah perintah bersyarat (*conditional statement separator*), sehingga perintah `.venv\Scripts\python.exe -m src.visualize_grid && ...` langsung ditolak oleh parser PowerShell dengan `ParserError` (`InvalidEndOfLine`, exit code `1`) sebelum skrip Python dieksekusi.
2. **Solusi yang Diterapkan:**
   Mengganti pemisah `&&` menjadi sintaks kompatibel PowerShell `; if ($LASTEXITCODE -eq 0) { ... }`:
   ```powershell
   .venv\Scripts\python.exe -m src.visualize_grid; if ($LASTEXITCODE -eq 0) { .venv\Scripts\python.exe -c "from pathlib import Path; from PIL import Image; p=Path('outputs/perbandingan_eksperimen.png'); im=Image.open(p); print('Exists:', p.exists(), 'Bytes:', p.stat().st_size, 'Size:', im.size, 'Mode:', im.mode)" }
   ```
3. **Hasil Verifikasi Akhir:**
   - Eksekusi `.venv\Scripts\python.exe -m src.visualize_grid` selesai dengan `Command exited with code 0`.
   - Keluaran konsol:
     ```text
     Saved comparison grid to outputs\perbandingan_eksperimen.png
     Exists: True Bytes: 9807670 Size: (2911, 1970) Mode: RGBA
     ```
   - Berkas [outputs/perbandingan_eksperimen.png](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/outputs/perbandingan_eksperimen.png) telah diverifikasi secara visual menampilkan ke-6 panel (`Content Image`, `Style Image`, `Exp 1: Baseline (Adam)`, `Exp 2: High Style`, `Exp 3: L-BFGS`, dan `Exp 4: AvgPool`) dengan tata letak 2 baris × 3 kolom dan judul yang terbaca jelas.
