# Log Interaksi Sesi 03: Modul Losses & Gram Matrix

- **Tanggal Sesi:** 2026-11-25
- **Target Modul:** `src/losses.py`
- **Keterangan Sesi:** Antigravity IDE Lokal

---

## 1. TAHAP INISIASI

### Prompt Lengkap

Prompt inisiasi & perencanaan modul:
```text
Merujuk langsung pada struktur output forward() di @src/vgg_extractor.py, rancang modul src/losses.py secara modular dari nol tanpa pustaka loss eksternal. Implementasikan: (1) compute_gram_matrix(feature_map) dengan mengubah tensor (B, C, H, W) menjadi matriks (C, H*W), mengalikan matriks dengan transposenya, lalu menormalisasinya dengan membagi terhadap C * H * W; (2) compute_content_loss berbasis MSE; (3) compute_style_loss berbasis MSE berbobot terhadap Gram Matrix di 5 lapisan gaya; (4) compute_total_variation_loss opsional untuk menekan bintik noise; dan (5) compute_total_loss yang menggabungkan alpha * content_loss + beta * style_loss + tv_weight * tv_loss. Jangan buat atau ubah file dulu, paparkan desain fungsi dan dimensinya di chat.
```

Prompt persetujuan penulisan kode:
```text
LANJUTKAN
```

### Alasan Khusus
1. **Kepatuhan Aturan Perencanaan Wajib (`AGENTS.md`):** Daftar fungsi, konstanta `DEFAULT_STYLE_WEIGHTS`, validasi parameter, alur logika, serta transformasi dimensi tensor input-output dipaparkan terlebih dahulu di chat sebelum membuat file `src/losses.py` setelah persetujuan `LANJUTKAN`.
2. **Implementasi Manual Tanpa Pustaka Loss Eksternal:** Seluruh perhitungan MSE pada `compute_content_loss` dan `compute_style_loss` ditulis eksplisit dari operasi tensor dasar (`torch.mean((a - b) * (a - b))`) tanpa menggunakan `torch.nn.MSELoss` maupun `torch.nn.functional.mse_loss`.
3. **Keselarasan Langsung dengan Output `VGG19FeatureExtractor.forward()`:** Fungsi `compute_content_loss` dan `compute_style_loss` menerima kamus fitur `dict[str, torch.Tensor]` yang dikembalikan oleh `VGG19FeatureExtractor.forward()`, dengan merujuk langsung pada `DEFAULT_CONTENT_LAYER_NAMES` (`"conv4_2"`) dan `DEFAULT_STYLE_LAYER_NAMES` (`"conv1_1"`, `"conv2_1"`, `"conv3_1"`, `"conv4_1"`, `"conv5_1"`).
4. **Perhitungan & Normalisasi Gram Matrix (`C * H * W`):** `compute_gram_matrix` mengubah tensor `(1, C, H, W)` menjadi `(C, H * W)`, menghitung perkalian matriks dengan transposenya (`torch.mm(features, features.t())`) menjadi `(C, C)`, lalu menormalisasinya dengan membagi terhadap `C * H * W`. Untuk `B > 1`, perkalian dilakukan per sampel menggunakan `torch.bmm` sehingga menghasilkan `(B, C, C)` tanpa mencampur statistik antar-gambar.
5. **Isolasi Gradien Target via `.detach()`:** Pada `compute_content_loss` dan `compute_style_loss`, fitur target dipanggil dengan `.detach()` sehingga gradien saat backpropagation hanya mengalir ke gambar yang sedang dioptimasi (`generated_img`).
6. **Total Variation Loss Anisotropik Berbasis `mean`:** `compute_total_variation_loss` menghitung rata-rata nilai absolut selisih piksel bertetangga secara vertikal (`H - 1`) dan horizontal (`W - 1`) agar besarnya penalti tidak meledak saat resolusi gambar berubah, sekaligus bersifat opsional pada `compute_total_loss` (`tv_loss: torch.Tensor | None = None`).

### Cuplikan Kode Krusial (`src/losses.py`)

```python
DEFAULT_STYLE_WEIGHTS: dict[str, float] = {
    layer_name: 1.0 / len(DEFAULT_STYLE_LAYER_NAMES)
    for layer_name in DEFAULT_STYLE_LAYER_NAMES
}


def compute_gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    if feature_map.ndim != 4:
        raise ValueError(
            f"Expected 4D feature map with shape (B, C, H, W), got {tuple(feature_map.shape)}."
        )

    batch_size, channels, height, width = feature_map.shape
    if batch_size <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"All feature map dimensions must be positive, got {tuple(feature_map.shape)}."
        )

    norm_factor = float(channels * height * width)

    if batch_size == 1:
        features = feature_map.reshape(channels, height * width)
        return torch.mm(features, features.t()) / norm_factor

    features = feature_map.reshape(batch_size, channels, height * width)
    return torch.bmm(features, features.transpose(1, 2)) / norm_factor
```

```python
def compute_content_loss(
    generated_features: dict[str, torch.Tensor],
    target_features: dict[str, torch.Tensor],
    content_layers: tuple[str, ...] = DEFAULT_CONTENT_LAYER_NAMES,
) -> torch.Tensor:
    layers = _validate_feature_layers(
        generated_features, target_features, content_layers, "content"
    )

    layer_losses: list[torch.Tensor] = []
    for layer_name in layers:
        gen_feat = generated_features[layer_name]
        target_feat = target_features[layer_name].detach()

        if gen_feat.shape != target_feat.shape:
            raise ValueError(
                f"Content feature shape mismatch at '{layer_name}': "
                f"{tuple(gen_feat.shape)} vs {tuple(target_feat.shape)}."
            )

        diff = gen_feat - target_feat
        layer_losses.append(torch.mean(diff * diff))

    return torch.stack(layer_losses).mean()
```

```python
def compute_style_loss(
    generated_features: dict[str, torch.Tensor],
    target_features: dict[str, torch.Tensor],
    style_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    weights = DEFAULT_STYLE_WEIGHTS if style_weights is None else style_weights
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("All style layer weights must be non-negative.")

    layers = _validate_feature_layers(
        generated_features, target_features, weights.keys(), "style"
    )

    ref_tensor = generated_features[layers[0]]
    total_style_loss = torch.zeros((), dtype=ref_tensor.dtype, device=ref_tensor.device)

    for layer_name in layers:
        weight = weights[layer_name]
        gen_feat = generated_features[layer_name]
        target_feat = target_features[layer_name].detach()
        ...
        gen_gram = compute_gram_matrix(gen_feat)
        target_gram = compute_gram_matrix(target_feat)
        diff = gen_gram - target_gram
        layer_loss = torch.mean(diff * diff)
        total_style_loss = total_style_loss + weight * layer_loss

    return total_style_loss
```

```python
def compute_total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    ...
    tv_loss = torch.zeros((), dtype=image.dtype, device=image.device)

    if height > 1:
        diff_h = image[:, :, 1:, :] - image[:, :, :-1, :]
        tv_loss = tv_loss + torch.mean(torch.abs(diff_h))

    if width > 1:
        diff_w = image[:, :, :, 1:] - image[:, :, :, :-1]
        tv_loss = tv_loss + torch.mean(torch.abs(diff_w))

    return tv_loss


def compute_total_loss(
    content_loss: torch.Tensor,
    style_loss: torch.Tensor,
    tv_loss: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1e6,
    tv_weight: float = 0.0,
) -> torch.Tensor:
    if alpha < 0.0 or beta < 0.0 or tv_weight < 0.0:
        raise ValueError("Loss weights alpha, beta, and tv_weight must be non-negative.")

    total_loss = alpha * content_loss + beta * style_loss
    if tv_loss is not None and tv_weight > 0.0:
        total_loss = total_loss + tv_weight * tv_loss

    return total_loss
```

---

## 2. TAHAP PENANGANAN EROR / DEBUGGING

### Status Eksekusi & Validasi Unit Test
Tidak ditemukan eror maupun bug saat implementasi dan pengujian modul [src/losses.py](file:///d:/SEMESTER%207/GENERATIVE%20AI/Mid%20Test/Vibe-Coding_Neural-Style-Transfer/src/losses.py); seluruh pengujian berjalan mulus pada eksekusi pertama.

### Rincian Validasi yang Dijalankan
Pengujian dijalankan melalui `.\.venv\Scripts\python.exe` dengan skenario berikut:
1. **`compute_gram_matrix`:**
   - Verifikasi bentuk output `(2, 2)` untuk input `(1, 2, 2, 2)` serta kesesuaian nilai numerik eksak (`[[3.75, 1.625], [1.625, 1.125]]`) terhadap perhitungan manual dengan pembagi `C * H * W = 8`.
   - Verifikasi bentuk output `(2, 4, 4)` untuk input batch `(2, 4, 3, 5)`.
2. **`compute_content_loss`:**
   - Verifikasi MSE manual menghasilkan `2.5` pada pasangan tensor uji dan `0.0` saat tensor identik.
3. **`compute_style_loss`:**
   - Verifikasi bobot 5 lapisan gaya pada `DEFAULT_STYLE_WEIGHTS` (`len == 5`), kompatibilitas resolusi spasial berbeda antara fitur hasil `(1, C, 16, 16)` dan target gaya `(1, C, 12, 20)`, serta loss bernilai `0.0` saat fitur identik.
4. **`compute_total_variation_loss` & `compute_total_loss`:**
   - Verifikasi TV loss `0.0` pada citra konstan `(1, 3, 8, 8)` dan `3.0` pada tensor uji `(1, 1, 2, 2)`, serta verifikasi kombinasi bobot `alpha`, `beta`, dan `tv_weight` (dengan maupun tanpa `tv_loss`).
5. **Uji Integrasi End-to-End dengan `VGG19FeatureExtractor`:**
   - Ekstraksi fitur konten dan gaya dari `VGG19FeatureExtractor(use_avg_pool=True)` pada tensor sintetis `(1, 3, 64, 64)`, dilanjutkan `total.backward()`.
   - Hasil keluaran konsol:
     ```text
     ALL_CHECKS_PASSED
     c_loss=0.000000, s_loss=0.000011, tv_loss=2.275450, total=10.701361
     ```
   - Terverifikasi `generated_img.grad` terisi dengan bentuk `(1, 3, 64, 64)` dan seluruh nilai berhingga (`torch.isfinite`), sementara seluruh parameter VGG-19 tetap beku tanpa gradien (`p.grad is None`).
