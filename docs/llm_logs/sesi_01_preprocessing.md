# Log Interaksi Sesi 01: Modul Preprocessing

- **Tanggal Sesi:** 2026-11-25
- **Target Modul:** `src/preprocessing.py`
- **Keterangan Sesi:** Antigravity Lokal

---

## 1. TAHAP INISIASI

### Prompt Lengkap Pengguna

Prompt perencanaan modul:
```text
Kita akan membangun modul pertama untuk proyek Neural Style Transfer, yaitu src/preprocessing.py. Saya butuh fungsi untuk memuat gambar content dan style menggunakan PyTorch dan Pillow (pastikan otomatis dikonversi ke RGB agar aman untuk file PNG), mengubah ukurannya ke resolusi target (default 256 atau 512 px) tanpa merusak konsistensi tensor (1, 3, H, W), menerapkan normalisasi ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), serta fungsi tensor_to_image untuk denormalisasi dan clamping ke [0, 1]. Jangan buat atau ubah file apa pun dulu. Paparkan rencana desain fungsi dan bentuk tensornya di chat ini, lalu tunggu persetujuan saya.
```

Prompt persetujuan eksekusi:
```text
LANJUTKAN
```

### Alasan Khusus Pendekatan Ini
1. **Kepatuhan Aturan Modular & Perencanaan Wajib (`AGENTS.md`):** Sebelum membuat file `src/preprocessing.py`, desain arsitektur (daftar fungsi, konstanta, bentuk tensor input-output, dan alur logika) dipaparkan terlebih dahulu di chat dan menunggu kata kunci persetujuan `LANJUTKAN` agar struktur API disepakati sejak awal.
2. **Keamanan Format Gambar dengan `.convert("RGB")`:** File gambar PNG sering memiliki 4 channel (`RGBA` dengan *alpha channel*) atau 1 channel (`L`/grayscale) maupun mode palette (`P`). Konversi eksplisit ke `"RGB"` memastikan jumlah channel input selalu tepat `3`.
3. **Konsistensi Bentuk Tensor `(1, 3, H, W)`:** Seluruh fungsi pemuatan gambar (`load_image` dan `load_content_and_style`) melakukan `unsqueeze(0)` setelah `ToTensor()` dan `Normalize(...)` sehingga tensor langsung berbentuk batch 4D `(1, 3, H, W)` dengan `dtype=torch.float32` yang siap masuk ke VGG-19.
4. **Normalisasi & Denormalisasi ImageNet yang Reversibel dan Aman:** Menggunakan statistik standar ImageNet (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`). Pada tahap rekonstruksi citra (`denormalize_tensor` dan `tensor_to_image`), tensor terlebih dahulu di-`.detach().clone().cpu()` agar tidak memodifikasi tensor hasil optimasi piksel secara *in-place* dan tetap aman dari autograd graph, lalu dikembalikan ke `[0.0, 1.0]` dengan `torch.clamp(denorm, 0.0, 1.0)` sebelum dikonversi menjadi `PIL.Image.Image`.

### Cuplikan Kode Krusial (`src/preprocessing.py`)

```python
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE: int = 512


def load_image(
    image_path: str | Path,
    target_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
    keep_aspect_ratio: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    transform = build_transform(
        target_size=target_size,
        keep_aspect_ratio=keep_aspect_ratio,
    )

    with Image.open(path) as img:
        rgb_image = img.convert("RGB")
        tensor = transform(rgb_image).unsqueeze(0)

    if device is not None:
        tensor = tensor.to(device)

    return tensor
```

```python
def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        if tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError(
                f"Expected 4D tensor with shape (1, 3, H, W), got {tuple(tensor.shape)}."
            )
        view_shape = (1, 3, 1, 1)
    elif tensor.ndim == 3:
        if tensor.shape[0] != 3:
            raise ValueError(
                f"Expected 3D tensor with shape (3, H, W), got {tuple(tensor.shape)}."
            )
        view_shape = (3, 1, 1)
    else:
        raise ValueError(
            f"Expected 3D or 4D tensor, got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}."
        )

    denorm = tensor.detach().clone().cpu().to(torch.float32)
    mean = torch.tensor(IMAGENET_MEAN, dtype=denorm.dtype).view(*view_shape)
    std = torch.tensor(IMAGENET_STD, dtype=denorm.dtype).view(*view_shape)

    denorm = denorm * std + mean
    return torch.clamp(denorm, 0.0, 1.0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    clamped = denormalize_tensor(tensor)
    image_tensor = clamped.squeeze(0) if clamped.ndim == 4 else clamped
    return transforms.ToPILImage()(image_tensor)
```

---

## 2. TAHAP PENANGANAN EROR / DEBUGGING

### Pesan Eror Konsol Utuh
*Tidak ada pesan eror konsol pada implementasi maupun pengujian `src/preprocessing.py`.*

Hasil eksekusi verifikasi inline (`torch 2.8.0+cpu`, `torchvision 0.23.0+cpu`, `PIL 11.1.0`):
```text
ALL PREPROCESSING CHECKS PASSED
```

### Prompt Perbaikan Pengguna
*Tidak ada prompt perbaikan — rencana desain langsung disetujui (`LANJUTKAN`) dan implementasi berjalan tanpa kegagalan.*

### Analisis Akar Masalah dan Solusinya
*Tidak diperlukan debugging karena seluruh skenario uji pada modul `src/preprocessing.py` berhasil pada percobaan pertama, meliputi:*
- Pembacaan PNG `RGBA` dan `L` (grayscale) menjadi tensor `torch.float32` berukuran `(1, 3, 512, 512)` dan `(1, 3, 256, 256)`.
- Konsistensi bentuk `(1, 3, H, W)` pada `load_content_and_style` untuk citra non-persegi.
- Akurasi transformasi normalisasi ImageNet serta denormalisasi dan *clamping* nilai ekstrem ke `[0.0, 1.0]` pada `denormalize_tensor` dan `tensor_to_image`.
- Penanganan kesalahan (`FileNotFoundError` untuk file tidak ditemukan dan `ValueError` untuk bentuk tensor yang tidak valid).
