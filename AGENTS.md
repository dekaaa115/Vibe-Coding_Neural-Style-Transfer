# Aturan Proyek: Neural Style Transfer (Vibe Coding from Scratch)

1. Modular Bertahap: Jangan membuat seluruh proyek sekaligus. Hanya buat atau ubah file yang diminta pada sesi aktif.
2. Tahap Perencanaan Wajib (Dilarang Edit File Dulu): Setiap kali saya meminta modul baru, fitur baru, atau perbaikan besar, JANGAN langsung membuat atau mengubah file apa pun. Tulis dulu rencana desain (daftar fungsi/kelas, bentuk tensor input-output, dan alur logika) di balasan chat, lalu berhenti dan tunggu sampai saya mengetik "LANJUTKAN" sebelum menulis kode ke file.
3. Implementasi Manual Konsep Inti: Gunakan bobot pretrained VGG-19 dari torchvision hanya sebagai ekstraktor fitur beku (requires_grad=False). Perhitungan Gram Matrix, Content Loss, Style Loss, dan loop optimasi piksel wajib ditulis eksplisit dari dasar.
4. Debugging Transparan: Jika saya mengirim pesan error, jelaskan akar masalahnya sebelum menerapkan perbaikan.
5. Dokumentasi Sesi: Saat saya minta menutup sesi, buat rangkuman log interaksi dalam format Markdown di folder `docs/llm_logs/`.