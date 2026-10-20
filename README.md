# Materi 6 — AI dan Citra Satelit untuk Pemantauan Deforestasi
**Studi kasus:** tim lingkungan ingin mengelompokkan dugaan penyebab perubahan tutupan lahan agar area yang perlu diperiksa dapat diprioritaskan.

**Implementasi:** model Random Forest dilatih dengan fitur simulasi perubahan NDVI/NBR dan jarak jalan. Kelas contoh: pertanian, pertambangan, pembangunan jalan, kebakaran, dan faktor alam.

**Menjalankan:**
```bash
pip install -r requirements.txt
python src/klasifikasi_deforestasi.py
```
**Batasan:** data bukan citra Sentinel asli dan bukan hasil lapangan. Prediksi tidak membuktikan penyebab deforestasi; sistem nyata memerlukan citra dan label terverifikasi, evaluasi akurasi, serta pemeriksaan manusia.
