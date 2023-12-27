# PMUDT Kelompok 1

## I Made Rian Wijaya (2008561065)
## I Nengah Oka Darmayasa (2008561070)

## Link Video Presentasi : https://youtu.be/3m6lvuAai3U?si=no57WuXKK-V1tZMI

# Cara Settup Project
1. Install semua depedensi library yang digunakan (direkomendasikan menggunakan Python versi 3.10)
2. Buat file config.py di root project yang berisi class seperti config-example.py. Sesuaikan koneksi database anda.
3. Jalankan perintah berikut di terminal secara berurutan untuk melakukan migrasi database
    - `flask shell`
    - `from database import db`
    - `from model.userModel import User`
    - `from model.historyModel import History`
    - `db.create_all()`
4. Jalankan aplikasi dengan menjalankan perintah `flask run` atau `flask run --debug` untuk memasukin mode debuging.
## Selamat aplikasi anda berjalan !