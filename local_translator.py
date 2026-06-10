import re
from difflib import get_close_matches
from functools import lru_cache


PHRASE_TRANSLATIONS = {
    "age": "Usia",
    "agent name": "Nama Agen",
    "agent shift": "Shift Agen",
    "arrival delay in minutes": "Keterlambatan Tiba (Menit)",
    "baggage handling": "Penanganan Bagasi",
    "batch size": "Ukuran Batch",
    "business travel": "Perjalanan Bisnis",
    "checkin service": "Layanan Check-in",
    "connected handling time": "Waktu Penanganan Terhubung",
    "csat score": "Skor Kepuasan Pelanggan",
    "customer": "ID Pelanggan",
    "customer city": "Kota Pelanggan",
    "customer id": "ID Pelanggan",
    "customer remarks": "Catatan Pelanggan",
    "customer satisfaction": "Kepuasan Pelanggan",
    "customer satisfaction score": "Skor Kepuasan Pelanggan",
    "customer segment": "Segmen Pelanggan",
    "customer support": "Dukungan Pelanggan",
    "customer type": "Tipe Pelanggan",
    "delivery experience": "Pengalaman Pengiriman",
    "delivery speed": "Kecepatan Pengiriman",
    "delivery time": "Waktu Pengiriman",
    "departure arrival time convenient": "Kenyamanan Waktu Berangkat/Tiba",
    "departure delay in minutes": "Keterlambatan Berangkat (Menit)",
    "ease of online booking": "Kemudahan Booking Online",
    "flight distance": "Jarak Penerbangan",
    "food and drink": "Makanan dan Minuman",
    "gate location": "Lokasi Gerbang",
    "how satisfied were you with the quality of the food at alis 1 5 where 1 extremely dissatisfied and 5 extremely satisfied": "Kepuasan Kualitas Makanan",
    "how satisfied were you with the speed of delivery at alis 1 5 where 1 extremely dissatisfied and 5 extremely satisfied": "Kepuasan Kecepatan Pengiriman",
    "how satisfied were you with your overall delivery experience at ali 1 5 where 1 extremely dissatisfied and 5 extremely satisfied": "Kepuasan Pengalaman Pengiriman",
    "inflight entertainment": "Hiburan di Pesawat",
    "inflight service": "Layanan Selama Penerbangan",
    "inflight wifi service": "Layanan WiFi di Pesawat",
    "issue reported at": "Waktu Keluhan Dilaporkan",
    "issue responded": "Waktu Keluhan Direspons",
    "item price": "Harga Barang",
    "leg room service": "Ruang Kaki",
    "max iteration": "Maksimal Iterasi",
    "missing value": "Nilai Hilang",
    "on board service": "Layanan di Pesawat",
    "online boarding": "Boarding Online",
    "order accurate": "Akurasi Pesanan",
    "order date time": "Tanggal dan Waktu Pesanan",
    "order id": "ID Pesanan",
    "payment method": "Metode Pembayaran",
    "personal travel": "Perjalanan Pribadi",
    "product category": "Kategori Produk",
    "rating": "Penilaian",
    "satisfaction": "Kepuasan",
    "satisfaction level": "Tingkat Kepuasan",
    "satisfaction score": "Skor Kepuasan",
    "seat comfort": "Kenyamanan Kursi",
    "service quality": "Kualitas Layanan",
    "survey response date": "Tanggal Respons Survei",
    "tenure bucket": "Kelompok Masa Kerja",
    "type of travel": "Jenis Perjalanan",
    "unique id": "ID Unik",
    "was your order accurate please respond yes or no": "Akurasi Pesanan",
}


VALUE_TRANSLATIONS = {
    "0-30": "0-30",
    "31-60": "31-60",
    "61-90": "61-90",
    ">90": ">90",
    "afternoon": "Sore",
    "app": "Aplikasi",
    "business": "Bisnis",
    "business travel": "Perjalanan Bisnis",
    "card": "Kartu",
    "cash": "Tunai",
    "chat": "Chat",
    "dissatisfied": "Tidak Puas",
    "disloyal customer": "Pelanggan Tidak Loyal",
    "eco": "Ekonomi",
    "eco plus": "Ekonomi Plus",
    "email": "Email",
    "evening": "Malam",
    "female": "Perempuan",
    "furniture": "Furnitur",
    "home": "Rumah",
    "inbound": "Masuk",
    "lifestyle": "Gaya Hidup",
    "loyal customer": "Pelanggan Loyal",
    "male": "Laki-laki",
    "member": "Anggota",
    "morning": "Pagi",
    "netral": "Netral",
    "neutral": "Netral",
    "neutral or dissatisfied": "Netral/Tidak Puas",
    "night": "Malam",
    "no": "Tidak",
    "not satisfied": "Tidak Puas",
    "on job training": "Pelatihan Kerja",
    "outcall": "Panggilan Keluar",
    "personal travel": "Perjalanan Pribadi",
    "phone": "Telepon",
    "puas": "Puas",
    "satisfaction": "Puas",
    "satisfied": "Puas",
    "split": "Split",
    "student": "Pelajar/Mahasiswa",
    "tidak puas": "Tidak Puas",
    "transfer": "Transfer",
    "unsatisfied": "Tidak Puas",
    "web": "Web",
    "worker": "Pekerja",
    "yes": "Ya",
}


WORD_TRANSLATIONS = {
    # Customer and identity
    "account": "akun",
    "address": "alamat",
    "agent": "agen",
    "area": "area",
    "branch": "cabang",
    "buyer": "pembeli",
    "city": "kota",
    "client": "klien",
    "consumer": "konsumen",
    "contact": "kontak",
    "customer": "pelanggan",
    "email": "email",
    "gender": "jenis kelamin",
    "group": "kelompok",
    "id": "ID",
    "identifier": "ID",
    "location": "lokasi",
    "manager": "manajer",
    "member": "anggota",
    "name": "nama",
    "phone": "telepon",
    "region": "wilayah",
    "segment": "segmen",
    "seller": "penjual",
    "supervisor": "supervisor",
    "type": "tipe",
    "unique": "unik",
    "user": "pengguna",
    "username": "nama pengguna",
    # Satisfaction and survey
    "accurate": "akurat",
    "accuracy": "akurasi",
    "answer": "jawaban",
    "bad": "buruk",
    "complaint": "keluhan",
    "csat": "CSAT",
    "dissatisfied": "tidak puas",
    "experience": "pengalaman",
    "feedback": "umpan balik",
    "good": "baik",
    "happy": "senang",
    "level": "tingkat",
    "neutral": "netral",
    "nps": "NPS",
    "overall": "keseluruhan",
    "question": "pertanyaan",
    "rating": "penilaian",
    "remark": "catatan",
    "remarks": "catatan",
    "response": "respons",
    "review": "ulasan",
    "score": "skor",
    "satisfied": "puas",
    "satisfaction": "kepuasan",
    "survey": "survei",
    "unhappy": "tidak senang",
    # Service and support
    "assigned": "ditugaskan",
    "call": "panggilan",
    "case": "kasus",
    "category": "kategori",
    "channel": "kanal",
    "chat": "chat",
    "closed": "ditutup",
    "delay": "keterlambatan",
    "department": "departemen",
    "duration": "durasi",
    "escalated": "dieskalasi",
    "handling": "penanganan",
    "help": "bantuan",
    "inbound": "masuk",
    "issue": "keluhan",
    "opened": "dibuka",
    "outbound": "keluar",
    "priority": "prioritas",
    "queue": "antrean",
    "reported": "dilaporkan",
    "resolved": "diselesaikan",
    "responded": "direspons",
    "resolution": "penyelesaian",
    "service": "layanan",
    "shift": "shift",
    "sla": "SLA",
    "status": "status",
    "sub": "sub",
    "support": "dukungan",
    "ticket": "tiket",
    "waiting": "menunggu",
    # E-commerce and product
    "amount": "jumlah",
    "brand": "merek",
    "cart": "keranjang",
    "cashback": "cashback",
    "catalog": "katalog",
    "checkout": "checkout",
    "coupon": "kupon",
    "delivery": "pengiriman",
    "discount": "diskon",
    "exchange": "penukaran",
    "fee": "biaya",
    "food": "makanan",
    "invoice": "faktur",
    "item": "barang",
    "logistic": "logistik",
    "offer": "penawaran",
    "order": "pesanan",
    "package": "paket",
    "payment": "pembayaran",
    "price": "harga",
    "product": "produk",
    "promo": "promo",
    "purchase": "pembelian",
    "quantity": "kuantitas",
    "refund": "pengembalian dana",
    "replacement": "penggantian",
    "return": "retur",
    "seller": "penjual",
    "shipping": "pengiriman",
    "shop": "toko",
    "sku": "SKU",
    "stock": "stok",
    "transaction": "transaksi",
    # Time
    "arrival": "kedatangan",
    "created": "dibuat",
    "date": "tanggal",
    "day": "hari",
    "hour": "jam",
    "minute": "menit",
    "month": "bulan",
    "period": "periode",
    "received": "diterima",
    "reported": "dilaporkan",
    "response": "respons",
    "second": "detik",
    "speed": "kecepatan",
    "submitted": "dikirim",
    "time": "waktu",
    "timestamp": "stempel waktu",
    "updated": "diperbarui",
    "week": "minggu",
    "year": "tahun",
    # Flight/travel dataset
    "airline": "maskapai",
    "arrival": "kedatangan",
    "baggage": "bagasi",
    "boarding": "boarding",
    "booking": "pemesanan",
    "business": "bisnis",
    "checkin": "check-in",
    "class": "kelas",
    "cleanliness": "kebersihan",
    "comfort": "kenyamanan",
    "departure": "keberangkatan",
    "distance": "jarak",
    "entertainment": "hiburan",
    "flight": "penerbangan",
    "gate": "gerbang",
    "inflight": "selama penerbangan",
    "leg": "kaki",
    "online": "online",
    "room": "ruang",
    "seat": "kursi",
    "travel": "perjalanan",
    "wifi": "WiFi",
    # ML and evaluation
    "accuracy": "akurasi",
    "actual": "aktual",
    "batch": "batch",
    "classification": "klasifikasi",
    "class": "kelas",
    "confusion": "confusion",
    "correlation": "korelasi",
    "epoch": "epoch",
    "evaluation": "evaluasi",
    "feature": "fitur",
    "f1": "F1",
    "iteration": "iterasi",
    "label": "label",
    "loss": "loss",
    "matrix": "matrix",
    "model": "model",
    "precision": "precision",
    "prediction": "prediksi",
    "recall": "recall",
    "target": "target",
    "test": "uji",
    "train": "latih",
    "training": "training",
    "validation": "validasi",
    # Common attributes
    "active": "aktif",
    "average": "rata-rata",
    "bucket": "kelompok",
    "cancelled": "dibatalkan",
    "complete": "selesai",
    "count": "jumlah",
    "current": "saat ini",
    "default": "bawaan",
    "description": "deskripsi",
    "detail": "detail",
    "failed": "gagal",
    "high": "tinggi",
    "low": "rendah",
    "medium": "sedang",
    "missing": "hilang",
    "new": "baru",
    "normal": "normal",
    "old": "lama",
    "pending": "tertunda",
    "previous": "sebelumnya",
    "quality": "kualitas",
    "rank": "peringkat",
    "reason": "alasan",
    "record": "baris data",
    "row": "baris",
    "source": "sumber",
    "total": "total",
    "valid": "valid",
    "value": "nilai",
}


def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_id(text):
    if text in {"ID", "CSAT", "NPS", "SLA", "SKU", "F1", "WiFi"}:
        return text
    return text.title()


def _lookup_word(word):
    if word in WORD_TRANSLATIONS:
        return WORD_TRANSLATIONS[word]

    if len(word) >= 5:
        match = get_close_matches(word, WORD_TRANSLATIONS.keys(), n=1, cutoff=0.88)
        if match:
            return WORD_TRANSLATIONS[match[0]]

    return word


@lru_cache(maxsize=4096)
def translate_column_name(column_name):
    normalized = normalize_text(column_name)
    if not normalized:
        return str(column_name)

    if normalized in PHRASE_TRANSLATIONS:
        return PHRASE_TRANSLATIONS[normalized]

    if "satisf" in normalized and "overall" in normalized and "delivery" in normalized:
        return "Kepuasan Pengalaman Pengiriman"
    if "satisf" in normalized and "quality" in normalized and "food" in normalized:
        return "Kepuasan Kualitas Makanan"
    if "satisf" in normalized and "speed" in normalized and "delivery" in normalized:
        return "Kepuasan Kecepatan Pengiriman"
    if "order" in normalized and "accurate" in normalized:
        return "Akurasi Pesanan"

    words = [_lookup_word(word) for word in normalized.split()]
    translated = " ".join(words)
    return " ".join(_title_id(part) for part in translated.split())


@lru_cache(maxsize=4096)
def translate_value_label(value):
    normalized = normalize_text(value)
    if not normalized:
        return str(value)
    if normalized in VALUE_TRANSLATIONS:
        return VALUE_TRANSLATIONS[normalized]
    if normalized in PHRASE_TRANSLATIONS:
        return PHRASE_TRANSLATIONS[normalized]

    words = [_lookup_word(word) for word in normalized.split()]
    translated = " ".join(words)
    return " ".join(_title_id(part) for part in translated.split())

