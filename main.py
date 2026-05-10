from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import anthropic
import json
import os

app = FastAPI(title="SIPANAS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ── Models ──────────────────────────────────────────────────────────────────
class ObservasiInput(BaseModel):
    lokasi: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    suhu: float               # °C
    kelembaban: float         # %
    kecepatan_angin: float    # km/jam
    tutupan_lahan: str
    catatan: Optional[str] = ""


class HotspotPoint(BaseModel):
    nama: str
    level: str                # tinggi | sedang | rendah
    suhu: float
    rh: float
    angin: float
    lat: float
    lon: float
    px: float                 # posisi relatif di peta (0-1)
    py: float
    keterangan: str


class AnalisisResponse(BaseModel):
    hotspots: list[HotspotPoint]
    level_keseluruhan: str
    analisis: str
    rekomendasi: str
    prediksi: str
    fire_weather_index: float  # FWI sederhana (0-100)


# ── Helper: hitung Fire Weather Index sederhana ──────────────────────────────
def hitung_fwi(suhu: float, rh: float, angin: float, lahan: str) -> float:
    """
    FWI sederhana berbasis Canadian FWI System (disederhanakan).
    Nilai 0-100: <25 rendah, 25-50 sedang, 50-75 tinggi, >75 ekstrem
    """
    # Fine Fuel Moisture Code (FFMC) proxy
    ffmc = max(0, (suhu * 0.8) + ((100 - rh) * 0.5) - 10)

    # Wind Effect
    wind_effect = angin * 0.6

    # Lahan multiplier
    lahan_mult = {
        "hutan gambut": 1.5,
        "hutan tropis": 1.1,
        "lahan pertanian": 0.9,
        "semak belukar": 1.2,
        "perkebunan sawit": 1.0,
        "savana/padang rumput": 1.3,
    }.get(lahan.lower(), 1.0)

    fwi = min(100, (ffmc + wind_effect) * lahan_mult * 0.5)
    return round(fwi, 1)


# ── Endpoint utama ───────────────────────────────────────────────────────────
@app.post("/api/analisis", response_model=AnalisisResponse)
async def analisis_karhutla(data: ObservasiInput):
    fwi = hitung_fwi(data.suhu, data.kelembaban, data.kecepatan_angin, data.tutupan_lahan)

    prompt = f"""Kamu adalah sistem AI untuk deteksi titik panas (heat spot) karhutla (kebakaran hutan dan lahan) di Indonesia.

Data Observasi:
- Lokasi: {data.lokasi}
- Koordinat: {data.lat or 'tidak diketahui'}, {data.lon or 'tidak diketahui'}
- Suhu permukaan: {data.suhu}°C
- Kelembaban relatif: {data.kelembaban}%
- Kecepatan angin: {data.kecepatan_angin} km/jam
- Jenis tutupan lahan: {data.tutupan_lahan}
- Fire Weather Index (FWI): {fwi}/100
- Catatan: {data.catatan or 'tidak ada'}

Tugas:
1. Deteksi 3-6 titik panas potensial di sekitar lokasi tersebut
2. Tentukan level risiko (tinggi/sedang/rendah) per titik
3. Buat analisis kondisi karhutla berdasarkan data dan FWI
4. Buat rekomendasi tindakan konkret
5. Prediksi arah penyebaran api

Respond ONLY with valid JSON, no backticks, no preamble:
{{
  "hotspots": [
    {{
      "nama": "Nama lokasi spesifik",
      "level": "tinggi|sedang|rendah",
      "suhu": number,
      "rh": number,
      "angin": number,
      "lat": number,
      "lon": number,
      "px": float 0.0-1.0,
      "py": float 0.0-1.0,
      "keterangan": "singkat"
    }}
  ],
  "level_keseluruhan": "tinggi|sedang|rendah",
  "analisis": "paragraf analisis kondisi",
  "rekomendasi": "paragraf rekomendasi tindakan",
  "prediksi": "paragraf prediksi penyebaran api"
}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text
        result = json.loads(raw.strip())
        result["fire_weather_index"] = fwi
        return AnalisisResponse(**result)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Gagal parse respons AI: {e}")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {e}")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "SIPANAS", "version": "1.0.0"}


@app.get("/")
def root():
    return {"message": "SIPANAS API — gunakan /docs untuk Swagger UI"}
