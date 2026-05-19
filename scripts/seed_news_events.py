"""One-shot seed script for data/raw/news_events.csv.

Writes 15 representative classified-news rows in the same schema the live
DeepSeek pipeline (`src/data/news.py:save_news_events`) produces, so the
dashboard's "Recent News (DeepSeek-classified)" panel has content to render
before/while the live API key is configured.

These rows are SYNTHETIC SEED DATA — the headlines and dates are plausible
representations of Turkish citrus-market news drawn from publicly-known macro
patterns; they are not scraped from Google News. Replace with live pipeline
output (`python -m src.auto_refresh --full` with `DEEPSEEK_API_KEY` set) for
production use.
"""
from __future__ import annotations

import pandas as pd

from src.config import RAW_DIR


ROWS = [
    # 2025 — Q1
    {
        "date": "2025-01-12",
        "title": "Doğu Akdeniz'de soğuk hava — narenciye üreticileri tedirgin",
        "link": "https://example.com/news/2025/01/cold-front-citrus",
        "relevant": True,
        "sentiment": "bullish",
        "event_type": "frost",
        "magnitude": 1,
        "confidence": 0.78,
        "llm_summary": "Cold front passing over Eastern Mediterranean prompts citrus growers to take frost-protection measures; limited damage reported so far.",
        "raw_summary": "Doğu Akdeniz bölgesini etkileyen soğuk hava dalgası narenciye üreticilerini don önlemlerine yöneltti.",
    },
    {
        "date": "2025-02-22",
        "title": "TCMB faiz kararı — politika faizi sabit tutuldu",
        "link": "https://example.com/news/2025/02/cbrt-rate",
        "relevant": True,
        "sentiment": "neutral",
        "event_type": "economic",
        "magnitude": 1,
        "confidence": 0.66,
        "llm_summary": "Central Bank of Türkiye holds policy rate steady; analysts read decision as continued tightening bias, supportive of TRY stability and stable input costs for growers.",
        "raw_summary": "TCMB politika faizini değiştirmedi; sıkı para politikasının sürdürüldüğü değerlendirildi.",
    },
    # 2025 — Q2
    {
        "date": "2025-04-18",
        "title": "Yıllık enflasyon %40'ın altına geriledi",
        "link": "https://example.com/news/2025/04/inflation-eases",
        "relevant": True,
        "sentiment": "bearish",
        "event_type": "economic",
        "magnitude": 2,
        "confidence": 0.74,
        "llm_summary": "Türkiye's annual CPI inflation falls below 40 percent for the first time since 2021, easing input-cost pressure on agricultural producers including citrus growers.",
        "raw_summary": "Türkiye'de yıllık enflasyon 2021'den bu yana ilk kez %40'ın altına indi; tarım girdi maliyetleri üzerindeki baskı azaldı.",
    },
    {
        "date": "2025-05-30",
        "title": "Süveyş gerginliği azalıyor — taşımacılık maliyetleri normalleşiyor",
        "link": "https://example.com/news/2025/05/suez-eases",
        "relevant": True,
        "sentiment": "bearish",
        "event_type": "trade",
        "magnitude": 1,
        "confidence": 0.71,
        "llm_summary": "Red Sea shipping disruption shows signs of easing; container freight rates to EU and Gulf markets are normalising, supporting Turkish citrus export margins.",
        "raw_summary": "Kızıldeniz'deki gerginliğin azalmasıyla AB ve Körfez güzergahında konteyner navlunları gerilemeye başladı.",
    },
    # 2025 — Q3
    {
        "date": "2025-07-08",
        "title": "Asgari ücrette ara zam — tarım işçilik maliyeti yükseliyor",
        "link": "https://example.com/news/2025/07/min-wage-mid-year",
        "relevant": True,
        "sentiment": "bullish",
        "event_type": "economic",
        "magnitude": 1,
        "confidence": 0.69,
        "llm_summary": "Mid-year minimum-wage adjustment in Türkiye lifts harvest-labour costs for citrus growers; pressure on farm-gate prices expected in autumn harvest.",
        "raw_summary": "Türkiye'de yıl ortası asgari ücret düzenlemesi narenciye üreticilerinin işçilik maliyetini artırdı.",
    },
    {
        "date": "2025-08-12",
        "title": "Antalya'da uzun süreli sıcak hava — meyve iriliği endişesi",
        "link": "https://example.com/news/2025/08/heat-wave-fruit-size",
        "relevant": True,
        "sentiment": "bullish",
        "event_type": "supply",
        "magnitude": 2,
        "confidence": 0.72,
        "llm_summary": "Prolonged heat wave over Antalya region raises concern about fruit sizing in citrus orchards; growers report increased irrigation needs ahead of the 2025-26 harvest.",
        "raw_summary": "Antalya'da süregelen sıcak hava narenciye bahçelerinde meyve iriliği için endişe yarattı.",
    },
    # 2025 — Q4
    {
        "date": "2025-10-15",
        "title": "Hasat öncesi kuraklık stresi Akdeniz'de devam ediyor",
        "link": "https://example.com/news/2025/10/drought-stress",
        "relevant": True,
        "sentiment": "bullish",
        "event_type": "drought",
        "magnitude": 2,
        "confidence": 0.70,
        "llm_summary": "Mediterranean drought stress persists into pre-harvest period; some early-variety citrus orchards report reduced yield projections for the 2025-26 season.",
        "raw_summary": "Akdeniz'de süregelen kuraklık stresi hasat öncesi narenciye bahçelerini etkiliyor.",
    },
    {
        "date": "2025-11-22",
        "title": "Erken sezon don uyarısı — Mersin ve Adana etkilenebilir",
        "link": "https://example.com/news/2025/11/frost-warning-early-season",
        "relevant": True,
        "sentiment": "bullish",
        "event_type": "frost",
        "magnitude": 2,
        "confidence": 0.76,
        "llm_summary": "Early-season frost warning issued for Mersin and Adana citrus regions; meteorology service projects sub-zero temperatures for three consecutive nights.",
        "raw_summary": "Meteoroloji Mersin ve Adana narenciye bölgeleri için erken sezon don uyarısı yayımladı.",
    },
    {
        "date": "2025-12-09",
        "title": "Körfez ülkelerinden Türk portakaline tatil sezonu talebi",
        "link": "https://example.com/news/2025/12/gulf-demand",
        "relevant": True,
        "sentiment": "bullish",
        "event_type": "demand",
        "magnitude": 1,
        "confidence": 0.65,
        "llm_summary": "Gulf-region importers boost Turkish orange orders ahead of the holiday season; exporters report tighter availability in Antalya wholesale markets.",
        "raw_summary": "Körfez ülkeleri tatil sezonu öncesi Türk portakal siparişlerini artırdı; Antalya Hal'inde arz daraldı.",
    },
    {
        "date": "2025-12-28",
        "title": "Rusya fitosaniter denetimlerini sıkılaştırdı",
        "link": "https://example.com/news/2025/12/russia-phyto-tighter",
        "relevant": True,
        "sentiment": "bearish",
        "event_type": "policy",
        "magnitude": 2,
        "confidence": 0.68,
        "llm_summary": "Russia tightens phytosanitary inspections on Turkish citrus shipments at land crossings; exporters report longer delays and additional inspection fees.",
        "raw_summary": "Rusya gümrüklerde Türk narenciyesi için fitosaniter kontrolleri sıkılaştırdı; ihracatçılar gecikme bildiriyor.",
    },
    # 2026 — Q1
    {
        "date": "2026-01-18",
        "title": "Antalya'da gece sıcaklıkları sıfırın altına düştü",
        "link": "https://example.com/news/2026/01/antalya-cold-spell",
        "relevant": True,
        "sentiment": "bullish",
        "event_type": "frost",
        "magnitude": 2,
        "confidence": 0.79,
        "llm_summary": "Antalya region records sub-zero night temperatures for first time this winter; localised citrus orchard damage reported in higher-elevation zones.",
        "raw_summary": "Antalya'da gece sıcaklıkları bu kış ilk kez sıfırın altına düştü; yüksek rakımlı bahçelerde lokal zararlar bildirildi.",
    },
    {
        "date": "2026-02-05",
        "title": "AB Sınır Karbon Düzenleme Mekanizması tartışmaları",
        "link": "https://example.com/news/2026/02/eu-cbam",
        "relevant": True,
        "sentiment": "bearish",
        "event_type": "policy",
        "magnitude": 1,
        "confidence": 0.60,
        "llm_summary": "European Union consultations on Carbon Border Adjustment Mechanism extension to agricultural inputs raise concerns about future export-cost competitiveness for Turkish citrus.",
        "raw_summary": "AB Sınır Karbon Düzenleme Mekanizması'nın tarım girdilerine genişletilmesi Türk narenciye ihracatçıları için maliyet endişesi yarattı.",
    },
    {
        "date": "2026-03-14",
        "title": "TL'de değer kaybı yeniden hızlandı",
        "link": "https://example.com/news/2026/03/try-weakness",
        "relevant": True,
        "sentiment": "bullish",
        "event_type": "economic",
        "magnitude": 2,
        "confidence": 0.73,
        "llm_summary": "Turkish lira resumes depreciation against the US dollar; imported agricultural inputs (fertilizer, fuel, pesticides) more expensive, lifting wholesale citrus prices.",
        "raw_summary": "TL ABD doları karşısında değer kaybetmeye yeniden başladı; ithal tarım girdileri pahalandı.",
    },
    {
        "date": "2026-04-12",
        "title": "Akdeniz'de bahar yağmurları toprak nemini iyileştirdi",
        "link": "https://example.com/news/2026/04/spring-rains",
        "relevant": True,
        "sentiment": "bearish",
        "event_type": "supply",
        "magnitude": 1,
        "confidence": 0.67,
        "llm_summary": "Spring rainfall over the Mediterranean basin restores soil moisture levels in citrus regions; growers report improved outlook for 2026-27 fruit set.",
        "raw_summary": "Bahar yağmurları Akdeniz narenciye bölgelerinde toprak nemini iyileştirdi; sonraki sezon için olumlu beklenti.",
    },
    {
        "date": "2026-05-02",
        "title": "Mısır portakal ihracat kotasını artırdı",
        "link": "https://example.com/news/2026/05/egypt-quota-up",
        "relevant": True,
        "sentiment": "bearish",
        "event_type": "trade",
        "magnitude": 2,
        "confidence": 0.71,
        "llm_summary": "Egypt raises seasonal orange-export quota; additional Mediterranean supply pressures Turkish exporter pricing in EU and Gulf markets.",
        "raw_summary": "Mısır mevsimlik portakal ihracat kotasını artırdı; AB ve Körfez pazarlarında Türk ihracatçılarına fiyat baskısı.",
    },
]


def main() -> None:
    df = pd.DataFrame(ROWS)
    df["date"] = pd.to_datetime(df["date"])
    out_path = RAW_DIR / "news_events.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Seeded {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
