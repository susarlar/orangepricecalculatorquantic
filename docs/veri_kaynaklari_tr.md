# Finike Portakal Fiyat Tahmin Modeli — Veri Kaynakları ve Özellikler

## Proje Özeti
Yapay zeka destekli makine öğrenimi modeli ile Finike portakal fiyatlarını 1-3 ay önceden tahmin etme. Hal fiyatları, hava durumu, uydu verileri, rakip ülke verileri, ithalatçı ülke düzenlemeleri ve daha fazlasını hesaba katan kapsamlı bir model.

---

## 1. Fiyat Verileri (Hedef + Referans)
- **Finike Hal günlük fiyatları** — ana hedef değişken
- **Antalya Hal fiyatları** — bölgesel karşılaştırma
- **Diğer büyük hal fiyatları** (Mersin, Adana, İstanbul) — arbitraj sinyali
- **Geçmiş fiyat mevsimselliği** (gecikmeli fiyatlar, hareketli ortalamalar)

## 2. Arz Tarafı — Üretim
- **Hasat takvimi** — Finike portakalı Aralık–Nisan arası yoğunlaşır (Washington Navel, Valencia)
- **Ekili alan ve ağaç sayıları** (TÜİK ilçe bazlı veriler)
- **Sezon bazlı verim tahminleri**
- **Ağaç yaş dağılımı** — olgun ağaçlar vs. yeni dikimler

## 3. Hava Durumu ve İklim
- **Sıcaklık** (don riski fiyatları en çok etkileyen faktör)
- **Yağış** — çiçeklenme döneminde kuraklık veya aşırı yağış
- **Nem** — hastalık baskısı
- **Don olayları** — geçmiş don hasarları arzı düşürür, fiyatları yükseltir
- **Büyüme derece günleri (GDD)** — olgunlaşma zamanlaması

## 4. Uydu / Uzaktan Algılama
- **NDVI** (bitki örtüsü indeksi) — Sentinel-2 uydusundan ağaç sağlığı ve taç vigor takibi
- **Yer yüzey sıcaklığı (LST)** — sıcaklık stresi tespiti
- **Toprak nemi** (SMAP/Sentinel-1) — sulama durumu göstergesi
- **Evapotranspirasyon** — su stresi göstergesi
- **Ekili alan değişim tespiti** — bahçe genişleme/sökümü yıllar bazında

## 5. Rakip ve İkame Ürün Fiyatları
- **Mersin portakal fiyatları** (en büyük rakip bölge)
- **Adana portakal fiyatları**
- **İthal portakal fiyatları** (Mısır, İspanya, Güney Afrika)
- **Mandalina/limon fiyatları** — ikame narenciye ürünleri
- **Diğer meyve fiyatları** — talep ikamesi

## 6. İthalat/İhracat ve Ticaret Politikası
- **Türkiye portakal ithalat hacimleri** (TÜİK dış ticaret verileri)
- **İthalat gümrük tarifeleri ve kotaları** — mevsimsel tarife değişiklikleri
- **Türkiye'den ihracat hacimleri** (Rusya, Irak, AB)
- **Rakip ülkelerin ihracat yasakları veya kota değişiklikleri**
- **Bitki sağlığı düzenlemeleri** — ani ithalat yasakları
- **Döviz kuru (USD/TRY, EUR/TRY)** — ithalat rekabet gücünü etkiler

## 7. Pazar ve Talep
- **Hal günlük işlem hacimleri** (günlük işlem gören kg miktarı)
- **Zincir market fiyatları** (BİM, A101, Migros, ŞOK)
- **Antalya'da nüfus/turizm mevsimselliği** — yerel talep
- **Meyve suyu endüstrisi talebi** — endüstriyel vs. taze tüketim ayrımı
- **Ramazan/bayram zamanlaması** — talep artışları

## 8. Girdi Maliyetleri
- **Gübre fiyatları**
- **Mazot/yakıt fiyatları** — nakliye maliyeti
- **İşçilik maliyetleri** — hasat işçiliği
- **Sulama suyu maliyetleri**
- **İlaç (pestisit) maliyetleri**

## 9. Haberler ve Duygu Analizi
- **Tarım haberleri** (don uyarıları, hastalık salgınları, politika değişiklikleri)
- **Ticaret politikası duyuruları**
- **Devlet teşvikleri veya destek programları**
- **Anahtar kelime trendleri** (Google Trends — "portakal fiyat")

## 10. Makroekonomik Veriler
- **TÜFE / enflasyon oranı** — genel fiyat baskısı
- **Üretici fiyat endeksi (ÜFE) — tarım**
- **Faiz oranları** — depolama maliyeti etkisi
- **Akaryakıt fiyatları** — lojistik maliyeti

## 11. Lojistik
- **Finike → büyük şehirler nakliye maliyetleri**
- **Soğuk hava deposu kapasitesi ve doluluk oranı**
- **Depolama süresi** — depolanan portakal = ertelenmiş arz

---

## 12. Rakip Üretici Ülkeler

### Akdeniz Havzası
- **Mısır** — Dünyanın en büyük portakal ihracatçısı. Düşük işçilik maliyeti, Türkiye'nin en büyük rakibi. Hasat: Kasım–Mayıs
- **Fas (Maroko)** — AB'ye yakınlık avantajı, büyüyen ihracat kapasitesi. Hasat: Kasım–Haziran
- **İspanya** — AB'nin en büyük üreticisi, premium kalite. Hasat: Kasım–Haziran
- **İtalya** — Sicilya kan portakalları, AB iç pazarına yönelik. Hasat: Aralık–Nisan
- **Yunanistan** — Küçük ama AB iç pazarında rakip. Hasat: Kasım–Mayıs
- **Tunus** — Büyüyen ihracat kapasitesi, AB ile serbest ticaret anlaşması
- **İsrail** — Jaffa portakalları, yüksek kalite segmenti
- **Lübnan** — Bölgesel rakip, sınırlı hacim

### Güney Yarımküre (Ters Mevsim — Yaz Dönemi Rakipleri)
- **Güney Afrika** — Haziran–Ekim arası AB ve Rusya'ya ihracat, Türk yazlık portakalıyla çakışır
- **Arjantin** — Büyük üretici, Rusya ve AB pazarında rakip
- **Şili** — AB'ye serbest ticaret anlaşmasıyla avantajlı giriş
- **Uruguay** — Küçük ama büyüyen ihracatçı
- **Avustralya** — Asya pazarına yönelik ama global fiyatları etkiler

### Diğer Önemli Üreticiler
- **Çin** — Dünyanın en büyük üreticisi ama çoğu iç tüketime gidiyor
- **ABD (Florida, Kaliforniya)** — Meyve suyu endüstrisi fiyatlarını belirler
- **Brezilya** — Dünyanın en büyük portakal suyu ihracatçısı, FCOJ fiyat referansı

### Takip Edilecek Rakip Ülke Verileri
- Üretim hacimleri (USDA FAS, FAO verileri)
- İhracat hacimleri ve hedef pazarlar
- Hasat takvimi çakışmaları
- Hastalık/zararlı salgınları (HLB/citrus greening, Akdeniz sineği)
- Don/kuraklık gibi iklim olayları (rakipte arz düşerse Türkiye fiyatı değişir)
- Döviz kurları (Mısır poundu, Fas dirhemi vs. TRY)

---

## 13. İthalatçı Ülke Düzenlemeleri ve Politikaları

### Avrupa Birliği (AB)
- **Giriş fiyat sistemi** — Portakal için minimum giriş fiyatı; altında ek gümrük vergisi uygulanır
- **Mevsimsel tarife takvimi** — Haziran–Kasım arası düşük tarife (AB üretimi yok), Aralık–Mayıs arası yüksek koruma
- **MRL (Maksimum Kalıntı Limitleri)** — Pestisit kalıntı sınırları, sık güncellenir
- **Bitki sağlığı kontrolleri** — CBS (Citrus Black Spot), Akdeniz sineği, HLB taramaları
- **Tercihli ticaret anlaşmaları** — Fas, Tunus, Mısır, İsrail, Güney Afrika ile anlaşmalar (Türkiye'ye karşı avantaj)
- **Türkiye-AB Gümrük Birliği** — Tarım ürünleri tam kapsamda değil, portakalda kısıtlamalar var
- **Organik sertifikasyon gereklilikleri**
- **EUDR (Ormansızlaşma Yönetmeliği)** — Gelecekte tarım ürünlerini etkileyebilir

### Rusya
- **İthalat yasakları/kısıtlamaları** — Politik kararlarla ani değişebilir (2015 Türkiye ambargosu örneği)
- **Rosselkhoznadzor kontrolleri** — Bitki sağlığı denetimleri, sık red kararları
- **Kota sistemleri** — Belirli ülkelere yönelik kotalar
- **Ruble kuru** — Satın alma gücünü doğrudan etkiler
- **Mısır ve Fas ile artan ticaret** — Türkiye alternatifi olarak
- **Lojistik rotalar** — Karadeniz üzerinden nakliye maliyetleri

### Ukrayna
- **Savaş sonrası pazar durumu** — Lojistik aksaklıklar, liman kapanışları
- **AB entegrasyon süreci** — Düzenlemelerin AB'ye uyumu
- **Döviz kontrolleri** — Hryvnia istikrarsızlığı
- **İthalat kapasitesi** — Azalan satın alma gücü

### Irak
- **Türkiye'nin en büyük portakal ihracat pazarlarından** — Çok kritik
- **Düzensiz düzenlemeler** — Ani ithalat yasakları olabiliyor
- **Ödeme güçlükleri** — Döviz transferi sorunları
- **Kara yolu lojistiği** — Habur sınır kapısı yoğunluğu
- **Rekabet** — İran portakallarıyla fiyat rekabeti

### Suudi Arabistan ve Körfez Ülkeleri (BAE, Katar, Kuveyt)
- **Gümrük tarifeleri** — GCC ortak tarife sistemi
- **Kalite standartları** — SASO/GSO standartları
- **Soğuk zincir gereklilikleri** — Sıcak iklimde sıkı şartlar
- **Mısır ve Güney Afrika ile rekabet**

### Diğer Önemli Pazarlar
- **Beyaz Rusya** — Rusya'ya transit rota olarak da kullanılır
- **Romanya, Bulgaristan** — AB içi ama Türk portakalı için yakın pazarlar
- **Sırbistan** — AB dışı Balkan pazarı

### Takip Edilecek Düzenleme Verileri
- Tarife değişiklik duyuruları (WTO bildirimleri)
- Bitki sağlığı red bildirimleri (EU RASFF sistemi)
- Ticaret anlaşması güncellemeleri
- İthalat yasağı/kota duyuruları
- MRL limit güncellemeleri
- Sınır kapısı durumu ve bekleme süreleri
- Ülke bazlı döviz kuru hareketleri

---

## Öncelik Tablosu (1-3 Aylık Tahmin İçin)

| Öncelik | Kategori | Neden |
|---------|----------|-------|
| **P0** | Hal fiyatları (geçmiş) | Temel sinyal |
| **P0** | Hava durumu (sıcaklık, don, yağış) | En büyük arz şoku etkeni |
| **P0** | Uydu NDVI | Gerçek zamanlı ürün sağlığı |
| **P1** | İthalat/ihracat hacimleri ve politika | Arz rekabeti |
| **P1** | Rakip bölge fiyatları | Piyasa dinamikleri |
| **P1** | Döviz kurları | İthalat fiyat eşiği |
| **P1** | Rakip ülke üretim/ihracat verileri | Global arz dengesi |
| **P1** | İthalatçı ülke düzenlemeleri | Talep tarafı şokları |
| **P2** | Girdi maliyetleri (yakıt, gübre) | Fiyat tabanı |
| **P2** | Haber duygu analizi | Erken uyarı sinyalleri |
| **P3** | Perakende fiyatları, makroekonomik | Talep tarafı ince ayar |

---

## Aşamalı Geliştirme Planı
- **Aşama 1:** Geçmiş hal fiyatları + hava durumu + uydu → temel model
- **Aşama 2:** Rakip fiyatları, ithalat, döviz kuru, rakip ülke verileri eklenmesi
- **Aşama 3:** Haber duygu analizi + talep sinyalleri + ithalatçı ülke düzenlemeleri
