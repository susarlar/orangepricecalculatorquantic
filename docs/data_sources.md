# Finike Orange Price Forecast — Data Sources and Features

## Project Summary
An AI-driven machine learning model that forecasts Finike orange prices 1–3 months ahead. A comprehensive model that accounts for wholesale (Hal) market prices, weather, satellite imagery, competitor-country supply, importer-country regulations, and more.

---

## 1. Price Data (Target + Reference)
- **Finike Hal daily prices** — main target variable
- **Antalya Hal prices** — regional comparator
- **Other major Hal prices** (Mersin, Adana, İstanbul) — arbitrage signal
- **Historical price seasonality** (lagged prices, moving averages)

## 2. Supply Side — Production
- **Harvest calendar** — Finike orange peak runs December–April (Washington Navel, Valencia)
- **Cultivated area and tree counts** (TÜİK district-level data)
- **Seasonal yield forecasts**
- **Tree-age distribution** — mature trees vs. new plantings

## 3. Weather and Climate
- **Temperature** (frost risk is the single biggest price driver)
- **Precipitation** — drought or excess rain during flowering
- **Humidity** — disease pressure
- **Frost events** — past frost damage cuts supply and lifts prices
- **Growing degree days (GDD)** — ripening timing

## 4. Satellite / Remote Sensing
- **NDVI** (vegetation index) from Sentinel-2 — tree health and canopy vigor
- **Land surface temperature (LST)** — heat-stress detection
- **Soil moisture** (SMAP/Sentinel-1) — irrigation status proxy
- **Evapotranspiration** — water-stress indicator
- **Cropland change detection** — orchard expansion or removal year over year

## 5. Competitor and Substitute Prices
- **Mersin orange prices** (largest competing region)
- **Adana orange prices**
- **Imported orange prices** (Egypt, Spain, South Africa)
- **Mandarin / lemon prices** — substitute citrus
- **Other fruit prices** — demand-side substitution

## 6. Imports / Exports and Trade Policy
- **Turkey orange import volumes** (TÜİK foreign-trade data)
- **Import tariffs and quotas** — seasonal tariff schedules
- **Turkey export volumes** (Russia, Iraq, EU)
- **Competitor export bans or quota changes**
- **Phytosanitary regulations** — sudden import bans
- **Exchange rates (USD/TRY, EUR/TRY)** — affect import competitiveness

## 7. Market and Demand
- **Daily Hal trading volumes** (kg traded per day)
- **Chain retailer prices** (BİM, A101, Migros, ŞOK)
- **Population / tourism seasonality in Antalya** — local demand
- **Juice industry demand** — industrial vs. fresh consumption split
- **Ramadan / Bayram timing** — demand spikes

## 8. Input Costs
- **Fertilizer prices**
- **Diesel / fuel prices** — transport cost
- **Labor costs** — harvest labor
- **Irrigation water costs**
- **Pesticide costs**

## 9. News and Sentiment
- **Agriculture news** (frost warnings, disease outbreaks, policy changes)
- **Trade policy announcements**
- **Government incentives or support programs**
- **Keyword trends** (Google Trends — "orange price")

## 10. Macroeconomic Data
- **CPI / inflation rate** — general price pressure
- **Producer Price Index (PPI) — agriculture**
- **Interest rates** — storage cost effect
- **Fuel prices** — logistics cost

## 11. Logistics
- **Finike → major-city transport costs**
- **Cold-storage capacity and occupancy**
- **Storage duration** — stored oranges = deferred supply

---

## 12. Competitor Producer Countries

### Mediterranean Basin
- **Egypt** — world's largest orange exporter. Low labor costs; Turkey's biggest competitor. Harvest: November–May
- **Morocco** — proximity to EU is an advantage; growing export capacity. Harvest: November–June
- **Spain** — EU's largest producer, premium quality. Harvest: November–June
- **Italy** — Sicilian blood oranges, EU domestic-market focus. Harvest: December–April
- **Greece** — small but competes inside the EU. Harvest: November–May
- **Tunisia** — growing export capacity, free-trade agreement with the EU
- **Israel** — Jaffa oranges, premium quality segment
- **Lebanon** — regional competitor, limited volume

### Southern Hemisphere (Counter-Season — Summer Competitors)
- **South Africa** — June–October exports to EU and Russia, overlaps with Turkish summer oranges
- **Argentina** — large producer, competes in Russian and EU markets
- **Chile** — preferential EU access through free-trade agreement
- **Uruguay** — small but growing exporter
- **Australia** — Asia-focused, but moves global prices

### Other Major Producers
- **China** — world's largest producer but mostly domestic consumption
- **USA (Florida, California)** — juice industry sets reference prices
- **Brazil** — world's largest orange-juice exporter, FCOJ price benchmark

### Competitor-Country Data to Track
- Production volumes (USDA FAS, FAO data)
- Export volumes and target markets
- Harvest-calendar overlaps
- Disease / pest outbreaks (HLB / citrus greening, Mediterranean fruit fly)
- Climate events such as frost or drought (a competitor supply drop moves Turkish prices)
- FX rates (Egyptian pound, Moroccan dirham vs. TRY)

---

## 13. Importer-Country Regulations and Policies

### European Union (EU)
- **Entry-price system** — minimum entry price for oranges; below it an additional duty applies
- **Seasonal tariff schedule** — low tariff June–November (no EU output), high protection December–May
- **MRLs (Maximum Residue Limits)** — pesticide residue limits, frequently updated
- **Phytosanitary controls** — Citrus Black Spot (CBS), Mediterranean fruit fly, HLB screening
- **Preferential trade agreements** — Morocco, Tunisia, Egypt, Israel, South Africa (advantage over Turkey)
- **Turkey–EU Customs Union** — does not fully cover agriculture; oranges face restrictions
- **Organic certification requirements**
- **EUDR (Deforestation Regulation)** — may affect agricultural products in the future

### Russia
- **Import bans / restrictions** — politically driven, can change suddenly (2015 Turkey embargo precedent)
- **Rosselkhoznadzor controls** — phytosanitary inspections, frequent rejections
- **Quota systems** — country-specific quotas
- **Ruble exchange rate** — directly affects purchasing power
- **Increasing trade with Egypt and Morocco** — Turkey's substitutes
- **Logistics routes** — Black Sea shipping costs

### Ukraine
- **Post-war market conditions** — logistics disruptions, port closures
- **EU integration process** — regulatory alignment with EU
- **FX controls** — hryvnia volatility
- **Import capacity** — declining purchasing power

### Iraq
- **One of Turkey's largest orange-export markets** — critical
- **Inconsistent regulations** — sudden import bans possible
- **Payment difficulties** — FX-transfer issues
- **Land-route logistics** — Habur border-crossing congestion
- **Competition** — price competition with Iranian oranges

### Saudi Arabia and Gulf Countries (UAE, Qatar, Kuwait)
- **Customs tariffs** — GCC common-tariff system
- **Quality standards** — SASO / GSO standards
- **Cold-chain requirements** — strict in hot climates
- **Competition** — Egypt and South Africa

### Other Significant Markets
- **Belarus** — also a transit route to Russia
- **Romania, Bulgaria** — EU-internal but close markets for Turkish oranges
- **Serbia** — non-EU Balkan market

### Regulatory Data to Track
- Tariff-change announcements (WTO notifications)
- Phytosanitary rejection notices (EU RASFF system)
- Trade-agreement updates
- Import-ban / quota announcements
- MRL limit updates
- Border-crossing status and wait times
- Country-by-country FX moves

---

## Priority Table (for 1–3 month forecasts)

| Priority | Category                                  | Why                              |
|----------|-------------------------------------------|----------------------------------|
| **P0**   | Hal prices (historical)                   | Baseline signal                  |
| **P0**   | Weather (temperature, frost, rainfall)    | Largest supply-shock driver      |
| **P0**   | Satellite NDVI                            | Real-time crop health            |
| **P1**   | Import / export volumes and policy        | Supply competition               |
| **P1**   | Competitor-region prices                  | Market dynamics                  |
| **P1**   | FX rates                                  | Import-price threshold           |
| **P1**   | Competitor-country production / exports   | Global supply balance            |
| **P1**   | Importer-country regulations              | Demand-side shocks               |
| **P2**   | Input costs (fuel, fertilizer)            | Price floor                      |
| **P2**   | News sentiment                            | Early-warning signals            |
| **P3**   | Retail prices, macroeconomic data         | Demand-side fine-tuning          |

---

## Phased Development Plan
- **Phase 1:** Historical Hal prices + weather + satellite → baseline model
- **Phase 2:** Add competitor prices, imports, FX rates, competitor-country data
- **Phase 3:** Add news sentiment + demand signals + importer-country regulations
