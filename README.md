# WhereFit

WhereFit helps you compare cities through the climate preferences that matter to you: heat, cold, humidity, rain, air quality, and difficult weather. It is an open-source Streamlit project for travel planning, long-term living exploration, and side-by-side city comparison.

Try the deployed app on [ModelScope Studio](https://www.modelscope.cn/studios/modelscope_mp_361258324/WhereFit).

> WhereFit is an exploratory comparison tool. Its scores are not safety ratings, event probabilities, or professional advice, and should not be used alone for housing, insurance, medical, emergency, or disaster-avoidance decisions.

## What it does

- Compares 77 included cities in 24 countries: 47 in China and 30 elsewhere.
- Supports Chinese and English, plus system, light, and dark themes.
- Offers Travel, Long-term Living, and City Comparison scenarios.
- Lets you browse included cities by country/region, province/region or city, and geo-climate label; filters work together rather than independently.
- Produces a ranked comparison, city climate snapshots, charts, a map, source notes, and a downloadable text report.
- Provides optional views for historical weather, short-term forecasts, recent air quality, historical hazard records, river-flow estimates, and aurora conditions.

## Data used in the main comparison

The bundled baseline covers all 77 cities. Climate values are derived from NASA POWER / MERRA-2 daily grid data for 2000–2025 and stored as monthly and annual city-centre summaries. The baseline includes temperature, humidity, precipitation, wind, heat and cold days, and related climate indicators. Apparent temperature is calculated from temperature, humidity, and wind; possible snow days are a temperature-and-precipitation estimate, not observed snowfall.

Long-term PM2.5 uses the bundled ACAG SatPM2.5 V6.GL.03 city extraction for 2015–2024. It is a gridded estimate that combines satellite, model, and monitoring information, rather than a reading from one monitoring station.

`data/city_seed.csv` supplies city names, coordinates, country and regional labels, aliases, and 1–5 reference levels. Those reference levels are only used if the corresponding bundled climate or air-quality value is unavailable.

## How the score works

WhereFit compares only the cities selected in the current session. Each component is converted to a 0–100 value, so the result is a relative preference match for that chosen city set, not a universal city ranking.

Climate Comfort combines temperature comfort (38%), humidity comfort (22%), rain friendliness (22%), and long-term air quality (18%). The Bad Weather Index combines heat, heavy rain, long-term PM2.5, and broad coastal/typhoon labels. A lower Bad Weather Index means fewer factors to pay attention to.

Your Preference Match combines Climate Comfort with the inverse of the Bad Weather Index. The comfort/weather weights are 75%/25% for Travel, 45%/55% for Long-term Living, and 60%/40% for City Comparison. In Travel mode, a valid short-term forecast can replace the baseline result with a forecast-based travel score; recent PM2.5 is used only in that short-term Travel calculation.

Historical hazard records do not change the main ranking. They are separate, on-demand reference views.

## Optional live and historical views

- Historical weather: Open-Meteo Historical Weather API, using the ERA5 model for the selected dates.
- Forecasts: Open-Meteo Forecast or MET Norway Locationforecast.
- Recent air quality: Open-Meteo Air Quality, used only for short-term Travel scoring when available.
- Earthquakes: USGS Earthquake Catalog.
- Tropical-cyclone tracks: NOAA IBTrACS, using the basin selected from the city location.
- Flood and landslide records: NASA EONET; nearby river-flow estimates: Open-Meteo Flood.
- Aurora: NOAA SWPC OVATION when requested, with a latitude-based fallback when no nowcast is queried.

Live responses are cached locally under `data/cache/`. API availability, coverage, and response speed can vary.

## Run locally

Python 3.11 is recommended because the runtime dependencies are pinned for that version.

```bash
git clone https://github.com/ZhuoyaoH/WhereFit.git
cd WhereFit
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app starts from `app.py`. The bundled city, climate, and air-quality files are already included; an ordinary comparison does not need to download multi-year global datasets.

## Important limits

- NASA POWER values are city-centre model-grid estimates, not station observations, administrative-area averages, or neighbourhood microclimates.
- ACAG values are averages over a 3×3 grid window around each city centre; they are not population-weighted exposure estimates.
- Full-year rain, heat, and cold counts are converted to monthly-equivalent rates before scoring so they are comparable with month-based thresholds.
- A missing public record does not mean that no local hazard exists. Track proximity, river flow, and aurora opportunity are not damage, safety, or event probabilities.
- Future climate projections, such as CMIP6 or SSP scenarios, are not part of this version.

## Data sources

- [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
- [ACAG SatPM2.5 V6.GL.03](https://registry.opendata.aws/surface-pm2-5-v6gl/)
- [Open-Meteo](https://open-meteo.com/en/docs)
- [MET Norway Locationforecast](https://api.met.no/weatherapi/locationforecast/2.0/documentation)
- [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/)
- [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
- [NASA EONET](https://eonet.gsfc.nasa.gov/docs/v3)
- [NOAA SWPC Aurora Forecast](https://www.spaceweather.gov/products/aurora-30-minute-forecast)

External sources remain subject to their own licences and terms. See `LICENSE` for this repository's licence.
