# Automatic high-resolution imagery for your saved location

## One-time setup

1. Obtain a Planet API key whose account has **SkySatCollect / ortho_visual archive download access** for your area and dates. The collector uses already entitled archive assets; it does not buy imagery or submit tasking orders. Downloads consume your provider quota.
2. On Windows, open **Edit environment variables for your account**, add a user variable named `PL_API_KEY`, and put your key in its value. Close and reopen the launcher so it inherits the variable. Do not put the key in a source file or share it in chat. On Linux/macOS, supply `PL_API_KEY` in the environment used to launch `python start_dashboard.py`.
3. Run `RUN_ALL.bat` (Windows) or `python start_dashboard.py`. Open **Aircraft Awareness**, use **Your collection location**, and allow the browser's location request. Review its reported accuracy. If the location is inaccurate, enter verified coordinates under **Setup & Auto Mode**.
4. Automatic collection and detection are enabled by default. Choose the area half-width (default 0.5 km), maximum scene age (default 90 days), and cloud threshold (default 20%). The service checks every 30 minutes; **Check for new high-resolution imagery now** requests an earlier check.

## What happens automatically

The app searches the latest SkySat archive scenes that overlap the selected area and cover its center, retaining only assets your account can download. It activates an entitled RGB asset, polls for readiness, downloads the GeoTIFF, and checks its CRS, RGB bands, pixel spacing and valid pixels at the saved coordinate. It crops the delivered pixel grid without resampling and records acquisition time, AOI and coverage metadata. The optional optical detector runs after collection; its checkpoint downloads on first use. Rechecking the same scene reuses the local crop and detector output. Changing the location starts a separate history.

The downloadable result is shown on Aircraft Awareness and stored under `app/data/highres_auto/`. `source.json` records the scene metadata; `detector.json` records detector provenance and outputs. API keys and signed delivery URLs are not written into these records. The supervisor manages the `highres` worker alongside the existing satellite services.

## What the status means

| Status | Meaning |
|---|---|
| LOCATION REQUIRED | Save a browser location fix or verified coordinates. |
| CREDENTIAL REQUIRED | Set PL_API_KEY and restart the launcher. |
| NO ACCESSIBLE COVERAGE | No entitled scene met the location, date and cloud filters. |
| ACTIVATING | Planet is preparing the archive asset; the service polls every 30 seconds. |
| DOWNLOADING / DETECTING | The crop or aircraft candidate analysis is in progress. |
| UP TO DATE | Latest selected archive scene is saved; this does not mean a new satellite image was acquired now. |
| PROVIDER ERROR / ERROR | Access, transfer or processing failed; a bounded retry follows. |

A browser fix is not guaranteed exact; the UI displays the device's reported uncertainty. The saved location remains fixed until updated. SkySat ortho_visual is provider-enhanced RGB, sampled at 0.5 m; that sampling is not an independent native-resolution measurement. Scene-wide cloud filtering does not guarantee that your particular location is cloud-free. Crops may only partially cover the requested area; the UI reports valid-center and full-crop coverage separately. No scene availability, clear weather or current-day acquisition is guaranteed. Missing detections never establish absence.

Transfers are limited to 1 GB per scene and need at least 1.25 GB free disk space. No imagery purchase, paid tasking, or aerial basemap substitution is performed. Your selected coordinates are sent to Planet for catalog search when this feature is enabled.

## Verification

Automated tests exercise location geometry, download entitlements, activation, credential/location waiting states, no-coverage behavior, the mocked download-to-detector handoff, scene caching and existing aircraft analysis. Actual Planet access, real raster cropping and real model inference were not run in this build environment because no entitled account/imagery or raster/model runtime was available.

Provider references: [Data API](https://docs.planet.com/develop/apis/data/), [search filters](https://docs.planet.com/develop/apis/data/item-search/), [activation and download](https://docs.planet.com/develop/apis/data/items/), [SkySat product details](https://docs.planet.com/data/imagery/skysat/).
