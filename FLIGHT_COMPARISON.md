# FlightRadar24 comparison

Select an observation on **Aircraft Map**, then open its **Flight comparison** tab.

The comparison uses the satellite image's capture time and a selected spatial radius. It does not compare old satellite imagery with today's live traffic. Records must pass both the distance and time tolerances. This is a proximity screen, not confirmed identification or calibrated matching probability; geolocation uncertainty, aircraft altitude, timestamp precision, and nearby aircraft can make associations wrong.

## Use the official API

1. Obtain an official FlightRadar24 API token with access to the required historical date. API subscriptions and history availability differ from consumer website/app subscriptions.
2. Set `FR24_API_TOKEN` in the environment of the app process, then restart the app. Do not commit or share the token.
3. Select a real observation, open **Flight comparison**, and choose **FlightRadar24 API**.
4. Set a matching radius and time tolerance. Click **Fetch historical flight records & compare**.

For the current terminal session:

```sh
# Linux / macOS: replace the placeholder locally
export FR24_API_TOKEN='your-api-token'
python start_dashboard.py
```

```powershell
# Windows PowerShell: replace the placeholder locally
$env:FR24_API_TOKEN='your-api-token'
python start_dashboard.py
```

If using a codespace with the lightweight testing dependencies, run the dashboard using the README's Streamlit command after setting the environment variable. Actual satellite collection and inference still need their separate dependencies and provider entitlements.

Each explicit API click sends one historical-position request for the selected image capture timestamp and a small bounding box around that observation. It uses your FR24 API credits and returns at most 100 records. Reaching that cap marks the comparison incomplete. The map's 10-second refresh does not make additional API calls. An API error is displayed as an error and is never treated as evidence of no matching flight.

## Use an existing snapshot

Choose **Upload snapshot**, download the CSV template, fill it with flight-position records you are entitled to use, and upload it. Official API JSON responses with a `data` array are also accepted.

Required CSV columns: `fr24_id,timestamp,lat,lon`. Aliases `flight_id,latitude,longitude` are accepted. Timestamps must contain a timezone or be UNIX seconds. Optional columns include `callsign,flight,reg,hex,source,alt,gspeed,track,squawk,type`. Units follow FR24: altitude in feet, ground speed in knots, track in degrees. Keep squawk codes as text to preserve leading zeros.

Limits: 5 MB and 5,000 positions. Uploaded provenance and coverage are not verified. An empty or incomplete snapshot cannot establish that the aircraft was not transmitting.

## Read the result

| Result | Meaning |
| --- | --- |
| POSSIBLE BROADCAST MATCH | One flight ID has a nearby, time-aligned record whose reported source is ADS-B or MLAT. Association with the imaged aircraft is unconfirmed. |
| POSSIBLE ESTIMATED MATCH | A nearby position is labeled estimated; it does not establish a transmission at that time. |
| POSSIBLE FLIGHT MATCH | A nearby record exists, but its source does not establish broadcast evidence. |
| AMBIGUOUS MATCH | Multiple flight IDs meet the limits. The app does not pick one identity. |
| NO MATCHING FLIGHT RECORD | Some records fit the time window, but none fit the distance limit. Transponder state is undetermined. |
| NO TIME-ALIGNED RECORDS | Records do not fit the capture-time window. Transponder state is undetermined. |
| NO RECORDS RETURNED | The response/snapshot contains no positions. Transponder state is undetermined. |

The panel displays reported callsign, registration, ICAO hex, squawk, altitude, speed, track, source, record timestamp, and separation where available. These belong to a possible flight-record match, not a confirmed identity assignment to the satellite detection.

**No match does not prove a transponder is switched off.** Coverage gaps, filtering, ground operations, stale records, incomplete queries, and detection/geolocation errors can all produce unmatched candidates. FlightRadar24 also distributes estimated positions, so not every displayed position is direct evidence of a contemporaneous transmission.

## Try it without a token

Turn on **Preview synthetic example**, select a marker, open **Flight comparison**, and click **Compare synthetic flight record**. The invented record is labeled synthetic, does not call FR24, and does not alter saved satellite evidence.

## Storage and verification

Comparison results are held only in the current Streamlit session, expire after 30 minutes, and can be cleared with **Clear flight comparison results**. They are not written to the evidence database or included in map exports. The source ZIP contains no token or real provider flight data.

The integration is tested with mocked HTTP responses and synthetic positions, including ambiguity, time mismatch, empty responses, provider errors, estimated sources, and API-token handling. A real authenticated FR24 query has not been run in this development session. Verify it with your entitled API token and independently confirmed imagery/flight records.

Official references:

- [API endpoint documentation](https://fr24api.flightradar24.com/docs/endpoints/overview)
- [Historical position resolution](https://fr24api.flightradar24.com/docs/endpoints/flight-positions-resolution)
- [FAQ and coverage/source limitations](https://fr24api.flightradar24.com/docs/faq)
- [Credits and limits](https://fr24api.flightradar24.com/docs/credit-overview)
- [Storage rules](https://fr24api.flightradar24.com/docs/storage-rules)
