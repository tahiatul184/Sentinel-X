import unittest

from weather_service import fetch_weather, summarize_weather


class WeatherServiceTests(unittest.TestCase):
    def payload(self):
        times=[f'2026-09-12T{h:02d}:00' for h in range(24)]
        return {
            'latitude':23.8,'longitude':90.4,'timezone':'UTC',
            'current':{'time':'2026-09-12T00:00','temperature_2m':31,'relative_humidity_2m':80,
                       'precipitation':2,'rain':2,'weather_code':61,'cloud_cover':80,
                       'wind_speed_10m':12,'wind_gusts_10m':25,'visibility':7000},
            'hourly':{
                'time':times,'temperature_2m':[31]*24,'precipitation':[1]*24,'rain':[1]*24,
                'weather_code':[95]+[61]*23,'cloud_cover':[80]*24,'visibility':[6000]*24,
                'wind_speed_10m':[15]*24,'wind_gusts_10m':[30]*24,'cape':[900]*24,
            }
        }

    def test_weather_summary_has_operational_context_fields(self):
        result=summarize_weather(self.payload(), fetched_at='2026-09-12T00:00:00+00:00')
        s=result['summary']
        self.assertEqual(s['rain_next_6h_mm'],6.0)
        self.assertEqual(s['rain_next_24h_mm'],24.0)
        self.assertEqual(s['min_visibility_next_6h_m'],6000.0)
        self.assertEqual(s['thunderstorm_hours_next_24h'],1)
        self.assertIn('not certified aviation weather',result['disclaimer'])

    def test_fetch_is_keyless_and_injectable(self):
        class Response:
            def raise_for_status(self): pass
            def json(self): return WeatherServiceTests().payload()
        calls={}
        def get(url,params=None,timeout=None):
            calls['url']=url;calls['params']=params;return Response()
        result=fetch_weather(23.8,90.4,get=get)
        self.assertIn('open-meteo.com',calls['url'])
        self.assertIn('visibility',calls['params']['current'])
        self.assertEqual(result['provider'],'Open-Meteo')


if __name__=='__main__':unittest.main()
