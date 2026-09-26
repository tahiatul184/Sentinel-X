"""Planet SkySat archive collection for the saved user location.

Only already entitled ortho_visual assets are activated/downloaded. No Orders
or Tasking API calls are made. Credentials never enter persisted state.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import math
from pathlib import Path
from urllib.parse import urlparse, quote

API = 'https://api.planet.com/data/v1'
ASSET = 'ortho_visual'
ITEM_TYPE = 'SkySatCollect'


class ProviderError(RuntimeError):
    pass


def location_aoi(cfg):
    if not cfg.get('auto_location_ready'):
        raise ValueError('Save your browser location or coordinates first.')
    lat, lon = float(cfg['auto_latitude']), float(cfg['auto_longitude'])
    radius = float(cfg.get('highres_radius_km', 0.5))
    if not all(math.isfinite(v) for v in (lat, lon, radius)) or not (-79 <= lat <= 83 and -180 <= lon <= 180 and 0.1 <= radius <= 1):
        raise ValueError('Invalid high-resolution location or area size.')
    dy = radius / 111.32
    dx = dy / math.cos(math.radians(lat))
    bbox = [lon-dx, lat-dy, lon+dx, lat+dy]
    if bbox[0] < -180 or bbox[2] > 180:
        raise ValueError('The collection area crosses the date line; reduce its size.')
    ring = [[bbox[0],bbox[1]], [bbox[2],bbox[1]], [bbox[2],bbox[3]], [bbox[0],bbox[3]], [bbox[0],bbox[1]]]
    return bbox, {'type':'Polygon', 'coordinates':[ring]}


def site_key(cfg):
    bbox, _ = location_aoi(cfg)
    return 'location-' + hashlib.sha256(repr(bbox).encode()).hexdigest()[:16]


def search_body(cfg, now=None):
    now = now or datetime.now(timezone.utc)
    _, geometry = location_aoi(cfg)
    days = int(cfg.get('highres_lookback_days', 90))
    if not 1 <= days <= 365:
        raise ValueError('High-resolution lookback must be 1–365 days.')
    cloud = float(cfg.get('highres_max_cloud', 0.2))
    if not math.isfinite(cloud) or not 0 <= cloud <= 1:
        raise ValueError('Cloud threshold must be between 0 and 1.')
    return {'item_types':[ITEM_TYPE], 'filter':{'type':'AndFilter', 'config':[
        {'type':'GeometryFilter','field_name':'geometry','config':geometry},
        {'type':'GeometryFilter','field_name':'geometry','config':{'type':'Point','coordinates':[cfg['auto_longitude'],cfg['auto_latitude']]}},
        {'type':'DateRangeFilter','field_name':'acquired','config':{
            'gte':(now-timedelta(days=days)).isoformat(), 'lte':now.isoformat()}},
        {'type':'RangeFilter','field_name':'cloud_cover','config':{'lte':cloud}},
        {'type':'PermissionFilter','config':['assets:download']},
        {'type':'AssetFilter','config':[ASSET]},
    ]}}


def check_api(response):
    if response.status_code in (401, 403):
        raise ProviderError('Planet credentials or SkySat download entitlement are missing or rejected.')
    if response.status_code == 429:
        raise ProviderError('Planet rate limit reached; the collector will retry later.')
    if response.status_code >= 300:
        raise ProviderError(f'Planet API returned HTTP {response.status_code}.')
    return response


class PlanetArchive:
    def __init__(self, api_key, session=None):
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.session.auth = (api_key, '')

    def request(self, method, url, **kwargs):
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.netloc != 'api.planet.com':
            raise ProviderError('Rejected unexpected Planet API endpoint.')
        return check_api(self.session.request(method, url, timeout=(10,60), allow_redirects=False, **kwargs))

    def search(self, cfg):
        result = self.request('POST', API+'/quick-search', params={'_sort':'acquired desc', '_page_size':50}, json=search_body(cfg)).json()
        # Bound each poll to one page. Newest accessible acquisitions are first.
        return [item for item in result.get('features', [])
                if f'assets.{ASSET}:download' in item.get('_permissions', [])]

    def active_asset(self, item):
        ident = quote(str(item['id']), safe='')
        url = f'{API}/item-types/{ITEM_TYPE}/items/{ident}/assets/'
        asset = self.request('GET', url).json().get(ASSET)
        if not asset:
            return None
        if asset.get('status') == 'active' and asset.get('location'):
            return asset['location']
        if asset.get('status') == 'inactive':
            self.request('GET', asset['_links']['activate'])
        return None


def download_asset(url, destination, *, max_bytes=1024**3, get=None):
    """Use a credential-free request for the signed delivery URL."""
    parsed = urlparse(url)
    host = parsed.hostname or ''
    allowed = any(host == suffix or host.endswith('.'+suffix) for suffix in
                  ('planet.com','amazonaws.com','googleapis.com','blob.core.windows.net'))
    if parsed.scheme != 'https' or not allowed or parsed.username or parsed.password:
        raise ProviderError('Rejected unexpected imagery delivery host.')
    if get is None:
        import requests
        get = requests.get
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.part')
    try:
        with get(url, stream=True, timeout=(15,60), allow_redirects=False) as response:
            if response.status_code != 200:
                raise ProviderError(f'Imagery download returned HTTP {response.status_code}.')
            if int(response.headers.get('Content-Length',0)) > max_bytes:
                raise ProviderError('Scene exceeds the 1 GB download limit.')
            total = 0
            with temp.open('wb') as stream:
                for chunk in response.iter_content(1024*1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ProviderError('Scene exceeds the 1 GB download limit.')
                    stream.write(chunk)
            if not total:
                raise ProviderError('Provider returned an empty image.')
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    return destination


def crop_rgb(source, destination, cfg):
    """Crop without resampling, retaining the delivered pixel grid and CRS."""
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds, transform
    from rasterio.windows import from_bounds, Window
    from rasterio.enums import ColorInterp
    from direct_candidate import _estimate_gsd
    bbox, _ = location_aoi(cfg)
    destination = Path(destination)
    temporary = destination.with_suffix('.partial.tif')
    try:
        with rasterio.open(source) as src:
            if not src.crs or src.count < 3 or src.transform.is_identity:
                raise ValueError('Provider asset is not a georeferenced RGB raster.')
            colors = list(src.colorinterp)
            if not all(c in colors for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
                raise ValueError('Provider asset is missing explicit RGB band metadata.')
            bands = [colors.index(c)+1 for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)]
            spacing = _estimate_gsd(src)
            if spacing is None or not 0 < spacing <= 2:
                raise ValueError('Provider product exceeds the 2 m pixel-spacing limit.')
            xs, ys = transform('EPSG:4326', src.crs, [cfg['auto_longitude']], [cfg['auto_latitude']])
            row, col = src.index(xs[0], ys[0])
            if not (0 <= row < src.height and 0 <= col < src.width):
                raise ValueError('Scene does not cover the saved location.')
            center = src.read(bands, window=Window(col,row,1,1), masked=True)
            if np.ma.getmaskarray(center).any():
                raise ValueError('Scene has no valid RGB pixels at the saved location.')
            bounds = transform_bounds('EPSG:4326', src.crs, *bbox, densify_pts=21)
            win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            clipped = win.intersection(Window(0,0,src.width,src.height))
            data = src.read(bands, window=clipped, masked=True)
            valid = ~np.any(np.ma.getmaskarray(data), axis=0)
            if not valid.any():
                raise ValueError('No valid imagery in the selected area.')
            profile = src.profile.copy()
            profile.update(count=3, width=data.shape[2], height=data.shape[1], transform=src.window_transform(clipped), compress='deflate')
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(temporary,'w',**profile) as dst:
                    dst.write(np.ma.filled(data,0))
                    dst.write_mask(valid.astype('uint8')*255)
                    dst.colorinterp = (ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            full = clipped == win and bool(valid.all())
        temporary.replace(destination)
        return {'product_pixel_spacing_m':spacing, 'valid_fraction':float(valid.mean()),
                'full_aoi_valid':full, 'location_pixel_valid':True,
                'processing_note':'SkySat ortho_visual is a provider-enhanced RGB product. Delivered pixel spacing is not an independent native-resolution measurement.'}
    finally:
        temporary.unlink(missing_ok=True)
