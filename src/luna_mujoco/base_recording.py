"""Freeze Base's renderer/assets and render only archived measured twin frames."""
import hashlib
import base64
import io
import json
import re
from urllib.parse import urljoin, urlparse

from PIL import Image

BASE = '/api/extensions/jodell.urdf-viewer/v1'


def freeze_renderer(robot, directory, scene):
    root = directory / 'videos' / 'base-viewer'
    root.mkdir(parents=True, exist_ok=True)
    index = {}
    origin = urlparse(robot.url)

    def fetch(path):
        url = urljoin(robot.url + '/', path)
        parsed = urlparse(url)
        if (parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc):
            raise ValueError('recording_asset_must_be_published_by_base')
        key = parsed.path + ('?' + parsed.query if parsed.query else '')
        if key in index:
            return
        content, content_type = robot.viewer_asset(url)
        name = hashlib.sha256(key.encode()).hexdigest()
        (root / name).write_bytes(content)
        index[key] = {'path': name, 'content_type': content_type}
        if parsed.path.endswith(('.js', '.mjs')):
            for dependency in re.findall(r'(?:from\s*|import\s*)[\'\"]([^\'\"]+)[\'\"]', content.decode('utf-8')):
                if dependency.startswith(('.', '/')):
                    fetch(urljoin(url, dependency))

    descriptor = robot.manifest()
    fetch(BASE + '/embed-view')
    fetch(BASE + '/embed')
    fetch(descriptor['renderer']['module_url'])
    fetch(descriptor['model']['scene_blob_url'])

    def assets(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('mesh_url', 'geometry_url', 'url') and isinstance(item, str) and item.startswith(('/', robot.url)):
                    fetch(item)
                elif isinstance(item, (dict, list)):
                    assets(item)
        elif isinstance(value, list):
            for item in value:
                assets(item)
    assets(scene.get('environment', {}))
    (root / 'resources.json').write_text(json.dumps(index, ensure_ascii=False), encoding='utf-8')
    return {'base_url': robot.url, 'renderer': 'jodell.urdf-viewer', 'assets': 'videos/base-viewer/resources.json'}


def render_recorded_scenes(directory, recording, packets):
    from playwright.sync_api import sync_playwright

    root = directory / 'videos' / 'base-viewer'
    index = json.loads((root / 'resources.json').read_text(encoding='utf-8'))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--use-gl=angle', '--use-angle=swiftshader',
            '--enable-unsafe-swiftshader', '--disable-dev-shm-usage', '--num-raster-threads=1'])
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 720}, device_scale_factor=1)

            def archived(route):
                parsed = urlparse(route.request.url)
                key = parsed.path + ('?' + parsed.query if parsed.query else '')
                item = index.get(key) or index.get(parsed.path)
                if item is None:
                    route.abort()
                else:
                    route.fulfill(body=(root / item['path']).read_bytes(), content_type=item['content_type'])
            page.route('**/*', archived)
            page.goto(recording['base_renderer']['base_url'] + BASE + '/embed-view?recording=1&appearance=neutral')
            page.wait_for_function('window.baseTwinReady || window.baseTwinError')
            error = page.evaluate('window.baseTwinError')
            if error:
                raise RuntimeError(error)
            loaded_scene = False
            for timestamp, packet in packets:
                if packet is None:
                    yield timestamp, Image.new('RGB', (1280, 720))
                    continue
                page.evaluate('scene => window.setRecordedScene(scene)', packet['base_scene'])
                if not loaded_scene:
                    page.wait_for_load_state('networkidle')
                    loaded_scene = True
                png = page.evaluate('window.renderRecordedFrame()').split(',', 1)[1]
                with Image.open(io.BytesIO(base64.b64decode(png, validate=True))) as image:
                    yield timestamp, image.convert('RGB')
        finally:
            browser.close()
