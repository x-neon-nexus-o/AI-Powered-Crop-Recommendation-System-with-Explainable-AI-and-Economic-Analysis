// CropAI Service Worker — v1
var CACHE_NAME = 'cropai-v1';
var OFFLINE_URL = '/offline';

// App shell: files pre-cached on install
var APP_SHELL = [
    '/',
    '/offline',
    '/static/css/style.css',
    '/static/js/main.js',
    '/static/js/dashboard.js',
    '/static/js/form_validation.js',
    '/static/manifest.json',
    '/static/images/icon-192.png',
    '/static/images/icon-512.png'
];

// CDN resources to cache on first use
var CDN_HOSTS = [
    'cdn.jsdelivr.net',
    'fonts.googleapis.com',
    'fonts.gstatic.com'
];

// ==================== INSTALL ====================
self.addEventListener('install', function (event) {
    event.waitUntil(
        caches.open(CACHE_NAME).then(function (cache) {
            console.log('[SW] Pre-caching app shell');
            return cache.addAll(APP_SHELL);
        })
    );
    // Activate immediately without waiting for old SW to finish
    self.skipWaiting();
});

// ==================== ACTIVATE ====================
self.addEventListener('activate', function (event) {
    event.waitUntil(
        caches.keys().then(function (names) {
            return Promise.all(
                names.filter(function (name) {
                    return name !== CACHE_NAME;
                }).map(function (name) {
                    console.log('[SW] Deleting old cache:', name);
                    return caches.delete(name);
                })
            );
        })
    );
    // Take control of all pages immediately
    self.clients.claim();
});

// ==================== FETCH ====================
self.addEventListener('fetch', function (event) {
    var url = new URL(event.request.url);

    // Skip non-GET requests (POST predictions, API calls, etc.)
    if (event.request.method !== 'GET') {
        return;
    }

    // Skip API endpoints — always need live data
    if (url.pathname.startsWith('/api/')) {
        return;
    }

    // Skip PDF export — needs live session
    if (url.pathname === '/export-pdf') {
        return;
    }

    // CDN resources — cache-first
    var isCDN = CDN_HOSTS.some(function (host) {
        return url.hostname === host;
    });
    if (isCDN) {
        event.respondWith(cacheFirst(event.request));
        return;
    }

    // Static assets (our own CSS, JS, images) — cache-first
    if (url.pathname.startsWith('/static/')) {
        event.respondWith(cacheFirst(event.request));
        return;
    }

    // HTML pages — network-first with offline fallback
    event.respondWith(networkFirst(event.request));
});

// ==================== STRATEGIES ====================

// Cache-first: return cached version, fall back to network
function cacheFirst(request) {
    return caches.match(request).then(function (cached) {
        if (cached) {
            return cached;
        }
        return fetch(request).then(function (response) {
            // Only cache valid responses
            if (!response || response.status !== 200) {
                return response;
            }
            var clone = response.clone();
            caches.open(CACHE_NAME).then(function (cache) {
                cache.put(request, clone);
            });
            return response;
        });
    });
}

// Network-first: try network, fall back to cache, then offline page
function networkFirst(request) {
    return fetch(request).then(function (response) {
        // Cache successful HTML responses for offline use
        if (response && response.status === 200) {
            var clone = response.clone();
            caches.open(CACHE_NAME).then(function (cache) {
                cache.put(request, clone);
            });
        }
        return response;
    }).catch(function () {
        // Network failed — try cache
        return caches.match(request).then(function (cached) {
            if (cached) {
                return cached;
            }
            // Nothing in cache — show offline page
            return caches.match(OFFLINE_URL);
        });
    });
}
