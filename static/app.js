(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const storage = {
    get(key, fallback = null) {
      try {
        const value = localStorage.getItem(`galleryflow:${key}`);
        return value === null ? fallback : JSON.parse(value);
      } catch (_) { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(`galleryflow:${key}`, JSON.stringify(value)); } catch (_) { /* private mode */ }
    }
  };

  const state = {
    view: 'discover',
    browseMode: 'url',
    galleries: [],
    page: 1,
    total: 0,
    sourceUrl: '',
    nextUrl: '',
    query: '',
    profiles: [],
    activeProfile: storage.get('active-profile', ''),
    profileChangeRequest: 0,
    historyRequest: 0,
    historyUrls: new Set(),
    jobs: [],
    jobFilter: 'all',
    gallery: null,
    selectedImages: new Set(),
    finderFeedbackGallerySelection: new Set(),
    finderFeedbackGalleryDirty: false,
    finderFeedbackGallerySaving: false,
    galleryReviewBusy: false,
    galleryDetailRequest: 0,
    galleryDetailCache: new Map(),
    galleryDetailPrefetches: new Map(),
    galleryDetailPrefetchQueue: [],
    galleryDetailPrefetchQueued: new Map(),
    galleryDetailPrefetchActive: 0,
    galleryDetailGlobalRevision: 0,
    galleryDetailRevisions: new Map(),
    galleryPreviewWarmups: new Map(),
    galleryPreviewWarmGeneration: 0,
    mediaLoadQueue: [],
    mediaLoadActive: new Set(),
    mediaLoadObserved: new Set(),
    mediaLoadByImage: new WeakMap(),
    mediaLoadDrainScheduled: false,
    mediaLoadLazyObserver: null,
    mediaLoadDomObserver: null,
    mediaLoadPeak: 0,
    mediaLoadCancelled: 0,
    galleryForegroundPeak: 0,
    galleryPreviewWarmPeak: 0,
    galleryNavigationRequest: 0,
    galleryMode: 'download',
    poseSelectedImages: new Set(),
    poseTags: [],
    poseDraft: { revision: 0, controls: { solo: '', couple: '', group: '' }, targets: [] },
    poseLoadedKey: '',
    poseLoading: false,
    poseDirty: false,
    poseSaving: false,
    poseSaveTimer: null,
    poseSavePromise: null,
    poseMutation: 0,
    poseApplying: false,
    poseApplyRequest: 0,
    poseExporting: false,
    poseExportRequest: 0,
    poseAssignment: 'target',
    galleryContext: null,
    lightboxIndex: -1,
    lightboxZoomed: false,
    lightboxTrigger: null,
    lightboxLoadToken: 0,
    filters: storage.get('filters', { showSaved: true, showIgnored: false }),
    density: storage.get('density', 'comfortable'),
    settings: {},
    loadingGalleries: false,
    loadingDetail: false,
    serverOnline: false,
    eventSource: null,
    eventConnected: false,
    queueTimer: null,
    jobEventTimer: null,
    finderEventTimer: null,
    finderPollTimer: null,
    finderScanRequest: 0,
    finderScanMutationEpoch: 0,
    healthTimer: null,
    requestController: null,
    galleryObserver: null,
    finderFolders: [],
    finderTags: [],
    finderStatus: null,
    finderCorpus: null,
    finderCorpusSupported: null,
    finderJoytagIndexSupported: null,
    finderJoytagIndexBusy: false,
    finderJoytagIndexRequest: 0,
    finderJoytagIndexTimer: null,
    finderFeedback: null,
    finderFeedbackSupported: null,
    finderFeedbackLoading: false,
    finderFeedbackBusy: false,
    finderFeedbackMutations: 0,
    finderResultMutationEpoch: 0,
    finderResultRefreshTimer: null,
    finderFeedbackError: '',
    finderFeedbackRequest: 0,
    finderFeedbackTimer: null,
    finderScans: [],
    finderScan: null,
    finderScanId: storage.get('finder-scan', ''),
    finderResults: [],
    finderReviewCounts: null,
    finderReview: 'pending',
    finderResultPage: 1,
    finderResultPageCount: 1,
    finderResultPageSize: 24,
    finderResultTotal: 0,
    finderResultLoading: false,
    finderResultRequest: 0,
    finderResultThresholdTimer: null,
    finderMode: storage.get('finder-mode', 'pose') === 'joytag' ? 'joytag' : 'pose',
    finderReferenceAnalysis: null,
    finderReferenceAnalysisSource: '',
    finderReferenceAnalysisLoading: false,
    finderReferenceAnalysisError: '',
    finderReferenceAnalysisRequest: 0,
    finderJoytagSelectedTag: '',
    finderJoytagRequiredTags: [],
    finderJoytagExcludedTags: [],
    finderJoytagTagFilter: '',
    finderJoytagThreshold: Math.max(0.05, Math.min(0.95, Number(storage.get('finder-joytag-threshold', 0.4)) || 0.4)),
    finderJoytagRejectThreshold: Math.max(0.05, Math.min(0.95, Number(storage.get('finder-joytag-reject-threshold', 0.4)) || 0.4)),
    finderJoytagAutoPoseLabel: '',
    finderLoaded: false,
    finderLoading: false,
    finderBusy: false,
    finderExtendPages: 5,
    finderContinuePages: 5,
    sortFolders: [],
    sortProfiles: [],
    sortSession: null,
    sortSessionId: storage.get('sort-session', ''),
    sortLoaded: false,
    sortLoading: false,
    sortBusy: false
  };

  const POSE_ROLES = ['solo', 'couple', 'group'];
  const FINDER_TERMINAL_STATES = ['completed', 'completed_with_errors', 'complete', 'done', 'failed', 'cancelled', 'canceled'];
  const FINDER_MAX_PAGES = 500;
  const FINDER_RESULTS_PAGE_SIZE = 24;
  const FINDER_JOYTAG_QUERY_LIMIT = 16;
  const FINDER_JOYTAG_CATALOG_RENDER_LIMIT = 80;
  const GALLERY_DETAIL_CACHE_LIMIT = 8;
  const GALLERY_DETAIL_CACHE_TTL = 60_000;
  const GALLERY_DETAIL_PREFETCH_CONCURRENCY = 1;
  const MEDIA_LOAD_CONCURRENCY = 3;
  const GALLERY_PREVIEW_WARM_CONCURRENCY = 2;
  const GALLERY_PREVIEW_WARM_LIMIT = 1;
  const GALLERY_PREVIEW_CACHE_LIMIT = 24;
  const FINDER_RANKING_VERSION = 'pose-precision-v2';
  const FINDER_JOYTAG_RANKING_VERSION = 'joytag-v1';
  const poseRoleLabel = role => ({ solo: 'Solo', couple: 'Couple', group: 'Group' }[role] || 'Solo');

  class ApiError extends Error {
    constructor(message, status = 0, data = null) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
      this.data = data;
    }
  }

  async function api(path, options = {}) {
    const request = {
      method: options.method || 'GET',
      headers: { Accept: 'application/json', ...(options.headers || {}) },
      signal: options.signal
    };
    if (options.body !== undefined) {
      request.headers['Content-Type'] = 'application/json';
      request.body = JSON.stringify(options.body);
    }

    let response;
    try {
      response = await fetch(path, request);
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      setServerState(false, 'Unavailable');
      throw new ApiError('Could not reach the server. Check that GalleryFlow is running.');
    }

    let data = null;
    if (response.status !== 204) {
      const contentType = response.headers.get('content-type') || '';
      try {
        data = contentType.includes('application/json') ? await response.json() : await response.text();
      } catch (_) { data = null; }
    }
    if (!response.ok) {
      const message = data?.detail || data?.message || data?.error || (typeof data === 'string' && data) || `Request failed (${response.status})`;
      throw new ApiError(message, response.status, data);
    }
    return data;
  }

  function withParams(path, params) {
    const url = new URL(path, window.location.origin);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== '' && value !== null && value !== undefined) url.searchParams.set(key, String(value));
    });
    return `${url.pathname}${url.search}`;
  }

  function apiItems(data, alternate = '') {
    if (Array.isArray(data)) return data;
    if (Array.isArray(data?.items)) return data.items;
    if (alternate && Array.isArray(data?.[alternate])) return data[alternate];
    return [];
  }

  function safeUrl(value) {
    if (!value) return '';
    if (/^data:image\/svg\+xml;base64,[a-z0-9+/=]+$/i.test(value) && value.length <= 200000) {
      return value;
    }
    if (/^data:image\/(?:jpeg|png|webp);base64,[a-z0-9+/=]+$/i.test(value) && value.length <= 500000) {
      return value;
    }
    try {
      const parsed = new URL(value, window.location.origin);
      return ['http:', 'https:'].includes(parsed.protocol) ? parsed.href : '';
    } catch (_) { return ''; }
  }

  function displayHost(value) {
    try { return new URL(value).hostname.replace(/^www\./, ''); } catch (_) { return 'Gallery source'; }
  }

  function formatNumber(value) {
    const number = Number(value || 0);
    return new Intl.NumberFormat(undefined, { notation: number > 9999 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(number);
  }

  function formatBytes(value) {
    const bytes = Number(value || 0);
    if (!bytes) return '—';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / (1024 ** index)).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
  }

  function relativeTime(value) {
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '';
    const seconds = Math.round((date.getTime() - Date.now()) / 1000);
    const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
    const ranges = [[60, 'second'], [60, 'minute'], [24, 'hour'], [7, 'day'], [4.35, 'week'], [12, 'month'], [Infinity, 'year']];
    let amount = seconds;
    for (const [size, unit] of ranges) {
      if (Math.abs(amount) < size) return formatter.format(Math.round(amount), unit);
      amount /= size;
    }
    return '';
  }

  function normalizeGallery(item, { useHistory = true } = {}) {
    const gallery = { ...item };
    const serverState = String(item.state || '').toLowerCase();
    gallery.id = item.id ?? item.gallery_id ?? item.url;
    gallery.url = item.url || item.gallery_url || '';
    gallery.title = item.title || item.name || 'Untitled gallery';
    gallery.thumbnailUrl = item.thumbnail_url || item.thumbnail || item.preview_url || '';
    gallery.imageCount = Number(item.total_images || item.image_count || item.count || (Array.isArray(item.images) ? item.images.length : 0));
    gallery.downloadedImages = Number(item.downloaded_images || 0);
    gallery.serverState = serverState || 'new';
    gallery.saved = Boolean(
      item.saved
      || item.downloaded
      || serverState === 'complete'
      || serverState === 'saved'
      || (useHistory && state.historyUrls.has(normalizeHistoryUrl(gallery.url)))
    );
    gallery.partial = Boolean(item.partial || serverState === 'partial' || (gallery.downloadedImages && !gallery.saved));
    gallery.ignored = Boolean(item.ignored || serverState === 'ignored');
    gallery.queued = serverState === 'queued' || state.jobs.some(job => !isTerminalJob(job) && (
      String(job.galleryId) === String(gallery.id) ||
      (job.galleryUrl && normalizeHistoryUrl(job.galleryUrl) === normalizeHistoryUrl(gallery.url))
    ));
    return gallery;
  }

  function normalizeDetail(item) {
    const gallery = normalizeGallery(item || {}, { useHistory: false });
    gallery.images = apiItems(item?.images || []).map((image, index) => {
      if (typeof image === 'string') return { url: image, previewUrl: image, fullUrl: image, filename: `Image ${index + 1}` };
      return {
        ...image,
        url: image.url || image.image_url || image.src || '',
        previewUrl: image.preview_url || image.thumbnail_url || image.url || '',
        fullUrl: image.full_url || image.fullUrl || image.preview_url || image.url || '',
        filename: image.filename || image.name || `Image ${index + 1}`
      };
    }).filter(image => image.url);
    gallery.imageCount = gallery.images.length || gallery.imageCount;
    gallery.downloadedImages = Number(item?.downloaded_images ?? gallery.images.filter(image => image.downloaded).length);
    gallery.partial = Boolean(item?.partial || (gallery.downloadedImages && !gallery.saved));
    return gallery;
  }

  function normalizeProfile(profile) {
    if (typeof profile === 'string') return { name: profile, count: 0, path: '' };
    return {
      ...profile,
      name: profile.name || profile.id || profile.profile || 'Default',
      count: Number(profile.count ?? profile.gallery_count ?? profile.download_count ?? profile.downloads ?? 0),
      directory: profile.directory || profile.folder || profile.name || '',
      path: profile.path || ''
    };
  }

  function normalizeJob(item) {
    let status = String(item.status || item.state || 'queued').toLowerCase();
    if (['done', 'finished', 'success'].includes(status)) status = 'completed';
    if (['pending', 'waiting'].includes(status)) status = 'queued';
    if (['error'].includes(status)) status = 'failed';
    const total = Number(item.total_images ?? item.total ?? item.image_count ?? item.urls?.length ?? 0);
    const complete = Number(item.completed_images ?? item.completed ?? item.downloaded_images ?? item.done ?? 0);
    let progress = Number(item.progress ?? (total ? (complete / total) * 100 : status === 'completed' ? 100 : 0));
    if (progress > 0 && progress <= 1 && item.progress !== undefined) progress *= 100;
    progress = Math.max(0, Math.min(100, progress));
    const poseRevision = item.pose_revision ?? item.poseRevision ?? null;
    return {
      ...item,
      id: item.id ?? item.job_id,
      galleryId: item.gallery_id ?? item.galleryId,
      galleryUrl: item.gallery_url || item.url || '',
      title: item.title || item.gallery_title || item.folder_name || 'Untitled gallery',
      thumbnailUrl: item.thumbnail_url || item.thumbnail || '',
      profile: item.profile || item.profile_name || 'Default',
      status,
      total,
      complete,
      progress,
      bytes: Number(item.bytes_downloaded ?? item.downloaded_bytes ?? 0),
      speed: Number(item.speed ?? item.bytes_per_second ?? 0),
      kind: item.kind === 'pose_export' ? 'pose_export' : 'download',
      pairCount: Number(item.pair_count ?? 0),
      poseRevision: poseRevision === null ? null : Number(poseRevision),
      error: item.error || item.message || '',
      createdAt: item.created_at || item.started_at || item.date_added || ''
    };
  }

  function normalizeHistoryUrl(value) {
    return String(value || '').replace(/^https?:\/\//, '').replace(/^www\./, '').replace(/\/$/, '').toLowerCase();
  }

  function isTerminalJob(job) { return ['completed', 'completed_with_errors', 'failed', 'cancelled', 'canceled'].includes(job.status); }
  function isActiveJob(job) { return ['starting', 'downloading', 'running', 'active', 'canceling'].includes(job.status); }

  function setServerState(online, detail = '') {
    state.serverOnline = online;
    const pill = $('#server-pill');
    pill.classList.toggle('is-online', online);
    pill.classList.toggle('is-offline', !online);
    $('#server-label').textContent = online ? 'Server online' : 'Server offline';
    $('#server-detail').textContent = detail || (online ? window.location.host : 'Unable to connect');
    $('#about-server').textContent = online ? detail || 'Online' : 'Offline';
  }

  function toast(title, message = '', type = 'success', timeout = 4200) {
    const item = document.createElement('div');
    item.className = `toast ${type}`;
    item.setAttribute('role', type === 'error' ? 'alert' : 'status');
    const icon = type === 'error' ? 'i-info' : type === 'info' ? 'i-info' : 'i-check';
    item.innerHTML = `<span class="toast-icon"><svg><use href="#${icon}"></use></svg></span><span class="toast-copy"><strong></strong><span></span></span><button type="button" aria-label="Dismiss"><svg><use href="#i-close"></use></svg></button>`;
    $('strong', item).textContent = title;
    $('.toast-copy span', item).textContent = message;
    const remove = () => {
      item.classList.add('is-leaving');
      window.setTimeout(() => item.remove(), 210);
    };
    $('button', item).addEventListener('click', remove);
    $('#toast-region').append(item);
    if (timeout) window.setTimeout(remove, timeout);
    return item;
  }

  function announce(message) { $('#aria-status').textContent = message; }

  function showNotice(message) {
    const notice = $('#discover-notice');
    $('span', notice).textContent = message;
    notice.hidden = false;
  }

  function hideNotice() { $('#discover-notice').hidden = true; }

  function errorMessage(error) {
    return error instanceof ApiError ? error.message : 'Something went wrong. Please try again.';
  }

  function setButtonBusy(button, busy, label = 'Working…') {
    if (!button) return;
    if (busy) {
      button.dataset.originalHtml = button.innerHTML;
      button.disabled = true;
      button.innerHTML = `<svg class="busy-icon"><use href="#i-refresh"></use></svg><span>${label}</span>`;
      $('.busy-icon', button)?.classList.add('spin');
    } else if (button.dataset.originalHtml) {
      button.innerHTML = button.dataset.originalHtml;
      button.disabled = false;
      delete button.dataset.originalHtml;
    } else {
      button.disabled = false;
    }
  }

  async function checkHealth(quiet = true) {
    try {
      const data = await api('/api/health');
      const version = data?.version ? `v${data.version}` : data?.status || 'Ready';
      setServerState(true, version);
    } catch (error) {
      setServerState(false);
      if (!quiet) toast('Server unavailable', errorMessage(error), 'error');
    }
  }

  function connectEvents() {
    if (!('EventSource' in window) || state.eventSource) return;
    const source = new EventSource('/api/events');
    state.eventSource = source;
    source.addEventListener('open', () => {
      state.eventConnected = true;
      scheduleJobPoll();
    });
    source.addEventListener('error', () => {
      state.eventConnected = false;
      scheduleJobPoll(5000);
    });
    source.addEventListener('job', () => {
      if (state.jobEventTimer !== null) return;
      state.jobEventTimer = window.setTimeout(() => {
        state.jobEventTimer = null;
        loadJobs({ quiet: true });
      }, 200);
    });
    source.addEventListener('gallery', event => {
      try {
        const change = JSON.parse(event.data || '{}');
        const gallery = state.galleries.find(item => normalizeHistoryUrl(item.url) === normalizeHistoryUrl(change.url));
        if (gallery && typeof change.ignored === 'boolean') {
          invalidateGalleryDetailCache(gallery.id);
          gallery.ignored = change.ignored;
          renderGalleries();
        } else {
          invalidateGalleryDetailByUrl(change.url);
          if (state.view === 'discover') loadGalleries({ quiet: true });
        }
      } catch (_) { /* the next refresh will reconcile state */ }
    });
    const refreshFinderFromEvent = event => {
      if (!state.finderScanId || state.finderEventTimer !== null) return;
      try {
        const change = JSON.parse(event.data || '{}');
        const changedId = change.scan_id ?? change.id ?? change.scan?.id;
        if (changedId !== undefined && String(changedId) !== String(state.finderScanId)) return;
      } catch (_) { /* refresh the active scan when an event has no JSON payload */ }
      state.finderEventTimer = window.setTimeout(() => {
        state.finderEventTimer = null;
        loadFinderScan({ quiet: true });
      }, 180);
    };
    source.addEventListener('finder', refreshFinderFromEvent);
    source.addEventListener('finder_scan', refreshFinderFromEvent);
    const refreshFinderCorpusFromEvent = () => loadFinderCorpus({ quiet: true });
    source.addEventListener('finder_corpus', refreshFinderCorpusFromEvent);
    source.addEventListener('finder_index', refreshFinderCorpusFromEvent);
    source.addEventListener('finder_joytag_index', event => {
      try {
        applyFinderJoytagIndexPayload(JSON.parse(event.data || '{}'));
      } catch (_) {
        loadFinderJoytagIndex({ quiet: true, force: true });
      }
    });
    source.addEventListener('settings', () => loadSettings());
  }

  async function loadSettings() {
    try {
      const data = await api('/api/settings');
      state.settings = data?.settings || data || {};
      applySettingsToForm();
      renderProfiles();
      if (storage.get('filters') === null) {
        state.filters.showIgnored = !(state.settings.hide_ignored ?? true);
        state.filters.showSaved = state.settings.show_saved ?? true;
        syncFilterControls();
      }
    } catch (error) {
      if (error.status !== 404) toast('Could not load settings', errorMessage(error), 'error');
    }
  }

  function applySettingsToForm() {
    const settings = state.settings;
    $('#setting-workers').value = settings.image_workers ?? settings.max_workers ?? 6;
    $('#setting-job-workers').value = settings.job_workers ?? 2;
    $('#setting-timeout').value = settings.request_timeout ?? settings.timeout ?? 30;
    $('#setting-density').value = state.density;
    $('#setting-hide-ignored').checked = !state.filters.showIgnored;
    $('#setting-show-saved').checked = state.filters.showSaved;
    $('#setting-root').value = settings.download_root ?? settings.base_folder ?? '';
    $('#setting-sort-root').value = settings.sort_root ?? settings.download_root ?? settings.base_folder ?? '';
    $('#root-prefix').textContent = `${settings.download_root || settings.base_folder || 'downloads'}/`;
  }

  async function saveSettings(event) {
    event.preventDefault();
    const button = $('#save-settings');
    const payload = {
      image_workers: Number($('#setting-workers').value),
      job_workers: Number($('#setting-job-workers').value),
      request_timeout: Number($('#setting-timeout').value),
    };
    setButtonBusy(button, true, 'Saving…');
    try {
      const data = await api('/api/settings', { method: 'PATCH', body: payload });
      state.settings = data?.settings || data || payload;
      state.filters = {
        showIgnored: !$('#setting-hide-ignored').checked,
        showSaved: $('#setting-show-saved').checked
      };
      state.density = $('#setting-density').value;
      storage.set('filters', state.filters);
      storage.set('density', state.density);
      syncFilterControls();
      $('#settings-status').textContent = 'All changes saved.';
      toast('Settings saved', data?.restart_required ? 'Gallery-job concurrency will apply after a server restart.' : 'Your server preferences are up to date.');
      applySettingsToForm();
      renderProfiles();
      loadGalleries({ quiet: true });
    } catch (error) {
      toast('Could not save settings', errorMessage(error), 'error');
    } finally { setButtonBusy(button, false); }
  }

  function commitActiveProfile(name) {
    const next = String(name || '');
    if (next === state.activeProfile) return false;
    state.activeProfile = next;
    storage.set('active-profile', next);
    state.profileChangeRequest += 1;
    state.galleryDetailRequest += 1;
    invalidateGalleryDetailCache(null, { abortForeground: true });
    return true;
  }

  async function loadProfiles({ quiet = false } = {}) {
    try {
      const previousProfile = state.activeProfile;
      const data = await api('/api/profiles');
      state.profiles = apiItems(data, 'profiles').map(normalizeProfile);
      if (state.profiles.length && !state.profiles.some(profile => profile.name === state.activeProfile)) {
        commitActiveProfile(data?.default_profile || state.settings.default_profile || state.profiles[0].name);
      }
      if (!state.profiles.length) commitActiveProfile('');
      renderProfileSelectors();
      renderProfiles();
      if (
        previousProfile !== state.activeProfile
        && $('#gallery-modal').open
        && state.gallery
      ) {
        await loadHistory();
        await refreshOpenGalleryDetail(state.activeProfile);
      }
    } catch (error) {
      state.profiles = [];
      renderProfileSelectors();
      renderProfiles();
      if (!quiet) toast('Could not load profiles', errorMessage(error), 'error');
    }
  }

  function renderProfileSelectors() {
    const selectors = [$('#active-profile'), $('#modal-profile-select')];
    selectors.forEach(select => {
      const previous = select.value || state.activeProfile;
      select.replaceChildren();
      if (!state.profiles.length) {
        const option = new Option('No profiles', '');
        option.disabled = true;
        select.add(option);
        return;
      }
      state.profiles.forEach(profile => select.add(new Option(profile.name, profile.name)));
      select.value = state.profiles.some(profile => profile.name === previous) ? previous : state.activeProfile;
    });
  }

  async function selectProfile(name, reload = true) {
    if (!name || name === state.activeProfile) return;
    if ($('#gallery-modal').open && state.poseLoadedKey) {
      await flushPoseDraft();
      if (state.poseDirty) {
        renderProfileSelectors();
        toast('Profile not changed', 'Save the current pose draft before changing its destination.', 'error');
        return;
      }
    }
    commitActiveProfile(name);
    const profileRequest = state.profileChangeRequest;
    renderProfileSelectors();
    $('#active-profile').value = name;
    $('#modal-profile-select').value = name;
    renderProfiles();
    const galleryOpen = $('#gallery-modal').open && state.gallery;
    if (galleryOpen) {
      state.poseLoadedKey = '';
      state.poseDraft = { revision: 0, controls: { solo: '', couple: '', group: '' }, targets: [] };
      state.poseSelectedImages = new Set();
      renderImages();
    }
    await loadHistory(name);
    if (
      profileRequest !== state.profileChangeRequest
      || name !== state.activeProfile
    ) return;
    if (galleryOpen) {
      await refreshOpenGalleryDetail(name);
      if (
        profileRequest !== state.profileChangeRequest
        || name !== state.activeProfile
      ) return;
      if (state.galleryMode === 'pose') loadPoseWorkspace();
    }
    if (reload) {
      await loadGalleries({ quiet: true });
      toast('Profile changed', `New downloads will be saved to “${name}”.`, 'info');
    }
  }

  async function loadHistory(profile = state.activeProfile) {
    const requestedProfile = String(profile || '');
    const historyRequest = ++state.historyRequest;
    if (!requestedProfile) {
      if (historyRequest === state.historyRequest && !state.activeProfile) {
        state.historyUrls = new Set();
      }
      return;
    }
    try {
      const data = await api(withParams('/api/history', { profile: requestedProfile }));
      if (
        historyRequest !== state.historyRequest
        || requestedProfile !== state.activeProfile
      ) return;
      const entries = apiItems(data, 'downloads');
      state.historyUrls = new Set(entries.map(entry => normalizeHistoryUrl(typeof entry === 'string' ? entry : entry.url || entry.gallery_url)));
    } catch (_) {
      if (
        historyRequest === state.historyRequest
        && requestedProfile === state.activeProfile
      ) state.historyUrls = new Set();
    }
  }

  function renderProfiles() {
    const grid = $('#profile-grid');
    grid.replaceChildren();
    $('#profiles-empty').hidden = Boolean(state.profiles.length);
    state.profiles.forEach(profile => {
      const fragment = $('#profile-card-template').content.cloneNode(true);
      const card = $('.profile-card', fragment);
      card.dataset.profile = profile.name;
      card.classList.toggle('is-active', profile.name === state.activeProfile);
      $('h2', card).textContent = profile.name;
      $('.profile-path', card).textContent = profile.path || `${state.settings.download_root || state.settings.base_folder || 'downloads'}/${profile.directory || profile.name}`;
      $('.profile-count', card).textContent = formatNumber(profile.count);
      const badge = $('.default-badge', card);
      badge.hidden = profile.name !== state.activeProfile;
      badge.textContent = 'Selected';
      const deleteAction = $('[data-profile-action="delete"]', card);
      if (profile.name.toLowerCase() === 'default') deleteAction.hidden = true;
      const useButton = $('.profile-use', card);
      if (profile.name === state.activeProfile) {
        useButton.textContent = 'Currently selected';
        useButton.disabled = true;
      }
      grid.append(fragment);
    });
  }

  function openProfileModal(profile = null) {
    $('#profile-modal-title').textContent = profile ? 'Rename profile' : 'New profile';
    $('#profile-id').value = profile?.name || '';
    $('#profile-name').value = profile?.name || '';
    $('#profile-folder').value = profile?.directory || (profile?.name ? safeFolderPreview(profile.name) : '');
    $('#profile-default').checked = profile?.name === state.activeProfile;
    $('#profile-modal').showModal();
    requestAnimationFrame(() => $('#profile-name').focus());
  }

  function safeFolderPreview(value) {
    return value.replace(/[^a-z0-9 _.-]+/gi, '').replace(/^[ .]+|[ .]+$/g, '').replace(/\s+/g, '_').slice(0, 120);
  }

  async function saveProfile(event) {
    event.preventDefault();
    const oldName = $('#profile-id').value;
    const name = $('#profile-name').value.trim();
    if (!name) return;
    const button = $('#save-profile');
    setButtonBusy(button, true, 'Saving…');
    try {
      if (oldName) {
        await api(`/api/profiles/${encodeURIComponent(oldName)}`, { method: 'PATCH', body: { new_name: name } });
        if (state.activeProfile === oldName) {
          commitActiveProfile(name);
        }
        toast('Profile renamed', `“${oldName}” is now “${name}”.`);
      } else {
        await api('/api/profiles', { method: 'POST', body: { name } });
        toast('Profile created', `“${name}” is ready for downloads.`);
      }
      if ($('#profile-default').checked || !state.activeProfile) {
        commitActiveProfile(name);
      }
      $('#profile-modal').close();
      await loadProfiles();
      if ($('#gallery-modal').open && state.gallery) {
        await loadHistory();
        await refreshOpenGalleryDetail(state.activeProfile);
      }
    } catch (error) {
      toast(oldName ? 'Could not rename profile' : 'Could not create profile', errorMessage(error), 'error');
    } finally { setButtonBusy(button, false); }
  }

  async function deleteProfile(profile) {
    if (!window.confirm(`Delete the “${profile.name}” profile? Existing files will not be removed.`)) return;
    try {
      await api(`/api/profiles/${encodeURIComponent(profile.name)}`, { method: 'DELETE' });
      if (state.activeProfile === profile.name) commitActiveProfile('');
      toast('Profile deleted', `“${profile.name}” was removed.`, 'info');
      await loadProfiles();
      await Promise.all([loadHistory(), loadGalleries({ quiet: true })]);
      if ($('#gallery-modal').open && state.gallery) {
        await refreshOpenGalleryDetail(state.activeProfile);
      }
    } catch (error) { toast('Could not delete profile', errorMessage(error), 'error'); }
  }

  function galleryQuery({ url = state.sourceUrl, page = 1 } = {}) {
    return withParams('/api/galleries', {
      url,
      q: state.query,
      page,
      profile: state.activeProfile,
      show_saved: state.filters.showSaved,
      show_ignored: state.filters.showIgnored
    });
  }

  async function loadGalleries({ quiet = false, append = false } = {}) {
    if (state.loadingGalleries || (append && !state.nextUrl)) return;
    if (state.requestController) state.requestController.abort();
    state.requestController = new AbortController();
    state.loadingGalleries = true;
    const requestedPage = append ? state.page + 1 : 1;
    const requestedUrl = append ? state.nextUrl : state.sourceUrl;
    const moreButton = $('#page-next');
    if (append) {
      setButtonBusy(moreButton, true, 'Loading…');
      $('#gallery-grid').setAttribute('aria-busy', 'true');
    } else {
      renderGallerySkeletons();
    }
    hideNotice();
    try {
      const data = await api(galleryQuery({ url: requestedUrl, page: requestedPage }), { signal: state.requestController.signal });
      const incoming = apiItems(data, 'galleries').map(normalizeGallery);
      if (append) {
        const seen = new Set(state.galleries.map(gallery => String(gallery.id || normalizeHistoryUrl(gallery.url))));
        incoming.forEach(gallery => {
          const identity = String(gallery.id || normalizeHistoryUrl(gallery.url));
          if (!seen.has(identity)) {
            seen.add(identity);
            state.galleries.push(gallery);
          }
        });
      } else {
        state.galleries = incoming;
      }
      state.total = state.galleries.length;
      state.page = Number(data?.page ?? requestedPage);
      if (!append) state.sourceUrl = data?.source_url || state.sourceUrl;
      state.nextUrl = data?.next_url || '';
      if (state.sourceUrl && state.browseMode === 'url') $('#source-input').value = state.sourceUrl;
      renderGalleries();
      announce(`${state.galleries.length} galleries loaded`);
      setServerState(true, $('#server-detail').textContent === 'Unable to connect' ? 'Ready' : $('#server-detail').textContent);
    } catch (error) {
      if (error.name === 'AbortError') return;
      if (!append) state.galleries = [];
      renderGalleries();
      showNotice(errorMessage(error));
      if (!quiet) toast(append ? 'Could not load more galleries' : 'Could not load galleries', errorMessage(error), 'error');
    } finally {
      state.loadingGalleries = false;
      $('#gallery-grid').setAttribute('aria-busy', 'false');
      state.requestController = null;
      if (append) setButtonBusy(moreButton, false);
      renderPagination();
      $('#refresh-button').classList.remove('is-spinning');
    }
  }

  function renderGallerySkeletons() {
    const grid = $('#gallery-grid');
    grid.replaceChildren();
    grid.setAttribute('aria-busy', 'true');
    $('#gallery-empty').hidden = true;
    $('#pagination').hidden = true;
    const count = window.innerWidth < 760 ? 6 : 8;
    for (let index = 0; index < count; index += 1) {
      const card = document.createElement('div');
      card.className = 'skeleton-card';
      card.setAttribute('aria-hidden', 'true');
      card.innerHTML = '<div class="skeleton-image"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line short"></div></div>';
      grid.append(card);
    }
  }

  function filteredGalleries() {
    return state.galleries.filter(gallery => (state.filters.showSaved || !gallery.saved) && (state.filters.showIgnored || !gallery.ignored));
  }

  function galleryStatus(gallery) {
    if (gallery.ignored) return { label: 'Ignored', className: 'ignored' };
    if (gallery.queued) return { label: 'Queued', className: 'queued' };
    if (gallery.partial) {
      const progress = gallery.imageCount ? ` ${formatNumber(gallery.downloadedImages)}/${formatNumber(gallery.imageCount)}` : '';
      return { label: `Partial${progress}`, className: 'partial' };
    }
    if (gallery.saved) return { label: 'Saved', className: 'saved' };
    return { label: 'New', className: 'new' };
  }

  function renderGalleries() {
    const grid = $('#gallery-grid');
    const galleries = filteredGalleries();
    grid.replaceChildren();
    grid.classList.toggle('is-compact', state.density === 'compact');
    galleries.forEach(gallery => {
      const fragment = $('#gallery-card-template').content.cloneNode(true);
      const card = $('.gallery-card', fragment);
      card.dataset.galleryId = String(gallery.id);
      card.classList.toggle('is-ignored', gallery.ignored);
      card.classList.toggle('is-queued', gallery.queued);
      const image = $('.card-image img', card);
      loadImage(image, gallery.thumbnailUrl, gallery.title);
      const status = galleryStatus(gallery);
      const statusNode = $('.card-status', card);
      statusNode.textContent = status.label;
      statusNode.className = `card-status ${status.className}`;
      $('.image-count b', card).textContent = gallery.imageCount ? formatNumber(gallery.imageCount) : 'View';
      $('.card-title', card).textContent = gallery.title;
      $('.card-meta', card).textContent = `${displayHost(gallery.url)}${gallery.imageCount ? ` · ${formatNumber(gallery.imageCount)} images` : ''}`;
      const ignoreButton = $('.card-ignore', card);
      ignoreButton.title = gallery.ignored ? 'Unignore gallery' : 'Ignore gallery';
      ignoreButton.setAttribute('aria-label', ignoreButton.title);
      card.querySelectorAll('.gallery-open, .gallery-open-text').forEach(button => button.dataset.galleryId = String(gallery.id));
      grid.append(fragment);
    });

    $('#gallery-empty').hidden = Boolean(galleries.length);
    const hiddenCount = state.galleries.length - galleries.length;
    if (!galleries.length && hiddenCount) {
      $('#gallery-empty h3').textContent = `${hiddenCount} ${hiddenCount === 1 ? 'gallery is' : 'galleries are'} hidden`;
      $('#gallery-empty p').textContent = 'Adjust the visibility filters to include saved or ignored galleries.';
    } else {
      $('#gallery-empty h3').textContent = 'No galleries here yet';
      $('#gallery-empty p').textContent = 'Paste a supported source URL above to discover galleries, or try a different search.';
    }

    const context = state.query ? `Search results for “${state.query}”` : state.sourceUrl ? displayHost(state.sourceUrl) : 'Recent galleries';
    $('#collection-title').textContent = context;
    $('#collection-summary').textContent = state.total ? `${formatNumber(state.total)} ${state.total === 1 ? 'gallery' : 'galleries'} loaded${hiddenCount ? ` · ${hiddenCount} hidden` : ''}` : 'No matching galleries';
    renderPagination();
  }

  function renderPagination() {
    const hasNext = Boolean(state.nextUrl);
    $('#pagination').hidden = !hasNext && state.page <= 1;
    $('#page-next').hidden = !hasNext;
    $('#page-next').disabled = !hasNext || state.loadingGalleries;
    $('#page-status').textContent = `${formatNumber(state.galleries.length)} ${state.galleries.length === 1 ? 'gallery' : 'galleries'} loaded`;
    $('#page-hint').textContent = hasNext
      ? `Through page ${state.page} · more load automatically as you scroll`
      : `All available results loaded through page ${state.page}`;
  }

  function syncMediaLoadDiagnostics() {
    const tasks = [
      ...state.mediaLoadActive,
      ...state.mediaLoadQueue,
      ...state.mediaLoadObserved
    ];
    const activeForeground = [...state.mediaLoadActive]
      .filter(task => task.kind === 'foreground').length;
    const activeWarm = [...state.mediaLoadActive].filter(task => task.kind === 'warm').length;
    const queuedWarm = state.mediaLoadQueue.filter(task => task.kind === 'warm').length;
    const stale = tasks.filter(task => task.generation !== state.galleryPreviewWarmGeneration).length;
    state.mediaLoadPeak = Math.max(state.mediaLoadPeak, state.mediaLoadActive.size);
    state.galleryForegroundPeak = Math.max(
      state.galleryForegroundPeak,
      activeForeground
    );
    state.galleryPreviewWarmPeak = Math.max(state.galleryPreviewWarmPeak, activeWarm);
    const root = document.documentElement;
    root.dataset.galleryMediaActive = String(state.mediaLoadActive.size);
    root.dataset.galleryMediaPeak = String(state.mediaLoadPeak);
    root.dataset.galleryForegroundActive = String(activeForeground);
    root.dataset.galleryForegroundPeak = String(state.galleryForegroundPeak);
    root.dataset.galleryWarmActive = String(activeWarm);
    root.dataset.galleryWarmQueued = String(queuedWarm);
    root.dataset.galleryWarmPeak = String(state.galleryPreviewWarmPeak);
    root.dataset.galleryMediaStale = String(stale);
    root.dataset.galleryMediaCancelled = String(state.mediaLoadCancelled);
    root.dataset.galleryMediaGeneration = String(state.galleryPreviewWarmGeneration);
  }

  function mediaLoadTaskIsCurrent(task) {
    if (
      !task
      || task.cancelled
      || task.generation !== state.galleryPreviewWarmGeneration
    ) return false;
    if (task.kind === 'warm') return true;
    if (!task.image.isConnected) return false;
    const galleryOpen = Boolean($('#gallery-modal')?.open);
    if (task.kind === 'critical') {
      return galleryOpen && (
        Boolean($('#lightbox-modal')?.open)
        || state.lightboxIndex >= 0
      );
    }
    if (task.kind === 'foreground') return galleryOpen || state.loadingDetail;
    return !galleryOpen;
  }

  function scheduleMediaLoadDrain() {
    if (state.mediaLoadDrainScheduled) return;
    state.mediaLoadDrainScheduled = true;
    window.queueMicrotask(() => {
      state.mediaLoadDrainScheduled = false;
      drainMediaLoadQueue();
    });
  }

  function removeQueuedMediaLoad(task) {
    const index = state.mediaLoadQueue.indexOf(task);
    if (index >= 0) state.mediaLoadQueue.splice(index, 1);
  }

  function removeObservedMediaLoad(task) {
    if (!task) return false;
    const removed = state.mediaLoadObserved.delete(task);
    state.mediaLoadLazyObserver?.unobserve(task.image);
    return removed;
  }

  function hasDomBoundMediaLoadTasks() {
    return [
      ...state.mediaLoadActive,
      ...state.mediaLoadQueue,
      ...state.mediaLoadObserved
    ].some(task => task.kind !== 'warm');
  }

  function maybeDisconnectMediaLoadDomObserver() {
    if (!hasDomBoundMediaLoadTasks()) state.mediaLoadDomObserver?.disconnect();
  }

  function observeMediaLoadDomRemovals() {
    if (!('MutationObserver' in window)) return;
    if (!state.mediaLoadDomObserver) {
      state.mediaLoadDomObserver = new MutationObserver(
        scheduleMediaLoadDrain
      );
    }
    state.mediaLoadDomObserver.observe(document.documentElement, {
      childList: true,
      subtree: true
    });
  }

  function mediaLoadUsesAbortableFetch(url) {
    try {
      const parsed = new URL(url, window.location.href);
      return (
        ['http:', 'https:'].includes(parsed.protocol)
        && parsed.origin === window.location.origin
      );
    } catch (_) {
      return false;
    }
  }

  function revokeMediaLoadObjectUrl(task) {
    const objectUrl = task?.objectUrl;
    if (!objectUrl) return;
    task.objectUrl = '';
    URL.revokeObjectURL(objectUrl);
  }

  function mediaLoadAttemptIsActive(task, attemptToken) {
    return Boolean(
      task
      && !task.cancelled
      && task.state === 'active'
      && attemptToken === task.attemptToken
    );
  }

  async function fetchMediaLoadTask(task, attemptToken, controller) {
    try {
      const response = await fetch(task.url, {
        signal: controller.signal,
        credentials: 'same-origin',
        cache: 'force-cache'
      });
      if (
        !mediaLoadAttemptIsActive(task, attemptToken)
        || task.abortController !== controller
      ) return;
      if (!response.ok) {
        throw new Error(`Media request failed (${response.status})`);
      }
      const blob = await response.blob();
      if (
        !mediaLoadAttemptIsActive(task, attemptToken)
        || task.abortController !== controller
      ) return;
      const objectUrl = URL.createObjectURL(blob);
      if (!mediaLoadAttemptIsActive(task, attemptToken)) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      task.abortController = null;
      task.objectUrl = objectUrl;
      task.image.src = objectUrl;
    } catch (_) {
      if (
        controller.signal.aborted
        || !mediaLoadAttemptIsActive(task, attemptToken)
      ) return;
      if (task.abortController === controller) task.abortController = null;
      finishMediaLoadTask(task, false, attemptToken);
    }
  }

  function cancelMediaLoadTask(task, { drain = true } = {}) {
    if (!task || task.cancelled || task.state === 'done') return;
    const wasActive = state.mediaLoadActive.delete(task);
    removeQueuedMediaLoad(task);
    removeObservedMediaLoad(task);
    task.attemptToken += 1;
    task.abortController?.abort();
    task.abortController = null;
    if (task.onLoad) task.image.removeEventListener('load', task.onLoad);
    if (task.onError) task.image.removeEventListener('error', task.onError);
    task.onLoad = null;
    task.onError = null;
    revokeMediaLoadObjectUrl(task);
    if (wasActive) {
      state.mediaLoadCancelled += 1;
      task.image.removeAttribute('src');
    }
    task.cancelled = true;
    task.state = 'cancelled';
    if (state.mediaLoadByImage.get(task.image) === task) {
      state.mediaLoadByImage.delete(task.image);
    }
    maybeDisconnectMediaLoadDomObserver();
    syncMediaLoadDiagnostics();
    if (drain) scheduleMediaLoadDrain();
  }

  function cancelMediaLoadForImage(image, options = {}) {
    const task = image && state.mediaLoadByImage.get(image);
    if (task) cancelMediaLoadTask(task, options);
  }

  function finishMediaLoadTask(task, loaded, attemptToken) {
    if (
      !task
      || task.cancelled
      || task.state !== 'active'
      || attemptToken !== task.attemptToken
    ) return;
    state.mediaLoadActive.delete(task);
    task.image.removeEventListener('load', task.onLoad);
    task.image.removeEventListener('error', task.onError);
    task.onLoad = null;
    task.onError = null;
    task.abortController = null;
    revokeMediaLoadObjectUrl(task);
    task.state = 'done';
    if (state.mediaLoadByImage.get(task.image) === task) {
      state.mediaLoadByImage.delete(task.image);
    }
    if (loaded) {
      task.image.previousElementSibling?.classList.add('is-loaded');
      if (task.kind === 'warm') {
        state.galleryPreviewWarmups.delete(task.url);
        state.galleryPreviewWarmups.set(task.url, task.image);
        while (state.galleryPreviewWarmups.size > GALLERY_PREVIEW_CACHE_LIMIT) {
          state.galleryPreviewWarmups.delete(state.galleryPreviewWarmups.keys().next().value);
        }
      }
    } else {
      task.image.hidden = true;
      task.image.removeAttribute('src');
    }
    const failureCallback = !loaded && typeof task.failureCallback === 'function'
      ? task.failureCallback
      : null;
    maybeDisconnectMediaLoadDomObserver();
    syncMediaLoadDiagnostics();
    scheduleMediaLoadDrain();
    if (failureCallback) window.queueMicrotask(failureCallback);
  }

  function startMediaLoadTask(task) {
    if (!mediaLoadTaskIsCurrent(task)) {
      cancelMediaLoadTask(task, { drain: false });
      return false;
    }
    task.state = 'active';
    state.mediaLoadActive.add(task);
    const attemptToken = ++task.attemptToken;
    task.onLoad = () => finishMediaLoadTask(task, true, attemptToken);
    task.onError = () => finishMediaLoadTask(task, false, attemptToken);
    task.image.addEventListener('load', task.onLoad, { once: true });
    task.image.addEventListener('error', task.onError, { once: true });
    task.image.hidden = false;
    task.image.removeAttribute('src');
    if (mediaLoadUsesAbortableFetch(task.url)) {
      const controller = new AbortController();
      task.abortController = controller;
      fetchMediaLoadTask(task, attemptToken, controller);
    } else {
      task.image.src = task.url;
    }
    syncMediaLoadDiagnostics();
    return true;
  }

  function nextMediaLoadTask() {
    const critical = state.mediaLoadQueue.findIndex(task => task.kind === 'critical');
    const foreground = state.mediaLoadQueue.findIndex(task => task.kind === 'foreground');
    const normal = state.mediaLoadQueue.findIndex(task => task.kind === 'normal');
    const warm = state.mediaLoadQueue.findIndex(task => task.kind === 'warm');
    const activeForeground = [...state.mediaLoadActive]
      .filter(task => task.kind === 'foreground').length;
    const activeWarm = [...state.mediaLoadActive]
      .filter(task => task.kind === 'warm').length;
    if (critical >= 0) return state.mediaLoadQueue.splice(critical, 1)[0];
    if (foreground >= 0 && activeForeground < 2) {
      return state.mediaLoadQueue.splice(foreground, 1)[0];
    }
    if (normal >= 0) return state.mediaLoadQueue.splice(normal, 1)[0];
    if (warm >= 0 && activeWarm < GALLERY_PREVIEW_WARM_CONCURRENCY) {
      return state.mediaLoadQueue.splice(warm, 1)[0];
    }
    return null;
  }

  function drainMediaLoadQueue() {
    [...state.mediaLoadActive].forEach(task => {
      if (!mediaLoadTaskIsCurrent(task)) cancelMediaLoadTask(task, { drain: false });
    });
    [...state.mediaLoadObserved].forEach(task => {
      if (!mediaLoadTaskIsCurrent(task)) cancelMediaLoadTask(task, { drain: false });
    });
    [...state.mediaLoadQueue].forEach(task => {
      if (!mediaLoadTaskIsCurrent(task)) cancelMediaLoadTask(task, { drain: false });
    });
    while (state.mediaLoadActive.size < MEDIA_LOAD_CONCURRENCY) {
      const task = nextMediaLoadTask();
      if (!task) break;
      startMediaLoadTask(task);
    }
    syncMediaLoadDiagnostics();
  }

  function enqueueMediaLoadTask(task) {
    if (!mediaLoadTaskIsCurrent(task)) {
      cancelMediaLoadTask(task);
      return;
    }
    removeObservedMediaLoad(task);
    task.state = 'queued';
    if (task.kind !== 'warm') observeMediaLoadDomRemovals();
    if (
      ['critical', 'foreground'].includes(task.kind)
      && state.mediaLoadActive.size >= MEDIA_LOAD_CONCURRENCY
    ) {
      const warm = [...state.mediaLoadActive].find(item => item.kind === 'warm');
      if (warm) cancelMediaLoadTask(warm, { drain: false });
    }
    state.mediaLoadQueue.push(task);
    syncMediaLoadDiagnostics();
    scheduleMediaLoadDrain();
  }

  function observeLazyMediaLoad(task) {
    if (!('IntersectionObserver' in window)) {
      enqueueMediaLoadTask(task);
      return;
    }
    if (!state.mediaLoadLazyObserver) {
      state.mediaLoadLazyObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          const pending = state.mediaLoadByImage.get(entry.target);
          if (!entry.target.isConnected) {
            if (pending) cancelMediaLoadTask(pending);
            else state.mediaLoadLazyObserver.unobserve(entry.target);
            return;
          }
          if (!entry.isIntersecting) return;
          if (pending) removeObservedMediaLoad(pending);
          else state.mediaLoadLazyObserver.unobserve(entry.target);
          if (pending?.state === 'observing') enqueueMediaLoadTask(pending);
        });
      }, { rootMargin: '600px 300px' });
    }
    task.state = 'observing';
    state.mediaLoadObserved.add(task);
    state.mediaLoadLazyObserver.observe(task.image);
    observeMediaLoadDomRemovals();
    syncMediaLoadDiagnostics();
  }

  function queueMediaLoad(
    image,
    source,
    alt = '',
    {
      kind = 'normal',
      lazy = image.loading === 'lazy',
      generation = state.galleryPreviewWarmGeneration,
      onError = null
    } = {}
  ) {
    const url = safeUrl(source);
    image.alt = alt;
    const previous = state.mediaLoadByImage.get(image);
    if (previous) cancelMediaLoadTask(previous, { drain: false });
    if (!url) {
      image.removeAttribute('src');
      image.hidden = true;
      return null;
    }
    if (kind === 'normal' && $('#gallery-modal')?.open) {
      // The Finder grid is re-rendered when the modal closes. Avoid letting
      // thumbnails behind the dialog compete with review actions in the meantime.
      image.removeAttribute('src');
      image.hidden = false;
      return null;
    }
    image.hidden = false;
    const task = {
      image,
      url,
      kind,
      generation,
      state: 'new',
      cancelled: false,
      attemptToken: 0,
      abortController: null,
      objectUrl: '',
      onLoad: null,
      onError: null,
      failureCallback: onError
    };
    state.mediaLoadByImage.set(image, task);
    if (kind !== 'warm') {
      observeMediaLoadDomRemovals();
      window.queueMicrotask(() => {
        if (state.mediaLoadByImage.get(image) !== task) return;
        if (!mediaLoadTaskIsCurrent(task)) {
          cancelMediaLoadTask(task);
          return;
        }
        if (lazy) observeLazyMediaLoad(task);
        else enqueueMediaLoadTask(task);
      });
    } else {
      enqueueMediaLoadTask(task);
    }
    return task;
  }

  function loadImage(image, source, alt = '', options = {}) {
    return queueMediaLoad(image, source, alt, options);
  }

  function advanceGalleryMediaGeneration() {
    state.galleryPreviewWarmGeneration += 1;
    [
      ...state.mediaLoadActive,
      ...state.mediaLoadQueue,
      ...state.mediaLoadObserved
    ].forEach(task => {
      cancelMediaLoadTask(task, { drain: false });
    });
    state.mediaLoadQueue = [];
    state.mediaLoadObserved.clear();
    state.mediaLoadDomObserver?.disconnect();
    state.mediaLoadPeak = 0;
    state.galleryForegroundPeak = 0;
    state.galleryPreviewWarmPeak = 0;
    syncMediaLoadDiagnostics();
    return state.galleryPreviewWarmGeneration;
  }

  async function handleSourceSubmit(event) {
    event.preventDefault();
    const input = $('#source-input');
    const value = input.value.trim();
    if (!value) {
      input.focus();
      toast(state.browseMode === 'url' ? 'Paste a source URL' : 'Enter a search', 'The field cannot be empty.', 'info');
      return;
    }
    if (state.browseMode === 'url') {
      const url = safeUrl(value);
      if (!url) {
        input.setCustomValidity('Enter a complete http or https URL.');
        input.reportValidity();
        input.setCustomValidity('');
        return;
      }
      state.sourceUrl = url;
      state.query = '';
    } else {
      state.query = value;
      state.sourceUrl = '';
    }
    state.page = 1;
    state.nextUrl = '';
    await loadGalleries();
  }

  function setBrowseMode(mode) {
    state.browseMode = mode;
    $$('.mode-button').forEach(button => {
      const active = button.dataset.mode === mode;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', String(active));
    });
    const input = $('#source-input');
    const leading = $('.input-leading-icon use');
    const submit = $('#source-submit span');
    if (mode === 'url') {
      input.type = 'url';
      input.placeholder = 'https://www.pornpics.com/…';
      input.setAttribute('aria-label', 'Source URL');
      input.value = state.sourceUrl;
      leading.setAttribute('href', '#i-link');
      submit.textContent = 'Browse';
      $('#source-hint').lastChild.textContent = ' Paste a gallery, category, model, or search-result URL. Everything is fetched by the server.';
    } else {
      input.type = 'search';
      input.placeholder = 'Search gallery titles…';
      input.setAttribute('aria-label', 'Search galleries');
      input.value = state.query;
      leading.setAttribute('href', '#i-search');
      submit.textContent = 'Search';
      $('#source-hint').lastChild.textContent = ' Search PornPics live by title or keyword. Results are fetched by the server.';
    }
    $('#clear-source').hidden = !input.value;
    input.focus();
  }

  async function loadMoreGalleries() {
    await loadGalleries({ append: true });
  }

  function setupGalleryAutoLoad() {
    if (!('IntersectionObserver' in window)) return;
    state.galleryObserver = new IntersectionObserver(entries => {
      if (
        entries.some(entry => entry.isIntersecting)
        && state.view === 'discover'
        && state.nextUrl
        && !state.loadingGalleries
      ) loadGalleries({ append: true, quiet: true });
    }, { rootMargin: '500px 0px' });
    state.galleryObserver.observe($('#pagination'));
  }

  async function toggleIgnore(gallery, button = null) {
    const nextValue = !gallery.ignored;
    if (button) button.disabled = true;
    invalidateGalleryDetailCache(gallery.id);
    gallery.ignored = nextValue;
    renderGalleries();
    if (state.gallery && String(state.gallery.id) === String(gallery.id)) {
      state.gallery.ignored = nextValue;
      renderGallerySummary();
    }
    try {
      const data = await api(`/api/galleries/${encodeURIComponent(gallery.id)}`, { method: 'PATCH', body: { ignored: nextValue } });
      gallery.ignored = data?.ignored ?? nextValue;
      renderGalleries();
      toast(nextValue ? 'Gallery ignored' : 'Gallery restored', nextValue ? 'It will stay hidden with the current filter.' : 'It is visible in Discover again.', 'info');
    } catch (error) {
      gallery.ignored = !nextValue;
      renderGalleries();
      if (state.gallery && String(state.gallery.id) === String(gallery.id)) {
        state.gallery.ignored = !nextValue;
        renderGallerySummary();
      }
      toast('Could not update gallery', errorMessage(error), 'error');
    } finally { if (button) button.disabled = false; }
  }

  function finderSuggestionForImage(image, index) {
    const suggestions = Array.isArray(state.galleryContext?.suggestions) ? state.galleryContext.suggestions : [];
    return suggestions.find(suggestion => {
      const targetUrl = String(suggestion.imageUrl || suggestion.image_url || '');
      return (targetUrl && [image.url, image.fullUrl, image.previewUrl].includes(targetUrl))
        || (Number(suggestion.ordinal || 0) === index + 1);
    }) || null;
  }

  function scrollToFinderSuggestion() {
    if (!state.galleryContext?.suggestions?.length) return;
    window.requestAnimationFrame(() => {
      const option = $$('.image-option', $('#image-grid')).find(item => item.classList.contains('is-finder-suggestion'));
      option?.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'smooth' });
    });
  }

  function finderFeedbackGalleryResult() {
    const key = state.galleryContext?.finderFeedbackResultKey;
    if (key === undefined || key === null) return null;
    const active = state.galleryContext?.activeFinderResult;
    if (active && String(active.key) === String(key)) return active;
    return state.finderResults.find(result => String(result.key) === String(key))
      || state.galleryContext?.finderReviewQueue?.results?.find(result => String(result.key) === String(key))
      || null;
  }

  function finderFeedbackGalleryAvailable() {
    return Boolean(finderFeedbackGalleryResult());
  }

  function finderReviewLabel(review) {
    return {
      pending: 'To review',
      accepted: 'Accepted',
      maybe: 'Maybe',
      rejected: 'Rejected'
    }[review] || 'To review';
  }

  function createFinderGalleryReviewQueue(result) {
    const review = state.finderReview;
    const threshold = Math.max(0, Math.min(1, Number($('#finder-result-threshold').value || 0)));
    let results = state.finderResults.filter(item => item.review === review && item.score >= threshold);
    if (!results.some(item => String(item.key) === String(result.key))) results = [...results, result];
    const index = Math.max(0, results.findIndex(item => String(item.key) === String(result.key)));
    const baseOffset = Math.max(0, (state.finderResultPage - 1) * state.finderResultPageSize);
    const total = Math.max(results.length, Number(state.finderResultTotal || 0));
    return {
      scanId: String(state.finderScan?.id || ''),
      review,
      threshold,
      results,
      index,
      baseOffset,
      total,
      pageSize: Math.max(1, Number(state.finderResultPageSize || FINDER_RESULTS_PAGE_SIZE)),
      exhaustedBefore: baseOffset <= 0,
      exhaustedAfter: !finderScanIsRunning() && baseOffset + results.length >= total,
      forwardProbeOffset: baseOffset + results.length,
      forwardProbeRemovedCount: 0,
      loading: false
    };
  }

  function renderFinderGalleryReview() {
    const rail = $('#gallery-review-rail');
    const result = finderFeedbackGalleryResult();
    const queue = state.galleryContext?.finderReviewQueue;
    const available = Boolean(result && queue && String(queue.scanId) === String(state.finderScan?.id || ''));
    const workspaceLocked = Boolean(
      state.poseApplying
      || state.poseExporting
      || (available && (state.galleryReviewBusy || queue.loading || state.finderFeedbackGallerySaving))
    );
    const modalBody = $('.gallery-modal-body');
    const modalFooter = $('.gallery-footer');
    if (modalBody) modalBody.inert = workspaceLocked;
    if (modalFooter) modalFooter.inert = workspaceLocked;
    $('#gallery-modal').classList.toggle('is-review-transition', workspaceLocked);
    $('#gallery-modal').setAttribute('aria-busy', String(workspaceLocked));
    rail.hidden = !available;
    if (!available) return;
    const index = Math.max(0, Math.min(queue.results.length - 1, Number(queue.index || 0)));
    const position = Math.max(1, Number(queue.baseOffset || 0) + index + 1);
    const total = Math.max(position, Number(queue.total || queue.results.length || 1));
    const feedbackCount = state.finderFeedbackGallerySelection.size;
    const review = normalizeFinderReview(result.review);
    const busy = state.galleryReviewBusy
      || queue.loading
      || state.loadingDetail
      || state.finderFeedbackBusy
      || state.finderFeedbackGallerySaving
      || state.poseApplying
      || state.poseExporting
      || Boolean(result.feedbackSaving);
    $('#gallery-review-position').textContent = `${formatNumber(position)} of ${formatNumber(total)}`;
    $('#gallery-review-queue').textContent = `${finderReviewLabel(queue.review)} queue · use ← →`;
    $('#gallery-review-feedback-count').textContent = formatNumber(feedbackCount);
    $('#gallery-review-status').textContent = finderReviewLabel(review);
    rail.classList.toggle('is-accepted', review === 'accepted');
    rail.classList.toggle('is-maybe', review === 'maybe');
    rail.classList.toggle('is-rejected', review === 'rejected');
    $$('[data-gallery-finder-review]', rail).forEach(button => {
      const buttonReview = button.dataset.galleryFinderReview;
      const active = buttonReview === review;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
      button.disabled = busy
        || (buttonReview === 'accepted' && !feedbackCount)
        || (active && !state.finderFeedbackGalleryDirty);
    });
    const accept = $('#gallery-review-accept');
    accept.title = feedbackCount
      ? `Accept with ${feedbackCount} selected Finder feedback ${feedbackCount === 1 ? 'image' : 'images'}`
      : 'Choose at least one image in Finder review before accepting';
    $('#gallery-review-maybe').title = 'Keep this gallery neutral for a later decision';
    $('#gallery-review-reject').title = feedbackCount
      ? `Reject with ${feedbackCount} selected negative feedback ${feedbackCount === 1 ? 'image' : 'images'}`
      : 'Reject the gallery without image-level feedback';
    const previous = $('#gallery-review-previous');
    const next = $('#gallery-review-next');
    previous.disabled = busy || (index <= 0 && queue.exhaustedBefore);
    next.disabled = busy || (index >= queue.results.length - 1 && queue.exhaustedAfter);
    previous.setAttribute('aria-label', `Previous gallery in ${finderReviewLabel(queue.review).toLowerCase()} queue`);
    next.setAttribute('aria-label', `Next gallery in ${finderReviewLabel(queue.review).toLowerCase()} queue`);
  }

  function restoreFinderFeedbackGallerySelection() {
    const result = finderFeedbackGalleryResult();
    const galleryUrls = new Set((state.gallery?.images || []).map(image => image.url));
    state.finderFeedbackGallerySelection = new Set(
      (result?.feedbackImageUrls || []).filter(url => galleryUrls.has(url)).slice(0, 3)
    );
    state.finderFeedbackGalleryDirty = false;
  }

  function confirmDiscardFinderFeedbackGalleryChanges(message = 'Discard the unsaved Finder feedback selection?') {
    if (state.finderFeedbackGallerySaving) {
      toast('Feedback selection is saving', 'Wait for the save to finish before closing or changing galleries. An in-flight save cannot be cancelled.', 'info');
      return false;
    }
    if (!state.finderFeedbackGalleryDirty) return true;
    if (!window.confirm(message)) return false;
    restoreFinderFeedbackGallerySelection();
    return true;
  }

  function galleryDetailCacheKey(id, profile = state.activeProfile) {
    return JSON.stringify([String(profile || ''), String(id ?? '')]);
  }

  function cachedGalleryDetail(id, profile = state.activeProfile) {
    const key = galleryDetailCacheKey(id, profile);
    const entry = state.galleryDetailCache.get(key);
    if (!entry) return null;
    if (Date.now() - entry.cachedAt > GALLERY_DETAIL_CACHE_TTL) {
      state.galleryDetailCache.delete(key);
      return null;
    }
    state.galleryDetailCache.delete(key);
    state.galleryDetailCache.set(key, entry);
    return entry.data;
  }

  function storeGalleryDetail(id, profile, data) {
    const key = galleryDetailCacheKey(id, profile);
    state.galleryDetailCache.delete(key);
    state.galleryDetailCache.set(key, {
      id: String(id ?? ''),
      profile: String(profile || ''),
      cachedAt: Date.now(),
      data
    });
    while (state.galleryDetailCache.size > GALLERY_DETAIL_CACHE_LIMIT) {
      state.galleryDetailCache.delete(state.galleryDetailCache.keys().next().value);
    }
    return data;
  }

  function invalidateGalleryDetailCache(id = null, { abortForeground = false } = {}) {
    const target = id === null || id === undefined ? null : String(id);
    if (id === null || id === undefined) {
      state.galleryDetailGlobalRevision += 1;
      state.galleryDetailCache.clear();
    } else {
      state.galleryDetailRevisions.set(
        target,
        Number(state.galleryDetailRevisions.get(target) || 0) + 1
      );
      [...state.galleryDetailCache.entries()].forEach(([key, entry]) => {
        if (entry.id === target) state.galleryDetailCache.delete(key);
      });
    }
    [...state.galleryDetailPrefetchQueued.entries()].forEach(([key, task]) => {
      if (target !== null && task.id !== target) return;
      task.cancelled = true;
      state.galleryDetailPrefetchQueued.delete(key);
      const error = new Error('Gallery detail prefetch invalidated');
      error.name = 'AbortError';
      task.reject(error);
    });
    state.galleryDetailPrefetches.forEach(entry => {
      if (
        (target === null || entry.id === target)
        && (entry.background || abortForeground)
      ) entry.controller.abort();
    });
    drainGalleryDetailPrefetchQueue();
  }

  function invalidateGalleryDetailByUrl(url) {
    const targetUrl = normalizeHistoryUrl(url);
    if (!targetUrl) return;
    const ids = new Set();
    state.galleryDetailCache.forEach(entry => {
      if (normalizeHistoryUrl(entry.data?.url || entry.data?.gallery_url) === targetUrl) {
        ids.add(entry.id);
      }
    });
    const current = state.gallery;
    if (current && normalizeHistoryUrl(current.url) === targetUrl) {
      ids.add(String(current.id));
    }
    ids.forEach(id => invalidateGalleryDetailCache(id));
  }

  function startGalleryDetailRequest(id, profile, { background = false } = {}) {
    const key = galleryDetailCacheKey(id, profile);
    const target = String(id ?? '');
    const globalRevision = state.galleryDetailGlobalRevision;
    const detailRevision = Number(state.galleryDetailRevisions.get(target) || 0);
    const controller = new AbortController();
    const entry = {
      id: target,
      profile: String(profile || ''),
      controller,
      background,
      promise: null
    };
    entry.promise = api(
      withParams(`/api/galleries/${encodeURIComponent(id)}`, { profile }),
      { signal: controller.signal }
    )
      .then(data => {
        if (controller.signal.aborted) {
          const error = new Error('Gallery detail request aborted');
          error.name = 'AbortError';
          throw error;
        }
        if (
          globalRevision !== state.galleryDetailGlobalRevision
          || detailRevision !== Number(state.galleryDetailRevisions.get(target) || 0)
        ) {
          const error = new Error('Gallery detail changed while loading');
          error.name = 'StaleGalleryDetailError';
          throw error;
        }
        return storeGalleryDetail(id, profile, data);
      })
      .finally(() => {
        if (state.galleryDetailPrefetches.get(key) === entry) {
          state.galleryDetailPrefetches.delete(key);
        }
      });
    state.galleryDetailPrefetches.set(key, entry);
    return entry.promise;
  }

  function drainGalleryDetailPrefetchQueue() {
    while (
      state.galleryDetailPrefetchActive < GALLERY_DETAIL_PREFETCH_CONCURRENCY
      && state.galleryDetailPrefetchQueue.length
    ) {
      const task = state.galleryDetailPrefetchQueue.shift();
      if (
        task.cancelled
        || state.galleryDetailPrefetchQueued.get(task.key) !== task
      ) continue;
      state.galleryDetailPrefetchQueued.delete(task.key);
      state.galleryDetailPrefetchActive += 1;
      startGalleryDetailRequest(task.id, task.profile, { background: true })
        .then(task.resolve, task.reject)
        .finally(() => {
          state.galleryDetailPrefetchActive = Math.max(
            0,
            state.galleryDetailPrefetchActive - 1
          );
          drainGalleryDetailPrefetchQueue();
        });
    }
  }

  function queueGalleryDetailPrefetch(id, profile) {
    const key = galleryDetailCacheKey(id, profile);
    const existing = state.galleryDetailPrefetchQueued.get(key);
    if (existing) return existing.promise;
    let resolve;
    let reject;
    const promise = new Promise((resolvePromise, rejectPromise) => {
      resolve = resolvePromise;
      reject = rejectPromise;
    });
    const task = {
      key,
      id: String(id ?? ''),
      profile: String(profile || ''),
      promise,
      resolve,
      reject,
      cancelled: false
    };
    state.galleryDetailPrefetchQueued.set(key, task);
    state.galleryDetailPrefetchQueue.push(task);
    drainGalleryDetailPrefetchQueue();
    return promise;
  }

  function retryInvalidatedGalleryDetail(request, id, profile, retryInvalidated) {
    if (!retryInvalidated) return request;
    return request.catch(error => {
      if (error.name !== 'StaleGalleryDetailError') throw error;
      return loadGalleryDetail(id, profile, {
        prefetch: false,
        retryInvalidated: false
      });
    });
  }

  function loadGalleryDetail(
    id,
    profile = state.activeProfile,
    { prefetch = false, retryInvalidated = true } = {}
  ) {
    const cached = cachedGalleryDetail(id, profile);
    if (cached) return Promise.resolve(cached);
    const key = galleryDetailCacheKey(id, profile);
    const inFlight = state.galleryDetailPrefetches.get(key);
    if (inFlight) {
      if (!prefetch) inFlight.background = false;
      return prefetch
        ? inFlight.promise
        : retryInvalidatedGalleryDetail(
          inFlight.promise,
          id,
          profile,
          retryInvalidated
        );
    }
    const queued = state.galleryDetailPrefetchQueued.get(key);
    if (prefetch) return queued?.promise || queueGalleryDetailPrefetch(id, profile);
    if (queued) {
      queued.cancelled = true;
      state.galleryDetailPrefetchQueued.delete(key);
      const request = startGalleryDetailRequest(id, profile);
      request.then(queued.resolve, queued.reject);
      return retryInvalidatedGalleryDetail(
        request,
        id,
        profile,
        retryInvalidated
      );
    }
    return retryInvalidatedGalleryDetail(
      startGalleryDetailRequest(id, profile),
      id,
      profile,
      retryInvalidated
    );
  }

  function cancelAdjacentGalleryPrefetches({ includeForeground = false } = {}) {
    advanceGalleryMediaGeneration();
    [...state.galleryDetailPrefetchQueued.values()].forEach(task => {
      task.cancelled = true;
      const error = new Error('Adjacent gallery prefetch cancelled');
      error.name = 'AbortError';
      task.reject(error);
    });
    state.galleryDetailPrefetchQueued.clear();
    state.galleryDetailPrefetches.forEach(entry => {
      if (entry.background || includeForeground) entry.controller.abort();
    });
    drainGalleryDetailPrefetchQueue();
  }

  function cancelForegroundGalleryDetailRequests() {
    state.galleryDetailPrefetches.forEach(entry => {
      if (!entry.background) entry.controller.abort();
    });
  }

  function retainAdjacentGalleryPrefetches(galleryIds, profile) {
    const desired = new Set(galleryIds.map(id => galleryDetailCacheKey(id, profile)));
    [...state.galleryDetailPrefetchQueued.entries()].forEach(([key, task]) => {
      if (desired.has(key)) return;
      task.cancelled = true;
      state.galleryDetailPrefetchQueued.delete(key);
      const error = new Error('Adjacent gallery prefetch superseded');
      error.name = 'AbortError';
      task.reject(error);
    });
    state.galleryDetailPrefetches.forEach((entry, key) => {
      if (entry.background && !desired.has(key)) entry.controller.abort();
    });
    drainGalleryDetailPrefetchQueue();
  }

  function warmGalleryPreviews(data, result = null, generation = state.galleryPreviewWarmGeneration) {
    if (!data) return;
    if (generation !== state.galleryPreviewWarmGeneration) return;
    const connection = navigator.connection;
    if (
      document.visibilityState !== 'visible'
      || connection?.saveData
      || ['slow-2g', '2g'].includes(connection?.effectiveType)
    ) return;
    const detail = normalizeDetail(data);
    const preferredImages = (result?.matches || []).map(match => {
      const matchUrl = String(match.imageUrl || '');
      return detail.images.find(image => (
        matchUrl
        && [image.url, image.fullUrl, image.previewUrl].includes(matchUrl)
      ));
    }).filter(Boolean);
    const urls = [...new Set(
      [...preferredImages, ...detail.images]
        .map(image => safeUrl(image.previewUrl))
        .filter(Boolean)
    )].slice(0, GALLERY_PREVIEW_WARM_LIMIT);
    urls.forEach(url => {
      const existing = state.galleryPreviewWarmups.get(url);
      if (existing?.complete && existing.naturalWidth > 0) {
        state.galleryPreviewWarmups.delete(url);
        state.galleryPreviewWarmups.set(url, existing);
        return;
      }
      const duplicate = [...state.mediaLoadActive, ...state.mediaLoadQueue]
        .some(task => task.kind === 'warm' && task.url === url);
      if (duplicate) return;
      const image = new Image();
      image.decoding = 'async';
      image.fetchPriority = 'low';
      queueMediaLoad(image, url, '', { kind: 'warm', lazy: false, generation });
    });
  }

  function scheduleGalleryPreviewWarm(data, result, generation) {
    const warm = () => {
      if (generation === state.galleryPreviewWarmGeneration) {
        warmGalleryPreviews(data, result, generation);
      }
    };
    if ('requestIdleCallback' in window) {
      window.requestIdleCallback(warm, { timeout: 500 });
    } else {
      window.setTimeout(warm, 0);
    }
  }

  function prefetchAdjacentFinderGalleries(context = state.galleryContext) {
    const queue = context?.finderReviewQueue;
    if (!queue?.results?.length) return;
    const generation = state.galleryPreviewWarmGeneration;
    const index = Math.max(0, Math.min(queue.results.length - 1, Number(queue.index || 0)));
    const profile = state.activeProfile;
    const neighbors = [queue.results[index - 1], queue.results[index + 1]]
      .filter(result => result?.galleryId !== undefined && result?.galleryId !== null);
    retainAdjacentGalleryPrefetches(
      neighbors.map(result => String(result.galleryId)),
      profile
    );
    const seen = new Set();
    neighbors.forEach(result => {
      const id = String(result.galleryId);
      if (seen.has(id)) return;
      seen.add(id);
      loadGalleryDetail(id, profile, { prefetch: true })
        .then(data => scheduleGalleryPreviewWarm(data, result, generation))
        .catch(() => { /* Navigation retries failed background fetches normally. */ });
    });
  }

  function snapshotOpenLightboxMedia() {
    const dialog = $('#lightbox-modal');
    if (!dialog?.open || state.lightboxIndex < 0) return null;
    const image = state.gallery?.images?.[state.lightboxIndex];
    return {
      index: state.lightboxIndex,
      identity: String(image?.url || image?.fullUrl || image?.previewUrl || '')
    };
  }

  function reloadOpenLightboxMedia(snapshot) {
    const dialog = $('#lightbox-modal');
    if (!snapshot || !dialog?.open) return;
    const images = state.gallery?.images || [];
    if (!images.length) {
      closeModal(dialog);
      return;
    }
    const matchingIndex = snapshot.identity
      ? images.findIndex(image => (
        String(image.url || image.fullUrl || image.previewUrl || '') === snapshot.identity
      ))
      : -1;
    state.lightboxIndex = matchingIndex >= 0
      ? matchingIndex
      : Math.max(0, Math.min(images.length - 1, snapshot.index));
    renderLightboxImage();
  }

  async function refreshOpenGalleryDetail(profile = state.activeProfile) {
    const dialog = $('#gallery-modal');
    const galleryId = state.gallery?.id;
    if (!dialog.open || galleryId === undefined || galleryId === null) return false;
    const detailProfile = String(profile || '');
    const detailRequest = ++state.galleryDetailRequest;
    state.loadingDetail = true;
    $('#image-grid').setAttribute('aria-busy', 'true');
    renderFinderGalleryReview();
    try {
      const data = await loadGalleryDetail(galleryId, detailProfile);
      if (
        detailRequest !== state.galleryDetailRequest
        || detailProfile !== state.activeProfile
        || !dialog.open
        || String(state.gallery?.id) !== String(galleryId)
      ) return false;
      const lightboxSnapshot = snapshotOpenLightboxMedia();
      advanceGalleryMediaGeneration();
      const detail = normalizeDetail(data);
      const imageUrls = new Set(detail.images.map(image => image.url));
      state.gallery = detail;
      if (state.galleryMode === 'download') {
        const pendingImages = detail.images.filter(image => !image.downloaded);
        state.selectedImages = new Set(
          (pendingImages.length ? pendingImages : detail.images).map(image => image.url)
        );
      } else {
        state.selectedImages = new Set([...state.selectedImages].filter(url => imageUrls.has(url)));
      }
      state.finderFeedbackGallerySelection = new Set(
        [...state.finderFeedbackGallerySelection].filter(url => imageUrls.has(url))
      );
      state.poseSelectedImages = new Set(
        [...state.poseSelectedImages].filter(url => imageUrls.has(url))
      );
      const listItem = state.galleries.find(item => String(item.id) === String(galleryId));
      if (listItem) {
        Object.assign(listItem, {
          saved: detail.saved,
          ignored: detail.ignored,
          imageCount: detail.imageCount,
          thumbnailUrl: detail.thumbnailUrl || listItem.thumbnailUrl
        });
      }
      renderGallerySummary();
      renderImages();
      reloadOpenLightboxMedia(lightboxSnapshot);
      renderGalleries();
      $('#gallery-modal-kicker').textContent = displayHost(detail.url);
      $('#gallery-modal-title').textContent = detail.title;
      prefetchAdjacentFinderGalleries();
      return true;
    } catch (error) {
      if (error.name !== 'AbortError') {
        toast('Could not refresh gallery', errorMessage(error), 'error');
      }
      return false;
    } finally {
      if (
        detailRequest === state.galleryDetailRequest
        && detailProfile === state.activeProfile
        && dialog.open
      ) {
        state.loadingDetail = false;
        $('#image-grid').setAttribute('aria-busy', 'false');
        updateSelectionUi();
        renderFinderGalleryReview();
      }
    }
  }

  async function openGallery(id, context = null) {
    if (!confirmDiscardFinderFeedbackGalleryChanges('Discard the unsaved Finder feedback selection and open another gallery?')) return;
    if (state.poseLoadedKey && (state.poseDirty || state.poseSaving)) {
      await flushPoseDraft();
      if (state.poseDirty) {
        toast('Pose draft still has unsaved changes', 'Resolve the save error before opening another gallery.', 'error');
        return;
      }
    }
    const summarySource = context?.summary || state.galleries.find(item => String(item.id) === String(id));
    const summary = summarySource ? normalizeGallery(summarySource) : null;
    if (!summary) return;
    advanceGalleryMediaGeneration();
    cancelForegroundGalleryDetailRequests();
    const detailProfile = state.activeProfile;
    const detailRequest = ++state.galleryDetailRequest;
    state.poseApplyRequest += 1;
    state.poseApplying = false;
    state.poseExportRequest += 1;
    state.poseExporting = false;
    setButtonBusy($('#pose-export'), false);
    window.clearTimeout(state.poseSaveTimer);
    state.poseSaving = false;
    state.poseSavePromise = null;
    state.poseLoading = false;
    state.galleryContext = context ? { ...context, suggestions: Array.isArray(context.suggestions) ? context.suggestions : [] } : null;
    state.gallery = { ...summary, images: [] };
    state.selectedImages = new Set();
    state.finderFeedbackGallerySelection = new Set();
    state.finderFeedbackGalleryDirty = false;
    state.finderFeedbackGallerySaving = false;
    state.poseSelectedImages = new Set();
    state.poseTags = [];
    state.poseDraft = { revision: 0, controls: { solo: '', couple: '', group: '' }, targets: [] };
    state.poseLoadedKey = '';
    state.poseDirty = false;
    state.poseMutation = 0;
    window.clearTimeout(state.poseSaveTimer);
    $('#pose-tag-input').value = '';
    $('#pose-control-role').value = 'solo';
    const feedbackRequested = context?.mode === 'feedback'
      && context?.finderFeedbackResultKey !== undefined
      && context?.finderFeedbackResultKey !== null;
    const requestedMode = feedbackRequested ? 'feedback' : context?.mode === 'pose' ? 'pose' : 'download';
    if (requestedMode === 'pose' && state.galleryContext?.suggestions.length) state.poseAssignment = 'target';
    setGalleryMode(requestedMode, { load: false, render: false });
    state.loadingDetail = true;
    $('#gallery-modal-title').textContent = summary.title;
    $('#gallery-modal-kicker').textContent = 'Loading gallery';
    $('#image-grid').replaceChildren();
    $('#image-grid').setAttribute('aria-busy', 'true');
    $('#images-empty').hidden = true;
    renderImageSkeletons();
    renderGallerySummary();
    updateSelectionUi();
    const dialog = $('#gallery-modal');
    if (!dialog.open) dialog.showModal();
    try {
      const data = await loadGalleryDetail(id, detailProfile);
      if (
        detailRequest !== state.galleryDetailRequest
        || detailProfile !== state.activeProfile
        || !dialog.open
      ) return;
      state.gallery = normalizeDetail(data);
      const listItem = state.galleries.find(item => String(item.id) === String(id));
      if (listItem) Object.assign(listItem, { saved: state.gallery.saved, ignored: state.gallery.ignored, imageCount: state.gallery.imageCount, thumbnailUrl: state.gallery.thumbnailUrl || listItem.thumbnailUrl });
      const pendingImages = state.gallery.images.filter(image => !image.downloaded);
      state.selectedImages = new Set((pendingImages.length ? pendingImages : state.gallery.images).map(image => image.url));
      if (context?.finderFeedbackResultKey !== undefined && context?.finderFeedbackResultKey !== null) {
        const galleryUrls = new Set(state.gallery.images.map(image => image.url));
        state.finderFeedbackGallerySelection = new Set(
          (Array.isArray(context?.feedbackImageUrls) ? context.feedbackImageUrls : [])
            .map(String)
            .filter(url => galleryUrls.has(url))
            .slice(0, 3)
        );
        state.finderFeedbackGalleryDirty = false;
      }
      if (feedbackRequested) {
        setGalleryMode('feedback', { load: false, render: false });
      }
      if (requestedMode === 'pose') {
        state.poseSelectedImages = new Set(state.gallery.images
          .map((image, index) => finderSuggestionForImage(image, index) ? image.url : '')
          .filter(Boolean));
        const poseTag = context?.poseTag;
        if (poseTag?.label) $('#pose-tag-input').value = poseTag.label;
        $('#pose-control-role').value = POSE_ROLES.includes(poseTag?.defaultRole) ? poseTag.defaultRole : 'solo';
        setGalleryMode('pose', { load: true, render: false });
      }
      renderGallerySummary();
      renderImages();
      renderGalleries();
      $('#gallery-modal-kicker').textContent = displayHost(state.gallery.url);
      $('#gallery-modal-title').textContent = state.gallery.title;
      if (requestedMode === 'pose' || requestedMode === 'feedback') scrollToFinderSuggestion();
      prefetchAdjacentFinderGalleries();
    } catch (error) {
      if (
        detailRequest !== state.galleryDetailRequest
        || detailProfile !== state.activeProfile
        || !dialog.open
      ) return;
      $('#image-grid').replaceChildren();
      $('#images-empty').hidden = false;
      $('#selection-summary').textContent = errorMessage(error);
      toast('Could not open gallery', errorMessage(error), 'error');
    } finally {
      if (
        detailRequest !== state.galleryDetailRequest
        || detailProfile !== state.activeProfile
        || !dialog.open
      ) return;
      state.loadingDetail = false;
      $('#image-grid').setAttribute('aria-busy', 'false');
      updateSelectionUi();
      renderFinderGalleryReview();
    }
  }

  function renderImageSkeletons() {
    const grid = $('#image-grid');
    for (let index = 0; index < 10; index += 1) {
      const skeleton = document.createElement('div');
      skeleton.className = 'image-option skeleton-image';
      skeleton.setAttribute('aria-hidden', 'true');
      grid.append(skeleton);
    }
  }

  function renderGallerySummary() {
    const gallery = state.gallery;
    if (!gallery) return;
    const cover = $('#summary-cover');
    cover.replaceChildren();
    const placeholder = document.createElement('div');
    placeholder.className = 'image-placeholder';
    placeholder.innerHTML = '<svg><use href="#i-image"></use></svg>';
    cover.append(placeholder);
    const image = document.createElement('img');
    cover.append(image);
    loadImage(
      image,
      gallery.thumbnailUrl || gallery.images?.[0]?.previewUrl,
      gallery.title,
      { kind: 'foreground', lazy: false }
    );
    const status = galleryStatus(gallery);
    $('#summary-status').innerHTML = `<span class="status-badge ${status.className}">${status.label}</span>`;
    $('#summary-image-count').textContent = gallery.imageCount ? formatNumber(gallery.imageCount) : '—';
    $('#summary-source').textContent = displayHost(gallery.url);
    $('#summary-profile').textContent = gallery.saved ? state.activeProfile || 'Saved' : 'Not saved';
    const sourceLink = $('#gallery-source-link');
    const sourceUrl = safeUrl(gallery.url);
    sourceLink.href = sourceUrl || '#';
    sourceLink.hidden = !sourceUrl;
    const button = $('#modal-ignore');
    $('span', button).textContent = gallery.ignored ? 'Unignore gallery' : 'Ignore gallery';
    $('use', button).setAttribute('href', gallery.ignored ? '#i-eye' : '#i-eye-off');
  }

  function normalizePoseTag(item) {
    return {
      ...item,
      id: item?.id,
      label: String(item?.label || item?.name || '').trim(),
      slug: String(item?.slug || ''),
      defaultRole: POSE_ROLES.includes(item?.default_role) ? item.default_role : 'solo'
    };
  }

  function normalizePoseDraft(item) {
    const draft = item?.draft || item || {};
    const controls = draft.controls || {};
    return {
      revision: Number(draft.revision || 0),
      controls: Object.fromEntries(POSE_ROLES.map(role => [role, typeof controls[role] === 'string' ? controls[role] : ''])),
      targets: (Array.isArray(draft.targets) ? draft.targets : []).map(target => ({
        imageUrl: String(target.image_url || ''),
        ordinal: Number(target.ordinal || 0),
        poseTagId: target.pose_tag_id,
        poseSlug: String(target.pose_slug || ''),
        poseLabel: String(target.pose_label || ''),
        role: POSE_ROLES.includes(target.role) ? target.role : 'solo'
      })).filter(target => target.imageUrl && target.poseTagId !== undefined && target.poseTagId !== null)
    };
  }

  function currentPoseKey(gallery = state.gallery, profile = $('#modal-profile-select')?.value || state.activeProfile) {
    if (gallery?.id === undefined || gallery?.id === null || !profile) return '';
    return `${gallery.id}\n${profile}`;
  }

  function poseExportMatchesGallery(job, gallery = state.gallery) {
    if (!gallery || job?.kind !== 'pose_export') return false;
    if (
      job.galleryId !== undefined
      && job.galleryId !== null
      && gallery.id !== undefined
      && gallery.id !== null
      && String(job.galleryId) === String(gallery.id)
    ) return true;
    return Boolean(
      job.galleryUrl
      && gallery.url
      && normalizeHistoryUrl(job.galleryUrl) === normalizeHistoryUrl(gallery.url)
    );
  }

  function poseExportForDraft(
    gallery = state.gallery,
    profile = $('#modal-profile-select')?.value || state.activeProfile,
    revision = state.poseDraft.revision
  ) {
    const galleryJobs = state.jobs.filter(job => poseExportMatchesGallery(job, gallery));
    const active = galleryJobs.find(job => !isTerminalJob(job));
    if (active) {
      const exact = active.profile === profile
        && active.poseRevision !== null
        && Number(active.poseRevision) === Number(revision);
      return { state: exact ? 'queued' : 'active', job: active };
    }
    const completed = galleryJobs.find(job => (
      job.status === 'completed'
      && job.profile === profile
      && job.poseRevision !== null
      && Number(job.poseRevision) === Number(revision)
    ));
    return completed ? { state: 'exported', job: completed } : null;
  }

  function jobProgressRank(job) {
    if (isTerminalJob(job)) return 5;
    return {
      queued: 0,
      starting: 1,
      downloading: 2,
      running: 2,
      active: 2,
      canceling: 3
    }[job?.status] ?? 0;
  }

  function upsertJob(item) {
    if (!item) return null;
    const incoming = normalizeJob(item);
    if (incoming.id === undefined || incoming.id === null) return null;
    const existing = state.jobs.find(job => String(job.id) === String(incoming.id));
    const job = existing && jobProgressRank(existing) > jobProgressRank(incoming)
      ? { ...incoming, ...existing, poseRevision: existing.poseRevision ?? incoming.poseRevision }
      : incoming;
    state.jobs = [job, ...state.jobs.filter(candidate => String(candidate.id) !== String(job.id))];
    return job;
  }

  function renderOpenPoseExportState() {
    if ($('#gallery-modal').open && state.galleryMode === 'pose') {
      renderPosePreflight();
      renderFinderGalleryReview();
    }
  }

  function poseTargetFor(url) {
    return state.poseDraft.targets.find(target => target.imageUrl === url) || null;
  }

  function poseControlFor(url) {
    return POSE_ROLES.find(role => state.poseDraft.controls[role] === url) || '';
  }

  function poseAssignmentFor(url) {
    const controlRole = poseControlFor(url);
    if (controlRole) return { type: 'control', role: controlRole };
    const target = poseTargetFor(url);
    return target ? { type: 'target', ...target } : null;
  }

  function poseTagForInput(value) {
    const query = String(value || '').trim().toLocaleLowerCase();
    if (!query) return null;
    return state.poseTags.find(tag => tag.label.toLocaleLowerCase() === query || tag.slug.toLocaleLowerCase() === query) || null;
  }

  function renderPoseTagOptions() {
    const list = $('#pose-tag-options');
    list.replaceChildren();
    [...state.poseTags].sort((a, b) => a.label.localeCompare(b.label)).forEach(tag => {
      const option = document.createElement('option');
      option.value = tag.label;
      option.label = `${tag.label} · ${poseRoleLabel(tag.defaultRole)} control`;
      list.append(option);
    });
  }

  function renderPoseBadge(option, image) {
    $('.pose-role-badge', option)?.remove();
    option.classList.remove('has-pose-target', 'has-pose-control');
    if (state.galleryMode !== 'pose') return;
    const assignment = poseAssignmentFor(image.url);
    if (!assignment) return;
    const badge = document.createElement('span');
    badge.className = `pose-role-badge ${assignment.type}`;
    const marker = document.createElement('b');
    marker.textContent = assignment.type === 'target' ? 'T' : 'C';
    const label = document.createElement('span');
    label.textContent = assignment.type === 'target' ? (assignment.poseLabel || assignment.poseSlug || 'Target') : poseRoleLabel(assignment.role);
    badge.title = assignment.type === 'target'
      ? `Target: ${label.textContent} · ${poseRoleLabel(assignment.role)} control`
      : `${poseRoleLabel(assignment.role)} control`;
    badge.append(marker, label);
    option.append(badge);
    option.classList.add(assignment.type === 'target' ? 'has-pose-target' : 'has-pose-control');
  }

  function posePreflight() {
    const targets = state.poseDraft.targets;
    const assignedControls = POSE_ROLES.filter(role => state.poseDraft.controls[role]);
    const issues = [];
    if (!targets.length) issues.push('Add at least one target');
    targets.forEach(target => {
      if (!target.poseTagId) issues.push('A target has no pose');
      if (!state.poseDraft.controls[target.role]) issues.push(`${poseRoleLabel(target.role)} control is missing`);
    });
    if (!($('#modal-profile-select')?.value || state.activeProfile)) issues.push('Choose a destination profile');
    return { targets: targets.length, controls: assignedControls.length, issues: [...new Set(issues)] };
  }

  function renderPosePreflight() {
    const result = posePreflight();
    const button = $('#pose-export');
    const priorExport = poseExportForDraft();
    const blocksExport = ['queued', 'active'].includes(priorExport?.state)
      || (priorExport?.state === 'exported' && !state.poseDirty);
    let detail = `Ready to build ${formatNumber(result.targets)} paired image${result.targets === 1 ? '' : 's'}.`;
    if (result.issues.length) detail = result.issues.join(' · ');
    else if (priorExport?.state === 'queued') detail = 'This saved draft already has a pose dataset queued.';
    else if (priorExport?.state === 'active') detail = 'Another pose export for this gallery is still active.';
    else if (priorExport?.state === 'exported' && !state.poseDirty) detail = 'This saved draft has already been exported.';
    $('#pose-target-count').textContent = formatNumber(result.targets);
    $('#pose-control-count').textContent = formatNumber(result.controls);
    $('#pose-issue-count').textContent = formatNumber(result.issues.length);
    $('#pose-issue-count').classList.toggle('has-issues', Boolean(result.issues.length));
    $('#pose-preflight-detail').textContent = detail;
    if (!button.dataset.originalHtml) {
      const label = $('span', button);
      if (label) {
        if (priorExport?.state === 'queued') label.textContent = 'Queued';
        else if (priorExport?.state === 'active') label.textContent = 'Export active';
        else if (priorExport?.state === 'exported' && !state.poseDirty) label.textContent = 'Exported';
        else label.textContent = 'Download & organize';
      }
    }
    button.disabled = state.poseLoading
      || state.poseSaving
      || state.poseApplying
      || state.poseExporting
      || blocksExport
      || Boolean(result.issues.length);
  }

  function renderPoseSaveStatus(message = '') {
    const status = $('#pose-save-status');
    if (message) status.textContent = message;
    else if (state.poseLoading) status.textContent = 'Loading draft…';
    else if (state.poseSaving) status.textContent = 'Saving…';
    else if (state.poseDirty) status.textContent = 'Changes pending…';
    else if (state.poseLoadedKey) status.textContent = 'Draft saved';
    else status.textContent = 'Draft not loaded';
    status.classList.toggle('is-saving', state.poseSaving || state.poseDirty);
  }

  function renderPoseToolbar() {
    const isTarget = state.poseAssignment === 'target';
    const checked = state.poseSelectedImages.size;
    $$('[data-pose-assignment]').forEach(button => {
      const active = button.dataset.poseAssignment === state.poseAssignment;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
      button.disabled = state.poseLoading || state.poseApplying || state.poseExporting;
    });
    $('#pose-target-fields').hidden = !isTarget;
    $('#pose-control-hint').hidden = isTarget;
    const apply = $('#pose-apply-checked');
    const label = $('span', apply);
    label.textContent = checked ? `Apply to ${formatNumber(checked)} checked` : 'Apply to checked';
    const missingTag = isTarget && !$('#pose-tag-input').value.trim();
    apply.disabled = state.poseLoading || state.poseApplying || state.poseExporting || !checked || missingTag || (!isTarget && checked !== 1);
    $('#pose-clear-checked').disabled = state.poseApplying || state.poseExporting || !checked || ![...state.poseSelectedImages].some(url => poseAssignmentFor(url));
    renderPoseSaveStatus();
    renderPosePreflight();
  }

  function renderLightboxPoseDock() {
    const dock = $('#lightbox-pose-dock');
    const image = state.gallery?.images?.[state.lightboxIndex];
    dock.hidden = state.galleryMode !== 'pose' || !image;
    $('#lightbox-footer-hint').textContent = state.galleryMode === 'pose'
      ? 'Tag this image, then use the arrows to continue through the gallery'
      : 'Click or tap the image to toggle fit and actual size';
    if (dock.hidden) return;
    const assignment = poseAssignmentFor(image.url);
    $('#lightbox-pose-title').textContent = !assignment
      ? 'Not assigned'
      : assignment.type === 'control'
        ? `${poseRoleLabel(assignment.role)} control`
        : `Target · ${assignment.poseLabel || assignment.poseSlug}`;
    $$('[data-lightbox-control]').forEach(button => {
      const active = assignment?.type === 'control' && assignment.role === button.dataset.lightboxControl;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
      button.disabled = state.poseApplying || state.poseExporting;
    });
    $('#lightbox-pose-tag-input').value = assignment?.type === 'target' ? assignment.poseLabel : '';
    $('#lightbox-pose-control-role').value = assignment?.type === 'target' ? assignment.role : 'solo';
    updateLightboxTargetAvailability();
    $('#lightbox-clear-pose').disabled = state.poseApplying || state.poseExporting || !assignment;
  }

  function updateLightboxTargetAvailability() {
    $('#lightbox-set-target').disabled = state.poseApplying || state.poseExporting;
    $('#lightbox-set-target').title = 'Set this image as a pose target';
  }

  function renderPoseWorkspace() {
    renderPoseTagOptions();
    renderPoseToolbar();
    renderLightboxPoseDock();
  }

  async function loadPoseWorkspace({ force = false } = {}) {
    const key = currentPoseKey();
    if (!key || (key === state.poseLoadedKey && !force) || state.poseLoading) return;
    state.poseLoading = true;
    renderPoseWorkspace();
    const [galleryId, profile] = key.split('\n');
    try {
      const [tagsData, draftData] = await Promise.all([
        api('/api/pose-tags'),
        api(withParams(`/api/galleries/${encodeURIComponent(galleryId)}/pose-draft`, { profile }))
      ]);
      if (key !== currentPoseKey()) return;
      state.poseTags = apiItems(tagsData).map(normalizePoseTag).filter(tag => tag.id !== undefined && tag.label);
      state.poseDraft = normalizePoseDraft(draftData);
      state.poseLoadedKey = key;
      state.poseDirty = false;
      renderImages();
      renderPoseWorkspace();
    } catch (error) {
      if (key !== currentPoseKey()) return;
      renderPoseSaveStatus('Draft unavailable');
      toast('Could not load pose draft', errorMessage(error), 'error');
    } finally {
      if (key === currentPoseKey()) {
        state.poseLoading = false;
        renderPoseWorkspace();
      }
    }
  }

  function setGalleryMode(mode, { load = true, render = true } = {}) {
    const feedbackAvailable = finderFeedbackGalleryAvailable();
    const nextMode = mode === 'feedback' && feedbackAvailable ? 'feedback' : mode === 'pose' ? 'pose' : 'download';
    if (state.finderFeedbackGallerySaving && nextMode !== state.galleryMode) {
      toast('Feedback selection is saving', 'Wait for the save to finish before switching workflows.', 'info');
      return false;
    }
    if (state.galleryReviewBusy && nextMode !== state.galleryMode) return false;
    state.galleryMode = nextMode;
    const poseMode = state.galleryMode === 'pose';
    const feedbackMode = state.galleryMode === 'feedback';
    $('#gallery-modal').classList.toggle('is-pose-mode', poseMode);
    $('#gallery-modal').classList.toggle('is-feedback-mode', feedbackMode);
    $('#finder-feedback-gallery-tab').hidden = !feedbackAvailable;
    $$('[data-gallery-mode]').forEach(button => {
      const active = button.dataset.galleryMode === state.galleryMode;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', String(active));
    });
    $('#pose-toolbar').hidden = !poseMode;
    $('#finder-feedback-gallery-toolbar').hidden = !feedbackMode;
    $('#download-selection-total').hidden = poseMode || feedbackMode;
    $('#finder-feedback-gallery-total').hidden = !feedbackMode;
    $('#pose-preflight').hidden = !poseMode;
    $('#queue-download').hidden = poseMode || feedbackMode;
    $('#pose-export').hidden = !poseMode;
    const feedbackResult = finderFeedbackGalleryResult();
    const reviewedFeedback = ['accepted', 'rejected'].includes(feedbackResult?.review);
    $('#finder-feedback-gallery-save').hidden = !feedbackMode || !reviewedFeedback;
    $('.modal-profile').hidden = feedbackMode;
    $('#modal-profile-label').textContent = poseMode ? 'Organize in' : 'Download to';
    $('#image-picker-title').textContent = feedbackMode ? 'Review gallery feedback' : poseMode ? 'Prepare pose pairs' : 'Choose images';
    $('#select-all').hidden = feedbackMode;
    $('.picker-actions > span').hidden = feedbackMode;
    $('#select-all').textContent = poseMode ? 'Check all' : 'Select all';
    $('#select-none').textContent = feedbackMode ? 'Clear selection' : poseMode ? 'Uncheck all' : 'Clear';
    $('#finder-feedback-gallery-review').textContent = feedbackResult?.review || 'Feedback';
    const feedbackHelp = $('#finder-feedback-gallery-toolbar p');
    if (feedbackHelp) {
      feedbackHelp.textContent = ['accepted', 'rejected'].includes(feedbackResult?.review)
        ? 'Pick up to 3 images from anywhere in this gallery, then save the edited feedback selection.'
        : 'Pick up to 3 images from anywhere in this gallery, then use Accept, Maybe, or Reject above.';
    }
    $('#finder-feedback-prepare-pose').hidden = feedbackResult?.review !== 'accepted';
    if (render) renderImages();
    renderPoseWorkspace();
    renderFinderGalleryReview();
    if (poseMode && load) loadPoseWorkspace();
    return true;
  }

  function prepareFinderPoseFromFeedback() {
    const result = finderFeedbackGalleryResult();
    if (!result || result.review !== 'accepted') return;
    const poseTag = state.galleryContext?.poseTag || finderPoseTagForScan();
    state.poseSelectedImages = new Set(state.finderFeedbackGallerySelection);
    state.poseAssignment = 'target';
    if (poseTag?.label) $('#pose-tag-input').value = poseTag.label;
    $('#pose-control-role').value = POSE_ROLES.includes(poseTag?.defaultRole) ? poseTag.defaultRole : 'solo';
    setGalleryMode('pose');
    const role = $('#pose-control-role').value;
    toast(
      'Target candidates highlighted',
      `Uncheck the target ${state.poseSelectedImages.size === 1 ? 'candidate' : 'candidates'}, check one ${poseRoleLabel(role).toLowerCase()} control image, and use Set ${poseRoleLabel(role).toLowerCase()}. Then re-check the highlighted targets. Nothing has been assigned yet.`,
      'info',
      7000
    );
    announce('Pose dataset mode opened. Set a matching control, then apply the highlighted Finder images as targets.');
  }

  async function handlePoseAssignmentButton(button) {
    if (state.galleryReviewBusy) return;
    const assignment = button.dataset.poseAssignment;
    if (!['target', ...POSE_ROLES].includes(assignment)) return;
    state.poseAssignment = assignment;
    renderPoseToolbar();
    if (state.poseLoading) {
      toast('Pose draft is loading', 'Try the assignment again when the draft is ready.', 'info');
      return;
    }
    const selected = state.poseSelectedImages.size;
    if (!selected) {
      toast(
        assignment === 'target' ? 'Check target images' : `Check one ${poseRoleLabel(assignment).toLowerCase()} control`,
        assignment === 'target'
          ? 'Check one or more images, then use Apply targets again.'
          : 'Check exactly one image in the gallery, then use this control button again.',
        'info'
      );
      return;
    }
    if (assignment !== 'target') {
      if (selected !== 1) {
        toast('Choose one control image', `A ${poseRoleLabel(assignment).toLowerCase()} control uses exactly one checked image.`, 'info');
        return;
      }
      await applyPoseAssignment(state.poseSelectedImages, assignment, { button, clearChecked: true });
      return;
    }
    if (!$('#pose-tag-input').value.trim()) {
      toast('Name the pose', 'Choose an existing pose tag or enter a new one before applying targets.', 'info');
      $('#pose-tag-input').focus();
      return;
    }
    await applyPoseAssignment(state.poseSelectedImages, 'target', { button, clearChecked: true });
  }

  function poseDraftBody(expectedRevision = state.poseDraft.revision) {
    return {
      expected_revision: Number(expectedRevision || 0),
      controls: Object.fromEntries(POSE_ROLES.map(role => [role, state.poseDraft.controls[role] || null])),
      targets: state.poseDraft.targets.map(target => ({ image_url: target.imageUrl, pose_tag_id: target.poseTagId, role: target.role }))
    };
  }

  function markPoseDraftDirty() {
    state.poseDirty = true;
    state.poseMutation += 1;
    renderPoseWorkspace();
    window.clearTimeout(state.poseSaveTimer);
    state.poseSaveTimer = window.setTimeout(() => savePoseDraft(), 650);
  }

  async function savePoseDraft() {
    window.clearTimeout(state.poseSaveTimer);
    if (state.poseSaving) return state.poseSavePromise;
    if (!state.poseDirty || !state.poseLoadedKey) return null;
    const key = state.poseLoadedKey;
    const [galleryId, profile] = key.split('\n');
    state.poseSaving = true;
    renderPoseWorkspace();
    state.poseSavePromise = (async () => {
      while (state.poseDirty && state.poseLoadedKey === key) {
        const mutation = state.poseMutation;
        state.poseDirty = false;
        try {
          const data = await api(withParams(`/api/galleries/${encodeURIComponent(galleryId)}/pose-draft`, { profile }), {
            method: 'PUT',
            body: poseDraftBody()
          });
          if (state.poseLoadedKey !== key) return;
          state.poseDraft.revision = Number((data?.draft || data)?.revision ?? state.poseDraft.revision + 1);
          if (mutation !== state.poseMutation) state.poseDirty = true;
        } catch (error) {
          if (error.status === 409 && state.poseLoadedKey === key) {
            const latestData = error.data?.draft
              ? { draft: error.data.draft }
              : await api(withParams(`/api/galleries/${encodeURIComponent(galleryId)}/pose-draft`, { profile }));
            if (state.poseLoadedKey !== key) return;
            state.poseDraft = normalizePoseDraft(latestData);
            state.poseDirty = false;
            state.poseMutation += 1;
            renderImages();
            renderPoseWorkspace();
            toast('Newer pose draft loaded', 'Your conflicting local edit was not saved. Review the newer draft and apply the change again.', 'info', 7000);
            return;
          }
          state.poseDirty = true;
          renderPoseSaveStatus('Draft not saved');
          throw error;
        }
      }
    })();
    try {
      await state.poseSavePromise;
    } catch (error) {
      toast('Could not save pose draft', errorMessage(error), 'error');
    } finally {
      state.poseSaving = false;
      state.poseSavePromise = null;
      renderPoseWorkspace();
    }
    return null;
  }

  async function flushPoseDraft() {
    window.clearTimeout(state.poseSaveTimer);
    if (state.poseSaving && state.poseSavePromise) await state.poseSavePromise.catch(() => null);
    if (state.poseDirty) await savePoseDraft();
  }

  async function ensurePoseTag(label, defaultRole) {
    const cleanLabel = String(label || '').trim().replace(/\s+/g, ' ');
    if (!cleanLabel) throw new ApiError('Enter a pose name first.');
    const existing = poseTagForInput(cleanLabel);
    if (existing) return existing;
    const data = await api('/api/pose-tags', { method: 'POST', body: { label: cleanLabel, default_role: defaultRole } });
    const tag = normalizePoseTag(data?.tag || data);
    state.poseTags.push(tag);
    renderPoseTagOptions();
    toast('Pose created', `${tag.label} defaults to the ${poseRoleLabel(tag.defaultRole).toLowerCase()} control.`, 'success');
    return tag;
  }

  function clearPoseAssignment(url) {
    POSE_ROLES.forEach(role => { if (state.poseDraft.controls[role] === url) state.poseDraft.controls[role] = ''; });
    state.poseDraft.targets = state.poseDraft.targets.filter(target => target.imageUrl !== url);
  }

  function setPoseControl(url, role) {
    const replaced = state.poseDraft.controls[role];
    clearPoseAssignment(url);
    state.poseDraft.controls[role] = url;
    return replaced && replaced !== url;
  }

  function setPoseTarget(url, tag, role) {
    clearPoseAssignment(url);
    state.poseDraft.targets.push({
      imageUrl: url,
      ordinal: (state.gallery?.images || []).findIndex(image => image.url === url) + 1,
      poseTagId: tag.id,
      poseSlug: tag.slug,
      poseLabel: tag.label,
      role
    });
  }

  async function applyPoseAssignment(urls, assignment, { button = null, clearChecked = false } = {}) {
    const selected = [...urls];
    if (!selected.length || state.poseLoading || state.poseApplying || state.poseExporting || state.galleryReviewBusy) return false;
    if (assignment !== 'target' && selected.length !== 1) {
      toast('Choose one control image', 'Each Solo, Couple, or Group slot uses exactly one control.', 'info');
      return false;
    }
    if (assignment === 'target') {
      const occupiedControl = selected.map(poseControlFor).find(Boolean);
      if (occupiedControl) {
        toast('This image is a control', `Replace the ${poseRoleLabel(occupiedControl).toLowerCase()} control before tagging it as a target.`, 'info');
        return false;
      }
    } else {
      const previousRole = poseControlFor(selected[0]);
      if (previousRole && previousRole !== assignment && state.poseDraft.targets.some(target => target.role === previousRole)) {
        toast('This control is still in use', `Replace the ${poseRoleLabel(previousRole).toLowerCase()} control before changing its role.`, 'info');
        return false;
      }
    }
    const poseKey = currentPoseKey();
    const applyRequest = ++state.poseApplyRequest;
    const stillCurrent = () => (
      applyRequest === state.poseApplyRequest
      && $('#gallery-modal').open
      && poseKey === currentPoseKey()
    );
    state.poseApplying = true;
    renderPoseWorkspace();
    renderFinderGalleryReview();
    if (button) setButtonBusy(button, true, assignment === 'target' ? 'Tagging…' : 'Assigning…');
    try {
      if (assignment === 'target') {
        const input = button?.closest('#lightbox-pose-dock') ? $('#lightbox-pose-tag-input') : $('#pose-tag-input');
        const roleSelect = button?.closest('#lightbox-pose-dock') ? $('#lightbox-pose-control-role') : $('#pose-control-role');
        const role = POSE_ROLES.includes(roleSelect.value) ? roleSelect.value : 'solo';
        const tag = await ensurePoseTag(input.value, role);
        if (!stillCurrent()) return false;
        selected.forEach(url => setPoseTarget(url, tag, role));
        announce(`${selected.length} image${selected.length === 1 ? '' : 's'} assigned as ${tag.label} targets`);
      } else {
        const replaced = setPoseControl(selected[0], assignment);
        toast(`${poseRoleLabel(assignment)} control set`, replaced ? 'The previous image in this control slot was replaced.' : 'Targets can now use this control.', 'success');
      }
      if (clearChecked) state.poseSelectedImages.clear();
      markPoseDraftDirty();
      renderImages();
      renderPoseWorkspace();
      return true;
    } catch (error) {
      if (!stillCurrent()) return false;
      toast('Could not assign image', errorMessage(error), 'error');
      return false;
    } finally {
      if (button) setButtonBusy(button, false);
      if (applyRequest === state.poseApplyRequest) {
        state.poseApplying = false;
        renderPoseWorkspace();
        renderFinderGalleryReview();
      }
    }
  }

  function clearCheckedPoseAssignments() {
    if (!state.poseSelectedImages.size) return;
    const blockedRole = POSE_ROLES.find(role => (
      state.poseSelectedImages.has(state.poseDraft.controls[role]) &&
      state.poseDraft.targets.some(target => target.role === role && !state.poseSelectedImages.has(target.imageUrl))
    ));
    if (blockedRole) {
      toast('Control is still in use', `Also check its ${poseRoleLabel(blockedRole).toLowerCase()} targets, or replace that control first.`, 'info');
      return;
    }
    state.poseSelectedImages.forEach(clearPoseAssignment);
    const count = state.poseSelectedImages.size;
    markPoseDraftDirty();
    renderImages();
    renderPoseWorkspace();
    announce(`Assignments removed from ${count} image${count === 1 ? '' : 's'}`);
  }

  function syncPoseTagDefault(input, select) {
    const tag = poseTagForInput(input.value);
    if (tag) select.value = tag.defaultRole;
    if (input.id === 'lightbox-pose-tag-input') updateLightboxTargetAvailability();
    else renderPoseToolbar();
  }

  function renderImages() {
    const grid = $('#image-grid');
    grid.replaceChildren();
    const images = state.gallery?.images || [];
    const activeSelection = state.galleryMode === 'feedback'
      ? state.finderFeedbackGallerySelection
      : state.galleryMode === 'pose' ? state.poseSelectedImages : state.selectedImages;
    const pendingFeedbackUrls = new Set(
      state.galleryMode === 'feedback'
        ? finderFeedbackGalleryResult()?.feedbackPendingImageUrls || []
        : []
    );
    $('#images-empty').hidden = Boolean(images.length);
    images.forEach((image, index) => {
      const option = document.createElement('div');
      option.className = `image-option${activeSelection.has(image.url) ? ' is-selected' : ''}`;
      option.classList.toggle('is-downloaded', Boolean(image.downloaded));
      option.dataset.imageUrl = image.url;
      option.dataset.imageIndex = String(index);
      option.title = image.filename;
      const input = document.createElement('input');
      input.id = `gallery-image-${index}`;
      input.type = 'checkbox';
      input.checked = activeSelection.has(image.url);
      input.setAttribute(
        'aria-label',
        state.galleryMode === 'feedback'
          ? `Use ${image.filename} as Finder pose feedback`
          : state.galleryMode === 'pose'
            ? `Check ${image.filename} for pose tagging`
            : `Select ${image.filename} for download`
      );
      const previewButton = document.createElement('button');
      previewButton.className = 'image-preview-button';
      previewButton.type = 'button';
      previewButton.setAttribute('aria-label', `View ${image.filename} full size, image ${index + 1} of ${images.length}`);
      const placeholder = document.createElement('div');
      placeholder.className = 'image-placeholder';
      placeholder.innerHTML = '<svg><use href="#i-image"></use></svg>';
      const preview = document.createElement('img');
      preview.loading = 'lazy';
      preview.decoding = 'async';
      loadImage(preview, image.previewUrl, image.filename, { kind: 'foreground' });
      const previewHint = document.createElement('span');
      previewHint.className = 'image-preview-hint';
      previewHint.innerHTML = '<svg><use href="#i-maximize"></use></svg><span>Full size</span>';
      previewButton.append(placeholder, preview, previewHint);
      const check = document.createElement('label');
      check.className = 'image-check';
      check.htmlFor = input.id;
      check.setAttribute('aria-label', input.getAttribute('aria-label'));
      check.innerHTML = '<svg><use href="#i-check"></use></svg>';
      const number = document.createElement('span');
      number.className = 'image-number';
      number.textContent = String(index + 1).padStart(2, '0');
      option.append(input, previewButton, check, number);
      if (activeSelection.has(image.url) && pendingFeedbackUrls.has(image.url)) {
        option.classList.add('is-feedback-manual-only');
        const manualOnly = document.createElement('span');
        manualOnly.className = 'finder-feedback-manual-badge';
        manualOnly.textContent = 'Saved · pose pending';
        manualOnly.title = 'Saved for your review; this image is not currently pose-usable and does not currently affect automatic ranking';
        option.append(manualOnly);
      }
      if (image.downloaded) {
        const saved = document.createElement('span');
        saved.className = 'image-saved';
        saved.innerHTML = '<svg><use href="#i-check"></use></svg> Saved';
        option.append(saved);
      }
      renderPoseBadge(option, image);
      const finderSuggestion = finderSuggestionForImage(image, index);
      const finderFeedbackCandidate = state.galleryMode === 'pose'
        && Boolean(state.galleryContext?.finderFeedbackResultKey)
        && finderFeedbackGalleryResult()?.review === 'accepted'
        && state.finderFeedbackGallerySelection.has(image.url);
      if (finderSuggestion || finderFeedbackCandidate) {
        option.classList.add('is-finder-suggestion');
        option.classList.toggle('is-finder-feedback-candidate', finderFeedbackCandidate);
        const badge = document.createElement('span');
        badge.className = 'finder-suggestion-badge';
        const score = Number(finderSuggestion?.score);
        badge.innerHTML = '<svg><use href="#i-spark"></use></svg><span></span>';
        $('span', badge).textContent = finderFeedbackCandidate
          ? 'Finder target candidate'
          : Number.isFinite(score) ? `Finder · ${score.toFixed(2)} similarity` : 'Finder suggestion';
        option.append(badge);
      }
      grid.append(option);
    });
    updateSelectionUi();
  }

  function setLightboxZoom(zoomed) {
    state.lightboxZoomed = Boolean(zoomed);
    const modal = $('#lightbox-modal');
    modal.classList.toggle('is-zoomed', state.lightboxZoomed);
    const button = $('#lightbox-zoom');
    button.setAttribute('aria-pressed', String(state.lightboxZoomed));
    $('#lightbox-zoom-label').textContent = state.lightboxZoomed ? 'Fit image' : 'Actual size';
    $('use', button).setAttribute('href', state.lightboxZoomed ? '#i-minimize' : '#i-maximize');
    if (!state.lightboxZoomed) $('#lightbox-stage').scrollTo({ top: 0, left: 0 });
  }

  function renderLightboxImage() {
    const images = state.gallery?.images || [];
    if (!images.length || state.lightboxIndex < 0) return;
    state.lightboxIndex = ((state.lightboxIndex % images.length) + images.length) % images.length;
    const image = images[state.lightboxIndex];
    setLightboxZoom(false);

    $('#lightbox-counter').textContent = `Image ${state.lightboxIndex + 1} of ${images.length}`;
    $('#lightbox-title').textContent = image.filename;
    const sourceLink = $('#lightbox-source-link');
    const sourceUrl = safeUrl(image.fullUrl || image.previewUrl);
    sourceLink.href = sourceUrl || '#';
    sourceLink.hidden = !sourceUrl;
    $('#lightbox-previous').disabled = images.length < 2;
    $('#lightbox-next').disabled = images.length < 2;

    const placeholder = $('#lightbox-stage .image-placeholder');
    placeholder.classList.remove('is-loaded', 'is-error');
    const previous = $('#lightbox-image');
    const loadToken = ++state.lightboxLoadToken;
    const display = document.createElement('img');
    display.id = 'lightbox-image';
    display.alt = `${image.filename}, full-resolution preview`;
    display.decoding = 'async';
    cancelMediaLoadForImage(previous, { drain: false });
    previous.removeAttribute('src');
    previous.replaceWith(display);

    const candidates = [...new Set([safeUrl(image.fullUrl), safeUrl(image.previewUrl)].filter(Boolean))];
    let candidateIndex = 0;
    const loadCandidate = () => {
      const source = candidates[candidateIndex];
      if (!source) {
        cancelMediaLoadForImage(display);
        display.hidden = true;
        display.removeAttribute('src');
        placeholder.classList.add('is-error');
        return;
      }
      display.hidden = false;
      loadImage(display, source, display.alt, {
        kind: 'critical',
        lazy: false,
        onError: () => {
          if (loadToken !== state.lightboxLoadToken) return;
          candidateIndex += 1;
          loadCandidate();
        }
      });
    };
    display.addEventListener('load', () => {
      if (loadToken === state.lightboxLoadToken) placeholder.classList.add('is-loaded');
    });
    loadCandidate();
    renderLightboxPoseDock();
    announce(`Viewing image ${state.lightboxIndex + 1} of ${images.length}: ${image.filename}`);
  }

  function openLightbox(index, trigger = null) {
    const images = state.gallery?.images || [];
    if (!images[index]) return;
    state.lightboxIndex = index;
    state.lightboxTrigger = trigger;
    renderLightboxImage();
    const dialog = $('#lightbox-modal');
    if (!dialog.open) dialog.showModal();
  }

  function navigateLightbox(offset) {
    const images = state.gallery?.images || [];
    if (images.length < 2) return;
    state.lightboxIndex += offset;
    renderLightboxImage();
  }

  function resetLightbox() {
    const trigger = state.lightboxTrigger;
    state.lightboxLoadToken += 1;
    state.lightboxIndex = -1;
    state.lightboxTrigger = null;
    setLightboxZoom(false);
    const image = $('#lightbox-image');
    cancelMediaLoadForImage(image);
    image?.removeAttribute('src');
    $('#lightbox-stage .image-placeholder')?.classList.remove('is-loaded', 'is-error');
    if ($('#gallery-modal').open && trigger?.isConnected) trigger.focus({ preventScroll: true });
  }

  function toggleImage(url, checked) {
    if (state.galleryReviewBusy) return;
    const feedbackMode = state.galleryMode === 'feedback';
    const selection = feedbackMode
      ? state.finderFeedbackGallerySelection
      : state.galleryMode === 'pose' ? state.poseSelectedImages : state.selectedImages;
    if (feedbackMode && checked && !selection.has(url) && selection.size >= 3) {
      const option = $$('.image-option', $('#image-grid')).find(item => item.dataset.imageUrl === url);
      const input = option && $('input', option);
      if (input) input.checked = false;
      toast('Choose up to 3 images', 'Remove one feedback image before adding another.', 'info');
      announce('Finder feedback selection is limited to 3 images.');
      return;
    }
    const changed = checked ? !selection.has(url) : selection.has(url);
    if (checked) selection.add(url);
    else selection.delete(url);
    const option = $$('.image-option', $('#image-grid')).find(item => item.dataset.imageUrl === url);
    option?.classList.toggle('is-selected', checked);
    if (feedbackMode && changed) state.finderFeedbackGalleryDirty = true;
    updateSelectionUi();
  }

  function selectAllImages(selected) {
    if (state.galleryReviewBusy) return;
    if (state.galleryMode === 'feedback' && selected) {
      toast('Choose feedback images individually', 'You can select up to 3 images from this gallery.', 'info');
      return;
    }
    const selection = new Set(selected ? (state.gallery?.images || []).map(image => image.url) : []);
    if (state.galleryMode === 'feedback') {
      if (state.finderFeedbackGallerySelection.size) state.finderFeedbackGalleryDirty = true;
      state.finderFeedbackGallerySelection = selection;
    } else if (state.galleryMode === 'pose') state.poseSelectedImages = selection;
    else state.selectedImages = selection;
    $$('.image-option', $('#image-grid')).forEach(option => {
      const checked = selection.has(option.dataset.imageUrl);
      option.classList.toggle('is-selected', checked);
      const input = $('input', option);
      if (input) input.checked = checked;
    });
    updateSelectionUi();
  }

  function updateSelectionUi() {
    const feedbackMode = state.galleryMode === 'feedback';
    const selection = feedbackMode
      ? state.finderFeedbackGallerySelection
      : state.galleryMode === 'pose' ? state.poseSelectedImages : state.selectedImages;
    const selected = selection.size;
    const total = state.gallery?.images?.length || 0;
    $('#selected-count').textContent = formatNumber(selected);
    $('#finder-feedback-gallery-count').textContent = formatNumber(selected);
    const downloaded = state.gallery?.downloadedImages || 0;
    const feedbackResult = finderFeedbackGalleryResult();
    const pendingFeedbackCount = feedbackResult?.feedbackPendingImageUrls?.length || 0;
    $('#selection-summary').textContent = state.loadingDetail
      ? 'Scanning the source page…'
      : feedbackMode
        ? `${formatNumber(selected)} of 3 feedback images selected${
          state.finderFeedbackGalleryDirty
            ? ' · unsaved changes'
            : pendingFeedbackCount
              ? ` · selection saved · ${formatNumber(pendingFeedbackCount)} pose-pending (not currently pose-usable)`
              : feedbackResult?.feedbackAnalysisProvided ? ' · selection saved and eligible for future ranking' : ' · selection saved'
        }`
        : state.galleryMode === 'pose'
          ? state.galleryContext?.finderFeedbackResultKey
            && feedbackResult?.review === 'accepted'
            && state.finderFeedbackGallerySelection.size
            ? selected
              ? state.poseDraft.controls[$('#pose-control-role').value]
                ? `${formatNumber(selected)} checked · control ready—apply the highlighted Finder target ${selected === 1 ? 'candidate' : 'candidates'}`
                : `${formatNumber(selected)} target ${selected === 1 ? 'candidate' : 'candidates'} checked · uncheck ${selected === 1 ? 'it' : 'them'} and check one ${poseRoleLabel($('#pose-control-role').value).toLowerCase()} control first`
              : state.poseDraft.controls[$('#pose-control-role').value]
                ? `Control ready · re-check the highlighted Finder target ${state.finderFeedbackGallerySelection.size === 1 ? 'candidate' : 'candidates'}, then apply targets`
                : `Finder target ${state.finderFeedbackGallerySelection.size === 1 ? 'candidate is' : 'candidates are'} highlighted · check one control image first`
            : `${formatNumber(selected)} checked for bulk tagging · ${formatNumber(state.poseDraft.targets.length)} targets assigned`
          : `${formatNumber(selected)} of ${formatNumber(total)} selected${downloaded ? ` · ${formatNumber(downloaded)} already saved` : ''}`;
    $('#queue-download').disabled = state.loadingDetail || !selected || !state.activeProfile;
    const saveFeedback = $('#finder-feedback-gallery-save');
    saveFeedback.disabled = state.loadingDetail
      || state.finderFeedbackGallerySaving
      || Boolean(feedbackResult?.feedbackSaving)
      || !state.finderFeedbackGalleryDirty
      || !['accepted', 'rejected'].includes(feedbackResult?.review)
      || (feedbackResult?.review === 'accepted' && !selected);
    saveFeedback.title = feedbackResult?.review === 'accepted' && !selected
      ? 'Accepted feedback needs at least one selected image'
      : !['accepted', 'rejected'].includes(feedbackResult?.review)
        ? 'Use Accept or Reject above to save this selection'
        : state.finderFeedbackGalleryDirty ? 'Save this Finder feedback selection' : 'Selection is already saved';
    renderFinderGalleryReview();
    renderPoseToolbar();
  }

  async function saveFinderGalleryFeedbackSelection() {
    const result = finderFeedbackGalleryResult();
    const review = result?.review;
    if (
      !result
      || !['accepted', 'rejected'].includes(review)
      || state.finderFeedbackGallerySaving
      || state.finderFeedbackBusy
      || state.galleryReviewBusy
      || result.feedbackSaving
    ) return;
    const feedbackImageUrls = [...state.finderFeedbackGallerySelection];
    const resultKey = String(result.key);
    const galleryId = String(state.gallery?.id ?? '');
    if (review === 'accepted' && !feedbackImageUrls.length) {
      toast('Choose a matching image', 'Accepted feedback needs at least one selected gallery image.', 'info');
      return;
    }
    const button = $('#finder-feedback-gallery-save');
    state.finderFeedbackGallerySaving = true;
    setButtonBusy(button, true, 'Saving…');
    updateSelectionUi();
    try {
      const saved = await reviewFinderResult(result, review, null, { feedbackImageUrls });
      if (!saved) return;
      const sameGallery = String(state.gallery?.id ?? '') === galleryId
        && String(state.galleryContext?.finderFeedbackResultKey ?? '') === resultKey;
      if (!sameGallery) return;
      state.finderFeedbackGalleryDirty = false;
      if (state.galleryContext) state.galleryContext.feedbackImageUrls = [...feedbackImageUrls];
      const updated = finderFeedbackGalleryResult();
      const galleryUrls = new Set((state.gallery?.images || []).map(image => image.url));
      state.finderFeedbackGallerySelection = new Set(
        (updated?.feedbackImageUrls || feedbackImageUrls).filter(url => galleryUrls.has(url)).slice(0, 3)
      );
      const pending = updated?.feedbackPendingImageUrls?.length || 0;
      const savedCount = state.finderFeedbackGallerySelection.size;
      toast(
        'Feedback selection saved',
        pending
          ? `All ${formatNumber(savedCount)} selected ${savedCount === 1 ? 'image was' : 'images were'} saved. ${formatNumber(pending)} ${pending === 1 ? 'is' : 'are'} not currently pose-usable, so ${pending === 1 ? 'it remains' : 'they remain'} selected for review but ${pending === 1 ? 'does' : 'do'} not currently affect automatic ranking.`
          : `${formatNumber(savedCount)} ${review} pose-feedback ${savedCount === 1 ? 'image' : 'images'} saved and eligible for future ranking.`,
        'success',
        pending ? 8500 : 4500
      );
      announce(`Finder feedback selection saved for ${result.title}.`);
      renderImages();
    } finally {
      state.finderFeedbackGallerySaving = false;
      setButtonBusy(button, false);
      updateSelectionUi();
    }
  }

  async function queueGallery() {
    const gallery = state.gallery;
    if (
      !gallery
      || !state.selectedImages.size
      || state.galleryReviewBusy
      || state.finderFeedbackGallerySaving
      || state.poseApplying
    ) return;
    const button = $('#queue-download');
    const profile = $('#modal-profile-select').value || state.activeProfile;
    const payload = {
      title: gallery.title,
      profile,
      image_urls: [...state.selectedImages]
    };
    if (gallery.id !== undefined && gallery.id !== null) payload.gallery_id = gallery.id;
    else payload.gallery_url = gallery.url;
    setButtonBusy(button, true, 'Queuing…');
    try {
      await api('/api/downloads', { method: 'POST', body: payload });
      const count = state.selectedImages.size;
      invalidateGalleryDetailCache(gallery.id);
      const item = state.galleries.find(candidate => String(candidate.id) === String(gallery.id));
      if (item) item.queued = true;
      closeModal($('#lightbox-modal'));
      $('#gallery-modal').close();
      renderGalleries();
      toast('Added to queue', `${formatNumber(count)} images will download to “${profile}”.`);
      await loadJobs({ quiet: true });
    } catch (error) {
      toast('Could not start download', errorMessage(error), 'error');
    } finally { setButtonBusy(button, false); }
  }

  async function exportPoseDataset() {
    const gallery = state.gallery;
    const profile = $('#modal-profile-select').value || state.activeProfile;
    const currentExport = poseExportForDraft(gallery, profile);
    if (
      !gallery
      || !profile
      || state.poseExporting
      || ['queued', 'active'].includes(currentExport?.state)
      || (currentExport?.state === 'exported' && !state.poseDirty)
    ) return;
    const button = $('#pose-export');
    const poseKey = currentPoseKey(gallery, profile);
    const exportRequest = ++state.poseExportRequest;
    let expectedRevision = null;
    const stillCurrent = () => (
      exportRequest === state.poseExportRequest
      && $('#gallery-modal').open
      && poseKey === currentPoseKey()
    );
    state.poseExporting = true;
    renderPoseWorkspace();
    renderFinderGalleryReview();
    setButtonBusy(button, true, 'Preparing…');
    try {
      await flushPoseDraft();
      if (!stillCurrent()) return;
      if (state.poseDirty) throw new ApiError('The pose draft could not be saved. Try again before exporting.');
      const preflight = posePreflight();
      if (preflight.issues.length) throw new ApiError(preflight.issues.join(' · '));
      expectedRevision = Number(state.poseDraft.revision);
      const priorExport = poseExportForDraft(gallery, profile, expectedRevision);
      if (priorExport) return;
      const data = await api('/api/pose-exports', {
        method: 'POST',
        body: { gallery_id: gallery.id, profile, expected_revision: expectedRevision }
      });
      const pairs = Number(data?.job?.pair_count ?? preflight.targets);
      const queuedJob = upsertJob(data?.job);
      if (queuedJob) {
        renderJobs();
        renderGalleries();
      }
      if (stillCurrent()) {
        state.poseExporting = false;
        setButtonBusy(button, false);
        renderPoseWorkspace();
        renderFinderGalleryReview();
      }
      toast('Pose dataset queued', `${formatNumber(pairs)} pair${pairs === 1 ? '' : 's'} will download and organize in “${profile}”. Use Next when you are ready for another gallery.`, 'success');
      await loadJobs({ quiet: true });
    } catch (error) {
      const activeJob = error instanceof ApiError && error.status === 409
        ? upsertJob(error.data?.job)
        : null;
      if (activeJob) {
        renderJobs();
        renderGalleries();
        const exact = poseExportMatchesGallery(activeJob, gallery)
          && activeJob.profile === profile
          && expectedRevision !== null
          && activeJob.poseRevision !== null
          && Number(activeJob.poseRevision) === expectedRevision;
        toast(
          exact ? 'Pose dataset already queued' : 'Another pose export is active',
          exact
            ? 'The existing queued job was restored in this review window.'
            : 'Finish or cancel the active export for this gallery before queuing this draft.',
          'info'
        );
        await loadJobs({ quiet: true });
      } else {
        toast('Could not export pose dataset', errorMessage(error), 'error');
      }
    } finally {
      if (exportRequest === state.poseExportRequest) {
        state.poseExporting = false;
        setButtonBusy(button, false);
        renderPoseWorkspace();
        renderFinderGalleryReview();
      }
    }
  }

  async function loadJobs({ quiet = false } = {}) {
    try {
      const data = await api('/api/downloads');
      const previousJobs = new Map(state.jobs.map(job => [String(job.id), job]));
      const nextJobs = apiItems(data, 'downloads').map(normalizeJob);
      nextJobs.forEach(job => {
        const previous = previousJobs.get(String(job.id));
        if (
          previous
          && previous.status === job.status
          && previous.complete === job.complete
        ) return;
        if (job.galleryId !== undefined && job.galleryId !== null) {
          invalidateGalleryDetailCache(job.galleryId);
          return;
        }
        const listItem = state.galleries.find(gallery => (
          job.galleryUrl
          && normalizeHistoryUrl(gallery.url) === normalizeHistoryUrl(job.galleryUrl)
        ));
        if (listItem) invalidateGalleryDetailCache(listItem.id);
      });
      state.jobs = nextJobs;
      state.galleries.forEach(gallery => {
        gallery.queued = state.jobs.some(job => !isTerminalJob(job) && (
          String(job.galleryId) === String(gallery.id) ||
          (job.galleryUrl && normalizeHistoryUrl(job.galleryUrl) === normalizeHistoryUrl(gallery.url))
        ));
      });
      renderJobs();
      renderGalleries();
      renderOpenPoseExportState();
      scheduleJobPoll();
    } catch (error) {
      if (!quiet) toast('Could not load queue', errorMessage(error), 'error');
      scheduleJobPoll(8000);
    }
  }

  function scheduleJobPoll(delay = null) {
    window.clearTimeout(state.queueTimer);
    const hasWork = state.jobs.some(job => !isTerminalJob(job));
    const fallbackDelay = state.eventConnected ? (hasWork ? 15000 : 30000) : (hasWork ? 1800 : 10000);
    state.queueTimer = window.setTimeout(() => loadJobs({ quiet: true }), delay ?? fallbackDelay);
  }

  function jobFilterMatch(job) {
    if (state.jobFilter === 'active') return !isTerminalJob(job);
    if (state.jobFilter === 'complete') return isTerminalJob(job);
    return true;
  }

  function renderJobs() {
    const list = $('#job-list');
    const jobs = state.jobs.filter(jobFilterMatch);
    list.replaceChildren();
    jobs.forEach(job => {
      const fragment = $('#job-template').content.cloneNode(true);
      const row = $('.job-row', fragment);
      row.dataset.jobId = String(job.id);
      row.classList.toggle('is-completed', job.status === 'completed');
      row.classList.toggle('is-partial', job.status === 'completed_with_errors');
      row.classList.toggle('is-failed', job.status === 'failed');
      row.classList.toggle('is-pose-export', job.kind === 'pose_export');
      if (job.kind === 'pose_export') $('use', $('.job-thumb', row)).setAttribute('href', '#i-layers');
      loadImage($('.job-thumb img', row), job.thumbnailUrl, '');
      $('.job-heading h3', row).textContent = job.kind === 'pose_export' ? (job.title || 'Pose dataset') : job.title;
      $('.job-heading p', row).textContent = job.kind === 'pose_export'
        ? `Pose dataset · ${formatNumber(job.pairCount)} pair${job.pairCount === 1 ? '' : 's'} · ${job.profile}${job.createdAt ? ` · ${relativeTime(job.createdAt)}` : ''}`
        : `${job.profile}${job.createdAt ? ` · ${relativeTime(job.createdAt)}` : ''}`;
      const stateLabel = $('.job-state', row);
      const displayStatus = job.status.replaceAll('_', ' ');
      stateLabel.textContent = displayStatus;
      stateLabel.className = `job-state ${job.status === 'completed_with_errors' ? 'partial' : job.status}`;
      $('.job-progress > span', row).style.width = `${job.progress}%`;
      const counts = job.kind === 'pose_export'
        ? job.total
          ? `${formatNumber(job.complete)} / ${formatNumber(job.total)} source images · ${formatNumber(job.pairCount)} pairs`
          : `${formatNumber(job.pairCount)} pairs queued`
        : job.total ? `${formatNumber(job.complete)} / ${formatNumber(job.total)} images` : `${Math.round(job.progress)}%`;
      $('.job-progress-label', row).textContent = counts;
      $('.job-speed', row).textContent = job.speed ? `${formatBytes(job.speed)}/s` : job.bytes ? formatBytes(job.bytes) : '';
      const error = $('.job-error', row);
      error.hidden = !job.error && !(job.status === 'completed_with_errors' && job.failed);
      error.textContent = job.error || (job.failed ? `${formatNumber(job.failed)} images failed` : '');
      $('.job-toggle', row).hidden = true;
      $('.job-remove', row).title = isTerminalJob(job) ? 'Remove' : job.kind === 'pose_export' ? 'Cancel pose export' : 'Cancel download';
      $('.job-remove', row).setAttribute('aria-label', $('.job-remove', row).title);
      list.append(fragment);
    });

    const active = state.jobs.filter(job => !isTerminalJob(job));
    const downloading = state.jobs.filter(isActiveJob).length;
    const pending = active.length - downloading;
    const completed = state.jobs.filter(job => ['completed', 'completed_with_errors'].includes(job.status));
    $('#stat-active').textContent = formatNumber(downloading);
    $('#stat-pending').textContent = formatNumber(Math.max(0, pending));
    $('#stat-completed').textContent = formatNumber(completed.length);
    $('#stat-data').textContent = formatNumber(completed.reduce((total, job) => total + job.complete, 0));
    $('#queue-summary').textContent = active.length ? `${formatNumber(active.length)} ${active.length === 1 ? 'transfer' : 'transfers'} in progress` : state.jobs.length ? `${formatNumber(state.jobs.length)} recent transfers` : 'Nothing downloading';
    $('#queue-empty').hidden = Boolean(jobs.length);
    const queueCount = active.length;
    $('#nav-queue-count').hidden = !queueCount;
    $('#nav-queue-count').textContent = queueCount > 99 ? '99+' : String(queueCount);
    $('#mobile-queue-count').hidden = !queueCount;
  }

  async function removeJob(job) {
    try {
      await api(`/api/downloads/${encodeURIComponent(job.id)}`, { method: 'DELETE' });
      state.jobs = state.jobs.filter(item => String(item.id) !== String(job.id));
      renderJobs();
      renderOpenPoseExportState();
      const title = isTerminalJob(job) ? 'Transfer removed' : job.kind === 'pose_export' ? 'Pose export cancelled' : 'Download cancelled';
      toast(title, job.title, 'info');
    } catch (error) {
      const title = !isTerminalJob(job) && job.kind === 'pose_export' ? 'Could not cancel pose export' : 'Could not remove transfer';
      toast(title, errorMessage(error), 'error');
    }
  }

  async function clearCompleted() {
    const jobs = state.jobs.filter(isTerminalJob);
    if (!jobs.length) {
      toast('Nothing to clear', 'There are no completed transfers.', 'info');
      return;
    }
    const button = $('#clear-completed');
    setButtonBusy(button, true, 'Clearing…');
    const results = await Promise.allSettled(jobs.map(job => api(`/api/downloads/${encodeURIComponent(job.id)}`, { method: 'DELETE' })));
    const removedIds = new Set(jobs.filter((_, index) => results[index].status === 'fulfilled').map(job => String(job.id)));
    state.jobs = state.jobs.filter(job => !removedIds.has(String(job.id)));
    renderJobs();
    renderOpenPoseExportState();
    setButtonBusy(button, false);
    const failures = results.filter(result => result.status === 'rejected').length;
    if (failures) toast('Some transfers could not be cleared', `${failures} entries remain.`, 'error');
    else toast('Completed transfers cleared', `${jobs.length} ${jobs.length === 1 ? 'entry' : 'entries'} removed.`, 'info');
  }

  function normalizeFinderFolder(item) {
    if (typeof item === 'string') return { path: item, name: item.split('/').filter(Boolean).pop() || item, imageCount: 0 };
    const path = String(item?.path || item?.relative_path || item?.directory || '');
    return {
      ...item,
      path,
      name: String(item?.name || path.split('/').filter(Boolean).pop() || path || 'Examples'),
      imageCount: Number(item?.image_count ?? item?.count ?? item?.images ?? 0)
    };
  }

  function optionalBoolean(value) {
    if (value === undefined || value === null || value === '') return null;
    if (typeof value === 'string') {
      const normalized = value.trim().toLowerCase();
      if (['false', '0', 'no', 'off'].includes(normalized)) return false;
      if (['true', '1', 'yes', 'on'].includes(normalized)) return true;
    }
    return Boolean(value);
  }

  function finderCorpusCount(...values) {
    for (const value of values) {
      if (value === undefined || value === null || value === '' || typeof value === 'boolean') continue;
      const number = Number(value);
      if (Number.isFinite(number)) return Math.max(0, Math.round(number));
    }
    return 0;
  }

  function finderCorpusPercent(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number)
      ? Math.max(0, Math.min(100, number))
      : Math.max(0, Math.min(100, Number(fallback) || 0));
  }

  function normalizeFinderJoytagCoverage(item) {
    const coverage = item?.coverage && typeof item.coverage === 'object'
      ? item.coverage
      : item;
    if (!coverage || typeof coverage !== 'object' || Array.isArray(coverage)) return null;
    const recognized = [
      'model_key', 'total_images', 'cached_images', 'missing_images', 'percent',
      'cache_entries', 'cache_bytes'
    ].some(key => coverage[key] !== undefined);
    if (!recognized) return null;
    const totalImages = finderCorpusCount(coverage.total_images);
    const cachedImages = finderCorpusCount(coverage.cached_images);
    const missingImages = coverage.missing_images === undefined
      ? Math.max(0, totalImages - cachedImages)
      : finderCorpusCount(coverage.missing_images);
    const derivedPercent = totalImages
      ? (Math.min(totalImages, cachedImages) / totalImages) * 100
      : 0;
    return {
      modelKey: String(coverage.model_key || ''),
      totalImages,
      cachedImages,
      missingImages,
      percent: finderCorpusPercent(coverage.percent, derivedPercent),
      cacheEntries: finderCorpusCount(coverage.cache_entries),
      cacheBytes: finderCorpusCount(coverage.cache_bytes)
    };
  }

  function normalizeFinderJoytagIndexJob(item) {
    const job = item?.job && typeof item.job === 'object' ? item.job : item;
    if (!job || typeof job !== 'object' || Array.isArray(job)) return null;
    const status = String(job.status || '').trim().toLowerCase();
    if (!status) return null;
    const errors = Array.isArray(job.errors)
      ? job.errors.map(value => (
          typeof value === 'string'
            ? value
            : value && typeof value === 'object'
              ? String(value.error || value.message || value.detail || '')
              : ''
        )).filter(Boolean)
      : [];
    return {
      id: String(job.id || ''),
      status,
      modelKey: String(job.model_key || ''),
      totalImages: finderCorpusCount(job.total_images),
      cachedImagesAtStart: finderCorpusCount(job.cached_images_at_start),
      processedImages: finderCorpusCount(job.processed_images),
      indexedImages: finderCorpusCount(job.indexed_images),
      failedImages: finderCorpusCount(job.failed_images),
      remainingImages: finderCorpusCount(job.remaining_images),
      progress: finderCorpusPercent(job.progress),
      cancelRequested: Boolean(job.cancel_requested),
      error: String(job.error || ''),
      errors,
      createdAt: String(job.created_at || ''),
      updatedAt: String(job.updated_at || ''),
      finishedAt: String(job.finished_at || '')
    };
  }

  function finderJoytagIndexIsActive(job = state.finderCorpus?.joytagIndexJob) {
    return Boolean(job && ['queued', 'running', 'canceling'].includes(job.status));
  }

  function normalizeFinderCorpus(item) {
    if (!item || typeof item !== 'object') return null;
    const wrapped = item.corpus && typeof item.corpus === 'object'
      ? item.corpus
      : item.index && typeof item.index === 'object'
        ? item.index
        : item.finder?.corpus && typeof item.finder.corpus === 'object' ? item.finder.corpus : item;
    const recognized = [
      'galleries', 'gallery_count', 'images', 'image_count', 'complete',
      'partial', 'ready', 'cache_entries', 'cache_bytes', 'storage_bytes',
      'joytag', 'joytag_index_job'
    ].some(key => wrapped[key] !== undefined);
    if (!recognized) return null;
    const galleries = finderCorpusCount(wrapped.galleries, wrapped.gallery_count, wrapped.indexed_galleries);
    const images = finderCorpusCount(wrapped.images, wrapped.image_count, wrapped.indexed_images);
    const complete = finderCorpusCount(wrapped.complete, wrapped.complete_galleries, wrapped.fully_indexed);
    const partial = finderCorpusCount(wrapped.partial, wrapped.partial_galleries, wrapped.historic_partial);
    const ready = finderCorpusCount(wrapped.ready, wrapped.ready_images, wrapped.embedded_images, wrapped.images_ready);
    return {
      galleries,
      images,
      complete,
      partial,
      ready,
      cacheEntries: finderCorpusCount(wrapped.cache_entries, wrapped.cached_images, wrapped.entries),
      cacheBytes: finderCorpusCount(wrapped.cache_bytes, wrapped.storage_bytes, wrapped.size_bytes, wrapped.bytes),
      maxCacheEntries: finderCorpusCount(wrapped.max_cache_entries, wrapped.cache_entry_limit, wrapped.max_entries),
      maxCacheBytes: finderCorpusCount(wrapped.max_cache_bytes, wrapped.cache_byte_limit, wrapped.max_bytes),
      joytag: normalizeFinderJoytagCoverage(wrapped.joytag),
      joytagIndexJob: normalizeFinderJoytagIndexJob(wrapped.joytag_index_job)
    };
  }

  function normalizeFinderFeedback(item, fallbackTag = null) {
    if (!item || typeof item !== 'object') return null;
    const wrapped = item.feedback && typeof item.feedback === 'object'
      ? item.feedback
      : item.finder_feedback && typeof item.finder_feedback === 'object'
        ? item.finder_feedback
        : item;
    const recognized = [
      'accepted', 'accepted_count', 'positive_count', 'rejected',
      'rejected_count', 'negative_count', 'accepted_samples', 'rejected_samples',
      'accepted_galleries', 'rejected_galleries', 'enabled', 'active', 'total'
    ].some(key => wrapped[key] !== undefined);
    if (!recognized) return null;
    const accepted = finderCorpusCount(
      wrapped.accepted_samples,
      wrapped.accepted,
      wrapped.accepted_count,
      wrapped.positive_count,
      wrapped.positive_examples
    );
    const rejected = finderCorpusCount(
      wrapped.rejected_samples,
      wrapped.rejected,
      wrapped.rejected_count,
      wrapped.negative_count,
      wrapped.negative_examples
    );
    const active = optionalBoolean(
      wrapped.active
      ?? wrapped.enabled
      ?? wrapped.applied
      ?? wrapped.ready
    );
    return {
      poseTagId: wrapped.pose_tag_id ?? wrapped.tag_id ?? fallbackTag?.id,
      poseTagLabel: String(wrapped.pose_tag_label || wrapped.tag_label || fallbackTag?.label || ''),
      accepted,
      rejected,
      acceptedGalleries: finderCorpusCount(wrapped.accepted_galleries, wrapped.positive_galleries),
      rejectedGalleries: finderCorpusCount(wrapped.rejected_galleries, wrapped.negative_galleries),
      usableAcceptedGalleries: finderCorpusCount(wrapped.usable_accepted_galleries, wrapped.ready_accepted_galleries),
      usableRejectedGalleries: finderCorpusCount(wrapped.usable_rejected_galleries, wrapped.ready_rejected_galleries),
      usableAcceptedSamples: finderCorpusCount(wrapped.usable_accepted_samples, wrapped.ready_accepted_samples),
      usableRejectedSamples: finderCorpusCount(wrapped.usable_rejected_samples, wrapped.ready_rejected_samples),
      total: Math.max(
        accepted + rejected,
        finderCorpusCount(wrapped.total, wrapped.feedback_count)
      ),
      active: active === null ? accepted + rejected > 0 : active,
      minimumGalleries: finderCorpusCount(
        wrapped.min_galleries_per_state,
        wrapped.minimum_galleries,
        wrapped.activation_threshold
      ),
      maximumGalleries: finderCorpusCount(wrapped.max_galleries_per_state, wrapped.maximum_galleries),
      maximumAdjustment: normalizeFinderScore(wrapped.max_adjustment, 0),
      revision: finderCorpusCount(wrapped.revision),
      updatedAt: wrapped.updated_at || ''
    };
  }

  function normalizeFinderInferenceBatch(item) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return null;
    const normalizeSize = value => Number.isSafeInteger(value) && value >= 0
      ? Math.max(1, value)
      : null;
    const normalized = {
      configured: normalizeSize(item.configured),
      appearance: normalizeSize(item.appearance),
      pose: normalizeSize(item.pose)
    };
    return Object.values(normalized).some(value => value !== null) ? normalized : null;
  }

  function normalizeFinderStatus(item) {
    const data = item?.finder || item || {};
    const model = data.model && typeof data.model === 'object' ? data.model : {};
    const rawStatus = typeof data.status === 'string' ? data.status : typeof model.status === 'string' ? model.status : '';
    const status = String(rawStatus || (data.ready || data.available || model.ready || model.available ? 'ready' : 'unavailable')).toLowerCase();
    const serviceAvailable = data.available ?? model.available;
    const reportedModelReady = data.model_ready ?? model.model_ready ?? data.ready ?? model.ready;
    const modelReady = reportedModelReady === undefined ? Boolean(serviceAvailable) : Boolean(reportedModelReady);
    const error = String(data.error || model.error || '');
    const poseError = String(data.pose_error || model.pose_error || '');
    const ready = serviceAvailable !== undefined
      ? Boolean(serviceAvailable)
      : reportedModelReady === undefined ? ['ready', 'available', 'loaded', 'ok'].includes(status) : modelReady;
    const providers = data.providers && typeof data.providers === 'object' ? data.providers : {};
    const appearanceProvider = providers.appearance && typeof providers.appearance === 'object' ? providers.appearance : {};
    const poseProvider = providers.pose && typeof providers.pose === 'object' ? providers.pose : {};
    const joytag = data.joytag && typeof data.joytag === 'object' ? data.joytag : {};
    const joytagProvider = joytag.provider && typeof joytag.provider === 'object'
      ? joytag.provider
      : providers.joytag && typeof providers.joytag === 'object' ? providers.joytag : {};
    const providerDetails = [];
    if (appearanceProvider.cpu_fallback) providerDetails.push('Appearance using CPU fallback');
    if (poseProvider.fallback) providerDetails.push('Pose using CPU fallback');
    if (poseProvider.message) providerDetails.push(String(poseProvider.message));
    const details = [
      error ? `Model error: ${error}` : '',
      poseError ? `Pose unavailable: ${poseError}` : '',
      data.detail || data.message || model.detail || model.description || '',
      data.device || model.device || '',
      data.backend || model.backend || '',
      ...providerDetails
    ].filter(Boolean);
    return {
      ready,
      modelReady,
      error,
      poseError,
      status,
      name: String(data.model_name || model.name || model.label || data.name || String(data.model_path || '').split('/').pop() || 'Similarity model'),
      detail: details.join(' · ') || (modelReady ? 'Ready to compare images' : ready ? 'Model downloads automatically on the first scan' : 'Model unavailable'),
      defaultSourceUrl: String(data.default_source_url || data.source_url || ''),
      folderRoot: String(data.folder_root || model.folder_root || ''),
      corpus: normalizeFinderCorpus(data.corpus || model.corpus),
      inferenceBatch: normalizeFinderInferenceBatch(data.inference_batch ?? model.inference_batch),
      joytagAvailable: joytag.available === undefined ? null : Boolean(joytag.available),
      joytagReady: Boolean(joytag.ready),
      joytagError: String(joytag.error || ''),
      joytagModelKey: String(joytag.model_key || ''),
      joytagProvider: String(joytagProvider.active || joytagProvider.provider || '')
    };
  }

  function normalizeFinderTagList(value, { limit = FINDER_JOYTAG_QUERY_LIMIT } = {}) {
    if (!Array.isArray(value)) return [];
    const seen = new Set();
    const tags = [];
    for (const raw of value) {
      const tag = String(raw || '').trim();
      if (!tag || seen.has(tag)) continue;
      seen.add(tag);
      tags.push(tag);
      if (tags.length >= limit) break;
    }
    return tags;
  }

  function normalizeFinderReferenceAnalysis(item) {
    const analysis = item?.analysis && typeof item.analysis === 'object' ? item.analysis : item;
    if (!analysis || typeof analysis !== 'object' || Array.isArray(analysis)) return null;
    const fingerprint = String(analysis.fingerprint || '');
    const directory = String(analysis.directory || '');
    const rawTags = Array.isArray(analysis.tags) ? analysis.tags : [];
    const rawImages = Array.isArray(analysis.images) ? analysis.images : [];
    if (!fingerprint || !directory || !rawTags.length || !rawImages.length) return null;
    const tags = rawTags.map((item, index) => {
      const tag = String(item?.tag || '').trim();
      if (!tag) return null;
      return {
        tag,
        index: Math.max(0, Number.parseInt(item?.index ?? index, 10) || 0),
        average: normalizeFinderScore(item?.average, 0),
        minimum: normalizeFinderScore(item?.minimum, 0),
        maximum: normalizeFinderScore(item?.maximum, 0),
        median: normalizeFinderScore(item?.median, 0),
        hitsAtPointFour: Math.max(0, Number(item?.hits_at_0_4 || 0)),
        imageCount: Math.max(0, Number(item?.image_count || analysis.image_count || rawImages.length))
      };
    }).filter(Boolean);
    if (!tags.length) return null;
    const tagCatalog = normalizeFinderTagList(
      [
        ...(Array.isArray(analysis.tag_catalog) ? analysis.tag_catalog : []),
        ...tags.map(item => item.tag)
      ],
      { limit: Number.MAX_SAFE_INTEGER }
    );
    const knownTags = new Set(tagCatalog);
    const images = rawImages.map(item => {
      const scores = {};
      if (item?.scores && typeof item.scores === 'object' && !Array.isArray(item.scores)) {
        Object.entries(item.scores).forEach(([tag, value]) => {
          if (!knownTags.has(tag)) return;
          const score = normalizeFinderScore(value);
          if (score !== null) scores[tag] = score;
        });
      }
      return {
        name: String(item?.name || 'Reference image'),
        previewUrl: String(item?.preview_url || ''),
        scores
      };
    });
    return {
      directory,
      fingerprint,
      imageCount: Math.max(1, Number(analysis.image_count || images.length)),
      modelKey: String(analysis.model_key || ''),
      provider: String(analysis.provider || ''),
      quantization: String(analysis.quantization || ''),
      bytesPerCachedImage: Math.max(0, Number(analysis.bytes_per_cached_image || 0)),
      tags,
      tagCatalog,
      images
    };
  }

  function normalizeFinderReview(value) {
    const review = String(value || 'pending').toLowerCase();
    if (['accepted', 'accept', 'approved'].includes(review)) return 'accepted';
    if (['maybe', 'unsure', 'uncertain', 'neutral'].includes(review)) return 'maybe';
    if (['rejected', 'reject', 'dismissed'].includes(review)) return 'rejected';
    return 'pending';
  }

  function normalizeFinderReviewCounts(value) {
    if (!value || typeof value !== 'object') return null;
    const counts = value.review_counts && typeof value.review_counts === 'object'
      ? value.review_counts
      : value.counts && typeof value.counts === 'object' ? value.counts : value;
    if (!['pending', 'accepted', 'maybe', 'rejected'].some(key => counts[key] !== undefined)) return null;
    const normalized = {
      pending: Math.max(0, Number(counts.pending || 0)),
      accepted: Math.max(0, Number(counts.accepted || 0)),
      maybe: Math.max(0, Number(counts.maybe || 0)),
      rejected: Math.max(0, Number(counts.rejected || 0))
    };
    normalized.total = Math.max(
      normalized.pending + normalized.accepted + normalized.maybe + normalized.rejected,
      Number(counts.total || 0)
    );
    return normalized;
  }

  function normalizeFinderScan(item) {
    if (!item) return null;
    const scan = item.scan || item;
    const config = scan.config || {};
    const progress = scan.progress && typeof scan.progress === 'object' ? scan.progress : {};
    const reviewCounts = scan.review_counts || scan.counts || {};
    let percentage = Number(scan.progress_percent ?? progress.percent ?? (typeof scan.progress === 'number' ? scan.progress : NaN));
    const pagesTotal = Number(scan.pages_total ?? scan.total_pages ?? scan.page_limit ?? config.page_limit ?? config.pages ?? scan.pages ?? 0);
    const pagesScanned = Number(scan.pages_completed ?? scan.pages_scanned ?? scan.completed_pages ?? progress.pages_completed ?? progress.pages_scanned ?? progress.completed ?? scan.current_page ?? 0);
    const nextUrl = String(scan.next_url || progress.next_url || '');
    const reportedContinuation = scan.has_next_page ?? scan.has_more ?? scan.continuation_available ?? scan.can_extend;
    const hasNextPage = !Boolean(scan.source_exhausted ?? scan.exhausted)
      && (reportedContinuation === undefined ? Boolean(nextUrl) : Boolean(reportedContinuation));
    const rankingVersion = String(scan.ranking_version || config.ranking_version || 'appearance-first-v1');
    const rankingCurrent = scan.ranking_current === undefined
      ? [FINDER_RANKING_VERSION, FINDER_JOYTAG_RANKING_VERSION].includes(rankingVersion)
      : Boolean(scan.ranking_current);
    const searchMode = ['joytag', 'tag'].includes(String(scan.search_mode || config.search_mode || scan.mode || config.mode || '').toLowerCase())
      || rankingVersion === FINDER_JOYTAG_RANKING_VERSION ? 'joytag' : 'pose';
    const scanCorpus = scan.corpus && typeof scan.corpus === 'object' ? scan.corpus : {};
    const corpusSearchRaw = scan.corpus_search_complete
      ?? progress.corpus_search_complete
      ?? scanCorpus.search_complete
      ?? scanCorpus.searchComplete;
    const corpusImagesRaw = scan.corpus_images_scored
      ?? progress.corpus_images_scored
      ?? scanCorpus.images_scored;
    const corpusGalleriesRaw = scan.corpus_galleries_scored
      ?? progress.corpus_galleries_scored
      ?? scanCorpus.galleries_scored;
    const corpusProgressAvailable = corpusSearchRaw !== undefined
      || corpusImagesRaw !== undefined
      || corpusGalleriesRaw !== undefined;
    if (!Number.isFinite(percentage)) percentage = pagesTotal ? (pagesScanned / pagesTotal) * 100 : 0;
    if (percentage > 0 && percentage <= 1) percentage *= 100;
    const poseTag = scan.pose_tag && typeof scan.pose_tag === 'object' ? scan.pose_tag : {};
    const joytagTag = String(scan.joytag_tag || config.joytag_tag || '').trim();
    const joytagRequiredTags = normalizeFinderTagList(
      scan.joytag_required_tags ?? config.joytag_required_tags
    );
    if (!joytagRequiredTags.length && joytagTag) joytagRequiredTags.push(joytagTag);
    const requiredSet = new Set(joytagRequiredTags);
    const joytagExcludedTags = normalizeFinderTagList(
      scan.joytag_excluded_tags ?? config.joytag_excluded_tags
    ).filter(tag => !requiredSet.has(tag));
    const joytagRejectThreshold = normalizeFinderScore(
      scan.joytag_reject_threshold ?? config.joytag_reject_threshold,
      0.4
    );
    return {
      ...scan,
      id: scan.id ?? scan.scan_id,
      status: String(scan.status || scan.state || 'queued').toLowerCase(),
      examplesFolder: String(scan.example_directory || scan.examples_folder || config.example_directory || config.examples_folder || scan.folder || ''),
      poseTagId: scan.pose_tag_id ?? config.pose_tag_id ?? poseTag.id,
      poseTagLabel: String(scan.pose_tag_label || config.pose_tag_label || poseTag.label || poseTag.name || ''),
      poseTagSlug: String(scan.pose_tag_slug || poseTag.slug || ''),
      poseDefaultRole: POSE_ROLES.includes(scan.pose_default_role || poseTag.default_role) ? (scan.pose_default_role || poseTag.default_role) : 'solo',
      searchMode,
      joytagTag: joytagRequiredTags[0] || joytagTag,
      joytagRequiredTags,
      joytagExcludedTags,
      joytagRejectThreshold,
      referenceFingerprint: String(scan.reference_fingerprint || config.reference_fingerprint || ''),
      sourceUrl: String(scan.source_url || config.source_url || scan.url || ''),
      nextUrl,
      hasNextPage,
      continuable: optionalBoolean(scan.continuable),
      rankingVersion,
      rankingCurrent,
      corpusProgressAvailable,
      corpusSearchComplete: optionalBoolean(corpusSearchRaw),
      corpusImagesScored: finderCorpusCount(corpusImagesRaw),
      corpusGalleriesScored: finderCorpusCount(corpusGalleriesRaw),
      pages: Number(scan.page_limit ?? config.page_limit ?? pagesTotal),
      pagesScanned,
      galleriesScanned: Number(scan.processed_galleries ?? scan.galleries_scanned ?? progress.processed_galleries ?? progress.galleries_scanned ?? progress.galleries ?? 0),
      imagesScanned: Number(scan.processed_images ?? scan.images_scanned ?? progress.processed_images ?? progress.images_scanned ?? progress.images ?? 0),
      totalGalleries: Number(scan.total_galleries ?? progress.total_galleries ?? 0),
      failedGalleries: Number(scan.failed_galleries ?? progress.failed_galleries ?? 0),
      candidateCount: Number(scan.candidate_count ?? scan.results_count ?? progress.candidates ?? reviewCounts.total ?? 0),
      pendingCount: Number(scan.pending_count ?? reviewCounts.pending ?? 0),
      acceptedCount: Number(scan.accepted_count ?? reviewCounts.accepted ?? 0),
      maybeCount: Number(scan.maybe_count ?? reviewCounts.maybe ?? 0),
      rejectedCount: Number(scan.rejected_count ?? reviewCounts.rejected ?? 0),
      minSimilarity: Number(scan.minimum_score ?? scan.min_similarity ?? scan.minimum_similarity ?? config.minimum_score ?? config.min_similarity ?? config.minimum_similarity ?? 0.68),
      percentage: Math.max(0, Math.min(100, percentage)),
      error: String(scan.error || scan.error_message || ''),
      createdAt: scan.created_at || scan.started_at || '',
      updatedAt: scan.updated_at || scan.finished_at || ''
    };
  }

  function encodeFinderGalleryId(url) {
    try {
      const bytes = new TextEncoder().encode(String(url || ''));
      let binary = '';
      bytes.forEach(byte => { binary += String.fromCharCode(byte); });
      return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
    } catch (_) { return ''; }
  }

  function normalizeFinderScore(value, fallback = null) {
    if (value === null || value === undefined || value === '') return fallback;
    let score = Number(value);
    if (!Number.isFinite(score)) return fallback;
    if (score > 1 && score <= 100) score /= 100;
    return Math.max(0, Math.min(1, score));
  }

  function firstFinderScore(...values) {
    for (const value of values) {
      const score = normalizeFinderScore(value);
      if (score !== null) return score;
    }
    return null;
  }

  function normalizeFinderTagScores(value, fallback = {}) {
    const source = value && typeof value === 'object' && !Array.isArray(value)
      ? value
      : {};
    const scores = {};
    Object.entries(source).forEach(([rawTag, rawScore]) => {
      const tag = String(rawTag || '').trim();
      const score = normalizeFinderScore(rawScore);
      if (tag && score !== null) scores[tag] = score;
    });
    if (!Object.keys(scores).length && fallback && typeof fallback === 'object') {
      Object.entries(fallback).forEach(([rawTag, rawScore]) => {
        const tag = String(rawTag || '').trim();
        const score = normalizeFinderScore(rawScore);
        if (tag && score !== null) scores[tag] = score;
      });
    }
    return scores;
  }

  function normalizeFinderAdjustment(value, fallback = 0) {
    if (value === null || value === undefined || value === '') return fallback;
    let adjustment = Number(value);
    if (!Number.isFinite(adjustment)) return fallback;
    if (Math.abs(adjustment) > 1 && Math.abs(adjustment) <= 100) adjustment /= 100;
    return Math.max(-1, Math.min(1, adjustment));
  }

  function finderFeedbackAdjustmentLabel(value) {
    const percentage = Math.abs(normalizeFinderAdjustment(value)) * 100;
    const sign = value < 0 ? '−' : '+';
    if (percentage > 0 && percentage < 0.01) return `${sign}<0.01%`;
    const formatted = new Intl.NumberFormat(undefined, {
      maximumFractionDigits: percentage < 0.1 ? 2 : percentage < 1 ? 1 : 0
    }).format(percentage);
    return `${sign}${formatted}%`;
  }

  function normalizeFinderTier(value, fallback = 1) {
    const tier = Number.parseInt(value, 10);
    return Number.isFinite(tier) ? Math.max(0, Math.min(3, tier)) : fallback;
  }

  function finderTierType(tier) {
    return ['pose_mismatch', 'visual_fallback', 'pose', 'exact'][normalizeFinderTier(tier)] || 'visual_fallback';
  }

  function normalizeFinderMatch(rawMatch, index, fallback = {}) {
    const match = typeof rawMatch === 'string' ? { image_url: rawMatch } : (rawMatch || {});
    const breakdown = match.score_breakdown && typeof match.score_breakdown === 'object'
      ? match.score_breakdown
      : match.scores && typeof match.scores === 'object' ? match.scores : {};
    const isExact = Boolean(match.is_exact ?? match.exact_match ?? (index === 0 ? fallback.isExact : false));
    const imageUrl = String(match.image_url || match.full_url || match.url || (index === 0 ? fallback.imageUrl : '') || '');
    const previewUrl = String(match.preview_url || match.thumbnail_url || match.preview || (index === 0 ? fallback.previewUrl : '') || '');
    const sourceKey = String(match.source_key || match.cache_source_key || match.descriptor_key || '');
    return {
      rank: Math.max(1, Number(match.rank ?? match.match_rank ?? index + 1) || index + 1),
      imageUrl,
      previewUrl,
      sourceKey,
      feedbackKey: sourceKey || imageUrl || previewUrl || `ordinal:${Number(match.ordinal ?? match.image_ordinal ?? match.index ?? index + 1)}`,
      ordinal: Number(match.ordinal ?? match.image_ordinal ?? match.index ?? (index === 0 ? fallback.ordinal : 0) ?? 0),
      score: firstFinderScore(match.score, match.similarity, match.combined_score, index === 0 ? fallback.score : null),
      baseScore: firstFinderScore(match.base_score, match.baseScore, index === 0 ? fallback.baseScore : null),
      feedbackAdjustment: normalizeFinderAdjustment(
        match.feedback_adjustment ?? match.feedbackAdjustment,
        index === 0 ? normalizeFinderAdjustment(fallback.feedbackAdjustment) : 0
      ),
      feedbackApplied: optionalBoolean(
        match.feedback_applied
        ?? match.feedbackApplied
        ?? (index === 0 ? fallback.feedbackApplied : null)
      ) ?? Math.abs(normalizeFinderAdjustment(
        match.feedback_adjustment ?? match.feedbackAdjustment,
        index === 0 ? normalizeFinderAdjustment(fallback.feedbackAdjustment) : 0
      )) > 1e-9,
      feedbackRevision: Math.max(
        0,
        Number.parseInt(
          match.feedback_revision
          ?? match.feedbackRevision
          ?? (index === 0 ? fallback.feedbackRevision : 0),
          10
        ) || 0
      ),
      exactScore: firstFinderScore(match.exact_score, match.duplicate_score, match.phash_score, breakdown.exact, breakdown.exact_score, index === 0 ? fallback.exactScore : null, isExact ? 1 : null),
      poseScore: firstFinderScore(match.pose_score, match.keypoint_score, match.geometry_score, breakdown.pose, breakdown.pose_score, index === 0 ? fallback.poseScore : null),
      appearanceScore: firstFinderScore(match.appearance_score, match.visual_score, match.dino_score, breakdown.appearance, breakdown.appearance_score, index === 0 ? fallback.appearanceScore : null),
      tag: String(match.tag || (index === 0 ? fallback.tag : '') || ''),
      tagScore: firstFinderScore(match.tag_score, index === 0 ? fallback.tagScore : null, match.match_type === 'tag' ? match.score : null),
      tagScores: normalizeFinderTagScores(
        match.tag_scores,
        index === 0 ? fallback.tagScores : {}
      ),
      personCount: Math.max(0, Number(match.person_count ?? match.people_count ?? match.persons_detected ?? (index === 0 ? fallback.personCount : 0) ?? 0) || 0),
      overlayUrl: String(match.skeleton_overlay_url || match.pose_overlay_url || match.overlay_url || (index === 0 ? fallback.overlayUrl : '') || ''),
      poseReliable: Boolean(match.pose_reliable ?? match.poseReliable ?? (index === 0 ? fallback.poseReliable : false)),
      rankingTier: normalizeFinderTier(
        match.ranking_tier ?? match.rank_tier,
        index === 0 ? normalizeFinderTier(fallback.rankingTier, isExact ? 3 : 1) : 1
      ),
      matchType: String(match.match_type || match.method || (index === 0 ? fallback.matchType : '') || '').toLowerCase(),
      isExact
    };
  }

  function normalizeFinderResult(item, index = 0) {
    const source = item?.gallery && typeof item.gallery === 'object' ? item.gallery : {};
    const best = item?.best_match && typeof item.best_match === 'object' ? item.best_match : {};
    const breakdown = item?.score_breakdown && typeof item.score_breakdown === 'object'
      ? item.score_breakdown
      : item?.scores && typeof item.scores === 'object' ? item.scores : {};
    const galleryUrl = item?.gallery_url || source.url || source.gallery_url || '';
    const galleryId = item?.gallery_id ?? source.gallery_id ?? source.id ?? encodeFinderGalleryId(galleryUrl);
    const isExact = Boolean(item?.is_exact ?? item?.exact_match ?? best.is_exact ?? best.exact_match);
    const fallback = {
      imageUrl: item?.best_image_url || best.image_url || best.full_url || best.url || '',
      previewUrl: item?.best_preview_url || best.preview_url || best.thumbnail_url || source.thumbnail_url || source.thumbnail || '',
      ordinal: Number(item?.best_ordinal ?? best.ordinal ?? best.image_ordinal ?? 0),
      score: firstFinderScore(item?.score, item?.similarity, item?.combined_score, best.score, best.similarity, best.combined_score),
      baseScore: firstFinderScore(item?.base_score, item?.baseScore, best.base_score, best.baseScore),
      feedbackAdjustment: normalizeFinderAdjustment(
        item?.feedback_adjustment
        ?? item?.feedbackAdjustment
        ?? best.feedback_adjustment
        ?? best.feedbackAdjustment
      ),
      feedbackApplied: optionalBoolean(
        item?.feedback_applied
        ?? item?.feedbackApplied
        ?? best.feedback_applied
        ?? best.feedbackApplied
      ),
      feedbackRevision: Math.max(
        0,
        Number.parseInt(
          item?.feedback_revision
          ?? item?.feedbackRevision
          ?? best.feedback_revision
          ?? best.feedbackRevision,
          10
        ) || 0
      ),
      exactScore: firstFinderScore(item?.exact_score, item?.duplicate_score, item?.phash_score, breakdown.exact, breakdown.exact_score, best.exact_score, isExact ? 1 : null),
      poseScore: firstFinderScore(item?.pose_score, item?.keypoint_score, item?.geometry_score, breakdown.pose, breakdown.pose_score, best.pose_score, best.keypoint_score),
      appearanceScore: firstFinderScore(item?.appearance_score, item?.visual_score, item?.dino_score, breakdown.appearance, breakdown.appearance_score, best.appearance_score, best.visual_score),
      tag: String(item?.tag || best.tag || ''),
      tagScore: firstFinderScore(item?.tag_score, best.tag_score, item?.match_type === 'tag' ? item?.score : null),
      tagScores: normalizeFinderTagScores(item?.tag_scores, best.tag_scores),
      personCount: Math.max(0, Number(item?.person_count ?? item?.people_count ?? item?.persons_detected ?? best.person_count ?? best.people_count ?? 0) || 0),
      overlayUrl: item?.skeleton_overlay_url || item?.pose_overlay_url || item?.overlay_url || best.skeleton_overlay_url || best.pose_overlay_url || best.overlay_url || '',
      poseReliable: Boolean(item?.pose_reliable ?? item?.poseReliable ?? best.pose_reliable ?? best.poseReliable),
      rankingTier: normalizeFinderTier(item?.ranking_tier ?? item?.rank_tier ?? best.ranking_tier ?? best.rank_tier, isExact ? 3 : 1),
      matchType: String(item?.match_type || item?.method || best.match_type || best.method || '').toLowerCase(),
      isExact
    };
    const matchCollections = [item?.matches, item?.top_matches, item?.best_matches, item?.candidate_matches, item?.candidates, item?.candidate_images];
    const rawMatches = matchCollections.find(value => Array.isArray(value) && value.length) || [];
    let matches = (rawMatches.length ? rawMatches : [best]).map((match, matchIndex) => normalizeFinderMatch(match, matchIndex, fallback));
    matches = matches
      .filter(match => match.imageUrl || match.previewUrl || match.overlayUrl)
      .sort((a, b) => b.rankingTier - a.rankingTier || (b.score ?? -1) - (a.score ?? -1) || a.rank - b.rank)
      .slice(0, 3);
    if (!matches.length && (fallback.imageUrl || fallback.previewUrl || fallback.overlayUrl)) matches = [normalizeFinderMatch({}, 0, fallback)];
    const primaryMatch = matches[0] || normalizeFinderMatch({}, 0, fallback);
    const gallery = normalizeGallery({
      ...source,
      id: galleryId,
      gallery_id: galleryId,
      url: galleryUrl,
      title: item?.title || source.title || source.name || 'Untitled gallery',
      thumbnail_url: primaryMatch.previewUrl || primaryMatch.imageUrl || source.thumbnail_url || source.thumbnail || '',
      image_count: item?.image_count ?? source.image_count ?? source.total_images ?? 0
    });
    const score = firstFinderScore(item?.score, item?.similarity, item?.combined_score, primaryMatch.score, fallback.score) ?? 0;
    const rankingTier = normalizeFinderTier(item?.ranking_tier ?? item?.rank_tier, primaryMatch.rankingTier);
    const matchType = String(item?.match_type || item?.method || best.match_type || best.method || primaryMatch.matchType || finderTierType(rankingTier)).toLowerCase();
    const origin = String(item?.origin || item?.result_origin || item?.source_origin || '').toLowerCase();
    const onlineScanned = optionalBoolean(item?.online_scanned ?? item?.scanned_online ?? item?.live_scanned);
    const indexedOnly = onlineScanned === false
      || (onlineScanned !== true && ['corpus', 'index', 'indexed', 'local'].includes(origin));
    const review = normalizeFinderReview(item?.review ?? item?.review_status);
    const suppliedFeedback = Array.isArray(item?.feedback_matches)
      ? item.feedback_matches
      : Array.isArray(item?.selected_matches) ? item.selected_matches : null;
    const suppliedFeedbackUrls = Array.isArray(item?.feedback_image_urls)
      ? item.feedback_image_urls.map(String)
      : Array.isArray(item?.selected_image_urls) ? item.selected_image_urls.map(String) : null;
    const suppliedFeedbackKeys = Array.isArray(item?.feedback_source_keys)
      ? item.feedback_source_keys.map(String)
      : suppliedFeedback
        ? suppliedFeedback.map(match => String(match?.source_key || match?.image_url || match?.url || '')).filter(Boolean)
        : null;
    const feedbackSelectionProvided = review === 'pending'
      ? Boolean(suppliedFeedbackUrls?.length || suppliedFeedbackKeys?.length)
      : Boolean(suppliedFeedbackUrls || suppliedFeedbackKeys);
    const feedbackMatchKeys = feedbackSelectionProvided
      ? matches.filter(match => (
        suppliedFeedbackUrls?.includes(match.imageUrl)
        || suppliedFeedbackKeys?.includes(match.sourceKey)
        || suppliedFeedbackKeys?.includes(match.feedbackKey)
      )).map(match => match.feedbackKey)
      : review === 'pending' ? matches.filter(match => match.imageUrl).map(match => match.feedbackKey) : [];
    let feedbackImageUrls;
    if (!feedbackSelectionProvided) {
      feedbackImageUrls = review === 'pending'
        ? matches.map(match => match.imageUrl).filter(Boolean).slice(0, 3)
        : [];
    } else {
      const explicitUrls = [...new Set((suppliedFeedbackUrls || []).filter(Boolean))];
      feedbackImageUrls = explicitUrls.length
        ? explicitUrls.slice(0, 3)
        : suppliedFeedbackKeys?.length
          ? matches
            .filter(match => feedbackMatchKeys.includes(match.feedbackKey))
            .map(match => match.imageUrl)
            .filter(Boolean)
            .slice(0, 3)
          : [];
    }
    const suppliedUsableFeedbackUrls = Array.isArray(item?.feedback_usable_image_urls)
      ? item.feedback_usable_image_urls.map(String).filter(Boolean)
      : null;
    const suppliedPendingFeedbackUrls = Array.isArray(item?.feedback_pending_image_urls)
      ? item.feedback_pending_image_urls.map(String).filter(Boolean)
      : null;
    return {
      ...gallery,
      key: item?.result_id ?? item?.id ?? galleryId,
      galleryId,
      rank: Number(item?.rank ?? index + 1),
      score,
      baseScore: firstFinderScore(fallback.baseScore, primaryMatch.baseScore, score),
      feedbackAdjustment: normalizeFinderAdjustment(
        item?.feedback_adjustment
        ?? item?.feedbackAdjustment,
        primaryMatch.feedbackAdjustment
      ),
      feedbackApplied: optionalBoolean(
        item?.feedback_applied
        ?? item?.feedbackApplied
        ?? primaryMatch.feedbackApplied
      ) ?? Math.abs(normalizeFinderAdjustment(
        item?.feedback_adjustment
        ?? item?.feedbackAdjustment,
        primaryMatch.feedbackAdjustment
      )) > 1e-9,
      feedbackRevision: Math.max(
        0,
        Number.parseInt(
          item?.feedback_revision
          ?? item?.feedbackRevision
          ?? primaryMatch.feedbackRevision,
          10
        ) || 0
      ),
      review,
      bestImageUrl: primaryMatch.imageUrl,
      bestPreviewUrl: primaryMatch.previewUrl || gallery.thumbnailUrl,
      bestOrdinal: primaryMatch.ordinal,
      matches,
      rankingTier,
      matchType,
      isExact: isExact || primaryMatch.isExact || ['exact', 'duplicate', 'near_duplicate'].includes(matchType),
      exactScore: firstFinderScore(fallback.exactScore, primaryMatch.exactScore),
      poseScore: firstFinderScore(fallback.poseScore, primaryMatch.poseScore),
      poseReliable: fallback.poseReliable || primaryMatch.poseReliable,
      appearanceScore: firstFinderScore(fallback.appearanceScore, primaryMatch.appearanceScore),
      tag: primaryMatch.tag || fallback.tag,
      tagScore: firstFinderScore(primaryMatch.tagScore, fallback.tagScore, matchType === 'tag' ? score : null),
      tagScores: Object.keys(primaryMatch.tagScores || {}).length
        ? primaryMatch.tagScores
        : fallback.tagScores,
      personCount: fallback.personCount || primaryMatch.personCount,
      hasOverlay: matches.some(match => Boolean(match.overlayUrl)),
      origin,
      onlineScanned,
      indexedOnly,
      feedbackMatchKeys,
      feedbackImageUrls,
      feedbackUsableImageUrls: [...new Set(suppliedUsableFeedbackUrls || [])].slice(0, 3),
      feedbackPendingImageUrls: [...new Set(suppliedPendingFeedbackUrls || [])].slice(0, 3),
      feedbackAnalysisProvided: suppliedUsableFeedbackUrls !== null || suppliedPendingFeedbackUrls !== null,
      feedbackSelectionProvided,
      feedbackSelectionDirty: false,
      feedbackSaving: false,
      matchCount: Number(item?.images_scored ?? item?.match_count ?? item?.matching_images ?? (Array.isArray(item?.candidate_images) ? item.candidate_images.length : item?.candidate_images) ?? 1)
    };
  }

  function finderScanIsTerminal(scan = state.finderScan) {
    return !scan || FINDER_TERMINAL_STATES.includes(scan.status);
  }

  function finderScanIsRunning(scan = state.finderScan) {
    return Boolean(scan) && ['queued', 'starting', 'preparing', 'running', 'scanning', 'active'].includes(scan.status);
  }

  function finderScanCanExtend(scan = state.finderScan) {
    if (!scan?.id || !scan.rankingCurrent || !scan.hasNextPage || Number(scan.pages || 0) >= FINDER_MAX_PAGES) return false;
    return ['completed', 'completed_with_errors', 'complete', 'done', 'paused', 'running', 'scanning', 'active'].includes(scan.status);
  }

  function finderScanCanContinue(scan = state.finderScan) {
    if (!scan?.id || !scan.rankingCurrent) return false;
    if (scan.continuable !== null && scan.continuable !== undefined) return scan.continuable;
    return finderScanSourceExhausted(scan) && Number(scan.pagesScanned || 0) < FINDER_MAX_PAGES;
  }

  function finderScanCanSwitchSource(scan = state.finderScan) {
    if (
      !scan?.id
      || !scan.rankingCurrent
      || finderScanSourceExhausted(scan)
      || Number(scan.pagesScanned || 0) >= FINDER_MAX_PAGES
    ) return false;
    return ['paused', 'failed', 'cancelled', 'canceled', 'completed', 'completed_with_errors', 'complete', 'done'].includes(scan.status);
  }

  function finderScanCanChangeSource(scan = state.finderScan) {
    if (
      !scan?.id
      || !scan.rankingCurrent
      || Number(scan.pagesScanned || 0) >= FINDER_MAX_PAGES
      || !['paused', 'failed', 'cancelled', 'canceled', 'completed', 'completed_with_errors', 'complete', 'done'].includes(scan.status)
    ) return false;
    return finderScanCanContinue(scan) || finderScanCanSwitchSource(scan);
  }

  function finderScanAtPageCap(scan = state.finderScan) {
    if (!scan?.id || !scan.rankingCurrent) return false;
    return Number(scan.pagesScanned || 0) >= FINDER_MAX_PAGES
      || (scan.hasNextPage && Number(scan.pages || 0) >= FINDER_MAX_PAGES);
  }

  function finderScanSourceExhausted(scan = state.finderScan) {
    if (!scan?.id || scan.hasNextPage) return false;
    const completed = ['completed', 'completed_with_errors', 'complete', 'done'].includes(scan.status);
    return completed && Number(scan.pagesScanned || 0) > 0;
  }

  function finderDefaultSource() {
    return state.sourceUrl || state.finderStatus?.defaultSourceUrl || state.settings.source_home || state.settings.default_source_url || 'https://www.pornpics.com/';
  }

  function finderTagForInput(value) {
    const query = String(value || '').trim().toLocaleLowerCase();
    if (!query) return null;
    return state.finderTags.find(tag => tag.label.toLocaleLowerCase() === query || tag.slug.toLocaleLowerCase() === query) || null;
  }

  function renderFinderFolders() {
    const list = $('#finder-folder-options');
    list.replaceChildren();
    [...state.finderFolders].sort((a, b) => a.path.localeCompare(b.path)).forEach(folder => {
      const count = folder.imageCount ? ` · ${formatNumber(folder.imageCount)} images` : '';
      const option = document.createElement('option');
      option.value = folder.path;
      option.label = `${folder.name}${count}`;
      list.append(option);
    });
  }

  function renderFinderTags() {
    const list = $('#finder-pose-tag-options');
    list.replaceChildren();
    [...state.finderTags].sort((a, b) => a.label.localeCompare(b.label)).forEach(tag => {
      const option = document.createElement('option');
      option.value = tag.label;
      option.label = `${tag.label} · ${poseRoleLabel(tag.defaultRole)} control`;
      list.append(option);
    });
  }

  function normalizeFinderMode(value) {
    return ['joytag', 'tag'].includes(String(value || '').toLowerCase()) ? 'joytag' : 'pose';
  }

  function finderConfigUsesJoyTag() {
    return state.finderMode === 'joytag';
  }

  function finderScanUsesJoyTag(scan = state.finderScan) {
    return normalizeFinderMode(scan?.searchMode) === 'joytag';
  }

  function finderFolderKey(value) {
    return String(value || '').trim().replace(/\/+$/, '');
  }

  function humanizeJoytagTag(value) {
    const words = String(value || '').trim().replaceAll('_', ' ').replace(/\s+/g, ' ');
    return words ? words[0].toUpperCase() + words.slice(1) : '';
  }

  function finderReferenceAnalysisIsCurrent() {
    return Boolean(
      state.finderReferenceAnalysis?.fingerprint
      && finderFolderKey(state.finderReferenceAnalysisSource)
      && finderFolderKey($('#finder-folder').value) === finderFolderKey(state.finderReferenceAnalysisSource)
    );
  }

  function finderJoytagData(tag = state.finderJoytagSelectedTag) {
    return state.finderReferenceAnalysis?.tags.find(item => item.tag === tag) || null;
  }

  function setFinderJoytagDatasetLabel(value, { automatic = false } = {}) {
    const label = String(value || '').trim().replace(/\s+/g, ' ');
    $('#finder-joytag-dataset-label').value = label;
    $('#finder-pose-tag').value = label;
    state.finderJoytagAutoPoseLabel = automatic ? label : '';
  }

  function syncFinderJoytagDatasetFromSelection() {
    const proposed = humanizeJoytagTag(state.finderJoytagRequiredTags[0]);
    if (!proposed) return;
    const current = $('#finder-joytag-dataset-label').value.trim().replace(/\s+/g, ' ');
    if (!current || current === state.finderJoytagAutoPoseLabel) {
      setFinderJoytagDatasetLabel(proposed, { automatic: true });
    }
  }

  function renderFinderMode() {
    const joytag = finderConfigUsesJoyTag();
    $$('[data-finder-mode]').forEach(button => {
      const active = normalizeFinderMode(button.dataset.finderMode) === state.finderMode;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
    });
    $('#finder-pose-config').hidden = joytag;
    $('#finder-tag-config').hidden = !joytag;
    $('#finder-analyze-references').hidden = !joytag;
    const poseThresholdField = $('#finder-min-similarity').closest('.field');
    poseThresholdField.hidden = joytag;
    $('#finder-min-similarity').disabled = joytag;
    poseThresholdField.closest('.finder-config-row').style.gridTemplateColumns = joytag ? '1fr' : '';
    $('#finder-mode-copy').textContent = joytag
      ? 'Require every positive JoyTag signal and optionally reject images that contain any excluded signal.'
      : 'Compare body geometry and use visual similarity only as fallback evidence.';
    $('#finder-title').textContent = joytag ? 'Tag Finder' : 'Pose Finder';
    $('#finder-title').nextElementSibling.textContent = joytag
      ? 'Build a precise JoyTag rule from required and excluded signals, then review only galleries with qualifying images.'
      : 'Teach GalleryFlow from a folder of examples, then review galleries ranked by pose evidence, with visual layout used only as a fallback.';
    $('#finder-start').querySelector('span').textContent = joytag ? 'Start tag scan' : 'Start scan';
    const welcome = $('#finder-welcome');
    $('h3', welcome).textContent = joytag ? 'Find galleries with matching tags' : 'Find galleries with matching poses';
    $('p', welcome).textContent = joytag
      ? 'Analyze a reference folder, require one or more signals, and optionally exclude common false-positive tags.'
      : 'Select an examples folder and pose tag. High-precision pose matches rank ahead of visual fallbacks, and every review decision is remembered.';
    renderFinderCorpus();
  }

  function setFinderMode(value, { persist = true } = {}) {
    const mode = normalizeFinderMode(value);
    if (state.finderScan && !finderScanIsTerminal() && mode !== state.finderMode) return;
    if (mode === 'joytag' && state.finderMode !== 'joytag') {
      const existing = $('#finder-joytag-dataset-label').value.trim()
        || $('#finder-pose-tag').value.trim();
      if (existing && !$('#finder-joytag-dataset-label').value.trim()) {
        setFinderJoytagDatasetLabel(existing, { automatic: true });
      }
    }
    state.finderMode = mode;
    if (persist) storage.set('finder-mode', mode);
    renderFinderMode();
    renderFinderReferenceAnalysis();
    renderFinderStatus();
    renderFinderResults();
    syncFinderConfigAvailability();
    if (mode === 'pose') loadFinderFeedback({ quiet: true, force: true });
  }

  function invalidateFinderReferenceAnalysis({ clearSelection = true } = {}) {
    state.finderReferenceAnalysisRequest += 1;
    state.finderReferenceAnalysis = null;
    state.finderReferenceAnalysisSource = '';
    state.finderReferenceAnalysisLoading = false;
    state.finderReferenceAnalysisError = '';
    if (clearSelection) {
      state.finderJoytagSelectedTag = '';
      state.finderJoytagRequiredTags = [];
      state.finderJoytagExcludedTags = [];
      const datasetLabel = $('#finder-joytag-dataset-label').value.trim();
      if (datasetLabel && datasetLabel === state.finderJoytagAutoPoseLabel) {
        setFinderJoytagDatasetLabel('');
      }
    }
    setButtonBusy($('#finder-analyze-references'), false);
    renderFinderReferenceAnalysis();
    syncFinderConfigAvailability();
  }

  function finderJoytagQueryRole(tag) {
    if (state.finderJoytagRequiredTags.includes(tag)) return 'required';
    if (state.finderJoytagExcludedTags.includes(tag)) return 'excluded';
    return '';
  }

  function finderJoytagCatalogHas(tag) {
    return Boolean(
      tag
      && state.finderReferenceAnalysis?.tagCatalog.includes(tag)
    );
  }

  function finderJoytagQueryChip(tag, role, { removable = true } = {}) {
    const chip = document.createElement('span');
    chip.className = `finder-joytag-query-chip is-${role}`;
    chip.setAttribute('role', 'listitem');
    const label = document.createElement('span');
    label.textContent = humanizeJoytagTag(tag);
    chip.append(label);
    if (removable) {
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.dataset.finderJoytagRemoveRole = role;
      remove.dataset.finderJoytagTag = tag;
      remove.title = `Remove ${humanizeJoytagTag(tag)} from ${role} tags`;
      remove.setAttribute('aria-label', remove.title);
      remove.innerHTML = '<svg aria-hidden="true"><use href="#i-close"></use></svg>';
      chip.append(remove);
    }
    return chip;
  }

  function renderFinderJoytagQueryList(container, tags, role) {
    container.replaceChildren();
    if (!tags.length) {
      const empty = document.createElement('span');
      empty.className = 'finder-joytag-query-empty';
      empty.textContent = role === 'required'
        ? 'Choose at least one tag'
        : 'No exclusions';
      container.append(empty);
      return;
    }
    tags.forEach(tag => container.append(finderJoytagQueryChip(tag, role)));
  }

  function setFinderJoytagTagRole(tag, role, { toggle = true, announceChange = true } = {}) {
    if (!finderJoytagCatalogHas(tag) || !['required', 'excluded', ''].includes(role)) return;
    const previousRole = finderJoytagQueryRole(tag);
    const nextRole = toggle && previousRole === role ? '' : role;
    if (
      nextRole
      && previousRole !== nextRole
      && (nextRole === 'required' ? state.finderJoytagRequiredTags : state.finderJoytagExcludedTags).length
        >= FINDER_JOYTAG_QUERY_LIMIT
    ) {
      toast(
        `Maximum ${nextRole} tags reached`,
        `A Tag Search can use up to ${FINDER_JOYTAG_QUERY_LIMIT} ${nextRole} tags.`,
        'info'
      );
      return;
    }
    const previousPrimary = state.finderJoytagRequiredTags[0] || '';
    state.finderJoytagRequiredTags = state.finderJoytagRequiredTags.filter(item => item !== tag);
    state.finderJoytagExcludedTags = state.finderJoytagExcludedTags.filter(item => item !== tag);
    if (nextRole === 'required') state.finderJoytagRequiredTags.push(tag);
    if (nextRole === 'excluded') state.finderJoytagExcludedTags.push(tag);
    state.finderJoytagSelectedTag = tag;
    if (state.finderJoytagRequiredTags.length) {
      syncFinderJoytagDatasetFromSelection();
    } else if (
      previousPrimary
      && $('#finder-joytag-dataset-label').value.trim() === state.finderJoytagAutoPoseLabel
    ) {
      setFinderJoytagDatasetLabel('');
    }
    renderFinderReferenceAnalysis();
    syncFinderConfigAvailability();
    if (announceChange) {
      const action = nextRole === 'required'
        ? 'required'
        : nextRole === 'excluded' ? 'excluded' : 'removed from the query';
      announce(`${humanizeJoytagTag(tag)} ${action}.`);
    }
  }

  function inspectFinderJoytagTag(tag) {
    if (!finderJoytagCatalogHas(tag)) return;
    if (!finderJoytagQueryRole(tag)) {
      setFinderJoytagTagRole(tag, 'required');
      return;
    }
    state.finderJoytagSelectedTag = tag;
    renderFinderReferenceAnalysis();
    announce(`${humanizeJoytagTag(tag)} reference scores shown.`);
  }

  function renderFinderReferenceAnalysis() {
    const card = $('#finder-joytag-analysis');
    const analysis = state.finderReferenceAnalysis;
    const loading = state.finderReferenceAnalysisLoading;
    const error = state.finderReferenceAnalysisError;
    const selectedTag = state.finderJoytagSelectedTag;
    const threshold = Math.max(0.05, Math.min(0.95, Number(state.finderJoytagThreshold || 0.4)));
    const rejectThreshold = Math.max(
      0.05,
      Math.min(0.95, Number(state.finderJoytagRejectThreshold || 0.4))
    );
    state.finderJoytagThreshold = threshold;
    state.finderJoytagRejectThreshold = rejectThreshold;
    card.classList.toggle('is-loading', loading);
    card.classList.toggle('is-ready', Boolean(analysis && !error));
    card.classList.toggle('is-error', Boolean(error));
    $('#finder-joytag-state').textContent = loading ? 'Analyzing' : error ? 'Retry' : analysis ? 'Ready' : 'Waiting';
    const requiredCount = state.finderJoytagRequiredTags.length;
    const excludedCount = state.finderJoytagExcludedTags.length;
    $('#finder-joytag-summary').textContent = loading
      ? 'Tagging every reference image in batches…'
      : error
        ? 'Reference analysis failed—review the message below.'
        : analysis
          ? `${formatNumber(analysis.imageCount)} references · ${requiredCount} required · ${excludedCount} excluded`
          : finderScanUsesJoyTag() && requiredCount
            ? `Saved query has ${requiredCount} required and ${excludedCount} excluded; analyze before starting another.`
            : 'Analyze the examples folder to build a tag query.';
    const errorElement = $('#finder-joytag-error');
    errorElement.hidden = !error;
    errorElement.textContent = error;
    const empty = $('#finder-joytag-empty');
    empty.hidden = Boolean(analysis) || loading;
    const savedQuery = $('#finder-joytag-saved-query');
    savedQuery.replaceChildren();
    if (!analysis && !loading && requiredCount) {
      state.finderJoytagRequiredTags.forEach(tag => (
        savedQuery.append(finderJoytagQueryChip(tag, 'required', { removable: false }))
      ));
      state.finderJoytagExcludedTags.forEach(tag => (
        savedQuery.append(finderJoytagQueryChip(tag, 'excluded', { removable: false }))
      ));
    }
    savedQuery.hidden = Boolean(analysis || loading || !requiredCount);
    if (!analysis && !loading) {
      $('strong', empty).textContent = error ? 'Analysis unavailable' : 'No analysis yet';
      $('small', empty).textContent = error
        ? 'Correct the folder or retry the analysis.'
        : finderScanUsesJoyTag() && requiredCount
          ? 'This saved scan query is restored below. Run Analyze to edit it or start a new tag scan.'
          : 'Choose a folder and run JoyTag to inspect its strongest tags.';
    }
    $('#finder-joytag-results').hidden = !analysis;
    $('#finder-joytag-threshold').value = threshold.toFixed(2);
    $('#finder-joytag-threshold-output').textContent = threshold.toFixed(2);
    $('#finder-joytag-reject-threshold').value = rejectThreshold.toFixed(2);
    $('#finder-joytag-reject-output').textContent = rejectThreshold.toFixed(2);
    if (!analysis) return;

    $('#finder-joytag-image-count').textContent = formatNumber(analysis.imageCount);
    const providerCopy = [analysis.provider, analysis.quantization].filter(Boolean).join(' · ');
    $('#finder-joytag-provider').textContent = providerCopy || 'Ready';
    $('#finder-joytag-provider').title = [
      analysis.modelKey,
      analysis.bytesPerCachedImage ? `${formatBytes(analysis.bytesPerCachedImage)} cached per image` : ''
    ].filter(Boolean).join(' · ');

    const filter = state.finderJoytagTagFilter.trim().toLocaleLowerCase();
    const topByTag = new Map(analysis.tags.map(item => [item.tag, item]));
    const selectedCatalogTags = [
      ...state.finderJoytagRequiredTags,
      ...state.finderJoytagExcludedTags
    ].filter(tag => analysis.tagCatalog.includes(tag));
    const sourceTags = filter
      ? analysis.tagCatalog.filter(tag => (
          tag.toLocaleLowerCase().includes(filter)
          || humanizeJoytagTag(tag).toLocaleLowerCase().includes(filter)
        ))
      : [...new Set([...selectedCatalogTags, ...analysis.tags.map(item => item.tag)])];
    const totalMatches = sourceTags.length;
    const tags = sourceTags.slice(0, FINDER_JOYTAG_CATALOG_RENDER_LIMIT);
    $('#finder-joytag-list-copy').textContent = filter
      ? `${formatNumber(Math.min(totalMatches, tags.length))} of ${formatNumber(totalMatches)} catalog matches`
      : `${formatNumber(analysis.tags.length)} top reference signals${selectedCatalogTags.some(tag => !topByTag.has(tag)) ? ' · selected catalog tags included' : ''}`;
    const tagList = $('#finder-joytag-tags');
    tagList.replaceChildren();
    tags.forEach(tag => {
      const item = topByTag.get(tag) || null;
      const role = finderJoytagQueryRole(tag);
      const comparisonThreshold = role === 'excluded' ? rejectThreshold : threshold;
      const values = item
        ? analysis.images
          .map(image => normalizeFinderScore(image.scores[tag]))
          .filter(score => score !== null)
        : [];
      const passing = values.filter(score => score >= comparisonThreshold).length;
      const row = document.createElement('article');
      row.className = `finder-joytag-tag${role ? ` is-${role}` : ''}`;
      row.setAttribute('role', 'listitem');
      row.title = item
        ? `${tag}: average ${item.average.toFixed(3)}, median ${item.median.toFixed(3)}`
        : `${tag}: catalog tag without reference statistics`;
      const choice = document.createElement('button');
      choice.type = 'button';
      choice.className = 'finder-joytag-tag-choice';
      choice.dataset.finderJoytagInspect = tag;
      choice.title = role
        ? `Inspect ${humanizeJoytagTag(tag)} reference scores`
        : `Require ${humanizeJoytagTag(tag)} and inspect its reference scores`;
      const name = document.createElement('span');
      name.textContent = humanizeJoytagTag(tag);
      const average = document.createElement('b');
      average.textContent = item ? finderScoreLabel(item.average) : 'Catalog';
      const coverage = document.createElement('small');
      coverage.textContent = item
        ? `${passing}/${analysis.imageCount} ${role === 'excluded' ? 'would reject' : 'meet threshold'}`
        : 'No reference scores · usable in the query';
      choice.append(name, average, coverage);
      const actions = document.createElement('div');
      actions.className = 'finder-joytag-tag-actions';
      actions.setAttribute('role', 'group');
      actions.setAttribute('aria-label', `${humanizeJoytagTag(tag)} query role`);
      ['required', 'excluded'].forEach(queryRole => {
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.finderJoytagRole = queryRole;
        button.dataset.finderJoytagTag = tag;
        button.setAttribute('aria-pressed', String(role === queryRole));
        button.textContent = queryRole === 'required' ? 'Require' : 'Exclude';
        actions.append(button);
      });
      row.append(choice, actions);
      tagList.append(row);
    });
    if (!tags.length) {
      const message = document.createElement('small');
      message.textContent = 'No JoyTag catalog tags match this search.';
      tagList.append(message);
    }

    renderFinderJoytagQueryList(
      $('#finder-joytag-required-tags'),
      state.finderJoytagRequiredTags,
      'required'
    );
    renderFinderJoytagQueryList(
      $('#finder-joytag-excluded-tags'),
      state.finderJoytagExcludedTags,
      'excluded'
    );
    $('#finder-joytag-query-title').textContent = requiredCount
      ? `Match all ${requiredCount} required ${requiredCount === 1 ? 'tag' : 'tags'}`
      : 'Choose at least one required tag';
    $('#finder-joytag-query-count').textContent = `${requiredCount} required · ${excludedCount} excluded`;

    const selected = finderJoytagData();
    const selectedInCatalog = analysis.tagCatalog.includes(selectedTag);
    const selectedRole = finderJoytagQueryRole(selectedTag);
    $('#finder-joytag-selected-tag').textContent = selectedInCatalog
      ? `${humanizeJoytagTag(selectedTag)}${selectedRole ? ` · ${selectedRole}` : ''}`
      : 'Choose a tag';
    $('#finder-joytag-average').textContent = selected ? selected.average.toFixed(3) : '—';
    $('#finder-joytag-minimum').textContent = selected ? selected.minimum.toFixed(3) : '—';
    $('#finder-joytag-maximum').textContent = selected ? selected.maximum.toFixed(3) : '—';
    const selectedScores = selected
      ? analysis.images.map(image => normalizeFinderScore(image.scores[selected.tag])).filter(score => score !== null)
      : [];
    const selectedThreshold = selectedRole === 'excluded' ? rejectThreshold : threshold;
    const passing = selectedScores.filter(score => score >= selectedThreshold).length;
    $('#finder-joytag-reference-coverage').textContent = !selectedInCatalog
      ? 'No signal'
      : !selected
        ? 'Catalog only'
        : selectedRole === 'excluded'
          ? `${passing} / ${analysis.imageCount} would reject`
          : `${passing} / ${analysis.imageCount} pass`;
    $('#finder-joytag-reference-copy').textContent = selected
      ? selectedRole === 'excluded'
        ? `Scores at or above ${rejectThreshold.toFixed(2)} would reject the image.`
        : `Scores at or above ${threshold.toFixed(2)} pass this required signal.`
      : selectedInCatalog
        ? 'This catalog-only tag has no scores in the compact reference analysis.'
        : 'Choose any tag to inspect its reference scores.';

    const grid = $('#finder-joytag-reference-grid');
    grid.replaceChildren();
    analysis.images.forEach(image => {
      const score = selected ? normalizeFinderScore(image.scores[selected.tag]) : null;
      const reference = document.createElement('article');
      reference.className = 'finder-joytag-reference';
      const meetsThreshold = score !== null && score >= selectedThreshold;
      reference.classList.toggle(
        'is-passing',
        score === null || selectedRole === 'excluded' ? !meetsThreshold : meetsThreshold
      );
      reference.classList.toggle(
        'is-rejecting',
        selectedRole === 'excluded' && meetsThreshold
      );
      reference.title = selected && score !== null
        ? `${image.name} · ${selected.tag} ${score.toFixed(3)}`
        : image.name;
      const preview = document.createElement('img');
      preview.loading = 'lazy';
      preview.decoding = 'async';
      loadImage(preview, image.previewUrl, image.name);
      const copy = document.createElement('span');
      const name = document.createElement('b');
      name.textContent = image.name;
      const value = document.createElement('strong');
      value.textContent = score === null ? '—' : score.toFixed(3);
      copy.append(name, value);
      reference.append(preview, copy);
      grid.append(reference);
    });
  }

  async function analyzeFinderReferences() {
    if (state.finderReferenceAnalysisLoading || state.finderBusy) return;
    const exampleDirectory = $('#finder-folder').value.trim();
    if (!exampleDirectory) {
      toast('Choose a reference folder', 'Enter a folder inside the Finder library before analyzing it.', 'info');
      $('#finder-folder').focus();
      return;
    }
    const previousTag = state.finderJoytagSelectedTag;
    const previousRequired = [...state.finderJoytagRequiredTags];
    const previousExcluded = [...state.finderJoytagExcludedTags];
    const sourceKey = finderFolderKey(exampleDirectory);
    const request = ++state.finderReferenceAnalysisRequest;
    state.finderReferenceAnalysis = null;
    state.finderReferenceAnalysisSource = exampleDirectory;
    state.finderReferenceAnalysisLoading = true;
    state.finderReferenceAnalysisError = '';
    renderFinderReferenceAnalysis();
    syncFinderConfigAvailability();
    const button = $('#finder-analyze-references');
    setButtonBusy(button, true, 'Analyzing…');
    try {
      const data = await api('/api/finder/reference-analysis', {
        method: 'POST',
        body: { example_directory: exampleDirectory, top_tags: 40 }
      });
      if (
        request !== state.finderReferenceAnalysisRequest
        || sourceKey !== finderFolderKey($('#finder-folder').value)
      ) return;
      const analysis = normalizeFinderReferenceAnalysis(data);
      if (!analysis) throw new ApiError('The server returned an invalid JoyTag reference analysis.');
      state.finderReferenceAnalysis = analysis;
      state.finderReferenceAnalysisSource = exampleDirectory;
      const catalog = new Set(analysis.tagCatalog);
      state.finderJoytagRequiredTags = previousRequired
        .filter(tag => catalog.has(tag))
        .slice(0, FINDER_JOYTAG_QUERY_LIMIT);
      const required = new Set(state.finderJoytagRequiredTags);
      state.finderJoytagExcludedTags = previousExcluded
        .filter(tag => catalog.has(tag) && !required.has(tag))
        .slice(0, FINDER_JOYTAG_QUERY_LIMIT);
      state.finderJoytagSelectedTag = catalog.has(previousTag)
        ? previousTag
        : state.finderJoytagRequiredTags[0] || '';
      state.finderReferenceAnalysisError = '';
      if (state.finderJoytagRequiredTags.length) syncFinderJoytagDatasetFromSelection();
      toast('Reference analysis ready', `${formatNumber(analysis.imageCount)} images tagged. Build the required and excluded query.`, 'success');
      announce(`JoyTag analysis complete for ${analysis.imageCount} reference images.`);
    } catch (error) {
      if (request !== state.finderReferenceAnalysisRequest) return;
      state.finderReferenceAnalysis = null;
      state.finderReferenceAnalysisError = errorMessage(error);
      toast('Could not analyze references', errorMessage(error), 'error');
    } finally {
      if (request === state.finderReferenceAnalysisRequest) {
        state.finderReferenceAnalysisLoading = false;
        setButtonBusy(button, false);
        renderFinderReferenceAnalysis();
        syncFinderConfigAvailability();
      }
    }
  }

  function finderFeedbackTag() {
    const label = $('#finder-pose-tag').value.trim().replace(/\s+/g, ' ');
    const existing = finderTagForInput(label);
    if (existing) return existing;
    if (
      label
      && state.finderScan?.poseTagId !== undefined
      && state.finderScan?.poseTagId !== null
      && label.toLocaleLowerCase() === state.finderScan.poseTagLabel.toLocaleLowerCase()
    ) {
      return finderPoseTagForScan(state.finderScan);
    }
    return label ? { id: null, label } : null;
  }

  function finderFeedbackIsSaving() {
    return state.finderFeedbackMutations > 0
      || state.finderResults.some(result => Boolean(result.feedbackSaving));
  }

  function finderFeedbackSaveBlocksReset(tag = finderFeedbackTag()) {
    return Boolean(
      tag?.id != null
      && String(state.finderScan?.poseTagId) === String(tag.id)
      && finderFeedbackIsSaving()
    );
  }

  function renderFinderFeedback() {
    const card = $('#finder-feedback-card');
    const tag = finderFeedbackTag();
    const feedback = tag?.id !== undefined
      && tag?.id !== null
      && String(state.finderFeedback?.poseTagId) === String(tag.id)
      ? state.finderFeedback
      : null;
    const accepted = feedback?.accepted ?? (tag?.id == null && tag ? 0 : null);
    const rejected = feedback?.rejected ?? (tag?.id == null && tag ? 0 : null);
    const total = feedback ? Math.max(feedback.total, feedback.accepted + feedback.rejected) : 0;
    card.classList.remove('is-active', 'is-collecting', 'is-unavailable');
    card.classList.toggle('is-active', Boolean(feedback?.active && total));
    card.classList.toggle('is-collecting', Boolean(feedback && total && !feedback.active));
    card.classList.toggle('is-unavailable', state.finderFeedbackSupported === false || Boolean(state.finderFeedbackError));
    $('#finder-feedback-title').textContent = tag?.label
      ? `${tag.label} feedback`
      : 'Select an existing pose';
    $('#finder-feedback-accepted').textContent = accepted === null ? '—' : formatNumber(accepted);
    $('#finder-feedback-rejected').textContent = rejected === null ? '—' : formatNumber(rejected);
    $('#finder-feedback-accepted').closest('.finder-feedback-count').title = feedback
      ? `${formatNumber(feedback.usableAcceptedSamples)} of ${formatNumber(feedback.accepted)} accepted samples are currently usable`
      : '';
    $('#finder-feedback-rejected').closest('.finder-feedback-count').title = feedback
      ? `${formatNumber(feedback.usableRejectedSamples)} of ${formatNumber(feedback.rejected)} rejected samples are currently usable`
      : '';
    const stateLabel = state.finderFeedbackLoading
      ? 'Loading'
      : state.finderFeedbackError
        ? 'Retry'
        : state.finderFeedbackSupported === false
          ? 'Unavailable'
          : !tag
            ? 'Waiting'
            : tag.id == null
              ? 'New pose'
              : !feedback
                ? 'Checking'
                : total
                  ? feedback.active ? 'In use' : 'Collecting'
                  : 'No feedback';
    $('#finder-feedback-state').textContent = stateLabel;
    const reset = $('#finder-feedback-reset');
    const reviewSaving = finderFeedbackSaveBlocksReset(tag);
    reset.disabled = state.finderFeedbackBusy || state.finderFeedbackLoading || reviewSaving || !feedback || !total;
    reset.title = reviewSaving ? 'Wait for the gallery review to finish saving' : '';
    const progress = feedback?.minimumGalleries
      ? `Accepted ${Math.min(feedback.usableAcceptedGalleries, feedback.minimumGalleries)}/${feedback.minimumGalleries} usable galleries · rejected ${Math.min(feedback.usableRejectedGalleries, feedback.minimumGalleries)}/${feedback.minimumGalleries}. `
      : '';
    $('#finder-feedback-copy').textContent = state.finderFeedbackError
      ? state.finderFeedbackError
      : state.finderFeedbackSupported === false
        ? 'This server does not expose pose-specific ranking feedback.'
        : !tag
          ? 'Choose an existing pose to see its reversible ranking feedback.'
          : tag.id == null
            ? 'Feedback begins after this pose is created and you review candidates. It is scoped to this pose and reversible.'
            : `${progress}Checked suggestions become pose feedback; unchecked suggestions are excluded. Reviews can adjust or veto future pose ranking—the vision models are not retrained.`;
  }

  function renderFinderStatus() {
    const model = state.finderStatus;
    const card = $('#finder-model-card');
    const root = model?.folderRoot;
    const normalizedRoot = root ? root.replace(/\/+$/, '') || '/' : '';
    const fullExample = normalizedRoot === '/' ? '/poses/matting-press' : `${normalizedRoot}/poses/matting-press`;
    $('#finder-folder-hint').textContent = normalizedRoot
      ? `Use poses/matting-press relative to ${normalizedRoot}, or paste ${fullExample}. Existing folders are suggestions only.`
      : 'Use a library-relative path such as poses/matting-press, or paste the full container path. Existing folders are suggestions only.';
    if (finderConfigUsesJoyTag()) {
      const available = model?.joytagAvailable !== false;
      const ready = Boolean(model?.joytagReady || state.finderReferenceAnalysis);
      const error = model?.joytagError || '';
      card.classList.toggle('is-ready', available && !error);
      card.classList.toggle('is-error', !available || Boolean(error));
      $('#finder-model-name').textContent = 'JoyTag image tagger';
      $('#finder-model-detail').textContent = error
        ? `Model error: ${error}`
        : [
            ready ? 'Ready for batched tag confidence search' : 'Loads automatically when references are analyzed',
            model?.joytagProvider || state.finderReferenceAnalysis?.provider || ''
          ].filter(Boolean).join(' · ');
      $('#finder-model-state').textContent = error ? 'Retry available' : ready ? 'Ready' : available ? 'Available' : 'Unavailable';
      return;
    }
    card.classList.toggle('is-ready', Boolean(model?.ready && !model.error));
    card.classList.toggle('is-error', Boolean(model && (!model.ready || model.error)));
    $('#finder-model-name').textContent = model?.name || 'Model unavailable';
    const appearanceBatch = model?.inferenceBatch?.appearance;
    const poseBatch = model?.inferenceBatch?.pose;
    const hasAppearanceBatch = Number.isSafeInteger(appearanceBatch);
    const hasPoseBatch = Number.isSafeInteger(poseBatch);
    let batchDetail = '';
    if (hasAppearanceBatch && hasPoseBatch) {
      batchDetail = appearanceBatch === poseBatch
        ? appearanceBatch > 1
          ? `Vision batches up to ${appearanceBatch} images`
          : 'Single-image inference'
        : `Appearance batches up to ${appearanceBatch} · Pose batches up to ${poseBatch}`;
    } else if (hasAppearanceBatch) {
      batchDetail = appearanceBatch > 1
        ? `Appearance batches up to ${appearanceBatch} images`
        : 'Appearance uses single-image inference';
    } else if (hasPoseBatch) {
      batchDetail = poseBatch > 1
        ? `Pose batches up to ${poseBatch} images`
        : 'Pose uses single-image inference';
    }
    $('#finder-model-detail').textContent = [
      model?.detail || 'Could not read model status',
      batchDetail
    ].filter(Boolean).join(' · ');
    $('#finder-model-state').textContent = model?.error ? 'Retry available' : model?.modelReady ? 'Ready' : model?.ready ? 'Available' : model ? model.status.replaceAll('_', ' ') : 'Offline';
  }

  function renderFinderJoytagIndex() {
    const panel = $('#finder-corpus-joytag');
    const joytagMode = finderConfigUsesJoyTag();
    panel.hidden = !joytagMode;
    if (!joytagMode) return;
    const coverage = state.finderCorpus?.joytag || null;
    const savedJob = state.finderCorpus?.joytagIndexJob || null;
    const active = finderJoytagIndexIsActive(savedJob);
    const job = active
      || !savedJob?.modelKey
      || !coverage?.modelKey
      || savedJob.modelKey === coverage.modelKey
      ? savedJob
      : null;
    const complete = Boolean(
      coverage
      && coverage.totalImages > 0
      && coverage.missingImages === 0
    );
    const failed = Boolean(
      job
      && ['failed', 'completed_with_errors'].includes(job.status)
    );
    panel.classList.toggle('is-running', active);
    panel.classList.toggle('is-complete', complete);
    panel.classList.toggle('is-failed', failed);
    panel.setAttribute('aria-busy', String(active || state.finderJoytagIndexBusy));
    $('#finder-corpus-joytag-cached').textContent = coverage
      ? formatNumber(coverage.cachedImages)
      : '—';
    $('#finder-corpus-joytag-total').textContent = coverage
      ? formatNumber(coverage.totalImages)
      : '—';

    const progress = active
      ? finderCorpusPercent(job?.progress)
      : finderCorpusPercent(coverage?.percent);
    const progressElement = $('#finder-corpus-index-progress');
    progressElement.setAttribute('aria-valuenow', String(progress));
    progressElement.setAttribute(
      'aria-valuetext',
      active && job
        ? `${formatNumber(job.processedImages)} processed, ${formatNumber(job.indexedImages)} indexed, ${formatNumber(job.failedImages)} failed`
        : coverage
          ? `${formatNumber(coverage.cachedImages)} of ${formatNumber(coverage.totalImages)} local images cached`
          : 'JoyTag local coverage unavailable'
    );
    $('#finder-corpus-index-progress-bar').style.width = `${progress}%`;

    const stateLabel = state.finderJoytagIndexSupported === false
      ? 'Unavailable'
      : active
        ? job.status === 'queued'
          ? 'Queued'
          : job.status === 'canceling' || job.cancelRequested ? 'Canceling' : 'Indexing'
        : job?.status === 'completed_with_errors'
          ? 'Completed with errors'
          : job?.status === 'canceled'
            ? 'Canceled'
            : job?.status === 'failed'
              ? 'Failed'
              : complete
                ? 'Complete'
                : coverage
                  ? coverage.totalImages
                    ? coverage.cachedImages ? 'Partial' : 'Not indexed'
                    : 'Empty'
                  : 'Checking';
    $('#finder-corpus-joytag-state').textContent = stateLabel;

    const progressCopy = $('#finder-corpus-index-progress-copy');
    if (active && job) {
      progressCopy.textContent = job.status === 'queued'
        ? `Waiting to index ${formatNumber(job.remainingImages)} uncached local images.`
        : `${formatNumber(job.processedImages)} processed · ${formatNumber(job.indexedImages)} cached · ${formatNumber(job.failedImages)} failed · ${formatNumber(job.remainingImages)} remaining`;
    } else if (!coverage) {
      progressCopy.textContent = state.finderJoytagIndexSupported === false
        ? 'This server does not expose explicit JoyTag corpus indexing.'
        : 'Checking cached image coverage…';
    } else if (!coverage.totalImages) {
      progressCopy.textContent = 'The Local Gallery Index does not contain any images yet.';
    } else if (complete) {
      progressCopy.textContent = `All ${formatNumber(coverage.totalImages)} local images are cached for this JoyTag model.`;
    } else if (job?.status === 'canceled') {
      progressCopy.textContent = `Indexing stopped safely. ${formatNumber(coverage.missingImages)} local images remain uncached.`;
    } else if (job?.status === 'failed') {
      progressCopy.textContent = `Indexing stopped after ${formatNumber(job.processedImages)} images. ${formatNumber(coverage.missingImages)} remain uncached.`;
    } else {
      progressCopy.textContent = `${formatNumber(coverage.cachedImages)} cached · ${formatNumber(coverage.missingImages)} uncached local images can be indexed.`;
    }

    const start = $('#finder-corpus-index-start');
    const cancel = $('#finder-corpus-index-cancel');
    start.hidden = active;
    cancel.hidden = !active;
    start.disabled = state.finderJoytagIndexBusy
      || state.finderJoytagIndexSupported === false
      || !coverage
      || coverage.missingImages === 0
      || state.finderStatus?.joytagAvailable === false;
    $('span', start).textContent = complete
      ? 'Local corpus indexed'
      : coverage?.cachedImages
        ? `Index ${formatNumber(coverage.missingImages)} remaining`
        : 'Index local corpus';
    start.title = state.finderJoytagIndexSupported === false
      ? 'Explicit JoyTag indexing is unavailable on this server'
      : state.finderStatus?.joytagAvailable === false
        ? state.finderStatus?.joytagError || 'JoyTag is unavailable'
        : !coverage
          ? 'Waiting for local corpus coverage'
          : !coverage.totalImages
            ? 'Add galleries to the Local Gallery Index first'
            : complete ? 'Every local image is already cached for this JoyTag model' : '';
    cancel.disabled = state.finderJoytagIndexBusy
      || Boolean(job?.cancelRequested)
      || job?.status === 'canceling';
    cancel.title = cancel.disabled && active ? 'Cancellation is already requested' : '';

    const error = $('#finder-corpus-index-error');
    const errorDetails = job?.error
      || (job?.errors?.length
        ? `${job.errors[0]}${job.errors.length > 1 ? ` · ${formatNumber(job.errors.length - 1)} more` : ''}`
        : '');
    error.hidden = !failed || !errorDetails;
    error.textContent = errorDetails;
  }

  function renderFinderCorpus() {
    const card = $('#finder-corpus-card');
    const corpus = state.finderCorpus;
    renderFinderJoytagIndex();
    card.classList.remove('is-ready', 'is-building', 'is-unavailable');
    const setCopy = (lead, detail) => {
      const copy = $('#finder-corpus-copy');
      const strong = document.createElement('strong');
      strong.textContent = lead;
      copy.replaceChildren(strong, document.createTextNode(` ${detail}`));
    };
    if (!corpus) {
      card.classList.toggle('is-unavailable', state.finderCorpusSupported === false);
      $('#finder-corpus-state').textContent = state.finderCorpusSupported === false ? 'Unavailable' : 'Checking';
      ['finder-corpus-galleries', 'finder-corpus-ready', 'finder-corpus-images', 'finder-corpus-bytes', 'finder-corpus-cache-entries'].forEach(id => {
        $(`#${id}`).textContent = '—';
      });
      $('#finder-corpus-coverage').hidden = true;
      setCopy(
        state.finderCorpusSupported === false ? 'Index status unavailable.' : 'Saved in /data.',
        state.finderCorpusSupported === false
          ? 'This server does not expose Local Gallery Index statistics.'
          : finderConfigUsesJoyTag()
            ? 'Tag searches use cached local JoyTag vectors only, skip uncached images, then go directly to the Source URL.'
            : 'Every new scan searches all indexed galleries first—not only the selected source—then explores that Source URL for more.'
      );
      return;
    }
    const hasIndex = corpus.galleries > 0 || corpus.images > 0;
    const partialReady = hasIndex && corpus.ready < corpus.images;
    card.classList.toggle('is-ready', hasIndex && !partialReady);
    card.classList.toggle('is-building', partialReady || (!hasIndex && corpus.cacheEntries > 0));
    $('#finder-corpus-state').textContent = hasIndex ? partialReady ? 'Partial' : 'Ready' : corpus.cacheEntries ? 'Cache saved' : 'Empty';
    $('#finder-corpus-galleries').textContent = formatNumber(corpus.galleries);
    $('#finder-corpus-ready').textContent = formatNumber(corpus.ready);
    $('#finder-corpus-images').textContent = formatNumber(corpus.images);
    $('#finder-corpus-bytes').textContent = formatBytes(corpus.cacheBytes);
    $('#finder-corpus-cache-entries').textContent = formatNumber(corpus.cacheEntries);
    const cacheStat = $('#finder-corpus-bytes').closest('span');
    const cacheLimits = [];
    if (corpus.maxCacheBytes) cacheLimits.push(`${formatBytes(corpus.cacheBytes)} of ${formatBytes(corpus.maxCacheBytes)}`);
    if (corpus.maxCacheEntries) cacheLimits.push(`${formatNumber(corpus.cacheEntries)} of ${formatNumber(corpus.maxCacheEntries)} entries`);
    cacheStat.title = cacheLimits.length ? `Descriptor cache: ${cacheLimits.join(' · ')}` : 'Descriptor cache usage';
    $('#finder-corpus-complete').textContent = formatNumber(corpus.complete);
    $('#finder-corpus-partial').textContent = formatNumber(corpus.partial);
    $('#finder-corpus-partial-wrap').hidden = !corpus.partial;
    $('#finder-corpus-coverage').hidden = !corpus.complete && !corpus.partial;
    setCopy(
      'Saved in /data.',
      finderConfigUsesJoyTag()
        ? 'Ordinary Tag searches use cached local images only and never fill missing cache entries automatically.'
        : hasIndex
          ? 'Every new scan searches all indexed galleries first—not only the selected source—then explores that Source URL for more.'
          : 'Your first scan will build the index; later poses can search it before exploring their Source URL.'
    );
  }

  function renderFinderScans() {
    const select = $('#finder-scan-select');
    const selected = state.finderScanId ? String(state.finderScanId) : '';
    select.replaceChildren(new Option('New scan', ''));
    const scans = [...state.finderScans];
    if (state.finderScan?.id && !scans.some(scan => String(scan.id) === String(state.finderScan.id))) scans.unshift(state.finderScan);
    scans.forEach(scan => {
      const date = scan.createdAt ? relativeTime(scan.createdAt) : '';
      const searchLabel = finderScanUsesJoyTag(scan)
        ? finderJoytagScanLabel(scan)
        : scan.poseTagLabel || 'Pose scan';
      const modeLabel = finderScanUsesJoyTag(scan) ? 'JoyTag' : 'Pose';
      const label = `${searchLabel} · ${modeLabel} · ${scan.status.replaceAll('_', ' ')}${date ? ` · ${date}` : ''}`;
      select.add(new Option(label, String(scan.id)));
    });
    if ([...select.options].some(option => option.value === selected)) select.value = selected;
  }

  function syncFinderConfigAvailability() {
    const locked = Boolean(state.finderScan && !finderScanIsTerminal());
    const feedbackMutationPending = state.finderFeedbackBusy || finderFeedbackIsSaving();
    const joytag = finderConfigUsesJoyTag();
    ['finder-folder', 'finder-pose-tag', 'finder-source', 'finder-pages'].forEach(id => { $(`#${id}`).disabled = locked || state.finderBusy; });
    $('#finder-min-similarity').disabled = locked || state.finderBusy || joytag;
    $('#finder-joytag-dataset-label').disabled = locked || state.finderBusy;
    $('#finder-joytag-threshold').disabled = locked || state.finderBusy || state.finderReferenceAnalysisLoading;
    $('#finder-joytag-reject-threshold').disabled = locked || state.finderBusy || state.finderReferenceAnalysisLoading;
    $('#finder-joytag-tag-filter').disabled = state.finderReferenceAnalysisLoading || !state.finderReferenceAnalysis;
    $$('[data-finder-joytag-role], [data-finder-joytag-remove-role], [data-finder-joytag-inspect]').forEach(button => {
      button.disabled = locked || state.finderBusy || state.finderReferenceAnalysisLoading;
    });
    $$('[data-finder-mode]').forEach(button => { button.disabled = locked || state.finderBusy; });
    $('#finder-use-current').disabled = locked || state.finderBusy;
    $('#finder-scan-select').disabled = state.finderLoading || feedbackMutationPending;
    const analysisCurrent = finderReferenceAnalysisIsCurrent();
    const joytagReady = state.finderStatus?.joytagAvailable !== false;
    const poseReady = Boolean(state.finderStatus?.ready);
    const modelReady = joytag ? joytagReady : poseReady;
    const hasConfig = Boolean(
      $('#finder-folder').value.trim()
      && $('#finder-pose-tag').value.trim()
      && $('#finder-source').value.trim()
      && (!joytag || (analysisCurrent && state.finderJoytagRequiredTags.length))
    );
    const analyze = $('#finder-analyze-references');
    analyze.disabled = locked
      || state.finderBusy
      || state.finderReferenceAnalysisLoading
      || !joytagReady
      || !$('#finder-folder').value.trim();
    analyze.title = !joytagReady
      ? state.finderStatus?.joytagError || 'JoyTag is unavailable'
      : !$('#finder-folder').value.trim() ? 'Choose an examples folder first' : '';
    $('#finder-start').hidden = locked;
    $('#finder-start').disabled = state.finderLoading
      || state.finderBusy
      || feedbackMutationPending
      || !modelReady
      || !hasConfig;
    $('#finder-start').title = joytag && !analysisCurrent
      ? 'Analyze the current examples folder before starting a tag scan'
      : joytag && !state.finderJoytagRequiredTags.length
        ? 'Choose at least one required JoyTag signal before starting'
        : !modelReady ? (joytag ? state.finderStatus?.joytagError : state.finderStatus?.detail) || 'Finder model unavailable' : '';
  }

  function applyFinderScanConfig(scan) {
    if (!scan) return;
    const scanMode = normalizeFinderMode(scan.searchMode);
    state.finderMode = scanMode;
    storage.set('finder-mode', scanMode);
    $('#finder-folder').value = scan.examplesFolder;
    $('#finder-pose-tag').value = scan.poseTagLabel;
    $('#finder-joytag-dataset-label').value = scan.poseTagLabel;
    state.finderJoytagAutoPoseLabel = '';
    state.finderJoytagRequiredTags = scanMode === 'joytag'
      ? [...scan.joytagRequiredTags]
      : [];
    state.finderJoytagExcludedTags = scanMode === 'joytag'
      ? [...scan.joytagExcludedTags]
      : [];
    state.finderJoytagSelectedTag = state.finderJoytagRequiredTags[0] || '';
    $('#finder-source').value = scan.sourceUrl || finderDefaultSource();
    $('#finder-pages').value = Math.max(1, Math.min(50, scan.pages || 5));
    $('#finder-min-similarity').value = scan.minSimilarity.toFixed(2);
    state.finderJoytagThreshold = Math.max(0.05, Math.min(0.95, scan.minSimilarity));
    $('#finder-joytag-threshold').value = state.finderJoytagThreshold.toFixed(2);
    $('#finder-joytag-threshold-output').textContent = state.finderJoytagThreshold.toFixed(2);
    state.finderJoytagRejectThreshold = Math.max(
      0.05,
      Math.min(0.95, scan.joytagRejectThreshold)
    );
    $('#finder-joytag-reject-threshold').value = state.finderJoytagRejectThreshold.toFixed(2);
    $('#finder-joytag-reject-output').textContent = state.finderJoytagRejectThreshold.toFixed(2);
    $('#finder-result-threshold').min = scanMode === 'joytag' ? '0.05' : '0.40';
    $('#finder-result-threshold').value = scan.minSimilarity.toFixed(2);
    $('#finder-min-output').textContent = scan.minSimilarity.toFixed(2);
    $('#finder-filter-output').textContent = scan.minSimilarity.toFixed(2);
    if (
      state.finderReferenceAnalysis
      && finderFolderKey(state.finderReferenceAnalysisSource) !== finderFolderKey(scan.examplesFolder)
    ) {
      state.finderReferenceAnalysis = null;
      state.finderReferenceAnalysisSource = '';
      state.finderReferenceAnalysisError = '';
    }
    renderFinderMode();
    renderFinderReferenceAnalysis();
  }

  function finderScoreLabel(score) {
    return `${Math.round((normalizeFinderScore(score, 0) || 0) * 100)}%`;
  }

  function finderJoytagQueryForScan(scan = state.finderScan) {
    const required = normalizeFinderTagList(
      scan?.joytagRequiredTags?.length
        ? scan.joytagRequiredTags
        : scan?.joytagTag ? [scan.joytagTag] : []
    );
    const requiredSet = new Set(required);
    const excluded = normalizeFinderTagList(scan?.joytagExcludedTags || [])
      .filter(tag => !requiredSet.has(tag));
    return { required, excluded };
  }

  function finderJoytagScanLabel(scan = state.finderScan) {
    const { required, excluded } = finderJoytagQueryForScan(scan);
    const primary = humanizeJoytagTag(required[0])
      || scan?.poseTagLabel
      || 'Tag scan';
    const requiredSuffix = required.length > 1 ? ` +${required.length - 1}` : '';
    const excludedSuffix = excluded.length ? ` · −${excluded.length}` : '';
    return `${primary}${requiredSuffix}${excludedSuffix}`;
  }

  function finderEvidenceLabel(item, { short = false } = {}) {
    if (finderScanUsesJoyTag() || item?.matchType === 'tag') {
      const score = finderScoreLabel(firstFinderScore(item?.tagScore, item?.score));
      const { required } = finderJoytagQueryForScan();
      if (required.length > 1) {
        return short
          ? `ALL ${required.length} · weakest ${score}`
          : `ALL ${required.length} required · weakest JoyTag confidence ${score}`;
      }
      const tag = humanizeJoytagTag(item?.tag || required[0] || state.finderScan?.joytagTag);
      return short
        ? `JoyTag confidence ${score}`
        : `${tag || 'Selected tag'} · JoyTag confidence ${score}`;
    }
    const tier = normalizeFinderTier(item?.rankingTier);
    const score = finderScoreLabel(item?.score);
    if (tier === 3) return `${short ? 'Exact' : 'Exact image'} ${score}`;
    if (tier === 2) return `Pose ${score}`;
    if (tier === 1) return `${short ? 'Visual' : 'Visual fallback'} ${score}`;
    return `${short ? 'Pose' : 'Pose mismatch'} ${score}`;
  }

  function finderEvidenceKind(item) {
    if (finderScanUsesJoyTag() || item?.matchType === 'tag') {
      const query = finderJoytagQueryForScan();
      return query.required.length > 1 || query.excluded.length
        ? 'JoyTag query'
        : 'JoyTag tag';
    }
    const tier = normalizeFinderTier(item?.rankingTier);
    if (tier === 3) return 'Exact image';
    if (tier === 2) return 'Pose match';
    if (tier === 1) return 'Visual fallback';
    return 'Pose mismatch';
  }

  function appendFinderMatchMedia(container, result) {
    const joytag = finderScanUsesJoyTag() || result.matchType === 'tag';
    const matches = result.matches?.length ? result.matches : [{
      rank: 1,
      imageUrl: result.bestImageUrl,
      previewUrl: result.bestPreviewUrl,
      ordinal: result.bestOrdinal,
      score: result.score,
      personCount: result.personCount,
      overlayUrl: ''
    }];
    container.classList.add(`has-${Math.min(3, Math.max(1, matches.length))}`);
    matches.slice(0, 3).forEach((match, matchIndex) => {
      const item = document.createElement('div');
      const selected = result.feedbackMatchKeys.includes(match.feedbackKey);
      item.className = `finder-match${selected ? ' is-feedback-selected' : ''}`;
      item.dataset.finderMatch = match.feedbackKey;
      const button = document.createElement('button');
      button.className = 'finder-match-open-target finder-card-open';
      button.type = 'button';
      button.dataset.finderAction = 'open';
      button.dataset.finderResult = String(result.key);
      const ordinalCopy = match.ordinal ? `image ${match.ordinal}` : `candidate ${matchIndex + 1}`;
      button.setAttribute('aria-label', `Open ${result.title}, ${ordinalCopy}`);
      button.innerHTML = '<span class="image-placeholder"><svg><use href="#i-image"></use></svg></span><img class="finder-match-image" alt="" loading="lazy" decoding="async"><img class="finder-skeleton-overlay" alt="" loading="lazy" decoding="async" hidden><span class="finder-match-position"></span><span class="finder-match-score" hidden></span><span class="finder-match-ordinal"></span><span class="finder-match-people" hidden></span><span class="finder-match-open"><svg><use href="#i-maximize"></use></svg></span>';
      loadImage($('.finder-match-image', button), match.previewUrl || match.imageUrl, `${result.title}, ${ordinalCopy}`);
      $('.finder-match-position', button).textContent = `#${matchIndex + 1}`;
      const matchScore = firstFinderScore(match.score, matchIndex === 0 ? result.score : null);
      const scoreBadge = $('.finder-match-score', button);
      scoreBadge.hidden = matchScore === null;
      if (matchScore !== null) {
        scoreBadge.textContent = finderEvidenceLabel(
          { ...match, score: matchScore },
          { short: true }
        );
      }
      $('.finder-match-ordinal', button).textContent = match.ordinal ? `Image ${String(match.ordinal).padStart(2, '0')}` : 'Candidate';
      const peopleBadge = $('.finder-match-people', button);
      peopleBadge.hidden = joytag || !match.personCount;
      if (!joytag && match.personCount) peopleBadge.textContent = `${match.personCount} ${match.personCount === 1 ? 'person' : 'people'}`;
      const overlayUrl = safeUrl(match.overlayUrl);
      if (overlayUrl) {
        const overlay = $('.finder-skeleton-overlay', button);
        overlay.hidden = false;
        if (overlayUrl.startsWith('data:')) overlay.src = overlayUrl;
        else loadImage(overlay, overlayUrl, '', { kind: 'normal' });
        overlay.addEventListener('error', () => {
          overlay.hidden = true;
          overlay.removeAttribute('src');
        }, { once: true });
      }
      const select = document.createElement('label');
      select.className = 'finder-match-select';
      select.title = joytag
        ? selected ? 'Selected as a matching image—click to exclude' : 'Excluded from this gallery review—click to include'
        : selected ? 'Selected for pose feedback—click to exclude' : 'Excluded from pose feedback—click to include';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = selected;
      checkbox.disabled = !match.imageUrl || result.feedbackSaving || state.finderFeedbackBusy;
      checkbox.dataset.finderFeedbackMatch = match.feedbackKey;
      checkbox.dataset.finderResult = String(result.key);
      checkbox.setAttribute(
        'aria-label',
        joytag
          ? match.imageUrl ? `Select ${ordinalCopy} as a matching image` : `${ordinalCopy} is unavailable`
          : match.imageUrl ? `Use ${ordinalCopy} as pose feedback` : `${ordinalCopy} is unavailable for pose feedback`
      );
      select.innerHTML = '<svg><use href="#i-check"></use></svg><span>Use</span>';
      select.prepend(checkbox);
      item.append(button, select);
      container.append(item);
    });
  }

  function renderFinderDiagnostics(card, result) {
    const breakdown = $('.finder-score-breakdown', card);
    if (finderScanUsesJoyTag() || result.matchType === 'tag') {
      const query = finderJoytagQueryForScan();
      const tagScores = normalizeFinderTagScores(result.tagScores);
      const required = query.required.length
        ? query.required
        : normalizeFinderTagList([result.tag || state.finderScan?.joytagTag]);
      required.forEach((tag, index) => {
        const score = firstFinderScore(
          tagScores[tag],
          index === 0 ? result.tagScore : null,
          index === 0 ? result.score : null
        );
        const badge = document.createElement('span');
        const label = humanizeJoytagTag(tag) || 'JoyTag';
        badge.className = 'finder-score-chip is-required';
        badge.title = score === null
          ? `${label} is required; this saved result has no per-tag diagnostic`
          : `${label} required confidence ${finderScoreLabel(score)}; must be at least ${finderScoreLabel(state.finderScan?.minSimilarity)}`;
        badge.innerHTML = '<i></i><span></span><b></b>';
        $('span', badge).textContent = required.length === 1
          ? `${label} · JoyTag`
          : `${label} · required`;
        $('b', badge).textContent = score === null ? '—' : finderScoreLabel(score);
        breakdown.append(badge);
      });
      query.excluded.forEach(tag => {
        const score = firstFinderScore(tagScores[tag]);
        const rejectThreshold = state.finderScan?.joytagRejectThreshold ?? 0.4;
        const badge = document.createElement('span');
        const label = humanizeJoytagTag(tag) || 'JoyTag';
        badge.className = 'finder-score-chip is-excluded';
        badge.title = score === null
          ? `${label} is excluded; this saved result has no per-tag diagnostic`
          : `${label} excluded confidence ${finderScoreLabel(score)}; the image is rejected at ${finderScoreLabel(rejectThreshold)} or higher`;
        badge.innerHTML = '<i></i><span></span><b></b>';
        $('span', badge).textContent = `${label} · excluded`;
        $('b', badge).textContent = score === null ? '—' : finderScoreLabel(score);
        breakdown.append(badge);
      });
      $('.finder-person-count', card).hidden = true;
      $('.finder-overlay-toggle', card).hidden = true;
      $('.finder-diagnostic-toolbar', card).hidden = !breakdown.children.length;
      return;
    }
    const scores = [
      ['exact', 'Exact', result.exactScore],
      ['pose', 'Pose', result.poseScore],
      ['appearance', 'Visual layout', result.appearanceScore]
    ];
    scores.forEach(([kind, label, score]) => {
      if (score === null || score === undefined) return;
      const badge = document.createElement('span');
      const displayLabel = kind === 'pose' && result.rankingTier === 0
        ? 'Pose mismatch'
        : kind === 'pose' && !result.poseReliable ? 'Pose uncertain' : label;
      badge.className = `finder-score-chip is-${kind}`;
      badge.title = `${displayLabel} score ${finderScoreLabel(score)}`;
      badge.innerHTML = '<i></i><span></span><b></b>';
      $('span', badge).textContent = displayLabel;
      $('b', badge).textContent = finderScoreLabel(score);
      breakdown.append(badge);
    });
    if (Math.abs(result.feedbackAdjustment) > 1e-9) {
      const adjustment = result.feedbackAdjustment;
      const badge = document.createElement('span');
      badge.className = `finder-score-chip is-feedback ${adjustment > 0 ? 'is-positive' : 'is-negative'}`;
      const revision = result.feedbackRevision ? ` revision ${result.feedbackRevision}` : '';
      const baseCopy = result.baseScore === null || result.baseScore === undefined
        ? ''
        : ` from ${finderScoreLabel(result.baseScore)} to ${finderScoreLabel(result.score)}`;
      badge.title = `Pose-specific feedback adjusted this result${baseCopy}. This scan uses feedback${revision} captured when it began; later reviews affect future scans.`;
      badge.innerHTML = '<i></i><span>Feedback</span><b></b>';
      $('b', badge).textContent = finderFeedbackAdjustmentLabel(adjustment);
      breakdown.append(badge);
    }
    const people = $('.finder-person-count', card);
    people.hidden = !result.personCount;
    if (result.personCount) $('b', people).textContent = `${result.personCount} ${result.personCount === 1 ? 'person' : 'people'}`;
    const overlay = $('.finder-overlay-toggle', card);
    overlay.hidden = !result.hasOverlay;
    $('.finder-diagnostic-toolbar', card).hidden = !breakdown.children.length && !result.personCount && !result.hasOverlay;
  }

  function renderFinderPagination() {
    const pagination = $('#finder-pagination');
    const total = Math.max(0, Number(state.finderResultTotal || 0));
    const pageSize = Math.max(1, Number(state.finderResultPageSize || FINDER_RESULTS_PAGE_SIZE));
    const pageCount = Math.max(1, Number(state.finderResultPageCount || Math.ceil(total / pageSize) || 1));
    const page = Math.max(1, Math.min(pageCount, Number(state.finderResultPage || 1)));
    const first = total ? (page - 1) * pageSize + 1 : 0;
    const last = total ? Math.min(total, first + state.finderResults.length - 1) : 0;
    pagination.hidden = !state.finderScan?.id || pageCount <= 1;
    $('#finder-page-status').textContent = `Page ${formatNumber(page)} of ${formatNumber(pageCount)}`;
    $('#finder-page-range').textContent = total
      ? `${formatNumber(first)}–${formatNumber(Math.max(first, last))} of ${formatNumber(total)} results`
      : '0 results';
    const previous = $('#finder-page-previous');
    const next = $('#finder-page-next');
    previous.disabled = state.finderResultLoading || page <= 1;
    next.disabled = state.finderResultLoading || page >= pageCount;
    previous.setAttribute('aria-label', `Previous Finder results page, page ${Math.max(1, page - 1)}`);
    next.setAttribute('aria-label', `Next Finder results page, page ${Math.min(pageCount, page + 1)}`);
  }

  function scrollToFinderResults() {
    const results = $('#finder-results');
    if (!results) return;
    const root = document.documentElement;
    const previousBehavior = root.style.scrollBehavior;
    root.style.scrollBehavior = 'auto';
    const top = results.getBoundingClientRect().top + window.scrollY - 84;
    window.scrollTo(0, Math.max(0, top));
    root.style.scrollBehavior = previousBehavior;
  }

  function renderFinderResults() {
    const ranked = [...state.finderResults];
    const joytagScan = finderScanUsesJoyTag();
    const resultThreshold = $('#finder-result-threshold');
    const resultThresholdMinimum = joytagScan ? 0.05 : 0.40;
    resultThreshold.min = resultThresholdMinimum.toFixed(2);
    if (Number(resultThreshold.value || 0) < resultThresholdMinimum) {
      resultThreshold.value = resultThresholdMinimum.toFixed(2);
    }
    const loadedCounts = {
      pending: ranked.filter(result => result.review === 'pending').length,
      accepted: ranked.filter(result => result.review === 'accepted').length,
      maybe: ranked.filter(result => result.review === 'maybe').length,
      rejected: ranked.filter(result => result.review === 'rejected').length
    };
    const counts = state.finderReviewCounts || loadedCounts;
    $('#finder-pending-count').textContent = formatNumber(counts.pending);
    $('#finder-accepted-count').textContent = formatNumber(counts.accepted);
    $('#finder-maybe-count').textContent = formatNumber(counts.maybe);
    $('#finder-rejected-count').textContent = formatNumber(counts.rejected);
    $$('[data-finder-review]').forEach(button => {
      const active = button.dataset.finderReview === state.finderReview;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', String(active));
    });
    const threshold = Number(resultThreshold.value || resultThresholdMinimum);
    const thresholdControl = resultThreshold.closest('.finder-threshold');
    const thresholdCopy = $('span', thresholdControl);
    if (thresholdCopy?.firstChild) thresholdCopy.firstChild.textContent = joytagScan ? 'Weakest required ≥ ' : 'Score ≥ ';
    thresholdControl.title = joytagScan
      ? 'Filter galleries by the weakest required JoyTag confidence'
      : 'Applied within each evidence tier';
    $('#finder-filter-output').textContent = threshold.toFixed(2);
    const results = ranked.filter(result => result.review === state.finderReview && result.score >= threshold);
    const grid = $('#finder-result-grid');
    grid.replaceChildren();
    results.forEach(result => {
      const fragment = $('#finder-card-template').content.cloneNode(true);
      const card = $('.finder-card', fragment);
      card.dataset.finderResult = String(result.key);
      card.dataset.finderRankingTier = String(result.rankingTier);
      card.classList.toggle('is-feedback-saving', Boolean(result.feedbackSaving));
      card.setAttribute('aria-busy', String(Boolean(result.feedbackSaving)));
      card.classList.toggle('is-high', result.score >= 0.85);
      card.classList.toggle('is-likely', result.score >= 0.70 && result.score < 0.85);
      card.classList.toggle('is-explore', result.score < 0.70);
      card.classList.toggle('is-accepted', result.review === 'accepted');
      card.classList.toggle('is-maybe', result.review === 'maybe');
      card.classList.toggle('is-rejected', result.review === 'rejected');
      card.classList.toggle('is-indexed', result.indexedOnly);
      appendFinderMatchMedia($('.finder-match-gallery', card), result);
      const resultRank = (state.finderResultPage - 1) * state.finderResultPageSize + ranked.indexOf(result) + 1;
      $('.finder-rank', card).textContent = `#${String(resultRank).padStart(2, '0')}`;
      $('.finder-similarity', card).textContent = finderEvidenceLabel(result);
      $('.finder-indexed-badge', card).hidden = !result.indexedOnly;
      const matchKind = $('.finder-match-kind', card);
      const kindCopy = finderEvidenceKind(result);
      matchKind.hidden = !kindCopy;
      matchKind.textContent = kindCopy;
      matchKind.title = joytagScan
        ? finderJoytagQueryForScan().required.length > 1
          ? 'Every required JoyTag signal passed; the card score is the weakest required confidence.'
          : `${humanizeJoytagTag(result.tag || state.finderScan?.joytagTag) || 'Selected tag'} confidence from JoyTag`
        : result.rankingTier === 1
          ? 'RTMO could not confirm matching person counts and enough body-and-limb evidence, so this candidate is ranked by visual layout below high-precision pose matches.'
          : result.rankingTier === 0
            ? 'RTMO found high-precision body evidence, but its geometry did not reach the pose-match floor.'
            : '';
      $('.finder-card-title', card).textContent = result.title;
      const matchCopy = `${formatNumber(result.matchCount)} ${result.matchCount === 1 ? 'image' : 'images'} compared`;
      $('.finder-card-meta', card).textContent = `${matchCopy}${result.imageCount ? ` · ${formatNumber(result.imageCount)} total` : ''}`;
      const selectedFeedback = result.matches.filter(match => result.feedbackMatchKeys.includes(match.feedbackKey)).length;
      const feedbackCopy = $('.finder-feedback-selection-copy', card);
      const savedFeedbackCount = result.feedbackImageUrls.length;
      const usableFeedbackCount = result.feedbackUsableImageUrls.length;
      const pendingFeedbackCount = result.feedbackPendingImageUrls.length;
      const reviewedWithFeedback = ['accepted', 'rejected'].includes(result.review);
      if (joytagScan) {
        feedbackCopy.textContent = result.feedbackSaving
          ? 'Saving gallery review and selected matching images…'
          : result.review === 'maybe'
            ? selectedFeedback
              ? `Maybe is neutral · ${selectedFeedback} matching ${selectedFeedback === 1 ? 'image' : 'images'} kept for a future decision`
              : 'Maybe is neutral · select images before changing to Accept'
            : reviewedWithFeedback
              ? result.feedbackSelectionDirty
                ? `${savedFeedbackCount} selected · unsaved changes—use Save selection`
                : savedFeedbackCount
                  ? `${savedFeedbackCount} matching ${savedFeedbackCount === 1 ? 'image' : 'images'} saved with this ${result.review} gallery · edit checks or open the gallery`
                  : `Gallery ${result.review} · no matching images saved`
              : selectedFeedback
                ? `${selectedFeedback} of ${result.matches.length} JoyTag ${result.matches.length === 1 ? 'suggestion' : 'suggestions'} selected · uncheck false positives`
                : 'No matching images selected · Accept requires at least one';
      } else {
        feedbackCopy.textContent = result.feedbackSaving
          ? 'Saving gallery review and pose feedback…'
          : state.finderFeedbackBusy
            ? 'Resetting pose feedback…'
            : result.review === 'maybe'
              ? selectedFeedback
                ? `Maybe is neutral—no feedback saved · ${selectedFeedback} checked for a future decision`
                : 'Maybe is neutral—no pose feedback · check images before changing to Accept'
              : reviewedWithFeedback
                ? result.feedbackSelectionDirty
                  ? `${savedFeedbackCount} selected · unsaved changes—use Save selection`
                  : pendingFeedbackCount
                    ? `${savedFeedbackCount} selected and saved · ${usableFeedbackCount} ranking-eligible · ${pendingFeedbackCount} pose-pending`
                    : savedFeedbackCount
                      ? `${savedFeedbackCount} ${result.review} feedback ${savedFeedbackCount === 1 ? 'image' : 'images'} ${result.feedbackAnalysisProvided ? 'saved and ranking-eligible' : 'saved'} · edit checks or open the gallery`
                      : `Gallery ${result.review} · no image-level pose feedback saved`
                : selectedFeedback
                  ? `${selectedFeedback} of ${result.matches.length} suggested ${result.matches.length === 1 ? 'image' : 'images'} checked for pose feedback · uncheck wrong images`
                  : 'No suggested images checked · Accept requires at least one';
      }
      renderFinderDiagnostics(card, result);
      $$('.finder-card-open', card).forEach(button => {
        button.dataset.finderAction = 'open';
        button.dataset.finderResult = String(result.key);
      });
      $$('[data-finder-action]', card).forEach(button => { button.dataset.finderResult = String(result.key); });
      const accept = $('.finder-accept', card);
      const maybe = $('.finder-maybe', card);
      const reject = $('.finder-reject', card);
      const saveSelection = $('.finder-save-selection', card);
      accept.classList.toggle('is-active', result.review === 'accepted');
      maybe.classList.toggle('is-active', result.review === 'maybe');
      reject.classList.toggle('is-active', result.review === 'rejected');
      const reviewLocked = state.finderBusy || state.finderFeedbackBusy || result.feedbackSaving;
      accept.disabled = reviewLocked || !savedFeedbackCount || result.review === 'accepted';
      maybe.disabled = reviewLocked || result.review === 'maybe';
      reject.disabled = reviewLocked || result.review === 'rejected';
      saveSelection.hidden = !reviewedWithFeedback;
      saveSelection.disabled = reviewLocked
        || !result.feedbackSelectionDirty
        || (result.review === 'accepted' && !savedFeedbackCount);
      accept.title = state.finderFeedbackBusy
        ? 'Wait for pose feedback reset to finish'
        : result.feedbackSaving
        ? 'Saving this gallery review'
        : !savedFeedbackCount ? 'Check at least one matching image before accepting' : '';
      if (joytagScan && state.finderFeedbackBusy) accept.title = 'Wait for the current Finder update to finish';
      reject.title = state.finderFeedbackBusy
        ? joytagScan ? 'Wait for the current Finder update to finish' : 'Wait for pose feedback reset to finish'
        : result.feedbackSaving ? 'Saving this gallery review' : '';
      maybe.title = state.finderFeedbackBusy
        ? joytagScan ? 'Wait for the current Finder update to finish' : 'Wait for pose feedback reset to finish'
        : joytagScan
          ? result.review === 'maybe' ? 'Maybe keeps this gallery neutral' : 'Keep this gallery for later review'
          : result.review === 'maybe' ? 'Maybe is neutral and creates no pose feedback' : 'Keep this gallery without using it as pose feedback';
      saveSelection.title = result.review === 'accepted' && !savedFeedbackCount
        ? 'Accepted feedback needs at least one selected image'
        : result.feedbackSelectionDirty ? 'Save the edited feedback image selection' : 'Selection is already saved';
      grid.append(fragment);
    });
    const empty = $('#finder-empty');
    empty.hidden = Boolean(results.length) || state.finderResultLoading;
    if (!results.length) {
      $('h3', empty).textContent = state.finderResultTotal ? 'No candidates on this page' : finderScanIsRunning() ? 'Scanning for candidates…' : 'No candidates found';
      $('p', empty).textContent = state.finderResultTotal
        ? 'Lower the display threshold or choose another review tab.'
        : finderScanIsRunning()
          ? 'Results will appear here as galleries are compared.'
          : joytagScan
            ? 'Try a lower JoyTag confidence threshold or a wider source.'
            : 'Try more examples, a lower minimum match score, or a wider source.';
    }
    renderFinderPagination();
  }

  function updateFinderExtendSummary({ commit = false } = {}) {
    const input = $('#finder-extend-pages');
    const parsed = Number.parseInt(input.value, 10);
    const currentPages = Math.max(0, Number(state.finderScan?.pages || 0), Number(state.finderScan?.pagesScanned || 0));
    const maximumAdditional = Math.max(1, Math.min(50, FINDER_MAX_PAGES - currentPages));
    const additionalPages = Math.max(1, Math.min(maximumAdditional, Number.isFinite(parsed) ? parsed : state.finderExtendPages || 5));
    state.finderExtendPages = additionalPages;
    input.max = String(maximumAdditional);
    if (commit) input.value = String(additionalPages);
    const resultingPages = currentPages + additionalPages;
    const capCopy = maximumAdditional < 50 ? ` · ${formatNumber(FINDER_MAX_PAGES)} maximum` : '';
    $('#finder-extend-summary').textContent = `${formatNumber(currentPages)} ${currentPages === 1 ? 'page' : 'pages'} → ${formatNumber(resultingPages)} pages${capCopy}`;
    return additionalPages;
  }

  function updateFinderContinueSummary({ commit = false } = {}) {
    const input = $('#finder-continue-pages');
    const parsed = Number.parseInt(input.value, 10);
    const completedPages = Math.max(0, Number(state.finderScan?.pagesScanned || 0));
    const maximumAdditional = Math.max(1, Math.min(50, FINDER_MAX_PAGES - completedPages));
    const additionalPages = Math.max(1, Math.min(
      maximumAdditional,
      Number.isFinite(parsed) ? parsed : state.finderContinuePages || 5
    ));
    state.finderContinuePages = additionalPages;
    input.max = String(maximumAdditional);
    if (commit) input.value = String(additionalPages);
    const candidateCount = Math.max(
      Number(state.finderReviewCounts?.total || 0),
      Number(state.finderScan?.candidateCount || 0)
    );
    const candidateCopy = `${formatNumber(candidateCount)} ${candidateCount === 1 ? 'candidate' : 'candidates'}`;
    const capCopy = maximumAdditional < 50 ? ` · ${formatNumber(FINDER_MAX_PAGES)} maximum` : '';
    $('#finder-continue-summary').textContent = `Up to ${formatNumber(additionalPages)} new ${additionalPages === 1 ? 'page' : 'pages'} · keep ${candidateCopy} and all reviews${capCopy}`;
    return additionalPages;
  }

  function renderFinderWorkspace() {
    renderFinderFolders();
    renderFinderTags();
    renderFinderMode();
    renderFinderReferenceAnalysis();
    renderFinderStatus();
    renderFinderFeedback();
    renderFinderCorpus();
    renderFinderScans();
    const scan = state.finderScan;
    const hasScan = Boolean(scan?.id);
    const legacyRanking = hasScan && !scan.rankingCurrent;
    const hasCorpusProgress = hasScan && Boolean(scan.corpusProgressAvailable);
    $('#finder-welcome').hidden = hasScan;
    $('#finder-results').hidden = !hasScan;
    $('#finder-progress-wrap').hidden = !hasScan;
    $('#finder-local-progress').hidden = !hasCorpusProgress;
    $('#finder-pause').hidden = !finderScanIsRunning(scan);
    const finderRetry = scan?.status === 'failed' && !legacyRanking;
    const finderResume = scan?.status === 'paused' && !legacyRanking;
    $('#finder-resume').hidden = !finderRetry && !finderResume;
    $('span', $('#finder-resume')).textContent = finderRetry ? 'Retry search' : 'Resume';
    $('use', $('#finder-resume')).setAttribute('href', finderRetry ? '#i-refresh' : '#i-play');
    $('#finder-resume').title = finderRetry
      ? 'Retry this scan from its saved source position'
      : 'Resume this scan from its saved progress';
    $('#finder-cancel').hidden = !hasScan || finderScanIsTerminal(scan) || scan?.status === 'canceling';
    $('#finder-ranking-note').hidden = !legacyRanking;
    ['finder-pause', 'finder-resume', 'finder-cancel'].forEach(id => { $(`#${id}`).disabled = state.finderBusy; });
    const canExtend = finderScanCanExtend(scan);
    const canChangeSource = finderScanCanChangeSource(scan);
    const switchingSource = finderScanCanSwitchSource(scan);
    const atPageCap = finderScanAtPageCap(scan);
    const joytagScan = finderScanUsesJoyTag(scan);
    const finderReady = joytagScan
      ? state.finderStatus?.joytagAvailable !== false
      : Boolean(state.finderStatus?.ready);
    const finderMutationPending = state.finderFeedbackBusy || finderFeedbackIsSaving();
    $('#finder-extend').hidden = !canExtend;
    $('#finder-continue').hidden = !canChangeSource;
    $('#finder-limit-note').hidden = !atPageCap;
    $('#finder-extend').classList.toggle('is-unavailable', canExtend && !finderReady);
    $('#finder-continue').classList.toggle('is-unavailable', canChangeSource && !finderReady);
    $('#finder-extend-pages').disabled = !canExtend || !finderReady || state.finderBusy;
    $('#finder-extend-button').disabled = !canExtend || !finderReady || state.finderBusy;
    $('#finder-continue-source').disabled = !canChangeSource || !finderReady || state.finderBusy || finderMutationPending;
    $('#finder-continue-pages').disabled = !canChangeSource || !finderReady || state.finderBusy || finderMutationPending;
    $('#finder-continue-button').disabled = !canChangeSource || !finderReady || state.finderBusy || finderMutationPending;
    $('#finder-continue-title').textContent = switchingSource ? 'Switch source' : 'Explore another source';
    const continueButtonLabel = $('#finder-continue-button-label');
    if (continueButtonLabel) {
      continueButtonLabel.textContent = switchingSource ? 'Switch source' : 'Continue same scan';
    }
    const unavailableTitle = finderReady ? '' : state.finderStatus?.detail || 'Finder is unavailable';
    $('#finder-extend-button').title = unavailableTitle;
    $('#finder-continue-button').title = unavailableTitle;
    if (!hasScan) {
      $('#finder-session-label').textContent = 'Configure a scan to begin';
      $('#finder-source-progress-copy').textContent = 'Exploring the selected Source URL for new galleries';
      syncFinderConfigAvailability();
      return;
    }
    if (hasCorpusProgress) {
      const corpusComplete = scan.corpusSearchComplete === true;
      const corpusHadRows = scan.corpusImagesScored > 0 || scan.corpusGalleriesScored > 0;
      const corpusStopped = !corpusComplete && finderScanIsTerminal(scan);
      const corpusPaused = !corpusComplete && scan.status === 'paused';
      $('#finder-local-progress').classList.toggle('is-complete', corpusComplete);
      $('#finder-local-galleries').textContent = formatNumber(scan.corpusGalleriesScored);
      $('#finder-local-images').textContent = formatNumber(scan.corpusImagesScored);
      $('#finder-local-progress-state').textContent = corpusComplete
        ? corpusHadRows ? 'Done' : 'No data'
        : corpusStopped ? 'Incomplete' : corpusPaused ? 'Paused' : 'Searching';
      $('#finder-local-progress-copy').textContent = joytagScan
        ? corpusComplete
          ? corpusHadRows
            ? 'Cached local JoyTag images searched before source exploration'
            : 'No cached JoyTag images; uncached local images were skipped'
          : corpusStopped
            ? 'Cached local JoyTag search stopped before completion'
            : corpusPaused
              ? 'Cached local JoyTag search is paused'
              : 'Searching cached local JoyTag images; uncached images are skipped'
        : corpusComplete
          ? corpusHadRows
            ? 'All saved galleries searched before live exploration'
            : 'No reusable indexed images were available for this scan'
          : corpusStopped
            ? 'Local index search stopped before completion'
            : corpusPaused
              ? 'Local index search is paused'
              : 'Searching every indexed gallery—not only this source';
    }
    if (canExtend) {
      updateFinderExtendSummary({ commit: true });
      if (!finderReady) $('#finder-extend-summary').textContent += ' · Finder unavailable';
    }
    if (canChangeSource) {
      updateFinderContinueSummary({ commit: true });
      if (!finderReady) $('#finder-continue-summary').textContent += ' · Finder unavailable';
    }
    const status = scan.status.replaceAll('_', ' ');
    const sourceExhausted = finderScanSourceExhausted(scan);
    const waitingForCorpus = hasCorpusProgress
      && scan.corpusSearchComplete !== true
      && !finderScanIsTerminal(scan)
      && scan.pagesScanned === 0;
    $('#finder-progress-wrap').classList.toggle('is-source-exhausted', sourceExhausted);
    $('#finder-source-progress-copy').textContent = waitingForCorpus
      ? joytagScan
        ? 'Starts immediately after the cached local JoyTag search'
        : 'Starts after the Local Gallery Index search'
      : sourceExhausted
        ? 'The selected Source URL has no more pages'
        : `Exploring ${displayHost(scan.sourceUrl)} for new galleries`;
    const sessionName = finderScanUsesJoyTag(scan)
      ? finderJoytagScanLabel(scan)
      : scan.poseTagLabel || 'Pose scan';
    $('#finder-session-label').textContent = `${sessionName} · ${finderScanUsesJoyTag(scan) ? 'JoyTag' : 'Pose'} · ${status}`;
    $('#finder-pages-scanned').textContent = formatNumber(scan.pagesScanned);
    $('#finder-pages-total').textContent = formatNumber(scan.pages || 0);
    $('#finder-pages-budget').hidden = sourceExhausted;
    $('#finder-pages-exhausted').hidden = !sourceExhausted;
    $('#finder-pages-exhausted-count').textContent = formatNumber(scan.pagesScanned);
    $('#finder-pages-exhausted-unit').textContent = scan.pagesScanned === 1 ? 'page' : 'pages';
    $('#finder-galleries-scanned').textContent = scan.totalGalleries
      ? `${formatNumber(scan.galleriesScanned)} / ${formatNumber(scan.totalGalleries)}`
      : formatNumber(scan.galleriesScanned);
    $('#finder-images-scanned').textContent = formatNumber(scan.imagesScanned);
    const visibleCandidates = state.finderResults.filter(result => result.score >= scan.minSimilarity).length;
    $('#finder-candidates-found').textContent = formatNumber(Math.max(scan.candidateCount, visibleCandidates));
    const progressState = waitingForCorpus
      ? 'Waiting'
      : sourceExhausted ? 'Source exhausted' : status[0]?.toUpperCase() + status.slice(1);
    $('#finder-progress-state').textContent = `${progressState}${scan.failedGalleries ? ` · ${formatNumber(scan.failedGalleries)} failed` : ''}`;
    $('#finder-progress-bar').style.width = `${scan.percentage}%`;
    $('.finder-progress').setAttribute('aria-valuenow', String(Math.round(scan.percentage)));
    const progressValueText = sourceExhausted
      ? `Source exhausted after ${scan.pagesScanned} ${scan.pagesScanned === 1 ? 'page' : 'pages'}`
      : `${scan.pagesScanned} of ${scan.pages || 0} pages`;
    $('.finder-progress').setAttribute('aria-valuetext', progressValueText);
    $('#finder-scan-error').hidden = !scan.error;
    $('#finder-scan-error').textContent = scan.error;
    renderFinderResults();
    syncFinderConfigAvailability();
  }

  function readFinderConfig({ validate = false } = {}) {
    const mode = state.finderMode;
    const joytag = mode === 'joytag';
    const exampleDirectory = $('#finder-folder').value.trim();
    const tagLabel = $('#finder-pose-tag').value.trim().replace(/\s+/g, ' ');
    const sourceInput = $('#finder-source').value.trim();
    const sourceUrl = /^https?:\/\//i.test(sourceInput) ? safeUrl(sourceInput) : '';
    const requestedPages = Number.parseInt($('#finder-pages').value || '5', 10);
    const pageLimit = Math.max(1, Math.min(50, Number.isFinite(requestedPages) ? requestedPages : 5));
    const minimumScore = joytag
      ? Math.max(0.05, Math.min(0.95, Number(state.finderJoytagThreshold || 0.4)))
      : Math.max(0.4, Math.min(0.95, Number($('#finder-min-similarity').value || 0.68)));
    const joytagRequiredTags = joytag
      ? normalizeFinderTagList(state.finderJoytagRequiredTags)
      : [];
    const requiredSet = new Set(joytagRequiredTags);
    const joytagExcludedTags = joytag
      ? normalizeFinderTagList(state.finderJoytagExcludedTags)
        .filter(tag => !requiredSet.has(tag))
      : [];
    const joytagRejectThreshold = joytag
      ? Math.max(0.05, Math.min(0.95, Number(state.finderJoytagRejectThreshold || 0.4)))
      : 0.4;
    state.finderJoytagRequiredTags = joytagRequiredTags;
    state.finderJoytagExcludedTags = joytagExcludedTags;
    state.finderJoytagRejectThreshold = joytagRejectThreshold;
    if (validate && !exampleDirectory) {
      const root = state.finderStatus?.folderRoot || 'the library root';
      toast('Enter an examples folder', `Use any folder inside ${root}, as a relative path or full container path.`, 'info');
      $('#finder-folder').focus();
      return null;
    }
    if (validate && joytag && !finderReferenceAnalysisIsCurrent()) {
      toast('Analyze this reference folder', 'Tag search needs a fresh JoyTag analysis of the folder currently shown.', 'info');
      $('#finder-analyze-references').focus();
      return null;
    }
    if (validate && joytag && !joytagRequiredTags.length) {
      toast('Choose a required JoyTag signal', 'Tag search needs at least one tag that every matching image must contain.', 'info');
      $('#finder-joytag-tags').scrollIntoView({ block: 'nearest' });
      return null;
    }
    if (validate && joytag) {
      const catalog = new Set(state.finderReferenceAnalysis?.tagCatalog || []);
      const unavailable = [...joytagRequiredTags, ...joytagExcludedTags]
        .filter(tag => !catalog.has(tag));
      if (unavailable.length) {
        toast(
          'Analyze the tag query again',
          `${humanizeJoytagTag(unavailable[0])} is not available in the current JoyTag catalog.`,
          'info'
        );
        $('#finder-joytag-tags').scrollIntoView({ block: 'nearest' });
        return null;
      }
    }
    if (validate && !tagLabel) {
      toast(
        joytag ? 'Name the dataset' : 'Name the pose',
        joytag ? 'Enter the label used for saved reviews and control/target preparation.' : 'Choose an existing pose tag or enter a new one.',
        'info'
      );
      (joytag ? $('#finder-joytag-dataset-label') : $('#finder-pose-tag')).focus();
      return null;
    }
    if (validate && !sourceUrl) {
      toast('Enter a source URL', 'Use a complete http or https gallery, category, model, search, or home URL.', 'info');
      $('#finder-source').focus();
      return null;
    }
    const modelReady = joytag
      ? state.finderStatus?.joytagAvailable !== false
      : Boolean(state.finderStatus?.ready);
    if (validate && !modelReady) {
      toast(
        joytag ? 'JoyTag is not ready' : 'Finder model is not ready',
        joytag
          ? state.finderStatus?.joytagError || 'Refresh after the tagger becomes available.'
          : state.finderStatus?.detail || 'Refresh after the model becomes available.',
        'info'
      );
      return null;
    }
    $('#finder-pages').value = String(pageLimit);
    return {
      mode,
      exampleDirectory,
      tagLabel,
      sourceUrl,
      pageLimit,
      minimumScore,
      joytagTag: joytag ? joytagRequiredTags[0] || null : null,
      joytagRequiredTags,
      joytagExcludedTags,
      joytagRejectThreshold,
      referenceFingerprint: joytag ? state.finderReferenceAnalysis?.fingerprint || null : null
    };
  }

  function readFinderContinueConfig({ validate = false } = {}) {
    const sourceInput = $('#finder-continue-source').value.trim();
    const sourceUrl = /^https?:\/\//i.test(sourceInput) ? safeUrl(sourceInput) : '';
    const additionalPages = updateFinderContinueSummary({ commit: true });
    if (validate && !sourceUrl) {
      toast('Enter another source URL', 'Use a complete http or https gallery, category, model, search, or home URL.', 'info');
      $('#finder-continue-source').focus();
      return null;
    }
    const joytag = finderScanUsesJoyTag();
    const modelReady = joytag
      ? state.finderStatus?.joytagAvailable !== false
      : Boolean(state.finderStatus?.ready);
    if (validate && !modelReady) {
      toast(
        joytag ? 'JoyTag is not ready' : 'Finder model is not ready',
        (joytag ? state.finderStatus?.joytagError : state.finderStatus?.detail)
          || 'Refresh after the model becomes available.',
        'info'
      );
      return null;
    }
    return { sourceUrl, additionalPages };
  }

  async function ensureFinderPoseTag(label) {
    const existing = finderTagForInput(label);
    if (existing) return existing;
    const data = await api('/api/pose-tags', { method: 'POST', body: { label, default_role: 'solo' } });
    const tag = normalizePoseTag(data?.tag || data);
    if (tag.id === undefined || !tag.label) throw new ApiError('The server did not return the new pose tag.');
    state.finderTags.push(tag);
    renderFinderTags();
    toast('Pose created', `${tag.label} defaults to the ${poseRoleLabel(tag.defaultRole).toLowerCase()} control.`, 'success');
    return tag;
  }

  function scheduleFinderPoll(delay = null) {
    window.clearTimeout(state.finderPollTimer);
    state.finderPollTimer = null;
    if (!state.finderScanId || finderScanIsTerminal()) return;
    const fallback = state.finderScan?.status === 'paused' ? 12000 : state.eventConnected ? 5000 : 1800;
    state.finderPollTimer = window.setTimeout(() => loadFinderScan({ quiet: true }), delay ?? fallback);
  }

  function scheduleFreshFinderResults(delay = 90) {
    window.clearTimeout(state.finderResultRefreshTimer);
    state.finderResultRefreshTimer = window.setTimeout(() => {
      state.finderResultRefreshTimer = null;
      if (!state.finderScanId) return;
      if (state.finderFeedbackBusy || finderFeedbackIsSaving()) {
        scheduleFreshFinderResults(120);
        return;
      }
      loadFinderResults({ quiet: true });
    }, delay);
  }

  function resetFinderResultPagination() {
    window.clearTimeout(state.finderResultThresholdTimer);
    state.finderResultThresholdTimer = null;
    state.finderResultRequest += 1;
    state.finderResultPage = 1;
    state.finderResultPageCount = 1;
    state.finderResultPageSize = FINDER_RESULTS_PAGE_SIZE;
    state.finderResultTotal = 0;
    state.finderResultLoading = false;
  }

  function scheduleFinderResultFilterLoad(delay = 180) {
    window.clearTimeout(state.finderResultThresholdTimer);
    state.finderResultThresholdTimer = window.setTimeout(() => {
      state.finderResultThresholdTimer = null;
      loadFinderResults({ quiet: true, page: 1 });
    }, delay);
  }

  async function loadFinderResults({ quiet = false, page = state.finderResultPage } = {}) {
    const scanId = state.finderScanId;
    const review = state.finderReview;
    const threshold = Math.max(0, Math.min(1, Number($('#finder-result-threshold').value || 0)));
    const pageSize = FINDER_RESULTS_PAGE_SIZE;
    const requestedPage = Math.max(1, Number.parseInt(page, 10) || 1);
    const request = ++state.finderResultRequest;
    const mutationEpoch = state.finderResultMutationEpoch;
    const mutationInFlight = state.finderFeedbackBusy || finderFeedbackIsSaving();
    if (!scanId) {
      state.finderResults = [];
      state.finderReviewCounts = null;
      resetFinderResultPagination();
      renderFinderWorkspace();
      return;
    }
    state.finderResultLoading = true;
    $('#finder-result-grid').setAttribute('aria-busy', 'true');
    renderFinderPagination();
    try {
      const data = await api(withParams(`/api/finder/scans/${encodeURIComponent(scanId)}/results`, {
        review,
        min_score: threshold,
        limit: pageSize,
        offset: (requestedPage - 1) * pageSize
      }));
      if (
        request !== state.finderResultRequest
        || String(scanId) !== String(state.finderScanId)
        || review !== state.finderReview
        || Math.abs(threshold - Number($('#finder-result-threshold').value || 0)) > 1e-9
      ) return;
      if (
        mutationInFlight
        || mutationEpoch !== state.finderResultMutationEpoch
        || state.finderFeedbackBusy
        || finderFeedbackIsSaving()
      ) {
        scheduleFreshFinderResults();
        return;
      }
      const total = Math.max(0, Number(data?.total ?? apiItems(data, 'results').length) || 0);
      const pageCount = Math.max(1, Number(data?.page_count || Math.ceil(total / pageSize) || 1));
      if (requestedPage > pageCount) {
        return loadFinderResults({ quiet, page: pageCount });
      }
      state.finderReviewCounts = normalizeFinderReviewCounts(data);
      const previousResults = new Map(state.finderResults.map(result => [String(result.key), result]));
      state.finderResults = apiItems(data, 'results').map(
        (result, index) => normalizeFinderResult(result, (requestedPage - 1) * pageSize + index)
      ).map(result => {
        const previous = previousResults.get(String(result.key));
        if (!previous?.feedbackSelectionDirty && !previous?.feedbackSaving) return result;
        return {
          ...result,
          review: previous.feedbackSaving ? previous.review : result.review,
          feedbackMatchKeys: [...previous.feedbackMatchKeys],
          feedbackImageUrls: [...previous.feedbackImageUrls],
          feedbackUsableImageUrls: [...previous.feedbackUsableImageUrls],
          feedbackPendingImageUrls: [...previous.feedbackPendingImageUrls],
          feedbackAnalysisProvided: previous.feedbackAnalysisProvided,
          feedbackSelectionProvided: previous.feedbackSelectionProvided,
          feedbackSelectionDirty: previous.feedbackSelectionDirty,
          feedbackSaving: previous.feedbackSaving
        };
      });
      state.finderResultPage = requestedPage;
      state.finderResultPageCount = pageCount;
      state.finderResultPageSize = pageSize;
      state.finderResultTotal = total;
      state.finderResultLoading = false;
      renderFinderWorkspace();
    } catch (error) {
      if (!quiet) toast('Could not load Finder results', errorMessage(error), 'error');
    } finally {
      if (request === state.finderResultRequest) {
        state.finderResultLoading = false;
        $('#finder-result-grid').setAttribute('aria-busy', 'false');
        renderFinderPagination();
      }
    }
  }

  async function loadFinderFeedback({ quiet = false, force = false } = {}) {
    if (finderConfigUsesJoyTag()) {
      state.finderFeedbackRequest += 1;
      state.finderFeedback = null;
      state.finderFeedbackLoading = false;
      state.finderFeedbackError = '';
      renderFinderFeedback();
      return;
    }
    const tag = finderFeedbackTag();
    window.clearTimeout(state.finderFeedbackTimer);
    state.finderFeedbackTimer = null;
    if (tag?.id === undefined || tag?.id === null) {
      state.finderFeedbackRequest += 1;
      state.finderFeedback = null;
      state.finderFeedbackLoading = false;
      state.finderFeedbackError = '';
      renderFinderFeedback();
      return;
    }
    if (
      !force
      && state.finderFeedback
      && String(state.finderFeedback.poseTagId) === String(tag.id)
    ) {
      renderFinderFeedback();
      return;
    }
    const request = ++state.finderFeedbackRequest;
    state.finderFeedbackLoading = true;
    state.finderFeedbackError = '';
    renderFinderFeedback();
    try {
      const data = await api(`/api/finder/feedback/${encodeURIComponent(tag.id)}`);
      if (request !== state.finderFeedbackRequest) return;
      const feedback = normalizeFinderFeedback(data, tag);
      if (!feedback) throw new ApiError('The server returned invalid pose-feedback statistics.');
      if (
        String(state.finderFeedback?.poseTagId) === String(feedback.poseTagId)
        && state.finderFeedback.revision > feedback.revision
      ) return;
      state.finderFeedback = feedback;
      state.finderFeedbackSupported = true;
      state.finderFeedbackError = '';
    } catch (error) {
      if (request !== state.finderFeedbackRequest) return;
      state.finderFeedback = null;
      if (error.status === 404) state.finderFeedbackSupported = false;
      else state.finderFeedbackError = errorMessage(error);
      if (!quiet && error.status !== 404) toast('Could not load pose feedback', errorMessage(error), 'error');
    } finally {
      if (request === state.finderFeedbackRequest) {
        state.finderFeedbackLoading = false;
        renderFinderFeedback();
      }
    }
  }

  function scheduleFinderFeedbackLoad(delay = 220) {
    window.clearTimeout(state.finderFeedbackTimer);
    state.finderFeedbackTimer = window.setTimeout(() => loadFinderFeedback({ quiet: true, force: true }), delay);
    renderFinderFeedback();
  }

  function applyFinderFeedbackResponse(data) {
    const tag = finderFeedbackTag();
    const candidate = data?.feedback
      || data?.finder_feedback
      || data?.result?.feedback
      || data?.result?.finder_feedback;
    const feedback = normalizeFinderFeedback(candidate, tag);
    if (!feedback || tag?.id == null || String(feedback.poseTagId) !== String(tag.id)) return false;
    state.finderFeedbackRequest += 1;
    state.finderFeedbackLoading = false;
    if (
      String(state.finderFeedback?.poseTagId) === String(feedback.poseTagId)
      && state.finderFeedback.revision > feedback.revision
    ) {
      renderFinderFeedback();
      return true;
    }
    state.finderFeedback = feedback;
    state.finderFeedbackSupported = true;
    state.finderFeedbackError = '';
    renderFinderFeedback();
    return true;
  }

  async function resetFinderFeedback() {
    const tag = finderFeedbackTag();
    const feedback = state.finderFeedback;
    if (
      tag?.id == null
      || !feedback
      || String(feedback.poseTagId) !== String(tag.id)
      || finderFeedbackSaveBlocksReset(tag)
      || state.finderFeedbackBusy
    ) return;
    const total = Math.max(feedback.total, feedback.accepted + feedback.rejected);
    if (!total) return;
    const sampleCopy = `${formatNumber(total)} saved feedback ${total === 1 ? 'sample' : 'samples'}`;
    if (!window.confirm(`Reset ${sampleCopy} for “${tag.label}”? This clears only this pose’s ranking feedback; galleries and cached images are not deleted.`)) return;
    const button = $('#finder-feedback-reset');
    state.finderResultMutationEpoch += 1;
    state.finderFeedbackBusy = true;
    setButtonBusy(button, true, 'Resetting…');
    renderFinderFeedback();
    renderFinderResults();
    syncFinderConfigAvailability();
    try {
      const data = await api(`/api/finder/feedback/${encodeURIComponent(tag.id)}`, { method: 'DELETE' });
      if (!applyFinderFeedbackResponse(data)) {
        state.finderFeedback = {
          ...feedback,
          accepted: 0,
          rejected: 0,
          acceptedGalleries: 0,
          rejectedGalleries: 0,
          usableAcceptedGalleries: 0,
          usableRejectedGalleries: 0,
          usableAcceptedSamples: 0,
          usableRejectedSamples: 0,
          total: 0,
          active: false
        };
      }
      if (String(state.finderScan?.poseTagId) === String(tag.id)) {
        state.finderResults.forEach(result => {
          if (result.review === 'pending') return;
          result.feedbackMatchKeys = [];
          result.feedbackImageUrls = [];
          result.feedbackUsableImageUrls = [];
          result.feedbackPendingImageUrls = [];
          result.feedbackAnalysisProvided = true;
          result.feedbackSelectionProvided = true;
          result.feedbackSelectionDirty = false;
        });
        renderFinderResults();
      }
      toast('Pose feedback reset', `Future “${tag.label}” scans will use the original ranking until new reviews are saved.`, 'info');
      announce(`${tag.label} ranking feedback reset.`);
    } catch (error) {
      toast('Could not reset pose feedback', errorMessage(error), 'error');
    } finally {
      state.finderFeedbackBusy = false;
      state.finderResultMutationEpoch += 1;
      scheduleFreshFinderResults();
      setButtonBusy(button, false);
      renderFinderFeedback();
      renderFinderResults();
      syncFinderConfigAvailability();
    }
  }

  function applyFinderJoytagIndexPayload(payload) {
    if (!payload || typeof payload !== 'object') return false;
    const coverage = normalizeFinderJoytagCoverage(payload.coverage);
    const hasJob = Object.prototype.hasOwnProperty.call(payload, 'job');
    const job = hasJob ? normalizeFinderJoytagIndexJob(payload.job) : null;
    if (!coverage && !hasJob) return false;
    const previousJob = state.finderCorpus?.joytagIndexJob || null;
    const previousActive = finderJoytagIndexIsActive(previousJob);
    state.finderCorpus = {
      galleries: 0,
      images: 0,
      complete: 0,
      partial: 0,
      ready: 0,
      cacheEntries: 0,
      cacheBytes: 0,
      maxCacheEntries: 0,
      maxCacheBytes: 0,
      ...(state.finderCorpus || {}),
      joytag: coverage || state.finderCorpus?.joytag || null,
      joytagIndexJob: hasJob ? job : previousJob
    };
    state.finderCorpusSupported = true;
    state.finderJoytagIndexSupported = true;
    renderFinderCorpus();
    scheduleFinderJoytagIndexPoll();
    if (
      previousActive
      && !finderJoytagIndexIsActive(state.finderCorpus.joytagIndexJob)
    ) {
      loadFinderCorpus({ quiet: true, force: true });
    }
    return true;
  }

  function scheduleFinderJoytagIndexPoll(delay = 1200) {
    window.clearTimeout(state.finderJoytagIndexTimer);
    state.finderJoytagIndexTimer = null;
    if (!finderJoytagIndexIsActive()) return;
    state.finderJoytagIndexTimer = window.setTimeout(() => {
      state.finderJoytagIndexTimer = null;
      loadFinderJoytagIndex({ quiet: true, force: true });
    }, delay);
  }

  async function loadFinderJoytagIndex({ quiet = false, force = false } = {}) {
    if (!force && state.finderJoytagIndexSupported === false) return;
    const request = ++state.finderJoytagIndexRequest;
    try {
      const data = await api('/api/finder/corpus/joytag-index');
      if (request !== state.finderJoytagIndexRequest) return;
      if (!applyFinderJoytagIndexPayload(data)) {
        throw new ApiError('The server returned invalid JoyTag corpus index status.');
      }
    } catch (error) {
      if (request !== state.finderJoytagIndexRequest) return;
      if (error.status === 404) {
        state.finderJoytagIndexSupported = false;
        window.clearTimeout(state.finderJoytagIndexTimer);
        state.finderJoytagIndexTimer = null;
      }
      renderFinderCorpus();
      if (!quiet && error.status !== 404) {
        toast('Could not refresh JoyTag indexing', errorMessage(error), 'error');
      }
    }
  }

  async function startFinderJoytagIndex() {
    const coverage = state.finderCorpus?.joytag || null;
    if (
      !finderConfigUsesJoyTag()
      || state.finderJoytagIndexBusy
      || state.finderJoytagIndexSupported === false
      || finderJoytagIndexIsActive()
      || !coverage
      || coverage.missingImages === 0
      || state.finderStatus?.joytagAvailable === false
    ) return;
    const button = $('#finder-corpus-index-start');
    state.finderJoytagIndexBusy = true;
    state.finderJoytagIndexRequest += 1;
    setButtonBusy(button, true, 'Starting…');
    renderFinderJoytagIndex();
    try {
      const data = await api('/api/finder/corpus/joytag-index', {
        method: 'POST'
      });
      if (!applyFinderJoytagIndexPayload(data)) {
        throw new ApiError('The server did not return JoyTag indexing status.');
      }
      const job = state.finderCorpus?.joytagIndexJob;
      if (finderJoytagIndexIsActive(job)) {
        toast(
          'Local JoyTag indexing started',
          `${formatNumber(job.remainingImages)} uncached local images are queued. Ordinary Tag searches can still use the existing cache.`,
          'success'
        );
        announce('Local JoyTag corpus indexing started.');
      } else {
        toast(
          'Local JoyTag cache is complete',
          'Every indexed local image is already cached for this JoyTag model.',
          'success'
        );
        loadFinderCorpus({ quiet: true, force: true });
      }
    } catch (error) {
      toast('Could not start JoyTag indexing', errorMessage(error), 'error');
    } finally {
      state.finderJoytagIndexBusy = false;
      setButtonBusy(button, false);
      renderFinderCorpus();
      scheduleFinderJoytagIndexPoll();
    }
  }

  async function cancelFinderJoytagIndex() {
    if (state.finderJoytagIndexBusy || !finderJoytagIndexIsActive()) return;
    const button = $('#finder-corpus-index-cancel');
    state.finderJoytagIndexBusy = true;
    state.finderJoytagIndexRequest += 1;
    setButtonBusy(button, true, 'Canceling…');
    renderFinderJoytagIndex();
    try {
      const data = await api('/api/finder/corpus/joytag-index', {
        method: 'DELETE'
      });
      if (!applyFinderJoytagIndexPayload(data)) {
        throw new ApiError('The server did not return JoyTag cancellation status.');
      }
      const stillActive = finderJoytagIndexIsActive();
      toast(
        stillActive ? 'JoyTag indexing is stopping' : 'JoyTag indexing canceled',
        'Already cached vectors are kept; you can index the remaining images later.',
        'info'
      );
      announce(
        stillActive
          ? 'Local JoyTag corpus indexing cancellation requested.'
          : 'Local JoyTag corpus indexing canceled.'
      );
    } catch (error) {
      if (error.status === 409) {
        await loadFinderJoytagIndex({ quiet: true, force: true });
        toast('JoyTag indexing already stopped', 'The latest corpus status has been loaded.', 'info');
      } else {
        toast('Could not cancel JoyTag indexing', errorMessage(error), 'error');
      }
    } finally {
      state.finderJoytagIndexBusy = false;
      setButtonBusy(button, false);
      renderFinderCorpus();
      scheduleFinderJoytagIndexPoll();
    }
  }

  async function loadFinderCorpus({ quiet = false, force = false } = {}) {
    if (!force && state.finderCorpusSupported === false) return;
    try {
      const data = await api('/api/finder/corpus');
      const corpus = normalizeFinderCorpus(data);
      if (!corpus) throw new ApiError('The server returned invalid Local Gallery Index statistics.');
      state.finderCorpus = corpus;
      state.finderCorpusSupported = true;
      state.finderJoytagIndexSupported = Boolean(corpus.joytag);
      renderFinderCorpus();
      scheduleFinderJoytagIndexPoll();
    } catch (error) {
      if (error.status === 404) {
        state.finderCorpusSupported = false;
        state.finderJoytagIndexSupported = false;
        if (!state.finderStatus?.corpus) state.finderCorpus = null;
      }
      renderFinderCorpus();
      if (!quiet && error.status !== 404) toast('Could not refresh the Local Gallery Index', errorMessage(error), 'error');
    }
  }

  function invalidateFinderScanLoads() {
    state.finderScanMutationEpoch += 1;
    state.finderScanRequest += 1;
  }

  async function loadFinderScan({ quiet = false, applyConfig = false } = {}) {
    const scanId = state.finderScanId;
    const mutationEpoch = state.finderScanMutationEpoch;
    const request = ++state.finderScanRequest;
    if (!scanId) {
      state.finderScan = null;
      state.finderResults = [];
      state.finderReviewCounts = null;
      resetFinderResultPagination();
      renderFinderWorkspace();
      return;
    }
    try {
      const data = await api(`/api/finder/scans/${encodeURIComponent(scanId)}`);
      if (
        String(scanId) !== String(state.finderScanId)
        || mutationEpoch !== state.finderScanMutationEpoch
        || request !== state.finderScanRequest
      ) return;
      const scan = normalizeFinderScan(data?.scan || data);
      if (!scan?.id) throw new ApiError('The server returned an invalid Finder scan.');
      state.finderScan = scan;
      const existing = state.finderScans.findIndex(item => String(item.id) === String(scan.id));
      if (existing >= 0) state.finderScans[existing] = scan;
      else state.finderScans.unshift(scan);
      if (applyConfig) applyFinderScanConfig(scan);
      renderFinderWorkspace();
      await Promise.all([
        loadFinderResults({ quiet: true }),
        loadFinderCorpus({ quiet: true }),
        loadFinderFeedback({ quiet: true })
      ]);
      scheduleFinderPoll();
    } catch (error) {
      if (error.status === 404) {
        state.finderScan = null;
        state.finderScanId = '';
        state.finderResults = [];
        state.finderReviewCounts = null;
        resetFinderResultPagination();
        storage.set('finder-scan', '');
        renderFinderWorkspace();
      } else if (!quiet) toast('Could not load Finder scan', errorMessage(error), 'error');
      scheduleFinderPoll(8000);
    }
  }

  async function loadFinderWorkspace({ quiet = false, preserveConfig = false } = {}) {
    if (state.finderLoading) return;
    state.finderLoading = true;
    syncFinderConfigAvailability();
    const requests = await Promise.allSettled([
      api('/api/finder/folders'),
      api('/api/finder/status'),
      api('/api/pose-tags'),
      api('/api/finder/scans'),
      api('/api/finder/corpus')
    ]);
    const [foldersResult, statusResult, tagsResult, scansResult, corpusResult] = requests;
    if (foldersResult.status === 'fulfilled') state.finderFolders = apiItems(foldersResult.value, 'folders').map(normalizeFinderFolder).filter(folder => folder.path);
    if (statusResult.status === 'fulfilled') state.finderStatus = normalizeFinderStatus(statusResult.value);
    else state.finderStatus = null;
    if (corpusResult.status === 'fulfilled') {
      state.finderCorpus = normalizeFinderCorpus(corpusResult.value);
      state.finderCorpusSupported = Boolean(state.finderCorpus);
      state.finderJoytagIndexSupported = state.finderCorpus
        ? Boolean(state.finderCorpus.joytag)
        : null;
    } else if (state.finderStatus?.corpus) {
      state.finderCorpus = state.finderStatus.corpus;
      state.finderCorpusSupported = true;
      state.finderJoytagIndexSupported = Boolean(state.finderCorpus.joytag);
    } else {
      state.finderCorpus = null;
      state.finderCorpusSupported = corpusResult.reason?.status === 404 ? false : null;
      state.finderJoytagIndexSupported = corpusResult.reason?.status === 404 ? false : null;
    }
    if (tagsResult.status === 'fulfilled') state.finderTags = apiItems(tagsResult.value).map(normalizePoseTag).filter(tag => tag.id !== undefined && tag.label);
    if (scansResult.status === 'fulfilled') state.finderScans = apiItems(scansResult.value, 'scans').map(normalizeFinderScan).filter(scan => scan?.id);
    state.finderLoaded = foldersResult.status === 'fulfilled' || scansResult.status === 'fulfilled';
    if (!$('#finder-source').value.trim()) $('#finder-source').value = finderDefaultSource();
    const stored = state.finderScanId && state.finderScans.find(scan => String(scan.id) === String(state.finderScanId));
    const active = state.finderScans.find(scan => !finderScanIsTerminal(scan));
    const selected = stored || active || state.finderScans[0] || null;
    if (selected?.id) {
      state.finderScanId = selected.id;
      storage.set('finder-scan', state.finderScanId);
    }
    renderFinderWorkspace();
    scheduleFinderJoytagIndexPoll();
    state.finderLoading = false;
    syncFinderConfigAvailability();
    if (selected?.id) await loadFinderScan({ quiet: true, applyConfig: !preserveConfig });
    else await loadFinderFeedback({ quiet: true, force: true });
    const failures = requests.slice(0, 4).filter(result => result.status === 'rejected');
    if (!quiet && failures.length) toast('Some Finder options are unavailable', errorMessage(failures[0].reason), 'error');
  }

  async function startFinderScan() {
    if (state.finderBusy || state.finderFeedbackBusy || finderFeedbackIsSaving()) return;
    if (
      finderFeedbackHasUnsavedSelections()
      && !window.confirm('Start a new scan and discard unsaved feedback-selection edits on the current results?')
    ) return;
    const config = readFinderConfig({ validate: true });
    if (!config) return;
    const button = $('#finder-start');
    state.finderBusy = true;
    setButtonBusy(button, true, 'Starting…');
    try {
      const tag = await ensureFinderPoseTag(config.tagLabel);
      const data = await api('/api/finder/scans', {
        method: 'POST',
        body: {
          example_directory: config.exampleDirectory,
          pose_tag_id: tag.id,
          source_url: config.sourceUrl,
          page_limit: config.pageLimit,
          minimum_score: config.minimumScore,
          mode: config.mode,
          joytag_tag: config.joytagTag,
          joytag_required_tags: config.joytagRequiredTags,
          joytag_excluded_tags: config.joytagExcludedTags,
          joytag_reject_threshold: config.joytagRejectThreshold,
          reference_fingerprint: config.referenceFingerprint
        }
      });
      const scan = normalizeFinderScan(data?.scan || data);
      if (!scan?.id) throw new ApiError('The server did not return a Finder scan ID.');
      if (!scan.poseTagLabel) scan.poseTagLabel = tag.label;
      if (!scan.poseTagId) scan.poseTagId = tag.id;
      scan.poseDefaultRole = tag.defaultRole;
      scan.searchMode = config.mode;
      scan.joytagTag = config.joytagTag || '';
      scan.joytagRequiredTags = [...config.joytagRequiredTags];
      scan.joytagExcludedTags = [...config.joytagExcludedTags];
      scan.joytagRejectThreshold = config.joytagRejectThreshold;
      scan.referenceFingerprint = config.referenceFingerprint || '';
      cancelAdjacentGalleryPrefetches();
      state.finderScan = scan;
      state.finderScanId = scan.id;
      state.finderResults = [];
      state.finderReviewCounts = { pending: 0, accepted: 0, maybe: 0, rejected: 0, total: 0 };
      state.finderReview = 'pending';
      resetFinderResultPagination();
      state.finderScans = [scan, ...state.finderScans.filter(item => String(item.id) !== String(scan.id))];
      storage.set('finder-scan', state.finderScanId);
      $('#finder-result-threshold').min = config.mode === 'joytag' ? '0.05' : '0.40';
      $('#finder-result-threshold').value = config.minimumScore.toFixed(2);
      $('#finder-filter-output').textContent = config.minimumScore.toFixed(2);
      toast(
        config.mode === 'joytag' ? 'JoyTag scan started' : 'Finder scan started',
        config.mode === 'joytag'
          ? `Searching up to ${config.pageLimit} pages for ${config.joytagRequiredTags.length === 1
            ? `“${humanizeJoytagTag(config.joytagTag)}”`
            : `all ${config.joytagRequiredTags.length} required tags`}${config.joytagExcludedTags.length
            ? `, excluding ${config.joytagExcludedTags.length} ${config.joytagExcludedTags.length === 1 ? 'tag' : 'tags'}`
            : ''}, at ${config.minimumScore.toFixed(2)} required confidence.`
          : `Scanning up to ${config.pageLimit} pages for “${tag.label}”.`,
        'success'
      );
      await loadFinderScan({ quiet: true });
    } catch (error) {
      toast('Could not start Finder', errorMessage(error), 'error');
    } finally {
      state.finderBusy = false;
      setButtonBusy(button, false);
      renderFinderWorkspace();
    }
  }

  async function extendFinderScan() {
    const scan = state.finderScan;
    if (!finderScanCanExtend(scan) || state.finderBusy) return;
    const additionalPages = updateFinderExtendSummary({ commit: true });
    const previousLimit = Math.max(0, Number(scan.pages || 0), Number(scan.pagesScanned || 0));
    const button = $('#finder-extend-button');
    invalidateFinderScanLoads();
    state.finderBusy = true;
    $('#finder-extend-pages').disabled = true;
    setButtonBusy(button, true, 'Extending…');
    try {
      const data = await api(`/api/finder/scans/${encodeURIComponent(scan.id)}/extend`, {
        method: 'POST',
        body: { additional_pages: additionalPages }
      });
      const updated = normalizeFinderScan(data?.scan || data);
      if (!updated?.id) throw new ApiError('The server did not return the extended Finder scan.');
      state.finderScan = updated;
      const existing = state.finderScans.findIndex(item => String(item.id) === String(updated.id));
      if (existing >= 0) state.finderScans[existing] = updated;
      else state.finderScans.unshift(updated);
      const newLimit = Math.max(previousLimit + additionalPages, Number(updated.pages || 0));
      const extensionDetail = updated.status === 'paused'
        ? `Page limit increased from ${formatNumber(previousLimit)} to ${formatNumber(newLimit)}. Resume when you are ready.`
        : `Continuing from ${formatNumber(previousLimit)} to ${formatNumber(newLimit)} pages. Existing results stay in place.`;
      toast('Finder search extended', extensionDetail, 'success');
      announce(`Finder search extended by ${additionalPages} pages.`);
    } catch (error) {
      toast('Could not extend Finder search', errorMessage(error), 'error');
    } finally {
      invalidateFinderScanLoads();
      state.finderBusy = false;
      setButtonBusy(button, false);
      renderFinderWorkspace();
      scheduleFinderPoll(300);
    }
  }

  async function continueFinderScan() {
    const scan = state.finderScan;
    if (
      !finderScanCanChangeSource(scan)
      || state.finderBusy
      || state.finderFeedbackBusy
      || finderFeedbackIsSaving()
    ) return;
    const switchingSource = finderScanCanSwitchSource(scan);
    const config = readFinderContinueConfig({ validate: true });
    if (!config) return;
    const button = $('#finder-continue-button');
    invalidateFinderScanLoads();
    state.finderBusy = true;
    $('#finder-continue-source').disabled = true;
    $('#finder-continue-pages').disabled = true;
    setButtonBusy(button, true, switchingSource ? 'Switching…' : 'Continuing…');
    try {
      const data = await api(`/api/finder/scans/${encodeURIComponent(scan.id)}/continue`, {
        method: 'POST',
        body: {
          source_url: config.sourceUrl,
          additional_pages: config.additionalPages
        }
      });
      const updated = normalizeFinderScan(data?.scan || data);
      if (!updated?.id || String(updated.id) !== String(scan.id)) {
        throw new ApiError('The server did not return the continued Finder scan.');
      }
      state.finderScan = updated;
      const existing = state.finderScans.findIndex(item => String(item.id) === String(updated.id));
      if (existing >= 0) state.finderScans[existing] = updated;
      else state.finderScans.unshift(updated);
      $('#finder-source').value = updated.sourceUrl || config.sourceUrl;
      $('#finder-continue-source').value = '';
      const candidateCount = Math.max(
        Number(state.finderReviewCounts?.total || 0),
        Number(updated.candidateCount || 0)
      );
      const keptCopy = `${formatNumber(candidateCount)} ${candidateCount === 1 ? 'candidate' : 'candidates'} and all reviews remain in this scan.`;
      toast(
        switchingSource ? 'Finder source switched' : 'Exploring another source',
        `Scanning up to ${config.additionalPages} new ${config.additionalPages === 1 ? 'page' : 'pages'} from ${displayHost(config.sourceUrl)}. ${keptCopy}`,
        'success'
      );
      announce(
        switchingSource
          ? `Finder source switched for up to ${config.additionalPages} pages.`
          : `Finder scan continued from another source for up to ${config.additionalPages} pages.`
      );
    } catch (error) {
      toast(switchingSource ? 'Could not switch Finder source' : 'Could not continue Finder search', errorMessage(error), 'error');
    } finally {
      invalidateFinderScanLoads();
      state.finderBusy = false;
      setButtonBusy(button, false);
      renderFinderWorkspace();
      scheduleFinderPoll(300);
    }
  }

  async function performFinderScanAction(action, button) {
    const scan = state.finderScan;
    if (!scan?.id || state.finderBusy || !['pause', 'resume', 'retry'].includes(action)) return;
    const retrying = action === 'retry';
    invalidateFinderScanLoads();
    state.finderBusy = true;
    setButtonBusy(button, true, action === 'pause' ? 'Pausing…' : retrying ? 'Retrying…' : 'Resuming…');
    try {
      const data = await api(`/api/finder/scans/${encodeURIComponent(scan.id)}/${action}`, { method: 'POST' });
      const updated = normalizeFinderScan(data?.scan || data);
      if (updated?.id) {
        state.finderScan = updated;
        const existing = state.finderScans.findIndex(item => String(item.id) === String(updated.id));
        if (existing >= 0) state.finderScans[existing] = updated;
        else state.finderScans.unshift(updated);
      } else await loadFinderScan({ quiet: true });
      toast(
        action === 'pause' ? 'Finder paused' : retrying ? 'Finder retry started' : 'Finder resumed',
        action === 'pause'
          ? 'Ranked results remain available for review.'
          : retrying
            ? 'Retrying the saved source position. Existing candidates and reviews remain in this scan.'
            : 'The server will continue from its saved progress.',
        'info'
      );
    } catch (error) {
      toast(
        action === 'pause' ? 'Could not pause Finder' : retrying ? 'Could not retry Finder' : 'Could not resume Finder',
        errorMessage(error),
        'error'
      );
    } finally {
      invalidateFinderScanLoads();
      state.finderBusy = false;
      setButtonBusy(button, false);
      renderFinderWorkspace();
      scheduleFinderPoll();
    }
  }

  async function cancelFinderScan() {
    const scan = state.finderScan;
    if (!scan?.id || finderScanIsTerminal(scan) || state.finderBusy) return;
    if (!window.confirm('Cancel this Finder scan? Results already found will remain available.')) return;
    const button = $('#finder-cancel');
    invalidateFinderScanLoads();
    state.finderBusy = true;
    button.disabled = true;
    try {
      const data = await api(`/api/finder/scans/${encodeURIComponent(scan.id)}`, { method: 'DELETE' });
      state.finderScan = data ? normalizeFinderScan(data?.scan || data) : { ...scan, status: 'cancelled' };
      if (!state.finderScan?.id) state.finderScan = { ...scan, status: 'cancelled' };
      toast('Finder scan cancelled', 'Existing candidates are still available for review.', 'info');
    } catch (error) {
      toast('Could not cancel Finder scan', errorMessage(error), 'error');
    } finally {
      invalidateFinderScanLoads();
      state.finderBusy = false;
      renderFinderWorkspace();
      scheduleFinderPoll();
    }
  }

  function recountFinderReviews() {
    if (!state.finderScan) return;
    const counts = state.finderReviewCounts || {
      pending: state.finderResults.filter(result => result.review === 'pending').length,
      accepted: state.finderResults.filter(result => result.review === 'accepted').length,
      maybe: state.finderResults.filter(result => result.review === 'maybe').length,
      rejected: state.finderResults.filter(result => result.review === 'rejected').length
    };
    state.finderScan.pendingCount = counts.pending;
    state.finderScan.acceptedCount = counts.accepted;
    state.finderScan.maybeCount = counts.maybe;
    state.finderScan.rejectedCount = counts.rejected;
  }

  function finderFeedbackHasUnsavedSelections() {
    return state.finderResults.some(result => result.feedbackSelectionDirty);
  }

  function adjustFinderReviewCounts(previousReview, nextReview) {
    if (!state.finderReviewCounts || previousReview === nextReview) return;
    if (state.finderReviewCounts[previousReview] !== undefined) {
      state.finderReviewCounts[previousReview] = Math.max(0, state.finderReviewCounts[previousReview] - 1);
    }
    if (state.finderReviewCounts[nextReview] !== undefined) {
      state.finderReviewCounts[nextReview] += 1;
    }
  }

  function finderResultCopies(result) {
    const key = String(result?.key ?? '');
    const copies = [];
    const add = item => {
      if (!item || String(item.key ?? '') !== key || copies.includes(item)) return;
      copies.push(item);
    };
    add(result);
    add(state.galleryContext?.activeFinderResult);
    (state.galleryContext?.finderReviewQueue?.results || []).forEach(add);
    state.finderResults.forEach(add);
    return copies;
  }

  function toggleFinderFeedbackMatch(input) {
    const result = state.finderResults.find(item => String(item.key) === String(input.dataset.finderResult));
    const matchKey = input.dataset.finderFeedbackMatch;
    if (!result || !matchKey || result.feedbackSaving || state.finderFeedbackBusy) return;
    const selected = new Set(result.feedbackMatchKeys);
    if (input.checked) selected.add(matchKey);
    else selected.delete(matchKey);
    const nextMatchKeys = result.matches
      .map(match => match.feedbackKey)
      .filter(key => selected.has(key));
    const topMatchUrls = new Set(result.matches.map(match => match.imageUrl).filter(Boolean));
    const outsideTopMatches = result.feedbackImageUrls.filter(url => !topMatchUrls.has(url));
    const selectedTopMatches = result.matches
      .filter(match => nextMatchKeys.includes(match.feedbackKey))
      .map(match => match.imageUrl)
      .filter(Boolean);
    const nextImageUrls = [...new Set([...outsideTopMatches, ...selectedTopMatches])];
    if (nextImageUrls.length > 3) {
      input.checked = false;
      toast('Choose up to 3 feedback images', 'This gallery already has selections outside the top suggestions. Open it to edit the complete selection.', 'info');
      renderFinderResults();
      return;
    }
    result.feedbackMatchKeys = nextMatchKeys;
    result.feedbackImageUrls = nextImageUrls;
    result.feedbackSelectionDirty = true;
    renderFinderResults();
    announce(
      finderScanUsesJoyTag()
        ? `${input.checked ? 'Included' : 'Excluded'} matching image ${result.feedbackMatchKeys.length} of ${result.matches.length} for this gallery review.`
        : `${input.checked ? 'Included' : 'Excluded'} suggested image ${result.feedbackMatchKeys.length} of ${result.matches.length} for pose feedback.`
    );
  }

  async function reviewFinderResult(result, review, button = null, { feedbackImageUrls: explicitFeedbackUrls = null } = {}) {
    if (!result || !['pending', 'accepted', 'maybe', 'rejected'].includes(review) || state.finderBusy || state.finderFeedbackBusy || result.feedbackSaving) return false;
    const scanId = String(state.finderScan?.id || '');
    if (!scanId) return false;
    const resultKey = String(result.key);
    const snapshot = {
      review: result.review,
      feedbackMatchKeys: [...result.feedbackMatchKeys],
      feedbackImageUrls: [...result.feedbackImageUrls],
      feedbackUsableImageUrls: [...result.feedbackUsableImageUrls],
      feedbackPendingImageUrls: [...result.feedbackPendingImageUrls],
      feedbackAnalysisProvided: result.feedbackAnalysisProvided,
      feedbackSelectionProvided: result.feedbackSelectionProvided,
      feedbackSelectionDirty: result.feedbackSelectionDirty
    };
    const resultCopies = finderResultCopies(result);
    if (snapshot.review === review && !snapshot.feedbackSelectionDirty && explicitFeedbackUrls === null) return false;
    const feedbackImageUrls = review === 'maybe'
      ? []
      : explicitFeedbackUrls === null
        ? [...result.feedbackImageUrls]
        : [...new Set(explicitFeedbackUrls.map(String).filter(Boolean))].slice(0, 3);
    if (review === 'accepted' && !feedbackImageUrls.length) {
      toast('Check a matching image', 'Accept needs at least one checked suggestion. Unchecked images are excluded from feedback.', 'info');
      return false;
    }
    const sameScan = () => String(state.finderScan?.id || '') === scanId;
    state.finderResultMutationEpoch += 1;
    state.finderFeedbackMutations += 1;
    resultCopies.forEach(copy => {
      copy.feedbackSaving = true;
      copy.review = review;
    });
    adjustFinderReviewCounts(snapshot.review, review);
    recountFinderReviews();
    renderFinderResults();
    renderFinderFeedback();
    renderFinderGalleryReview();
    syncFinderConfigAvailability();
    if (button) button.disabled = true;
    try {
      const data = await api(`/api/finder/scans/${encodeURIComponent(scanId)}/results/${encodeURIComponent(resultKey)}`, {
        method: 'PATCH',
        body: {
          review,
          feedback_image_urls: feedbackImageUrls
        }
      });
      if (!sameScan()) return false;
      const index = state.finderResults.findIndex(item => String(item.key) === resultKey);
      const current = index >= 0 ? state.finderResults[index] : result;
      let merged;
      if (data) {
        const updated = normalizeFinderResult(data?.result || data, result.rank - 1);
        merged = {
          ...current,
          ...updated,
          galleryId: updated.galleryId || current.galleryId,
          url: updated.url || current.url,
          title: updated.title === 'Untitled gallery' ? current.title : updated.title,
          bestImageUrl: updated.bestImageUrl || current.bestImageUrl,
          bestPreviewUrl: updated.bestPreviewUrl || current.bestPreviewUrl,
          imageCount: updated.imageCount || current.imageCount,
          matches: updated.matches?.length ? updated.matches : current.matches,
          exactScore: updated.exactScore ?? current.exactScore,
          poseScore: updated.poseScore ?? current.poseScore,
          appearanceScore: updated.appearanceScore ?? current.appearanceScore,
          personCount: updated.personCount || current.personCount,
          hasOverlay: updated.hasOverlay || current.hasOverlay,
          feedbackMatchKeys: updated.feedbackSelectionProvided ? updated.feedbackMatchKeys : snapshot.feedbackMatchKeys,
          feedbackImageUrls: updated.feedbackSelectionProvided ? updated.feedbackImageUrls : feedbackImageUrls,
          feedbackUsableImageUrls: updated.feedbackAnalysisProvided ? updated.feedbackUsableImageUrls : [],
          feedbackPendingImageUrls: updated.feedbackAnalysisProvided ? updated.feedbackPendingImageUrls : [],
          feedbackAnalysisProvided: updated.feedbackAnalysisProvided,
          feedbackSelectionProvided: updated.feedbackSelectionProvided || snapshot.feedbackSelectionProvided,
          feedbackSelectionDirty: false,
          feedbackSaving: false
        };
      } else {
        merged = {
          ...current,
          review,
          feedbackImageUrls: [...feedbackImageUrls],
          feedbackMatchKeys: current.matches
            .filter(match => feedbackImageUrls.includes(match.imageUrl))
            .map(match => match.feedbackKey),
          feedbackUsableImageUrls: [],
          feedbackPendingImageUrls: [],
          feedbackAnalysisProvided: false,
          feedbackSelectionProvided: true,
          feedbackSelectionDirty: false,
          feedbackSaving: false
        };
      }
      resultCopies.forEach(copy => Object.assign(copy, merged));
      if (index >= 0) Object.assign(state.finderResults[index], merged);
      recountFinderReviews();
      renderFinderWorkspace();
      renderFinderGalleryReview();
      if (!applyFinderFeedbackResponse(data)) loadFinderFeedback({ quiet: true, force: true });
      announce(`${result.title} ${review}`);
      return true;
    } catch (error) {
      if (sameScan()) {
        adjustFinderReviewCounts(review, snapshot.review);
        resultCopies.forEach(copy => Object.assign(copy, {
          review: snapshot.review,
          feedbackMatchKeys: [...snapshot.feedbackMatchKeys],
          feedbackImageUrls: [...snapshot.feedbackImageUrls],
          feedbackUsableImageUrls: [...snapshot.feedbackUsableImageUrls],
          feedbackPendingImageUrls: [...snapshot.feedbackPendingImageUrls],
          feedbackAnalysisProvided: snapshot.feedbackAnalysisProvided,
          feedbackSelectionProvided: snapshot.feedbackSelectionProvided,
          feedbackSelectionDirty: snapshot.feedbackSelectionDirty,
          feedbackSaving: false
        }));
        recountFinderReviews();
        renderFinderWorkspace();
        renderFinderGalleryReview();
      }
      toast('Could not save review', errorMessage(error), 'error');
      return false;
    } finally {
      state.finderFeedbackMutations = Math.max(0, state.finderFeedbackMutations - 1);
      state.finderResultMutationEpoch += 1;
      if (sameScan()) scheduleFreshFinderResults();
      if (sameScan()) {
        const stillSaving = resultCopies.some(copy => copy.feedbackSaving);
        resultCopies.forEach(copy => { copy.feedbackSaving = false; });
        if (stillSaving) {
          renderFinderResults();
        }
        renderFinderFeedback();
        renderFinderGalleryReview();
      }
      syncFinderConfigAvailability();
    }
  }

  function finderPoseTagForScan(scan = state.finderScan) {
    const existing = state.finderTags.find(tag => String(tag.id) === String(scan?.poseTagId));
    if (existing) return existing;
    return {
      id: scan?.poseTagId,
      label: scan?.poseTagLabel || 'Pose target',
      slug: scan?.poseTagSlug || '',
      defaultRole: POSE_ROLES.includes(scan?.poseDefaultRole) ? scan.poseDefaultRole : 'solo'
    };
  }

  function finderQueueResultStillMatches(queue, result) {
    return result.review === queue.review && result.score >= queue.threshold;
  }

  async function loadFinderGalleryQueueEdge(queue, direction) {
    if (
      !queue
      || String(queue.scanId) !== String(state.finderScan?.id || '')
      || ![-1, 1].includes(direction)
    ) return [];
    const existingKeys = new Set(queue.results.map(result => String(result.key)));
    const removedCount = queue.results.filter(result => !finderQueueResultStillMatches(queue, result)).length;
    let offset;
    let limit;
    if (direction > 0) {
      const computedOffset = queue.baseOffset
        + queue.results.filter(result => finderQueueResultStillMatches(queue, result)).length;
      const removedSinceProbe = removedCount - Number(queue.forwardProbeRemovedCount || 0);
      offset = Math.max(
        computedOffset,
        Math.max(0, Number(queue.forwardProbeOffset ?? computedOffset) - removedSinceProbe)
      );
      limit = queue.pageSize;
      let rawFetched = [];
      let fetched = [];
      let liveTotal = 0;
      for (let attempt = 0; attempt < 16; attempt += 1) {
        const data = await api(withParams(`/api/finder/scans/${encodeURIComponent(queue.scanId)}/results`, {
          review: queue.review,
          min_score: queue.threshold,
          limit,
          offset
        }));
        if (String(queue.scanId) !== String(state.finderScan?.id || '')) return [];
        rawFetched = apiItems(data, 'results');
        liveTotal = Math.max(0, Number(data?.total ?? rawFetched.length) || 0);
        fetched = rawFetched
          .map((item, index) => normalizeFinderResult(item, offset + index))
          .filter(result => !existingKeys.has(String(result.key)));
        queue.forwardProbeOffset = offset + rawFetched.length;
        queue.forwardProbeRemovedCount = removedCount;
        queue.total = Math.max(queue.total, queue.baseOffset + queue.results.length + fetched.length);
        const reachedLiveEnd = rawFetched.length < limit || offset + rawFetched.length >= liveTotal;
        if (fetched.length || reachedLiveEnd) {
          queue.results.push(...fetched);
          queue.exhaustedAfter = !finderScanIsRunning() && reachedLiveEnd;
          return fetched;
        }
        offset += rawFetched.length;
      }
      queue.exhaustedAfter = false;
      return [];
    } else if (finderScanIsRunning()) {
      // Offset pages move while an active scan inserts better-ranked results.
      // Probe from the live beginning until an already loaded result anchors the
      // stable modal queue, then prepend every unseen predecessor before it.
      const anchorKeys = new Set(queue.results.map(result => String(result.key)));
      if (!anchorKeys.size) {
        queue.exhaustedBefore = false;
        return [];
      }
      offset = 0;
      limit = Math.min(500, Math.max(queue.pageSize, queue.baseOffset + queue.pageSize));
      const fetched = [];
      const fetchedKeys = new Set(existingKeys);
      let liveTotal = 0;
      for (let attempt = 0; attempt < 32; attempt += 1) {
        const data = await api(withParams(`/api/finder/scans/${encodeURIComponent(queue.scanId)}/results`, {
          // Reviewed-away queue entries still provide a stable rank anchor.
          review: 'all',
          min_score: queue.threshold,
          limit,
          offset
        }));
        if (String(queue.scanId) !== String(state.finderScan?.id || '')) return [];
        const rawFetched = apiItems(data, 'results');
        liveTotal = Math.max(0, Number(data?.total ?? rawFetched.length) || 0);
        const normalized = rawFetched.map(
          (item, index) => normalizeFinderResult(item, offset + index)
        );
        const anchorIndex = normalized.findIndex(result => anchorKeys.has(String(result.key)));
        const predecessors = (anchorIndex >= 0 ? normalized.slice(0, anchorIndex) : normalized)
          .filter(result => finderQueueResultStillMatches(queue, result));
        predecessors.forEach(result => {
          const key = String(result.key);
          if (fetchedKeys.has(key)) return;
          fetchedKeys.add(key);
          fetched.push(result);
        });
        if (anchorIndex >= 0) {
          queue.results.unshift(...fetched);
          queue.index += fetched.length;
          queue.baseOffset = 0;
          const liveQueueTotal = Math.max(
            0,
            Number(data?.counts?.[queue.review] ?? queue.total) || 0
          );
          queue.total = Math.max(queue.total, liveQueueTotal + removedCount, queue.results.length);
          queue.exhaustedBefore = true;
          return fetched;
        }
        const reachedLiveEnd = rawFetched.length < limit || offset + rawFetched.length >= liveTotal;
        if (reachedLiveEnd || !rawFetched.length) break;
        offset += rawFetched.length;
      }
      // The known page may have been reviewed away while the scan was moving.
      // Keep the edge retryable instead of prepending an unanchored live slice.
      queue.exhaustedBefore = false;
      return [];
    } else {
      offset = Math.max(0, queue.baseOffset - queue.pageSize);
      limit = Math.max(0, queue.baseOffset - offset);
      if (!limit) {
        queue.exhaustedBefore = true;
        return [];
      }
    }
    const data = await api(withParams(`/api/finder/scans/${encodeURIComponent(queue.scanId)}/results`, {
      review: queue.review,
      min_score: queue.threshold,
      limit,
      offset
    }));
    if (String(queue.scanId) !== String(state.finderScan?.id || '')) return [];
    const rawFetched = apiItems(data, 'results');
    const fetched = rawFetched
      .map((item, index) => normalizeFinderResult(item, offset + index))
      .filter(result => !existingKeys.has(String(result.key)));
    queue.total = Math.max(queue.total, queue.baseOffset + queue.results.length + fetched.length);
    queue.results.unshift(...fetched);
    queue.index += fetched.length;
    queue.baseOffset = offset;
    queue.exhaustedBefore = offset <= 0;
    return fetched;
  }

  async function reviewFinderGallery(review, button = null) {
    const result = finderFeedbackGalleryResult();
    if (
      !result
      || !['accepted', 'maybe', 'rejected'].includes(review)
      || state.galleryReviewBusy
      || state.finderFeedbackGallerySaving
    ) return false;
    const feedbackImageUrls = review === 'maybe' ? [] : [...state.finderFeedbackGallerySelection];
    if (review === 'accepted' && !feedbackImageUrls.length) {
      toast('Choose a matching image', 'Open Finder review and select at least one image before accepting this gallery.', 'info');
      return false;
    }
    state.galleryReviewBusy = true;
    state.finderFeedbackGallerySaving = true;
    if (button) setButtonBusy(button, true, 'Saving…');
    renderFinderGalleryReview();
    try {
      const saved = await reviewFinderResult(result, review, null, { feedbackImageUrls });
      if (!saved) return false;
      if (state.galleryContext) {
        state.galleryContext.finderFeedbackReview = review;
        state.galleryContext.feedbackImageUrls = [...result.feedbackImageUrls];
      }
      state.finderFeedbackGallerySelection = new Set(result.feedbackImageUrls);
      state.finderFeedbackGalleryDirty = false;
      setGalleryMode(state.galleryMode, { load: false, render: false });
      renderImages();
      updateSelectionUi();
      const feedbackCount = result.feedbackImageUrls.length;
      toast(
        `Gallery ${finderReviewLabel(review).toLowerCase()}`,
        review === 'maybe'
          ? 'Kept neutral. Use the arrows when you are ready for another gallery.'
          : feedbackCount
            ? `${formatNumber(feedbackCount)} feedback ${feedbackCount === 1 ? 'image was' : 'images were'} saved. You can finish the pose assignments before moving on.`
            : 'The gallery decision was saved without image-level feedback.',
        'success'
      );
      return true;
    } finally {
      state.galleryReviewBusy = false;
      state.finderFeedbackGallerySaving = false;
      if (button) setButtonBusy(button, false);
      updateSelectionUi();
      renderFinderGalleryReview();
    }
  }

  async function navigateFinderGallery(direction) {
    const queue = state.galleryContext?.finderReviewQueue;
    if (
      !queue
      || ![-1, 1].includes(direction)
      || state.galleryReviewBusy
      || queue.loading
      || state.finderFeedbackGallerySaving
      || state.poseApplying
      || state.poseExporting
    ) return false;
    const discardFeedback = state.finderFeedbackGalleryDirty;
    if (discardFeedback && !window.confirm('Discard the unsaved Finder feedback selection and move to another gallery?')) return false;
    const navigationRequest = ++state.galleryNavigationRequest;
    const stillCurrent = () => (
      navigationRequest === state.galleryNavigationRequest
      && $('#gallery-modal').open
      && state.galleryContext?.finderReviewQueue === queue
    );
    state.galleryReviewBusy = true;
    queue.loading = true;
    renderFinderGalleryReview();
    const requestedMode = state.galleryMode;
    try {
      await flushPoseDraft();
      if (!stillCurrent()) return false;
      if (state.poseDirty) {
        toast('Pose draft still has unsaved changes', 'Resolve the save error before moving to another gallery.', 'error');
        return false;
      }
      let targetIndex = queue.index + direction;
      if (targetIndex < 0 && !queue.exhaustedBefore) {
        await loadFinderGalleryQueueEdge(queue, -1);
        if (!stillCurrent()) return false;
        targetIndex = queue.index - 1;
      } else if (targetIndex >= queue.results.length && !queue.exhaustedAfter) {
        await loadFinderGalleryQueueEdge(queue, 1);
        if (!stillCurrent()) return false;
        targetIndex = queue.index + 1;
      }
      const target = queue.results[targetIndex];
      if (!target) {
        const catchingUp = direction > 0 ? !queue.exhaustedAfter : !queue.exhaustedBefore;
        toast(
          catchingUp
            ? 'Review queue is catching up'
            : direction > 0 ? 'End of this review queue' : 'Start of this review queue',
          catchingUp
            ? `New scan results changed the ranking while this modal was open. Use ${direction > 0 ? 'Next' : 'Previous'} again to continue the stable queue.`
            : `There are no more ${finderReviewLabel(queue.review).toLowerCase()} galleries in that direction.`,
          'info'
        );
        renderFinderGalleryReview();
        return false;
      }
      if (!stillCurrent()) return false;
      if (state.poseApplying || state.poseExporting || state.finderFeedbackGallerySaving) return false;
      if (discardFeedback) restoreFinderFeedbackGallerySelection();
      queue.index = targetIndex;
      await openFinderResult(target, { reviewQueue: queue, mode: requestedMode });
      return true;
    } catch (error) {
      toast('Could not open the next gallery', errorMessage(error), 'error');
      return false;
    } finally {
      queue.loading = false;
      if (navigationRequest === state.galleryNavigationRequest) {
        state.galleryReviewBusy = false;
        renderFinderGalleryReview();
      }
    }
  }

  async function openFinderResult(result, { reviewQueue = null, mode = null } = {}) {
    if (!result?.galleryId) {
      toast('Gallery unavailable', 'This Finder result has no gallery identifier.', 'error');
      return;
    }
    const queue = reviewQueue || createFinderGalleryReviewQueue(result);
    const queueIndex = queue.results.findIndex(item => String(item.key) === String(result.key));
    if (queueIndex >= 0) queue.index = queueIndex;
    const feedbackReview = ['accepted', 'rejected'].includes(result.review);
    await openGallery(result.galleryId, {
      summary: {
        id: result.galleryId,
        url: result.url,
        title: result.title,
        thumbnail_url: result.bestPreviewUrl,
        image_count: result.imageCount
      },
      mode: mode || (feedbackReview ? 'feedback' : 'pose'),
      poseTag: finderPoseTagForScan(),
      finderFeedbackResultKey: result.key,
      finderFeedbackReview: result.review,
      activeFinderResult: result,
      finderReviewQueue: queue,
      feedbackImageUrls: [...result.feedbackImageUrls],
      suggestions: (result.matches?.length ? result.matches : [{ imageUrl: result.bestImageUrl, ordinal: result.bestOrdinal, score: result.score }]).map(match => ({
        imageUrl: match.imageUrl,
        ordinal: match.ordinal,
        score: firstFinderScore(match.score, result.score) ?? 0
      }))
    });
  }

  async function selectFinderScan(scanId) {
    if (state.finderFeedbackBusy || finderFeedbackIsSaving()) return;
    if (
      String(scanId || '') !== String(state.finderScanId || '')
      && finderFeedbackHasUnsavedSelections()
      && !window.confirm('Switch scans and discard unsaved feedback-selection edits on the current results?')
    ) {
      renderFinderScans();
      return;
    }
    const changedScan = String(scanId || '') !== String(state.finderScanId || '');
    if (changedScan) {
      cancelAdjacentGalleryPrefetches();
      $('#finder-continue-source').value = '';
      resetFinderResultPagination();
    }
    state.finderScanId = scanId;
    state.finderReviewCounts = null;
    storage.set('finder-scan', scanId);
    if (!scanId) {
      state.finderScan = null;
      state.finderResults = [];
      resetFinderResultPagination();
      window.clearTimeout(state.finderPollTimer);
      renderFinderWorkspace();
      await loadFinderFeedback({ quiet: true, force: true });
      return;
    }
    await loadFinderScan({ applyConfig: true });
  }

  function normalizeSortFolder(item) {
    if (typeof item === 'string') return { path: item, name: item.split('/').filter(Boolean).pop() || item, imageCount: 0 };
    const path = String(item?.path || item?.directory || '');
    return {
      ...item,
      path,
      name: item?.name || path.split('/').filter(Boolean).pop() || path || 'Unnamed folder',
      imageCount: Number(item?.image_count ?? item?.count ?? 0)
    };
  }

  function normalizeSortProfile(item) {
    return {
      ...item,
      name: String(item?.name || ''),
      targetDirectory: String(item?.target_directory || ''),
      controlDirectories: Array.isArray(item?.control_directories) ? item.control_directories.map(String) : [],
      mode: item?.mode === 'stem' ? 'stem' : 'time',
      thresholdSeconds: Number(item?.threshold_seconds ?? 50),
      addIds: Boolean(item?.add_ids ?? true)
    };
  }

  function normalizeSortSession(item) {
    if (!item) return null;
    const current = item.current ? {
      ...item.current,
      path: String(item.current.path || ''),
      name: item.current.name || String(item.current.path || '').split('/').pop() || 'Current target',
      previewUrl: item.current.preview_url || '',
      modifiedAt: item.current.modified_at || ''
    } : null;
    const matches = (Array.isArray(item.matches) ? item.matches : []).map(match => ({
      ...match,
      path: String(match.path || ''),
      name: match.name || String(match.path || '').split('/').pop() || 'Candidate',
      previewUrl: match.preview_url || '',
      folder: match.folder || '',
      deltaSeconds: match.delta_seconds === null || match.delta_seconds === undefined ? null : Number(match.delta_seconds)
    }));
    return {
      ...item,
      id: item.id,
      status: String(item.status || 'active').toLowerCase(),
      mode: item.mode === 'stem' ? 'stem' : 'time',
      targetDirectory: item.target_directory || '',
      controlDirectories: Array.isArray(item.control_directories) ? item.control_directories : [],
      thresholdSeconds: Number(item.threshold_seconds ?? 50),
      addIds: Boolean(item.add_ids),
      total: Number(item.total || 0),
      processed: Number(item.processed || 0),
      remaining: Number(item.remaining || 0),
      missing: Number(item.missing || 0),
      recovering: Number(item.recovering || 0),
      canUndo: Boolean(item.can_undo),
      current,
      matches
    };
  }

  function renderSortFolders() {
    const targetSelect = $('#sort-target');
    const selectedTarget = targetSelect.value;
    const selectedControls = new Set($$('#sort-control-list input:checked').map(input => input.value));
    targetSelect.replaceChildren(new Option('Choose a target folder…', ''));
    state.sortFolders.forEach(folder => {
      const count = folder.imageCount ? ` · ${formatNumber(folder.imageCount)} images` : '';
      targetSelect.add(new Option(`${folder.name}${count}`, folder.path));
    });
    if (state.sortFolders.some(folder => folder.path === selectedTarget)) targetSelect.value = selectedTarget;

    const list = $('#sort-control-list');
    list.replaceChildren();
    if (!state.sortFolders.length) {
      const empty = document.createElement('div');
      empty.className = 'sort-folder-loading';
      empty.textContent = 'No image folders found';
      list.append(empty);
    } else {
      state.sortFolders.forEach(folder => {
        const label = document.createElement('label');
        label.className = 'sort-folder-option';
        label.innerHTML = '<input type="checkbox"><span class="sort-folder-check"><svg><use href="#i-check"></use></svg></span><span class="sort-folder-copy"><b></b><small></small></span><span class="sort-folder-count"></span>';
        const input = $('input', label);
        input.value = folder.path;
        input.checked = selectedControls.has(folder.path);
        $('b', label).textContent = folder.name;
        $('small', label).textContent = folder.path;
        $('.sort-folder-count', label).textContent = formatNumber(folder.imageCount);
        list.append(label);
      });
    }
    $('#sort-root-label').textContent = state.sortRoot || 'Library folders';
    $('#sort-root-label').title = state.sortRoot || '';
    syncSortFolderAvailability();
  }

  function renderSortProfiles() {
    const select = $('#sort-profile-select');
    const selected = select.value;
    select.replaceChildren(new Option('Custom setup', ''));
    state.sortProfiles.forEach(profile => select.add(new Option(profile.name, profile.name)));
    if (state.sortProfiles.some(profile => profile.name === selected)) select.value = selected;
    $('#sort-profile-delete').disabled = !select.value;
  }

  function syncSortFolderAvailability() {
    const target = $('#sort-target').value;
    $$('.sort-folder-option').forEach(option => {
      const input = $('input', option);
      const isTarget = input.value === target;
      option.classList.toggle('is-target', isTarget);
      if (isTarget) input.checked = false;
    });
    const ready = Boolean(target) && state.sortFolders.length > 0;
    $('#sort-start').disabled = !ready;
    $('#sort-rescan').disabled = !ready;
  }

  function updateSortMode() {
    const mode = $('input[name="sort-mode"]:checked')?.value || 'time';
    const disabled = mode === 'stem';
    $('#sort-threshold').disabled = disabled;
    $('.sort-threshold-field').classList.toggle('is-disabled', disabled);
    $('#sort-control-help').textContent = disabled
      ? 'Select one or more folders for filename matching.'
      : 'Optional — leave empty to compare sibling folders automatically.';
  }

  function getSortConfig({ validate = false } = {}) {
    const targetDirectory = $('#sort-target').value;
    const controlDirectories = $$('#sort-control-list input:checked').map(input => input.value).filter(path => path !== targetDirectory);
    const mode = $('input[name="sort-mode"]:checked')?.value || 'time';
    if (validate && !targetDirectory) {
      toast('Choose a target folder', 'Select the folder whose images you want to classify.', 'info');
      $('#sort-target').focus();
      return null;
    }
    if (validate && mode === 'stem' && !controlDirectories.length) {
      toast('Choose a reference folder', 'Select at least one folder to compare with the target.', 'info');
      $('#sort-control-list input:not(:disabled)')?.focus();
      return null;
    }
    return {
      target_directory: targetDirectory,
      control_directories: controlDirectories,
      mode,
      threshold_seconds: Math.max(0, Number($('#sort-threshold').value || 0)),
      add_ids: $('#sort-add-ids').checked
    };
  }

  function applySortOptions(options) {
    if (!options) return;
    $('#sort-target').value = options.targetDirectory;
    $$('#sort-control-list input').forEach(input => { input.checked = options.controlDirectories.includes(input.value); });
    const mode = $(`input[name="sort-mode"][value="${options.mode}"]`);
    if (mode) mode.checked = true;
    $('#sort-threshold').value = options.thresholdSeconds;
    $('#sort-add-ids').checked = options.addIds;
    syncSortFolderAvailability();
    updateSortMode();
  }

  function applySortProfile(profile) {
    if (!profile) return;
    applySortOptions(profile);
    $('#sort-profile-name').value = profile.name;
  }

  async function loadSortWorkspace({ quiet = false, restoreSession = true } = {}) {
    if (state.sortLoading) return;
    state.sortLoading = true;
    try {
      const [folderData, profileData] = await Promise.all([
        api('/api/sort/folders'),
        api('/api/sort/profiles')
      ]);
      state.sortRoot = folderData?.root || '';
      state.sortFolders = apiItems(folderData).map(normalizeSortFolder).filter(folder => folder.path);
      state.sortProfiles = apiItems(profileData).map(normalizeSortProfile).filter(profile => profile.name);
      state.sortLoaded = true;
      renderSortFolders();
      renderSortProfiles();
      if (restoreSession && state.sortSessionId) await loadSortSession({ quiet: true });
      else renderSortSession();
    } catch (error) {
      state.sortLoaded = false;
      $('#sort-root-label').textContent = 'Folders unavailable';
      if (!quiet) toast('Could not load sorter', errorMessage(error), 'error');
    } finally {
      state.sortLoading = false;
    }
  }

  async function loadSortSession({ quiet = false } = {}) {
    if (!state.sortSessionId) {
      state.sortSession = null;
      renderSortSession();
      return;
    }
    try {
      const data = await api(`/api/sort/sessions/${encodeURIComponent(state.sortSessionId)}`);
      state.sortSession = normalizeSortSession(data?.session || data);
      if (state.sortSession?.id) {
        state.sortSessionId = state.sortSession.id;
        storage.set('sort-session', state.sortSessionId);
        applySortOptions(state.sortSession);
      }
      renderSortSession();
    } catch (error) {
      if (error.status === 404) {
        state.sortSession = null;
        state.sortSessionId = '';
        storage.set('sort-session', '');
        renderSortSession();
      } else if (!quiet) toast('Could not restore sorting session', errorMessage(error), 'error');
    }
  }

  function formatSortDate(value) {
    if (!value) return 'Current target';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return 'Current target';
    return date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  }

  function formatSortDelta(value, mode) {
    if (value === null || Number.isNaN(value)) return mode === 'stem' ? 'Filename' : 'Nearby';
    const seconds = Math.abs(value);
    if (seconds < 60) return `Δ ${seconds.toFixed(seconds < 10 && seconds % 1 ? 1 : 0)}s`;
    if (seconds < 3600) return `Δ ${(seconds / 60).toFixed(1)}m`;
    return `Δ ${(seconds / 3600).toFixed(1)}h`;
  }

  function renderSortSession() {
    const session = state.sortSession;
    const welcome = $('#sort-welcome');
    const decision = $('#sort-decision');
    const complete = $('#sort-complete');
    const progressWrap = $('#sort-progress-wrap');
    if (!session) {
      welcome.hidden = false;
      decision.hidden = true;
      complete.hidden = true;
      progressWrap.hidden = true;
      $('#sort-session-label').textContent = 'Ready when you are';
      $('#sort-welcome-title').textContent = 'Build a focused sorting queue';
      $('#sort-welcome-copy').textContent = 'Choose your folders and matching method, then start. Your active session is remembered on this browser.';
      $('#sort-undo').disabled = true;
      return;
    }

    const isComplete = ['complete', 'completed', 'done'].includes(session.status) || (!session.current && session.total > 0 && session.remaining === 0);
    welcome.hidden = Boolean(session.current) || isComplete;
    decision.hidden = !session.current;
    complete.hidden = !isComplete;
    progressWrap.hidden = false;
    const handled = session.processed + session.missing;
    const percentage = session.total ? Math.min(100, Math.round((handled / session.total) * 100)) : 0;
    $('#sort-processed').textContent = formatNumber(handled);
    $('#sort-total').textContent = formatNumber(session.total);
    $('#sort-remaining').textContent = `${formatNumber(session.remaining)} remaining`;
    $('#sort-progress-bar').style.width = `${percentage}%`;
    $('.sort-progress').setAttribute('aria-valuenow', String(percentage));
    $('#sort-session-label').textContent = session.recovering
      ? `Recovery pending · ${formatNumber(session.recovering)} ${session.recovering === 1 ? 'file' : 'files'}`
      : `${session.mode === 'stem' ? 'Filename' : 'Time'} matching · ${session.controlDirectories.length} reference ${session.controlDirectories.length === 1 ? 'folder' : 'folders'}`;
    $('#sort-welcome-title').textContent = session.recovering ? 'A file operation needs attention' : 'Build a focused sorting queue';
    $('#sort-welcome-copy').textContent = session.recovering
      ? 'The server preserved an ambiguous or temporarily unavailable file operation. Resolve the original/destination files on the server, then refresh this view.'
      : 'Choose your folders and matching method, then start. Your active session is remembered on this browser.';
    if (session.recovering) {
      $('#sort-start').disabled = true;
      $('#sort-rescan').disabled = true;
    }
    $('#sort-undo').disabled = !session.canUndo || state.sortBusy;
    $('#sort-complete-summary').textContent = session.status === 'superseded'
      ? 'A newer scan replaced this session.'
      : `${formatNumber(session.processed)} ${session.processed === 1 ? 'target has' : 'targets have'} been reviewed.${session.missing ? ` ${formatNumber(session.missing)} missing ${session.missing === 1 ? 'file was' : 'files were'} skipped.` : ''}`;

    if (!session.current) return;
    const current = session.current;
    const targetImage = $('#sort-target-preview');
    targetImage.previousElementSibling?.classList.remove('is-loaded');
    loadImage(targetImage, current.previewUrl, current.name);
    $('#sort-target-date').textContent = formatSortDate(current.modifiedAt);
    $('#sort-target-name').textContent = current.name;
    $('#sort-target-path').textContent = current.path;
    $('#sort-target-path').title = current.path;
    $('#sort-current-count').textContent = `${Math.min(session.total, handled + 1)} / ${session.total}`;
    $('#sort-match-summary').textContent = session.matches.length ? `${session.matches.length} likely ${session.matches.length === 1 ? 'match' : 'matches'}, ranked closest first.` : 'No likely reference image was found.';

    const grid = $('#sort-match-grid');
    grid.replaceChildren();
    if (!session.matches.length) {
      const empty = document.createElement('div');
      empty.className = 'sort-match-empty';
      empty.innerHTML = '<svg><use href="#i-search"></use></svg><span>No candidates inside this threshold</span>';
      grid.append(empty);
    } else {
      session.matches.forEach((match, index) => {
        const card = document.createElement('article');
        card.className = `sort-match-card${index === 0 ? ' is-best' : ''}`;
        card.innerHTML = '<div class="sort-match-media"><div class="image-placeholder"><svg><use href="#i-image"></use></svg></div><img alt="" loading="lazy"><span class="sort-match-rank"></span></div><div class="sort-match-copy"><div class="sort-match-meta"><span class="sort-match-folder"></span><span class="sort-match-delta"></span></div><h4></h4><p></p><div class="sort-match-actions"><button class="button sort-match-button" data-sort-kind="match" type="button"><svg><use href="#i-check"></use></svg><span>Match</span></button><button class="button sort-solo-button" data-sort-kind="solo" type="button"><svg><use href="#i-image"></use></svg><span>Solo</span></button></div></div>';
        loadImage($('img', card), match.previewUrl, match.name);
        $('.sort-match-rank', card).textContent = index === 0 ? 'Best match' : `Candidate ${index + 1}`;
        $('.sort-match-folder', card).textContent = match.folder || 'Reference';
        $('.sort-match-folder', card).title = match.folder || '';
        $('.sort-match-delta', card).textContent = formatSortDelta(match.deltaSeconds, session.mode);
        $('h4', card).textContent = match.name;
        $('p', card).textContent = match.path;
        $('p', card).title = match.path;
        $$('.sort-match-actions button', card).forEach(button => { button.dataset.sortMatchIndex = String(index); });
        grid.append(card);
      });
    }
    $$('.sort-action-bar .button, .sort-match-actions .button').forEach(button => { button.disabled = state.sortBusy; });
  }

  async function startSortSession(button = $('#sort-start')) {
    const config = getSortConfig({ validate: true });
    if (!config) return;
    setButtonBusy(button, true, 'Scanning…');
    try {
      const data = await api('/api/sort/sessions', { method: 'POST', body: config });
      state.sortSession = normalizeSortSession(data?.session || data);
      state.sortSessionId = state.sortSession?.id || '';
      storage.set('sort-session', state.sortSessionId);
      renderSortSession();
      toast('Sorting queue ready', `${formatNumber(state.sortSession?.total || 0)} targets found.`, 'success');
    } catch (error) {
      toast('Could not start sorting', errorMessage(error), 'error');
    } finally {
      setButtonBusy(button, false);
      syncSortFolderAvailability();
    }
  }

  async function performSortAction(kind, controlPath = null, button = null) {
    const session = state.sortSession;
    if (!session?.current || state.sortBusy) return;
    state.sortBusy = true;
    renderSortSession();
    if (button) setButtonBusy(button, true, 'Applying…');
    const body = { kind, expected_target: session.current.path };
    if (controlPath) body.control_path = controlPath;
    try {
      const data = await api(`/api/sort/sessions/${encodeURIComponent(session.id)}/actions`, { method: 'POST', body });
      state.sortSession = normalizeSortSession(data?.session || data);
      announce(kind === 'skip' ? 'Target skipped' : 'Sort decision applied');
    } catch (error) {
      toast('Could not apply decision', errorMessage(error), 'error');
      if (error.status === 409) await loadSortSession({ quiet: true });
    } finally {
      state.sortBusy = false;
      if (button) setButtonBusy(button, false);
      renderSortSession();
    }
  }

  async function undoSortAction() {
    const session = state.sortSession;
    if (!session?.canUndo || state.sortBusy) return;
    const button = $('#sort-undo');
    state.sortBusy = true;
    setButtonBusy(button, true, 'Undoing…');
    try {
      const data = await api(`/api/sort/sessions/${encodeURIComponent(session.id)}/undo`, { method: 'POST' });
      state.sortSession = normalizeSortSession(data?.session || data);
      toast('Last decision undone', 'The previous target is back on the desk.', 'info');
    } catch (error) {
      toast('Could not undo decision', errorMessage(error), 'error');
    } finally {
      state.sortBusy = false;
      setButtonBusy(button, false);
      renderSortSession();
    }
  }

  async function saveSortProfile() {
    const name = $('#sort-profile-name').value.trim();
    if (!name) {
      toast('Name this setup', 'Enter a short name before saving.', 'info');
      $('#sort-profile-name').focus();
      return;
    }
    const config = getSortConfig({ validate: true });
    if (!config) return;
    const button = $('#sort-profile-save');
    setButtonBusy(button, true, 'Saving…');
    try {
      await api('/api/sort/profiles', { method: 'POST', body: { name, ...config } });
      const data = await api('/api/sort/profiles');
      state.sortProfiles = apiItems(data).map(normalizeSortProfile).filter(profile => profile.name);
      renderSortProfiles();
      $('#sort-profile-select').value = name;
      $('#sort-profile-delete').disabled = false;
      toast('Setup saved', name, 'success');
    } catch (error) {
      toast('Could not save setup', errorMessage(error), 'error');
    } finally {
      setButtonBusy(button, false);
    }
  }

  async function deleteSortProfile() {
    const name = $('#sort-profile-select').value;
    if (!name) return;
    const button = $('#sort-profile-delete');
    setButtonBusy(button, true, 'Deleting…');
    try {
      await api(`/api/sort/profiles/${encodeURIComponent(name)}`, { method: 'DELETE' });
      state.sortProfiles = state.sortProfiles.filter(profile => profile.name !== name);
      renderSortProfiles();
      $('#sort-profile-select').value = '';
      $('#sort-profile-name').value = '';
      button.disabled = true;
      toast('Saved setup deleted', name, 'info');
    } catch (error) {
      toast('Could not delete setup', errorMessage(error), 'error');
    } finally {
      setButtonBusy(button, false);
      button.disabled = !$('#sort-profile-select').value;
    }
  }

  function setView(view, { updateHash = true } = {}) {
    if (!['discover', 'finder', 'queue', 'profiles', 'sort', 'settings'].includes(view)) view = 'discover';
    state.view = view;
    $$('[data-view-panel]').forEach(panel => {
      const active = panel.dataset.viewPanel === view;
      panel.hidden = !active;
      panel.classList.toggle('is-active', active);
    });
    $$('[data-view]').forEach(button => {
      const active = button.dataset.view === view;
      button.classList.toggle('is-active', active);
      if (button.classList.contains('nav-item')) {
        if (active) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
      }
    });
    if (updateHash) history.replaceState(null, '', `#${view}`);
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
    if (view === 'finder' && !state.finderLoaded) loadFinderWorkspace();
    if (view === 'queue') loadJobs({ quiet: true });
    if (view === 'profiles') loadProfiles({ quiet: true });
    if (view === 'sort' && !state.sortLoaded) loadSortWorkspace();
    if (view === 'settings' && !Object.keys(state.settings).length) loadSettings();
    announce(`${view[0].toUpperCase()}${view.slice(1)} view`);
  }

  async function refreshCurrent() {
    $('#refresh-button').classList.add('is-spinning');
    if (state.view === 'discover') await loadGalleries();
    else if (state.view === 'finder') await loadFinderWorkspace({ preserveConfig: true });
    else if (state.view === 'queue') await loadJobs();
    else if (state.view === 'profiles') await loadProfiles();
    else if (state.view === 'sort') await loadSortWorkspace({ restoreSession: true });
    else await Promise.all([loadSettings(), checkHealth(false)]);
    $('#refresh-button').classList.remove('is-spinning');
  }

  function closeModal(dialog) {
    if (
      dialog?.id === 'gallery-modal'
      && (state.finderFeedbackGalleryDirty || state.finderFeedbackGallerySaving)
      && !confirmDiscardFinderFeedbackGalleryChanges('Close the gallery and discard the unsaved Finder feedback selection?')
    ) return false;
    if (dialog?.open) dialog.close();
    return true;
  }

  function syncFilterControls() {
    $('#show-saved').checked = state.filters.showSaved;
    $('#show-ignored').checked = state.filters.showIgnored;
    const altered = Number(!state.filters.showSaved) + Number(state.filters.showIgnored);
    $('#filter-count').hidden = !altered;
    $('#filter-count').textContent = String(altered);
  }

  function bindEvents() {
    $$('[data-view]').forEach(button => button.addEventListener('click', () => setView(button.dataset.view)));
    $$('[data-route]').forEach(link => link.addEventListener('click', event => {
      event.preventDefault();
      setView(link.dataset.route);
    }));
    $$('[data-go-discover]').forEach(button => button.addEventListener('click', () => setView('discover')));
    $('#source-form').addEventListener('submit', handleSourceSubmit);
    $$('.mode-button').forEach(button => button.addEventListener('click', () => setBrowseMode(button.dataset.mode)));
    $('#source-input').addEventListener('input', event => { $('#clear-source').hidden = !event.target.value; });
    $('#clear-source').addEventListener('click', () => {
      $('#source-input').value = '';
      $('#clear-source').hidden = true;
      $('#source-input').focus();
    });
    $('#empty-focus-source').addEventListener('click', () => $('#source-input').focus());
    $('#discover-notice button').addEventListener('click', hideNotice);
    $('#filter-trigger').addEventListener('click', event => {
      event.stopPropagation();
      const popover = $('#filter-popover');
      popover.hidden = !popover.hidden;
      $('#filter-trigger').setAttribute('aria-expanded', String(!popover.hidden));
    });
    $('#filter-popover').addEventListener('click', event => event.stopPropagation());
    document.addEventListener('click', () => {
      $('#filter-popover').hidden = true;
      $('#filter-trigger').setAttribute('aria-expanded', 'false');
      $$('.card-menu-popover').forEach(menu => { menu.hidden = true; });
    });
    $('#show-saved').addEventListener('change', event => {
      state.filters.showSaved = event.target.checked;
      storage.set('filters', state.filters);
      syncFilterControls();
      state.page = 1;
      loadGalleries({ quiet: true });
    });
    $('#show-ignored').addEventListener('change', event => {
      state.filters.showIgnored = event.target.checked;
      storage.set('filters', state.filters);
      syncFilterControls();
      state.page = 1;
      loadGalleries({ quiet: true });
    });
    $$('.density-switch button').forEach(button => button.addEventListener('click', () => {
      state.density = button.dataset.density;
      storage.set('density', state.density);
      $$('.density-switch button').forEach(item => item.classList.toggle('is-active', item === button));
      renderGalleries();
    }));
    $('#gallery-grid').addEventListener('click', event => {
      const open = event.target.closest('[data-gallery-id]');
      const ignore = event.target.closest('.card-ignore');
      if (ignore) {
        event.stopPropagation();
        const card = ignore.closest('.gallery-card');
        const gallery = state.galleries.find(item => String(item.id) === card.dataset.galleryId);
        if (gallery) toggleIgnore(gallery, ignore);
      } else if (open) openGallery(open.dataset.galleryId);
    });
    $('#page-next').addEventListener('click', loadMoreGalleries);
    $('#finder-refresh').addEventListener('click', async () => {
      await loadFinderWorkspace({ preserveConfig: true });
      await loadFinderFeedback({ quiet: true, force: true });
    });
    $('#finder-use-current').addEventListener('click', () => {
      $('#finder-source').value = finderDefaultSource();
      syncFinderConfigAvailability();
      $('#finder-source').focus();
    });
    $$('[data-finder-mode]').forEach(button => button.addEventListener('click', () => {
      setFinderMode(button.dataset.finderMode);
    }));
    const handleFinderFolderInput = () => {
      const current = finderFolderKey($('#finder-folder').value);
      const analyzed = finderFolderKey(state.finderReferenceAnalysisSource);
      if (
        (
          state.finderReferenceAnalysis
          || state.finderReferenceAnalysisLoading
          || state.finderReferenceAnalysisError
        )
        && current !== analyzed
      ) {
        invalidateFinderReferenceAnalysis();
      } else {
        syncFinderConfigAvailability();
      }
    };
    $('#finder-folder').addEventListener('input', handleFinderFolderInput);
    $('#finder-folder').addEventListener('change', handleFinderFolderInput);
    ['finder-pose-tag', 'finder-source', 'finder-pages'].forEach(id => {
      $(`#${id}`).addEventListener('input', syncFinderConfigAvailability);
      $(`#${id}`).addEventListener('change', syncFinderConfigAvailability);
    });
    $('#finder-analyze-references').addEventListener('click', analyzeFinderReferences);
    $('#finder-joytag-dataset-label').addEventListener('input', event => {
      const value = event.currentTarget.value;
      $('#finder-pose-tag').value = value;
      state.finderJoytagAutoPoseLabel = '';
      syncFinderConfigAvailability();
    });
    $('#finder-joytag-dataset-label').addEventListener('change', event => {
      const normalized = event.currentTarget.value.trim().replace(/\s+/g, ' ');
      event.currentTarget.value = normalized;
      $('#finder-pose-tag').value = normalized;
      syncFinderConfigAvailability();
    });
    $('#finder-joytag-tag-filter').addEventListener('input', event => {
      state.finderJoytagTagFilter = event.currentTarget.value;
      renderFinderReferenceAnalysis();
    });
    $('#finder-joytag-results').addEventListener('click', event => {
      const roleButton = event.target.closest('[data-finder-joytag-role]');
      if (roleButton) {
        setFinderJoytagTagRole(
          roleButton.dataset.finderJoytagTag,
          roleButton.dataset.finderJoytagRole
        );
        return;
      }
      const removeButton = event.target.closest('[data-finder-joytag-remove-role]');
      if (removeButton) {
        setFinderJoytagTagRole(
          removeButton.dataset.finderJoytagTag,
          '',
          { toggle: false }
        );
        return;
      }
      const inspectButton = event.target.closest('[data-finder-joytag-inspect]');
      if (inspectButton) inspectFinderJoytagTag(inspectButton.dataset.finderJoytagInspect);
    });
    $('#finder-joytag-threshold').addEventListener('input', event => {
      state.finderJoytagThreshold = Math.max(0.05, Math.min(0.95, Number(event.currentTarget.value || 0.4)));
      storage.set('finder-joytag-threshold', state.finderJoytagThreshold);
      renderFinderReferenceAnalysis();
      syncFinderConfigAvailability();
    });
    $('#finder-joytag-reject-threshold').addEventListener('input', event => {
      state.finderJoytagRejectThreshold = Math.max(
        0.05,
        Math.min(0.95, Number(event.currentTarget.value || 0.4))
      );
      storage.set('finder-joytag-reject-threshold', state.finderJoytagRejectThreshold);
      renderFinderReferenceAnalysis();
      syncFinderConfigAvailability();
    });
    $('#finder-pose-tag').addEventListener('input', () => scheduleFinderFeedbackLoad());
    $('#finder-pose-tag').addEventListener('change', () => loadFinderFeedback({ quiet: true, force: true }));
    $('#finder-feedback-reset').addEventListener('click', resetFinderFeedback);
    $('#finder-corpus-index-start').addEventListener('click', startFinderJoytagIndex);
    $('#finder-corpus-index-cancel').addEventListener('click', cancelFinderJoytagIndex);
    $('#finder-min-similarity').addEventListener('input', event => {
      $('#finder-min-output').textContent = Number(event.currentTarget.value).toFixed(2);
    });
    $('#finder-result-threshold').addEventListener('input', event => {
      $('#finder-filter-output').textContent = Number(event.currentTarget.value || 0).toFixed(2);
      state.finderResultPage = 1;
      renderFinderResults();
      scheduleFinderResultFilterLoad();
    });
    $('#finder-result-threshold').addEventListener('change', () => {
      window.clearTimeout(state.finderResultThresholdTimer);
      state.finderResultThresholdTimer = null;
      loadFinderResults({ quiet: true, page: 1 });
    });
    $('#finder-scan-select').addEventListener('change', event => selectFinderScan(event.currentTarget.value));
    $('#finder-start').addEventListener('click', startFinderScan);
    $('#finder-extend-pages').addEventListener('input', () => updateFinderExtendSummary());
    $('#finder-extend-pages').addEventListener('change', () => updateFinderExtendSummary({ commit: true }));
    $('#finder-extend-pages').addEventListener('keydown', event => {
      if (event.key === 'Enter') extendFinderScan();
    });
    $('#finder-extend-button').addEventListener('click', extendFinderScan);
    $('#finder-continue-pages').addEventListener('input', () => updateFinderContinueSummary());
    $('#finder-continue-pages').addEventListener('change', () => updateFinderContinueSummary({ commit: true }));
    $('#finder-continue-pages').addEventListener('keydown', event => {
      if (event.key === 'Enter') continueFinderScan();
    });
    $('#finder-continue-source').addEventListener('keydown', event => {
      if (event.key === 'Enter') continueFinderScan();
    });
    $('#finder-continue-button').addEventListener('click', continueFinderScan);
    $('#finder-pause').addEventListener('click', event => performFinderScanAction('pause', event.currentTarget));
    $('#finder-resume').addEventListener('click', event => {
      performFinderScanAction(state.finderScan?.status === 'failed' ? 'retry' : 'resume', event.currentTarget);
    });
    $('#finder-cancel').addEventListener('click', cancelFinderScan);
    $$('[data-finder-review]').forEach(button => button.addEventListener('click', () => {
      if (button.dataset.finderReview === state.finderReview) return;
      state.finderReview = button.dataset.finderReview;
      state.finderResultPage = 1;
      const total = Math.max(0, Number(state.finderReviewCounts?.[state.finderReview] || 0));
      state.finderResultTotal = total;
      state.finderResultPageCount = Math.max(1, Math.ceil(total / FINDER_RESULTS_PAGE_SIZE));
      renderFinderResults();
      loadFinderResults({ quiet: true, page: 1 });
    }));
    $('#finder-page-previous').addEventListener('click', async () => {
      const currentPage = state.finderResultPage;
      await loadFinderResults({ page: currentPage - 1 });
      if (state.finderResultPage !== currentPage) scrollToFinderResults();
    });
    $('#finder-page-next').addEventListener('click', async () => {
      const currentPage = state.finderResultPage;
      await loadFinderResults({ page: currentPage + 1 });
      if (state.finderResultPage !== currentPage) scrollToFinderResults();
    });
    $('#finder-result-grid').addEventListener('click', event => {
      const button = event.target.closest('[data-finder-action]');
      if (!button) return;
      const result = state.finderResults.find(item => String(item.key) === String(button.dataset.finderResult));
      if (!result) return;
      const action = button.dataset.finderAction;
      if (action === 'overlay') {
        const card = button.closest('.finder-card');
        const visible = !card.classList.contains('is-overlay-visible');
        card.classList.toggle('is-overlay-visible', visible);
        button.setAttribute('aria-pressed', String(visible));
        $('span', button).textContent = visible ? 'Hide overlay' : 'Pose overlay';
        const use = $('use', button);
        if (use) use.setAttribute('href', visible ? '#i-eye-off' : '#i-eye');
      } else if (action === 'open') openFinderResult(result);
      else if (action === 'save-selection') reviewFinderResult(result, result.review, button);
      else reviewFinderResult(result, action, button);
    });
    $('#finder-result-grid').addEventListener('change', event => {
      const input = event.target.closest('input[data-finder-feedback-match]');
      if (input) toggleFinderFeedbackMatch(input);
    });
    $('#active-profile').addEventListener('change', event => selectProfile(event.target.value));
    $('#modal-profile-select').addEventListener('change', event => selectProfile(event.target.value, false));
    $('#refresh-button').addEventListener('click', refreshCurrent);
    $('#refresh-queue').addEventListener('click', () => loadJobs());
    $('#clear-completed').addEventListener('click', clearCompleted);
    $$('.segmented [data-job-filter]').forEach(button => button.addEventListener('click', () => {
      state.jobFilter = button.dataset.jobFilter;
      $$('.segmented [data-job-filter]').forEach(item => item.classList.toggle('is-active', item === button));
      renderJobs();
    }));
    $('#job-list').addEventListener('click', event => {
      const remove = event.target.closest('.job-remove');
      if (!remove) return;
      const id = remove.closest('.job-row').dataset.jobId;
      const job = state.jobs.find(item => String(item.id) === id);
      if (job) removeJob(job);
    });
    $('#new-profile').addEventListener('click', () => openProfileModal());
    $('#empty-new-profile').addEventListener('click', () => openProfileModal());
    $('#profile-name').addEventListener('input', event => {
      if (!$('#profile-id').value) $('#profile-folder').value = safeFolderPreview(event.target.value);
    });
    $('#profile-form').addEventListener('submit', saveProfile);
    $('#profile-grid').addEventListener('click', event => {
      const card = event.target.closest('.profile-card');
      if (!card) return;
      const profile = state.profiles.find(item => item.name === card.dataset.profile);
      if (!profile) return;
      const trigger = event.target.closest('.profile-menu-trigger');
      if (trigger) {
        event.stopPropagation();
        const popover = $('.card-menu-popover', card);
        const wasHidden = popover.hidden;
        $$('.card-menu-popover').forEach(menu => { menu.hidden = true; });
        popover.hidden = !wasHidden;
        return;
      }
      const action = event.target.closest('[data-profile-action]')?.dataset.profileAction;
      if (action) {
        event.stopPropagation();
        if (action === 'edit') openProfileModal(profile);
        else if (action === 'default') selectProfile(profile.name);
        else if (action === 'delete') deleteProfile(profile);
        return;
      }
      if (event.target.closest('.profile-use')) selectProfile(profile.name);
    });
    $('#sort-target').addEventListener('change', syncSortFolderAvailability);
    $('#sort-control-list').addEventListener('change', event => {
      if (event.target.matches('input[type="checkbox"]')) $('#sort-profile-select').value = '';
    });
    $$('input[name="sort-mode"]').forEach(input => input.addEventListener('change', () => {
      updateSortMode();
      $('#sort-profile-select').value = '';
      $('#sort-profile-delete').disabled = true;
    }));
    $('#sort-profile-select').addEventListener('change', event => {
      const profile = state.sortProfiles.find(item => item.name === event.target.value);
      $('#sort-profile-delete').disabled = !profile;
      if (profile) applySortProfile(profile);
      else $('#sort-profile-name').value = '';
    });
    $('#sort-start').addEventListener('click', event => startSortSession(event.currentTarget));
    $('#sort-rescan').addEventListener('click', event => startSortSession(event.currentTarget));
    $('#sort-complete-rescan').addEventListener('click', event => startSortSession(event.currentTarget));
    $('#sort-profile-save').addEventListener('click', saveSortProfile);
    $('#sort-profile-delete').addEventListener('click', deleteSortProfile);
    $('#sort-match-grid').addEventListener('click', event => {
      const button = event.target.closest('[data-sort-kind]');
      if (!button || !state.sortSession) return;
      const match = state.sortSession.matches[Number(button.dataset.sortMatchIndex)];
      if (match) performSortAction(button.dataset.sortKind, match.path, button);
    });
    $('#sort-action-none').addEventListener('click', event => performSortAction('no_control', null, event.currentTarget));
    $('#sort-action-skip').addEventListener('click', event => performSortAction('skip', null, event.currentTarget));
    $('#sort-undo').addEventListener('click', undoSortAction);
    $('#settings-form').addEventListener('submit', saveSettings);
    $('#settings-form').addEventListener('input', () => { $('#settings-status').textContent = 'You have unsaved changes.'; });
    $('#show-shortcuts').addEventListener('click', () => $('#shortcut-modal').showModal());
    $$('[data-close-modal]').forEach(button => button.addEventListener('click', () => closeModal(button.closest('dialog'))));
    $$('dialog').forEach(dialog => dialog.addEventListener('click', event => {
      if (event.target === dialog) closeModal(dialog);
    }));
    $('#select-all').addEventListener('click', () => selectAllImages(true));
    $('#select-none').addEventListener('click', () => selectAllImages(false));
    $$('[data-gallery-mode]').forEach(button => button.addEventListener('click', () => setGalleryMode(button.dataset.galleryMode)));
    $$('[data-gallery-finder-review]').forEach(button => button.addEventListener('click', () => {
      reviewFinderGallery(button.dataset.galleryFinderReview, button);
    }));
    $('#gallery-review-previous').addEventListener('click', () => navigateFinderGallery(-1));
    $('#gallery-review-next').addEventListener('click', () => navigateFinderGallery(1));
    $('#finder-feedback-gallery-save').addEventListener('click', saveFinderGalleryFeedbackSelection);
    $('#finder-feedback-prepare-pose').addEventListener('click', prepareFinderPoseFromFeedback);
    $$('[data-pose-assignment]').forEach(button => button.addEventListener('click', () => handlePoseAssignmentButton(button)));
    $('#pose-tag-input').addEventListener('input', event => syncPoseTagDefault(event.currentTarget, $('#pose-control-role')));
    $('#pose-tag-input').addEventListener('change', event => syncPoseTagDefault(event.currentTarget, $('#pose-control-role')));
    $('#pose-control-role').addEventListener('change', updateSelectionUi);
    $('#pose-apply-checked').addEventListener('click', event => applyPoseAssignment(
      state.poseSelectedImages,
      state.poseAssignment,
      { button: event.currentTarget, clearChecked: true }
    ));
    $('#pose-clear-checked').addEventListener('click', clearCheckedPoseAssignments);
    $('#image-grid').addEventListener('change', event => {
      if (event.target.matches('input[type="checkbox"]')) toggleImage(event.target.closest('.image-option').dataset.imageUrl, event.target.checked);
    });
    $('#image-grid').addEventListener('click', event => {
      const trigger = event.target.closest('.image-preview-button');
      if (!trigger) return;
      const option = trigger.closest('.image-option');
      openLightbox(Number(option.dataset.imageIndex), trigger);
    });
    $('#lightbox-previous').addEventListener('click', () => navigateLightbox(-1));
    $('#lightbox-next').addEventListener('click', () => navigateLightbox(1));
    $('#lightbox-zoom').addEventListener('click', () => setLightboxZoom(!state.lightboxZoomed));
    $('#lightbox-stage').addEventListener('click', event => {
      if (event.target.matches('#lightbox-image')) setLightboxZoom(!state.lightboxZoomed);
    });
    $$('[data-lightbox-control]').forEach(button => button.addEventListener('click', event => {
      const image = state.gallery?.images?.[state.lightboxIndex];
      if (image) applyPoseAssignment([image.url], event.currentTarget.dataset.lightboxControl, { button: event.currentTarget });
    }));
    $('#lightbox-pose-tag-input').addEventListener('input', event => syncPoseTagDefault(event.currentTarget, $('#lightbox-pose-control-role')));
    $('#lightbox-pose-tag-input').addEventListener('change', event => syncPoseTagDefault(event.currentTarget, $('#lightbox-pose-control-role')));
    $('#lightbox-pose-control-role').addEventListener('change', updateLightboxTargetAvailability);
    $('#lightbox-set-target').addEventListener('click', event => {
      const image = state.gallery?.images?.[state.lightboxIndex];
      if (image) applyPoseAssignment([image.url], 'target', { button: event.currentTarget });
    });
    $('#lightbox-clear-pose').addEventListener('click', () => {
      const image = state.gallery?.images?.[state.lightboxIndex];
      if (!image || !poseAssignmentFor(image.url)) return;
      const controlRole = poseControlFor(image.url);
      if (controlRole && state.poseDraft.targets.some(target => target.role === controlRole)) {
        toast('Control is still in use', `Replace the ${poseRoleLabel(controlRole).toLowerCase()} control before removing it.`, 'info');
        return;
      }
      clearPoseAssignment(image.url);
      markPoseDraftDirty();
      renderImages();
      renderPoseWorkspace();
      announce(`${image.filename} assignment removed`);
    });
    $('#lightbox-modal').addEventListener('close', resetLightbox);
    $('#gallery-modal').addEventListener('close', () => {
      state.galleryDetailRequest += 1;
      state.galleryNavigationRequest += 1;
      state.poseApplyRequest += 1;
      state.poseExportRequest += 1;
      cancelAdjacentGalleryPrefetches({ includeForeground: true });
      state.loadingDetail = false;
      closeModal($('#lightbox-modal'));
      flushPoseDraft();
      state.galleryContext = null;
      state.finderFeedbackGallerySelection = new Set();
      state.finderFeedbackGalleryDirty = false;
      state.finderFeedbackGallerySaving = false;
      state.galleryReviewBusy = false;
      state.poseApplying = false;
      state.poseExporting = false;
      setButtonBusy($('#pose-export'), false);
      renderFinderGalleryReview();
      if (state.view === 'finder') renderFinderResults();
      else if (state.view === 'discover') renderGalleries();
    });
    $('#gallery-modal').addEventListener('cancel', event => {
      if (!state.finderFeedbackGalleryDirty && !state.finderFeedbackGallerySaving) return;
      event.preventDefault();
      closeModal(event.currentTarget);
    });
    $('#modal-ignore').addEventListener('click', () => {
      if (!state.gallery) return;
      const listItem = state.galleries.find(item => String(item.id) === String(state.gallery.id)) || state.gallery;
      toggleIgnore(listItem, $('#modal-ignore'));
    });
    $('#queue-download').addEventListener('click', queueGallery);
    $('#pose-export').addEventListener('click', exportPoseDataset);
    window.addEventListener('scroll', () => $('.topbar').classList.toggle('is-scrolled', window.scrollY > 4), { passive: true });
    window.addEventListener('hashchange', () => setView(location.hash.slice(1), { updateHash: false }));
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) {
        checkHealth();
        loadJobs({ quiet: true });
        if (state.finderScanId) loadFinderScan({ quiet: true });
      }
    });
    document.addEventListener('keydown', handleKeyboard);
  }

  function handleKeyboard(event) {
    const target = event.target;
    const editing = target.matches('input, textarea, select, [contenteditable="true"]');
    const galleryOpen = $('#gallery-modal').open;
    const lightboxOpen = $('#lightbox-modal').open;
    const anyDialogOpen = Boolean($('dialog[open]'));
    const key = event.key.toLowerCase();
    if (lightboxOpen && !editing && event.key === 'ArrowLeft') {
      event.preventDefault();
      navigateLightbox(-1);
    } else if (lightboxOpen && !editing && event.key === 'ArrowRight') {
      event.preventDefault();
      navigateLightbox(1);
    } else if (galleryOpen && !lightboxOpen && !editing && !$('#gallery-review-rail').hidden && event.key === 'ArrowLeft') {
      event.preventDefault();
      navigateFinderGallery(-1);
    } else if (galleryOpen && !lightboxOpen && !editing && !$('#gallery-review-rail').hidden && event.key === 'ArrowRight') {
      event.preventDefault();
      navigateFinderGallery(1);
    } else if (state.view === 'sort' && !editing && !anyDialogOpen && !event.ctrlKey && !event.metaKey && !event.altKey && ['z', 'n', 's'].includes(key)) {
      event.preventDefault();
      if (key === 'z' && !$('#sort-undo').disabled) undoSortAction();
      else if (key === 'n' && state.sortSession?.current) performSortAction('no_control', null, $('#sort-action-none'));
      else if (key === 's' && state.sortSession?.current) performSortAction('skip', null, $('#sort-action-skip'));
    } else if (event.key === '/' && !editing && !galleryOpen) {
      event.preventDefault();
      setView('discover');
      $('#source-input').focus();
    } else if (key === 'r' && !editing && !galleryOpen) {
      event.preventDefault();
      refreshCurrent();
    } else if (galleryOpen && !lightboxOpen && key === 'a' && !editing) {
      event.preventDefault();
      selectAllImages(true);
    } else if (galleryOpen && !lightboxOpen && state.galleryMode === 'pose' && event.key === 'Enter' && !editing && !$('#pose-apply-checked').disabled) {
      event.preventDefault();
      $('#pose-apply-checked').click();
    } else if (galleryOpen && !lightboxOpen && state.galleryMode === 'download' && event.key === 'Enter' && !editing && !$('#queue-download').disabled) {
      event.preventDefault();
      queueGallery();
    }
  }

  async function init() {
    bindEvents();
    setupGalleryAutoLoad();
    syncFilterControls();
    $$('.density-switch button').forEach(button => button.classList.toggle('is-active', button.dataset.density === state.density));
    renderGallerySkeletons();
    const hashView = location.hash.slice(1);
    setView(['discover', 'finder', 'queue', 'profiles', 'sort', 'settings'].includes(hashView) ? hashView : 'discover', { updateHash: !hashView });
    connectEvents();
    await Promise.all([checkHealth(), loadSettings(), loadProfiles({ quiet: true }), loadJobs({ quiet: true })]);
    await loadHistory();
    await loadGalleries({ quiet: true });
    window.clearInterval(state.healthTimer);
    state.healthTimer = window.setInterval(() => checkHealth(), 30000);
  }

  init();
})();
