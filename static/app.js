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
    historyUrls: new Set(),
    jobs: [],
    jobFilter: 'all',
    gallery: null,
    selectedImages: new Set(),
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
    healthTimer: null,
    requestController: null,
    galleryObserver: null,
    finderFolders: [],
    finderTags: [],
    finderStatus: null,
    finderCorpus: null,
    finderCorpusSupported: null,
    finderFeedback: null,
    finderFeedbackSupported: null,
    finderFeedbackLoading: false,
    finderFeedbackBusy: false,
    finderFeedbackMutations: 0,
    finderFeedbackError: '',
    finderFeedbackRequest: 0,
    finderFeedbackTimer: null,
    finderScans: [],
    finderScan: null,
    finderScanId: storage.get('finder-scan', ''),
    finderResults: [],
    finderReview: 'pending',
    finderLoaded: false,
    finderLoading: false,
    finderBusy: false,
    finderExtendPages: 5,
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
  const FINDER_RANKING_VERSION = 'pose-first-v1';
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

  function normalizeGallery(item) {
    const gallery = { ...item };
    const serverState = String(item.state || '').toLowerCase();
    gallery.id = item.id ?? item.gallery_id ?? item.url;
    gallery.url = item.url || item.gallery_url || '';
    gallery.title = item.title || item.name || 'Untitled gallery';
    gallery.thumbnailUrl = item.thumbnail_url || item.thumbnail || item.preview_url || '';
    gallery.imageCount = Number(item.total_images || item.image_count || item.count || (Array.isArray(item.images) ? item.images.length : 0));
    gallery.downloadedImages = Number(item.downloaded_images || 0);
    gallery.serverState = serverState || 'new';
    gallery.saved = Boolean(item.saved || item.downloaded || serverState === 'complete' || serverState === 'saved' || state.historyUrls.has(normalizeHistoryUrl(gallery.url)));
    gallery.partial = Boolean(item.partial || serverState === 'partial' || (gallery.downloadedImages && !gallery.saved));
    gallery.ignored = Boolean(item.ignored || serverState === 'ignored');
    gallery.queued = serverState === 'queued' || state.jobs.some(job => !isTerminalJob(job) && (
      String(job.galleryId) === String(gallery.id) ||
      (job.galleryUrl && normalizeHistoryUrl(job.galleryUrl) === normalizeHistoryUrl(gallery.url))
    ));
    return gallery;
  }

  function normalizeDetail(item) {
    const gallery = normalizeGallery(item || {});
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
          gallery.ignored = change.ignored;
          renderGalleries();
        } else if (state.view === 'discover') loadGalleries({ quiet: true });
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

  async function loadProfiles({ quiet = false } = {}) {
    try {
      const data = await api('/api/profiles');
      state.profiles = apiItems(data, 'profiles').map(normalizeProfile);
      if (state.profiles.length && !state.profiles.some(profile => profile.name === state.activeProfile)) {
        state.activeProfile = data?.default_profile || state.settings.default_profile || state.profiles[0].name;
        storage.set('active-profile', state.activeProfile);
      }
      if (!state.profiles.length) state.activeProfile = '';
      renderProfileSelectors();
      renderProfiles();
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
    state.activeProfile = name;
    storage.set('active-profile', name);
    renderProfileSelectors();
    $('#active-profile').value = name;
    $('#modal-profile-select').value = name;
    renderProfiles();
    if ($('#gallery-modal').open && state.gallery) {
      state.poseLoadedKey = '';
      state.poseDraft = { revision: 0, controls: { solo: '', couple: '', group: '' }, targets: [] };
      state.poseSelectedImages = new Set();
      renderImages();
      if (state.galleryMode === 'pose') loadPoseWorkspace();
    }
    if (reload) {
      await Promise.all([loadHistory(), loadGalleries({ quiet: true })]);
      toast('Profile changed', `New downloads will be saved to “${name}”.`, 'info');
    }
  }

  async function loadHistory() {
    if (!state.activeProfile) {
      state.historyUrls = new Set();
      return;
    }
    try {
      const data = await api(withParams('/api/history', { profile: state.activeProfile }));
      const entries = apiItems(data, 'downloads');
      state.historyUrls = new Set(entries.map(entry => normalizeHistoryUrl(typeof entry === 'string' ? entry : entry.url || entry.gallery_url)));
    } catch (_) {
      state.historyUrls = new Set();
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
          state.activeProfile = name;
          storage.set('active-profile', name);
        }
        toast('Profile renamed', `“${oldName}” is now “${name}”.`);
      } else {
        await api('/api/profiles', { method: 'POST', body: { name } });
        toast('Profile created', `“${name}” is ready for downloads.`);
      }
      if ($('#profile-default').checked || !state.activeProfile) {
        state.activeProfile = name;
        storage.set('active-profile', name);
      }
      $('#profile-modal').close();
      await loadProfiles();
    } catch (error) {
      toast(oldName ? 'Could not rename profile' : 'Could not create profile', errorMessage(error), 'error');
    } finally { setButtonBusy(button, false); }
  }

  async function deleteProfile(profile) {
    if (!window.confirm(`Delete the “${profile.name}” profile? Existing files will not be removed.`)) return;
    try {
      await api(`/api/profiles/${encodeURIComponent(profile.name)}`, { method: 'DELETE' });
      if (state.activeProfile === profile.name) state.activeProfile = '';
      toast('Profile deleted', `“${profile.name}” was removed.`, 'info');
      await loadProfiles();
      await Promise.all([loadHistory(), loadGalleries({ quiet: true })]);
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

  function loadImage(image, source, alt = '') {
    const url = safeUrl(source);
    image.alt = alt;
    if (!url) {
      image.removeAttribute('src');
      image.hidden = true;
      return;
    }
    image.hidden = false;
    image.src = url;
    image.addEventListener('load', () => {
      image.previousElementSibling?.classList.add('is-loaded');
    }, { once: true });
    image.addEventListener('error', () => {
      image.hidden = true;
      image.removeAttribute('src');
    }, { once: true });
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

  async function openGallery(id, context = null) {
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
    window.clearTimeout(state.poseSaveTimer);
    state.poseSaving = false;
    state.poseSavePromise = null;
    state.poseLoading = false;
    state.galleryContext = context ? { ...context, suggestions: Array.isArray(context.suggestions) ? context.suggestions : [] } : null;
    state.gallery = { ...summary, images: [] };
    state.selectedImages = new Set();
    state.poseSelectedImages = new Set();
    state.poseTags = [];
    state.poseDraft = { revision: 0, controls: { solo: '', couple: '', group: '' }, targets: [] };
    state.poseLoadedKey = '';
    state.poseDirty = false;
    state.poseMutation = 0;
    window.clearTimeout(state.poseSaveTimer);
    $('#pose-tag-input').value = '';
    $('#pose-control-role').value = 'solo';
    const requestedMode = context?.mode === 'pose' ? 'pose' : 'download';
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
    $('#gallery-modal').showModal();
    try {
      const data = await api(withParams(`/api/galleries/${encodeURIComponent(id)}`, { profile: state.activeProfile }));
      state.gallery = normalizeDetail(data);
      const listItem = state.galleries.find(item => String(item.id) === String(id));
      if (listItem) Object.assign(listItem, { saved: state.gallery.saved, ignored: state.gallery.ignored, imageCount: state.gallery.imageCount, thumbnailUrl: state.gallery.thumbnailUrl || listItem.thumbnailUrl });
      const pendingImages = state.gallery.images.filter(image => !image.downloaded);
      state.selectedImages = new Set((pendingImages.length ? pendingImages : state.gallery.images).map(image => image.url));
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
      if (requestedMode === 'pose') scrollToFinderSuggestion();
    } catch (error) {
      $('#image-grid').replaceChildren();
      $('#images-empty').hidden = false;
      $('#selection-summary').textContent = errorMessage(error);
      toast('Could not open gallery', errorMessage(error), 'error');
    } finally {
      state.loadingDetail = false;
      $('#image-grid').setAttribute('aria-busy', 'false');
      updateSelectionUi();
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
    loadImage(image, gallery.thumbnailUrl || gallery.images?.[0]?.previewUrl, gallery.title);
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
    $('#pose-target-count').textContent = formatNumber(result.targets);
    $('#pose-control-count').textContent = formatNumber(result.controls);
    $('#pose-issue-count').textContent = formatNumber(result.issues.length);
    $('#pose-issue-count').classList.toggle('has-issues', Boolean(result.issues.length));
    $('#pose-preflight-detail').textContent = result.issues.length
      ? result.issues.join(' · ')
      : `Ready to build ${formatNumber(result.targets)} paired image${result.targets === 1 ? '' : 's'}.`;
    $('#pose-export').disabled = state.poseLoading || state.poseSaving || Boolean(result.issues.length);
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
    });
    $('#pose-target-fields').hidden = !isTarget;
    $('#pose-control-hint').hidden = isTarget;
    const apply = $('#pose-apply-checked');
    const label = $('span', apply);
    const targetRole = $('#pose-control-role').value;
    const missingControl = isTarget && !state.poseDraft.controls[targetRole];
    label.textContent = missingControl
      ? `Set ${poseRoleLabel(targetRole).toLowerCase()} control first`
      : checked ? `Apply to ${formatNumber(checked)} checked` : 'Apply to checked';
    const missingTag = isTarget && !$('#pose-tag-input').value.trim();
    apply.disabled = state.poseLoading || !checked || missingTag || missingControl || (!isTarget && checked !== 1);
    $('#pose-clear-checked').disabled = !checked || ![...state.poseSelectedImages].some(url => poseAssignmentFor(url));
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
    });
    $('#lightbox-pose-tag-input').value = assignment?.type === 'target' ? assignment.poseLabel : '';
    $('#lightbox-pose-control-role').value = assignment?.type === 'target' ? assignment.role : 'solo';
    updateLightboxTargetAvailability();
    $('#lightbox-clear-pose').disabled = !assignment;
  }

  function updateLightboxTargetAvailability() {
    const targetRole = $('#lightbox-pose-control-role').value;
    $('#lightbox-set-target').disabled = !state.poseDraft.controls[targetRole];
    $('#lightbox-set-target').title = state.poseDraft.controls[targetRole] ? 'Set this image as a pose target' : `Set a ${targetRole} control first`;
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
      renderPoseSaveStatus('Draft unavailable');
      toast('Could not load pose draft', errorMessage(error), 'error');
    } finally {
      state.poseLoading = false;
      renderPoseWorkspace();
    }
  }

  function setGalleryMode(mode, { load = true, render = true } = {}) {
    state.galleryMode = mode === 'pose' ? 'pose' : 'download';
    const poseMode = state.galleryMode === 'pose';
    $('#gallery-modal').classList.toggle('is-pose-mode', poseMode);
    $$('[data-gallery-mode]').forEach(button => {
      const active = button.dataset.galleryMode === state.galleryMode;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', String(active));
    });
    $('#pose-toolbar').hidden = !poseMode;
    $('#download-selection-total').hidden = poseMode;
    $('#pose-preflight').hidden = !poseMode;
    $('#queue-download').hidden = poseMode;
    $('#pose-export').hidden = !poseMode;
    $('#modal-profile-label').textContent = poseMode ? 'Organize in' : 'Download to';
    $('#image-picker-title').textContent = poseMode ? 'Prepare pose pairs' : 'Choose images';
    $('#select-all').textContent = poseMode ? 'Check all' : 'Select all';
    $('#select-none').textContent = poseMode ? 'Uncheck all' : 'Clear';
    if (render) renderImages();
    renderPoseWorkspace();
    if (poseMode && load) loadPoseWorkspace();
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
    if (!selected.length || state.poseLoading) return false;
    if (assignment !== 'target' && selected.length !== 1) {
      toast('Choose one control image', 'Each Solo, Couple, or Group slot uses exactly one control.', 'info');
      return false;
    }
    if (assignment === 'target') {
      const dockAction = button?.closest('#lightbox-pose-dock');
      const roleSelect = dockAction ? $('#lightbox-pose-control-role') : $('#pose-control-role');
      const role = POSE_ROLES.includes(roleSelect.value) ? roleSelect.value : 'solo';
      if (!state.poseDraft.controls[role]) {
        toast(`Set a ${poseRoleLabel(role).toLowerCase()} control first`, 'Every target needs its matching control before the draft can be saved.', 'info');
        return false;
      }
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
    if (button) setButtonBusy(button, true, assignment === 'target' ? 'Tagging…' : 'Assigning…');
    try {
      if (assignment === 'target') {
        const input = button?.closest('#lightbox-pose-dock') ? $('#lightbox-pose-tag-input') : $('#pose-tag-input');
        const roleSelect = button?.closest('#lightbox-pose-dock') ? $('#lightbox-pose-control-role') : $('#pose-control-role');
        const role = POSE_ROLES.includes(roleSelect.value) ? roleSelect.value : 'solo';
        const tag = await ensurePoseTag(input.value, role);
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
      toast('Could not assign image', errorMessage(error), 'error');
      return false;
    } finally {
      if (button) setButtonBusy(button, false);
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
    const activeSelection = state.galleryMode === 'pose' ? state.poseSelectedImages : state.selectedImages;
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
      input.setAttribute('aria-label', state.galleryMode === 'pose' ? `Check ${image.filename} for pose tagging` : `Select ${image.filename} for download`);
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
      loadImage(preview, image.previewUrl, image.filename);
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
      if (image.downloaded) {
        const saved = document.createElement('span');
        saved.className = 'image-saved';
        saved.innerHTML = '<svg><use href="#i-check"></use></svg> Saved';
        option.append(saved);
      }
      renderPoseBadge(option, image);
      const finderSuggestion = finderSuggestionForImage(image, index);
      if (finderSuggestion) {
        option.classList.add('is-finder-suggestion');
        const badge = document.createElement('span');
        badge.className = 'finder-suggestion-badge';
        const score = Number(finderSuggestion.score);
        badge.innerHTML = '<svg><use href="#i-spark"></use></svg><span></span>';
        $('span', badge).textContent = Number.isFinite(score) ? `Finder · ${score.toFixed(2)} similarity` : 'Finder suggestion';
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
    previous.removeAttribute('src');
    previous.replaceWith(display);

    const candidates = [...new Set([safeUrl(image.fullUrl), safeUrl(image.previewUrl)].filter(Boolean))];
    let candidateIndex = 0;
    const loadCandidate = () => {
      const source = candidates[candidateIndex];
      if (!source) {
        display.hidden = true;
        display.removeAttribute('src');
        placeholder.classList.add('is-error');
        return;
      }
      display.hidden = false;
      display.src = source;
    };
    display.addEventListener('load', () => {
      if (loadToken === state.lightboxLoadToken) placeholder.classList.add('is-loaded');
    });
    display.addEventListener('error', () => {
      if (loadToken !== state.lightboxLoadToken) return;
      candidateIndex += 1;
      loadCandidate();
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
    image?.removeAttribute('src');
    $('#lightbox-stage .image-placeholder')?.classList.remove('is-loaded', 'is-error');
    if ($('#gallery-modal').open && trigger?.isConnected) trigger.focus({ preventScroll: true });
  }

  function toggleImage(url, checked) {
    const selection = state.galleryMode === 'pose' ? state.poseSelectedImages : state.selectedImages;
    if (checked) selection.add(url);
    else selection.delete(url);
    const option = $$('.image-option', $('#image-grid')).find(item => item.dataset.imageUrl === url);
    option?.classList.toggle('is-selected', checked);
    updateSelectionUi();
  }

  function selectAllImages(selected) {
    const selection = new Set(selected ? (state.gallery?.images || []).map(image => image.url) : []);
    if (state.galleryMode === 'pose') state.poseSelectedImages = selection;
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
    const selection = state.galleryMode === 'pose' ? state.poseSelectedImages : state.selectedImages;
    const selected = selection.size;
    const total = state.gallery?.images?.length || 0;
    $('#selected-count').textContent = formatNumber(selected);
    const downloaded = state.gallery?.downloadedImages || 0;
    $('#selection-summary').textContent = state.loadingDetail
      ? 'Scanning the source page…'
      : state.galleryMode === 'pose'
        ? `${formatNumber(selected)} checked for bulk tagging · ${formatNumber(state.poseDraft.targets.length)} targets assigned`
        : `${formatNumber(selected)} of ${formatNumber(total)} selected${downloaded ? ` · ${formatNumber(downloaded)} already saved` : ''}`;
    $('#queue-download').disabled = state.loadingDetail || !selected || !state.activeProfile;
    renderPoseToolbar();
  }

  async function queueGallery() {
    const gallery = state.gallery;
    if (!gallery || !state.selectedImages.size) return;
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
    if (!gallery || !profile) return;
    const button = $('#pose-export');
    setButtonBusy(button, true, 'Preparing…');
    try {
      await flushPoseDraft();
      if (state.poseDirty) throw new ApiError('The pose draft could not be saved. Try again before exporting.');
      const preflight = posePreflight();
      if (preflight.issues.length) throw new ApiError(preflight.issues.join(' · '));
      const data = await api('/api/pose-exports', {
        method: 'POST',
        body: { gallery_id: gallery.id, profile, expected_revision: state.poseDraft.revision }
      });
      const pairs = Number(data?.job?.pair_count ?? preflight.targets);
      closeModal($('#lightbox-modal'));
      $('#gallery-modal').close();
      toast('Pose dataset queued', `${formatNumber(pairs)} pair${pairs === 1 ? '' : 's'} will download and organize in “${profile}”.`, 'success');
      await loadJobs({ quiet: true });
    } catch (error) {
      toast('Could not export pose dataset', errorMessage(error), 'error');
    } finally {
      setButtonBusy(button, false);
      renderPoseWorkspace();
    }
  }

  async function loadJobs({ quiet = false } = {}) {
    try {
      const data = await api('/api/downloads');
      state.jobs = apiItems(data, 'downloads').map(normalizeJob);
      state.galleries.forEach(gallery => {
        gallery.queued = state.jobs.some(job => !isTerminalJob(job) && (
          String(job.galleryId) === String(gallery.id) ||
          (job.galleryUrl && normalizeHistoryUrl(job.galleryUrl) === normalizeHistoryUrl(gallery.url))
        ));
      });
      renderJobs();
      renderGalleries();
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

  function normalizeFinderCorpus(item) {
    if (!item || typeof item !== 'object') return null;
    const wrapped = item.corpus && typeof item.corpus === 'object'
      ? item.corpus
      : item.index && typeof item.index === 'object'
        ? item.index
        : item.finder?.corpus && typeof item.finder.corpus === 'object' ? item.finder.corpus : item;
    const recognized = [
      'galleries', 'gallery_count', 'images', 'image_count', 'complete',
      'partial', 'ready', 'cache_entries', 'cache_bytes', 'storage_bytes'
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
      maxCacheBytes: finderCorpusCount(wrapped.max_cache_bytes, wrapped.cache_byte_limit, wrapped.max_bytes)
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
      corpus: normalizeFinderCorpus(data.corpus || model.corpus)
    };
  }

  function normalizeFinderReview(value) {
    const review = String(value || 'pending').toLowerCase();
    if (['accepted', 'accept', 'approved'].includes(review)) return 'accepted';
    if (['rejected', 'reject', 'dismissed'].includes(review)) return 'rejected';
    return 'pending';
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
      ? rankingVersion === FINDER_RANKING_VERSION
      : Boolean(scan.ranking_current);
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
    return {
      ...scan,
      id: scan.id ?? scan.scan_id,
      status: String(scan.status || scan.state || 'queued').toLowerCase(),
      examplesFolder: String(scan.example_directory || scan.examples_folder || config.example_directory || config.examples_folder || scan.folder || ''),
      poseTagId: scan.pose_tag_id ?? config.pose_tag_id ?? poseTag.id,
      poseTagLabel: String(scan.pose_tag_label || config.pose_tag_label || poseTag.label || poseTag.name || ''),
      poseTagSlug: String(scan.pose_tag_slug || poseTag.slug || ''),
      poseDefaultRole: POSE_ROLES.includes(scan.pose_default_role || poseTag.default_role) ? (scan.pose_default_role || poseTag.default_role) : 'solo',
      sourceUrl: String(scan.source_url || config.source_url || scan.url || ''),
      nextUrl,
      hasNextPage,
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
      rejectedCount: Number(scan.rejected_count ?? reviewCounts.rejected ?? 0),
      minSimilarity: Number(scan.minimum_score ?? scan.min_similarity ?? scan.minimum_similarity ?? config.minimum_score ?? config.min_similarity ?? config.minimum_similarity ?? 0.65),
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
      personCount: fallback.personCount || primaryMatch.personCount,
      hasOverlay: matches.some(match => Boolean(match.overlayUrl)),
      origin,
      onlineScanned,
      indexedOnly,
      feedbackMatchKeys,
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

  function finderScanAtPageCap(scan = state.finderScan) {
    if (!scan?.id || !scan.rankingCurrent || !scan.hasNextPage || Number(scan.pages || 0) < FINDER_MAX_PAGES) return false;
    return ['completed', 'completed_with_errors', 'complete', 'done', 'paused', 'running', 'scanning', 'active'].includes(scan.status);
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
            : `${progress}Checked suggestions become pose feedback; unchecked suggestions are excluded. Reviews adjust future ranking only—the vision models are not retrained.`;
  }

  function renderFinderStatus() {
    const model = state.finderStatus;
    const card = $('#finder-model-card');
    card.classList.toggle('is-ready', Boolean(model?.ready && !model.error));
    card.classList.toggle('is-error', Boolean(model && (!model.ready || model.error)));
    $('#finder-model-name').textContent = model?.name || 'Model unavailable';
    $('#finder-model-detail').textContent = model?.detail || 'Could not read model status';
    $('#finder-model-state').textContent = model?.error ? 'Retry available' : model?.modelReady ? 'Ready' : model?.ready ? 'Available' : model ? model.status.replaceAll('_', ' ') : 'Offline';
    const root = model?.folderRoot;
    const normalizedRoot = root ? root.replace(/\/+$/, '') || '/' : '';
    const fullExample = normalizedRoot === '/' ? '/poses/matting-press' : `${normalizedRoot}/poses/matting-press`;
    $('#finder-folder-hint').textContent = normalizedRoot
      ? `Use poses/matting-press relative to ${normalizedRoot}, or paste ${fullExample}. Existing folders are suggestions only.`
      : 'Use a library-relative path such as poses/matting-press, or paste the full container path. Existing folders are suggestions only.';
  }

  function renderFinderCorpus() {
    const card = $('#finder-corpus-card');
    const corpus = state.finderCorpus;
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
      hasIndex
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
      const label = `${scan.poseTagLabel || 'Pose scan'} · ${scan.status.replaceAll('_', ' ')}${date ? ` · ${date}` : ''}`;
      select.add(new Option(label, String(scan.id)));
    });
    if ([...select.options].some(option => option.value === selected)) select.value = selected;
  }

  function syncFinderConfigAvailability() {
    const locked = Boolean(state.finderScan && !finderScanIsTerminal());
    const feedbackMutationPending = state.finderFeedbackBusy || finderFeedbackIsSaving();
    ['finder-folder', 'finder-pose-tag', 'finder-source', 'finder-pages', 'finder-min-similarity'].forEach(id => { $(`#${id}`).disabled = locked || state.finderBusy; });
    $('#finder-use-current').disabled = locked || state.finderBusy;
    $('#finder-scan-select').disabled = state.finderLoading || feedbackMutationPending;
    const hasConfig = Boolean($('#finder-folder').value.trim() && $('#finder-pose-tag').value.trim() && $('#finder-source').value.trim());
    $('#finder-start').hidden = locked;
    $('#finder-start').disabled = state.finderLoading
      || state.finderBusy
      || feedbackMutationPending
      || !state.finderStatus?.ready
      || !hasConfig;
  }

  function applyFinderScanConfig(scan) {
    if (!scan) return;
    $('#finder-folder').value = scan.examplesFolder;
    $('#finder-pose-tag').value = scan.poseTagLabel;
    $('#finder-source').value = scan.sourceUrl || finderDefaultSource();
    $('#finder-pages').value = Math.max(1, Math.min(50, scan.pages || 5));
    $('#finder-min-similarity').value = scan.minSimilarity.toFixed(2);
    $('#finder-result-threshold').value = scan.minSimilarity.toFixed(2);
    $('#finder-min-output').textContent = scan.minSimilarity.toFixed(2);
    $('#finder-filter-output').textContent = scan.minSimilarity.toFixed(2);
  }

  function finderScoreLabel(score) {
    return `${Math.round((normalizeFinderScore(score, 0) || 0) * 100)}%`;
  }

  function finderEvidenceLabel(item, { short = false } = {}) {
    const tier = normalizeFinderTier(item?.rankingTier);
    const score = finderScoreLabel(item?.score);
    if (tier === 3) return `${short ? 'Exact' : 'Exact image'} ${score}`;
    if (tier === 2) return `Pose ${score}`;
    if (tier === 1) return `${short ? 'Visual' : 'Visual fallback'} ${score}`;
    return `${short ? 'Pose' : 'Pose mismatch'} ${score}`;
  }

  function finderEvidenceKind(item) {
    const tier = normalizeFinderTier(item?.rankingTier);
    if (tier === 3) return 'Exact image';
    if (tier === 2) return 'Pose match';
    if (tier === 1) return 'Visual fallback';
    return 'Pose mismatch';
  }

  function appendFinderMatchMedia(container, result) {
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
      peopleBadge.hidden = !match.personCount;
      if (match.personCount) peopleBadge.textContent = `${match.personCount} ${match.personCount === 1 ? 'person' : 'people'}`;
      const overlayUrl = safeUrl(match.overlayUrl);
      if (overlayUrl) {
        const overlay = $('.finder-skeleton-overlay', button);
        overlay.hidden = false;
        overlay.src = overlayUrl;
        overlay.addEventListener('error', () => {
          overlay.hidden = true;
          overlay.removeAttribute('src');
        }, { once: true });
      }
      const select = document.createElement('label');
      select.className = 'finder-match-select';
      select.title = selected ? 'Selected for pose feedback—click to exclude' : 'Excluded from pose feedback—click to include';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = selected;
      checkbox.disabled = !match.imageUrl || result.feedbackSaving || state.finderFeedbackBusy;
      checkbox.dataset.finderFeedbackMatch = match.feedbackKey;
      checkbox.dataset.finderResult = String(result.key);
      checkbox.setAttribute('aria-label', match.imageUrl ? `Use ${ordinalCopy} as pose feedback` : `${ordinalCopy} is unavailable for pose feedback`);
      select.innerHTML = '<svg><use href="#i-check"></use></svg><span>Use</span>';
      select.prepend(checkbox);
      item.append(button, select);
      container.append(item);
    });
  }

  function renderFinderDiagnostics(card, result) {
    const breakdown = $('.finder-score-breakdown', card);
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

  function renderFinderResults() {
    const ranked = [...state.finderResults].sort(
      (a, b) => b.rankingTier - a.rankingTier || b.score - a.score || (b.appearanceScore ?? -1) - (a.appearanceScore ?? -1) || a.rank - b.rank
    );
    const counts = {
      pending: ranked.filter(result => result.review === 'pending').length,
      accepted: ranked.filter(result => result.review === 'accepted').length,
      rejected: ranked.filter(result => result.review === 'rejected').length
    };
    $('#finder-pending-count').textContent = formatNumber(counts.pending);
    $('#finder-accepted-count').textContent = formatNumber(counts.accepted);
    $('#finder-rejected-count').textContent = formatNumber(counts.rejected);
    $$('[data-finder-review]').forEach(button => {
      const active = button.dataset.finderReview === state.finderReview;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', String(active));
    });
    const threshold = Number($('#finder-result-threshold').value || 0);
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
      card.classList.toggle('is-rejected', result.review === 'rejected');
      card.classList.toggle('is-indexed', result.indexedOnly);
      appendFinderMatchMedia($('.finder-match-gallery', card), result);
      $('.finder-rank', card).textContent = `#${String(ranked.indexOf(result) + 1).padStart(2, '0')}`;
      $('.finder-similarity', card).textContent = finderEvidenceLabel(result);
      $('.finder-indexed-badge', card).hidden = !result.indexedOnly;
      const matchKind = $('.finder-match-kind', card);
      const kindCopy = finderEvidenceKind(result);
      matchKind.hidden = !kindCopy;
      matchKind.textContent = kindCopy;
      matchKind.title = result.rankingTier === 1
        ? 'RTMO could not confirm enough joints, so this candidate is ranked by visual layout below reliable pose matches.'
        : result.rankingTier === 0
          ? 'RTMO found reliable joints, but their geometry did not reach the pose-match floor.'
          : '';
      $('.finder-card-title', card).textContent = result.title;
      const matchCopy = `${formatNumber(result.matchCount)} ${result.matchCount === 1 ? 'image' : 'images'} compared`;
      $('.finder-card-meta', card).textContent = `${matchCopy}${result.imageCount ? ` · ${formatNumber(result.imageCount)} total` : ''}`;
      const selectedFeedback = result.matches.filter(match => result.feedbackMatchKeys.includes(match.feedbackKey)).length;
      const feedbackCopy = $('.finder-feedback-selection-copy', card);
      feedbackCopy.textContent = result.feedbackSaving
        ? 'Saving gallery review and pose feedback…'
        : state.finderFeedbackBusy
          ? 'Resetting pose feedback…'
        : selectedFeedback
          ? `${selectedFeedback} of ${result.matches.length} suggested ${result.matches.length === 1 ? 'image' : 'images'} checked for pose feedback · uncheck wrong images`
          : `No suggested images checked · Accept requires at least one`;
      renderFinderDiagnostics(card, result);
      $$('.finder-card-open', card).forEach(button => {
        button.dataset.finderAction = 'open';
        button.dataset.finderResult = String(result.key);
      });
      $$('[data-finder-action]', card).forEach(button => { button.dataset.finderResult = String(result.key); });
      const accept = $('.finder-accept', card);
      const reject = $('.finder-reject', card);
      accept.classList.toggle('is-active', result.review === 'accepted');
      reject.classList.toggle('is-active', result.review === 'rejected');
      accept.disabled = state.finderBusy || state.finderFeedbackBusy || result.feedbackSaving || !selectedFeedback || (result.review === 'accepted' && !result.feedbackSelectionDirty);
      reject.disabled = state.finderBusy || state.finderFeedbackBusy || result.feedbackSaving || (result.review === 'rejected' && !result.feedbackSelectionDirty);
      accept.title = state.finderFeedbackBusy
        ? 'Wait for pose feedback reset to finish'
        : result.feedbackSaving
        ? 'Saving this gallery review'
        : !selectedFeedback ? 'Check at least one suggested image before accepting' : '';
      reject.title = state.finderFeedbackBusy
        ? 'Wait for pose feedback reset to finish'
        : result.feedbackSaving ? 'Saving this gallery review' : '';
      grid.append(fragment);
    });
    const empty = $('#finder-empty');
    empty.hidden = Boolean(results.length);
    if (!results.length) {
      $('h3', empty).textContent = state.finderResults.length ? 'No candidates in this view' : finderScanIsRunning() ? 'Scanning for candidates…' : 'No candidates found';
      $('p', empty).textContent = state.finderResults.length
        ? 'Lower the display threshold or choose another review tab.'
        : finderScanIsRunning()
          ? 'Results will appear here as galleries are compared.'
          : 'Try more examples, a lower minimum match score, or a wider source.';
    }
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

  function renderFinderWorkspace() {
    renderFinderFolders();
    renderFinderTags();
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
    $('#finder-resume').hidden = scan?.status !== 'paused' || legacyRanking;
    $('#finder-cancel').hidden = !hasScan || finderScanIsTerminal(scan) || scan?.status === 'canceling';
    $('#finder-ranking-note').hidden = !legacyRanking;
    ['finder-pause', 'finder-resume', 'finder-cancel'].forEach(id => { $(`#${id}`).disabled = state.finderBusy; });
    const canExtend = finderScanCanExtend(scan);
    const atPageCap = finderScanAtPageCap(scan);
    const finderReady = Boolean(state.finderStatus?.ready);
    $('#finder-extend').hidden = !canExtend;
    $('#finder-limit-note').hidden = !atPageCap;
    $('#finder-extend').classList.toggle('is-unavailable', canExtend && !finderReady);
    $('#finder-extend-pages').disabled = !canExtend || !finderReady || state.finderBusy;
    $('#finder-extend-button').disabled = !canExtend || !finderReady || state.finderBusy;
    const unavailableTitle = finderReady ? '' : state.finderStatus?.detail || 'Finder is unavailable';
    $('#finder-extend-button').title = unavailableTitle;
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
      $('#finder-local-progress-copy').textContent = corpusComplete
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
    const status = scan.status.replaceAll('_', ' ');
    const sourceExhausted = finderScanSourceExhausted(scan);
    const waitingForCorpus = hasCorpusProgress
      && scan.corpusSearchComplete !== true
      && !finderScanIsTerminal(scan)
      && scan.pagesScanned === 0;
    $('#finder-progress-wrap').classList.toggle('is-source-exhausted', sourceExhausted);
    $('#finder-source-progress-copy').textContent = waitingForCorpus
      ? 'Starts after the Local Gallery Index search'
      : sourceExhausted
        ? 'The selected Source URL has no more pages'
        : `Exploring ${displayHost(scan.sourceUrl)} for new galleries`;
    $('#finder-session-label').textContent = `${scan.poseTagLabel || 'Pose scan'} · ${status}`;
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
    const exampleDirectory = $('#finder-folder').value.trim();
    const tagLabel = $('#finder-pose-tag').value.trim().replace(/\s+/g, ' ');
    const sourceInput = $('#finder-source').value.trim();
    const sourceUrl = /^https?:\/\//i.test(sourceInput) ? safeUrl(sourceInput) : '';
    const requestedPages = Number.parseInt($('#finder-pages').value || '5', 10);
    const pageLimit = Math.max(1, Math.min(50, Number.isFinite(requestedPages) ? requestedPages : 5));
    const minimumScore = Math.max(0.4, Math.min(0.95, Number($('#finder-min-similarity').value || 0.65)));
    if (validate && !exampleDirectory) {
      const root = state.finderStatus?.folderRoot || 'the library root';
      toast('Enter an examples folder', `Use any folder inside ${root}, as a relative path or full container path.`, 'info');
      $('#finder-folder').focus();
      return null;
    }
    if (validate && !tagLabel) {
      toast('Name the pose', 'Choose an existing pose tag or enter a new one.', 'info');
      $('#finder-pose-tag').focus();
      return null;
    }
    if (validate && !sourceUrl) {
      toast('Enter a source URL', 'Use a complete http or https gallery, category, model, search, or home URL.', 'info');
      $('#finder-source').focus();
      return null;
    }
    if (validate && !state.finderStatus?.ready) {
      toast('Finder model is not ready', state.finderStatus?.detail || 'Refresh after the model becomes available.', 'info');
      return null;
    }
    $('#finder-pages').value = String(pageLimit);
    return { exampleDirectory, tagLabel, sourceUrl, pageLimit, minimumScore };
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

  async function loadFinderResults({ quiet = false } = {}) {
    const scanId = state.finderScanId;
    if (!scanId) {
      state.finderResults = [];
      renderFinderWorkspace();
      return;
    }
    $('#finder-result-grid').setAttribute('aria-busy', 'true');
    try {
      const data = await api(withParams(`/api/finder/scans/${encodeURIComponent(scanId)}/results`, {
        review: 'all',
        min_score: 0,
        limit: 500
      }));
      if (String(scanId) !== String(state.finderScanId)) return;
      const previousResults = new Map(state.finderResults.map(result => [String(result.key), result]));
      state.finderResults = apiItems(data, 'results').map(normalizeFinderResult).map(result => {
        const previous = previousResults.get(String(result.key));
        if (!previous?.feedbackSelectionDirty && !previous?.feedbackSaving) return result;
        return {
          ...result,
          review: previous.feedbackSaving ? previous.review : result.review,
          feedbackMatchKeys: [...previous.feedbackMatchKeys],
          feedbackSelectionProvided: previous.feedbackSelectionProvided,
          feedbackSelectionDirty: previous.feedbackSelectionDirty,
          feedbackSaving: previous.feedbackSaving
        };
      });
      renderFinderWorkspace();
    } catch (error) {
      if (!quiet) toast('Could not load Finder results', errorMessage(error), 'error');
    } finally {
      $('#finder-result-grid').setAttribute('aria-busy', 'false');
    }
  }

  async function loadFinderFeedback({ quiet = false, force = false } = {}) {
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
      setButtonBusy(button, false);
      renderFinderFeedback();
      renderFinderResults();
      syncFinderConfigAvailability();
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
      renderFinderCorpus();
    } catch (error) {
      if (error.status === 404) {
        state.finderCorpusSupported = false;
        if (!state.finderStatus?.corpus) state.finderCorpus = null;
      }
      renderFinderCorpus();
      if (!quiet && error.status !== 404) toast('Could not refresh the Local Gallery Index', errorMessage(error), 'error');
    }
  }

  async function loadFinderScan({ quiet = false, applyConfig = false } = {}) {
    const scanId = state.finderScanId;
    if (!scanId) {
      state.finderScan = null;
      state.finderResults = [];
      renderFinderWorkspace();
      return;
    }
    try {
      const data = await api(`/api/finder/scans/${encodeURIComponent(scanId)}`);
      if (String(scanId) !== String(state.finderScanId)) return;
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
    } else if (state.finderStatus?.corpus) {
      state.finderCorpus = state.finderStatus.corpus;
      state.finderCorpusSupported = true;
    } else {
      state.finderCorpus = null;
      state.finderCorpusSupported = corpusResult.reason?.status === 404 ? false : null;
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
    state.finderLoading = false;
    syncFinderConfigAvailability();
    if (selected?.id) await loadFinderScan({ quiet: true, applyConfig: !preserveConfig });
    else await loadFinderFeedback({ quiet: true, force: true });
    const failures = requests.slice(0, 4).filter(result => result.status === 'rejected');
    if (!quiet && failures.length) toast('Some Finder options are unavailable', errorMessage(failures[0].reason), 'error');
  }

  async function startFinderScan() {
    if (state.finderBusy || state.finderFeedbackBusy || finderFeedbackIsSaving()) return;
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
          minimum_score: config.minimumScore
        }
      });
      const scan = normalizeFinderScan(data?.scan || data);
      if (!scan?.id) throw new ApiError('The server did not return a Finder scan ID.');
      if (!scan.poseTagLabel) scan.poseTagLabel = tag.label;
      if (!scan.poseTagId) scan.poseTagId = tag.id;
      scan.poseDefaultRole = tag.defaultRole;
      state.finderScan = scan;
      state.finderScanId = scan.id;
      state.finderResults = [];
      state.finderReview = 'pending';
      state.finderScans = [scan, ...state.finderScans.filter(item => String(item.id) !== String(scan.id))];
      storage.set('finder-scan', state.finderScanId);
      $('#finder-result-threshold').value = config.minimumScore.toFixed(2);
      $('#finder-filter-output').textContent = config.minimumScore.toFixed(2);
      toast('Finder scan started', `Scanning up to ${config.pageLimit} pages for “${tag.label}”.`, 'success');
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
      state.finderBusy = false;
      setButtonBusy(button, false);
      renderFinderWorkspace();
      scheduleFinderPoll(300);
    }
  }

  async function performFinderScanAction(action, button) {
    const scan = state.finderScan;
    if (!scan?.id || state.finderBusy || !['pause', 'resume'].includes(action)) return;
    state.finderBusy = true;
    setButtonBusy(button, true, action === 'pause' ? 'Pausing…' : 'Resuming…');
    try {
      const data = await api(`/api/finder/scans/${encodeURIComponent(scan.id)}/${action}`, { method: 'POST' });
      const updated = normalizeFinderScan(data?.scan || data);
      if (updated?.id) state.finderScan = updated;
      else await loadFinderScan({ quiet: true });
      toast(action === 'pause' ? 'Finder paused' : 'Finder resumed', action === 'pause' ? 'Ranked results remain available for review.' : 'The server will continue from its saved progress.', 'info');
    } catch (error) {
      toast(action === 'pause' ? 'Could not pause Finder' : 'Could not resume Finder', errorMessage(error), 'error');
    } finally {
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
      state.finderBusy = false;
      renderFinderWorkspace();
      scheduleFinderPoll();
    }
  }

  function recountFinderReviews() {
    if (!state.finderScan) return;
    state.finderScan.pendingCount = state.finderResults.filter(result => result.review === 'pending').length;
    state.finderScan.acceptedCount = state.finderResults.filter(result => result.review === 'accepted').length;
    state.finderScan.rejectedCount = state.finderResults.filter(result => result.review === 'rejected').length;
  }

  function toggleFinderFeedbackMatch(input) {
    const result = state.finderResults.find(item => String(item.key) === String(input.dataset.finderResult));
    const matchKey = input.dataset.finderFeedbackMatch;
    if (!result || !matchKey || result.feedbackSaving || state.finderFeedbackBusy) return;
    const selected = new Set(result.feedbackMatchKeys);
    if (input.checked) selected.add(matchKey);
    else selected.delete(matchKey);
    result.feedbackMatchKeys = result.matches
      .map(match => match.feedbackKey)
      .filter(key => selected.has(key));
    result.feedbackSelectionDirty = true;
    renderFinderResults();
    announce(`${input.checked ? 'Included' : 'Excluded'} suggested image ${result.feedbackMatchKeys.length} of ${result.matches.length} for pose feedback.`);
  }

  async function reviewFinderResult(result, review, button = null) {
    if (!result || !['pending', 'accepted', 'rejected'].includes(review) || state.finderBusy || state.finderFeedbackBusy || result.feedbackSaving) return;
    const scanId = String(state.finderScan?.id || '');
    if (!scanId) return;
    const resultKey = String(result.key);
    const snapshot = {
      review: result.review,
      feedbackMatchKeys: [...result.feedbackMatchKeys],
      feedbackSelectionProvided: result.feedbackSelectionProvided,
      feedbackSelectionDirty: result.feedbackSelectionDirty
    };
    if (snapshot.review === review && !snapshot.feedbackSelectionDirty) return;
    const selectedMatches = result.matches.filter(match => result.feedbackMatchKeys.includes(match.feedbackKey));
    const feedbackImageUrls = selectedMatches.map(match => match.imageUrl).filter(Boolean);
    if (review === 'accepted' && !feedbackImageUrls.length) {
      toast('Check a matching image', 'Accept needs at least one checked suggestion. Unchecked images are excluded from feedback.', 'info');
      return;
    }
    const sameScan = () => String(state.finderScan?.id || '') === scanId;
    state.finderFeedbackMutations += 1;
    result.feedbackSaving = true;
    result.review = review;
    recountFinderReviews();
    renderFinderResults();
    renderFinderFeedback();
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
      if (!sameScan()) return;
      const index = state.finderResults.findIndex(item => String(item.key) === resultKey);
      const current = index >= 0 ? state.finderResults[index] : result;
      if (data) {
        const updated = normalizeFinderResult(data?.result || data, result.rank - 1);
        if (index >= 0) state.finderResults[index] = {
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
          feedbackSelectionProvided: updated.feedbackSelectionProvided || snapshot.feedbackSelectionProvided,
          feedbackSelectionDirty: false,
          feedbackSaving: false
        };
      } else {
        current.feedbackSelectionDirty = false;
        current.feedbackSaving = false;
      }
      recountFinderReviews();
      renderFinderWorkspace();
      if (!applyFinderFeedbackResponse(data)) loadFinderFeedback({ quiet: true, force: true });
      announce(`${result.title} ${review}`);
    } catch (error) {
      if (sameScan()) {
        const index = state.finderResults.findIndex(item => String(item.key) === resultKey);
        if (index >= 0) {
          state.finderResults[index] = {
            ...state.finderResults[index],
            review: snapshot.review,
            feedbackMatchKeys: snapshot.feedbackMatchKeys,
            feedbackSelectionProvided: snapshot.feedbackSelectionProvided,
            feedbackSelectionDirty: snapshot.feedbackSelectionDirty,
            feedbackSaving: false
          };
        }
        recountFinderReviews();
        renderFinderWorkspace();
      }
      toast('Could not save review', errorMessage(error), 'error');
    } finally {
      state.finderFeedbackMutations = Math.max(0, state.finderFeedbackMutations - 1);
      if (sameScan()) {
        const current = state.finderResults.find(item => String(item.key) === resultKey);
        if (current?.feedbackSaving) {
          current.feedbackSaving = false;
          renderFinderResults();
        }
        renderFinderFeedback();
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

  async function openFinderResult(result) {
    if (!result?.galleryId) {
      toast('Gallery unavailable', 'This Finder result has no gallery identifier.', 'error');
      return;
    }
    await openGallery(result.galleryId, {
      summary: {
        id: result.galleryId,
        url: result.url,
        title: result.title,
        thumbnail_url: result.bestPreviewUrl,
        image_count: result.imageCount
      },
      mode: 'pose',
      poseTag: finderPoseTagForScan(),
      suggestions: (result.matches?.length ? result.matches : [{ imageUrl: result.bestImageUrl, ordinal: result.bestOrdinal, score: result.score }]).map(match => ({
        imageUrl: match.imageUrl,
        ordinal: match.ordinal,
        score: firstFinderScore(match.score, result.score) ?? 0
      }))
    });
  }

  async function selectFinderScan(scanId) {
    if (state.finderFeedbackBusy || finderFeedbackIsSaving()) return;
    state.finderScanId = scanId;
    storage.set('finder-scan', scanId);
    if (!scanId) {
      state.finderScan = null;
      state.finderResults = [];
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
    if (dialog?.open) dialog.close();
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
    ['finder-folder', 'finder-pose-tag', 'finder-source', 'finder-pages'].forEach(id => {
      $(`#${id}`).addEventListener('input', syncFinderConfigAvailability);
      $(`#${id}`).addEventListener('change', syncFinderConfigAvailability);
    });
    $('#finder-pose-tag').addEventListener('input', () => scheduleFinderFeedbackLoad());
    $('#finder-pose-tag').addEventListener('change', () => loadFinderFeedback({ quiet: true, force: true }));
    $('#finder-feedback-reset').addEventListener('click', resetFinderFeedback);
    $('#finder-min-similarity').addEventListener('input', event => {
      $('#finder-min-output').textContent = Number(event.currentTarget.value).toFixed(2);
    });
    $('#finder-result-threshold').addEventListener('input', renderFinderResults);
    $('#finder-scan-select').addEventListener('change', event => selectFinderScan(event.currentTarget.value));
    $('#finder-start').addEventListener('click', startFinderScan);
    $('#finder-extend-pages').addEventListener('input', () => updateFinderExtendSummary());
    $('#finder-extend-pages').addEventListener('change', () => updateFinderExtendSummary({ commit: true }));
    $('#finder-extend-pages').addEventListener('keydown', event => {
      if (event.key === 'Enter') extendFinderScan();
    });
    $('#finder-extend-button').addEventListener('click', extendFinderScan);
    $('#finder-pause').addEventListener('click', event => performFinderScanAction('pause', event.currentTarget));
    $('#finder-resume').addEventListener('click', event => performFinderScanAction('resume', event.currentTarget));
    $('#finder-cancel').addEventListener('click', cancelFinderScan);
    $$('[data-finder-review]').forEach(button => button.addEventListener('click', () => {
      state.finderReview = button.dataset.finderReview;
      renderFinderResults();
    }));
    $('#finder-result-grid').addEventListener('click', event => {
      const button = event.target.closest('[data-finder-action]');
      if (!button) return;
      const result = state.finderResults.find(item => String(item.key) === String(button.dataset.finderResult));
      if (!result) return;
      if (button.dataset.finderAction === 'overlay') {
        const card = button.closest('.finder-card');
        const visible = !card.classList.contains('is-overlay-visible');
        card.classList.toggle('is-overlay-visible', visible);
        button.setAttribute('aria-pressed', String(visible));
        $('span', button).textContent = visible ? 'Hide overlay' : 'Pose overlay';
        const use = $('use', button);
        if (use) use.setAttribute('href', visible ? '#i-eye-off' : '#i-eye');
      } else if (button.dataset.finderAction === 'open') openFinderResult(result);
      else reviewFinderResult(result, button.dataset.finderAction, button);
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
    $$('[data-pose-assignment]').forEach(button => button.addEventListener('click', () => {
      state.poseAssignment = button.dataset.poseAssignment;
      renderPoseToolbar();
      if (state.poseAssignment === 'target') $('#pose-tag-input').focus();
    }));
    $('#pose-tag-input').addEventListener('input', event => syncPoseTagDefault(event.currentTarget, $('#pose-control-role')));
    $('#pose-tag-input').addEventListener('change', event => syncPoseTagDefault(event.currentTarget, $('#pose-control-role')));
    $('#pose-control-role').addEventListener('change', renderPoseToolbar);
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
      closeModal($('#lightbox-modal'));
      flushPoseDraft();
      state.galleryContext = null;
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
