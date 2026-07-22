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
    pages: 1,
    total: 0,
    sourceUrl: '',
    nextUrl: '',
    previousUrl: '',
    query: '',
    profiles: [],
    activeProfile: storage.get('active-profile', ''),
    historyUrls: new Set(),
    jobs: [],
    jobFilter: 'all',
    gallery: null,
    selectedImages: new Set(),
    filters: storage.get('filters', { showSaved: true, showIgnored: false }),
    density: storage.get('density', 'comfortable'),
    settings: {},
    loadingGalleries: false,
    loadingDetail: false,
    serverOnline: false,
    eventSource: null,
    eventConnected: false,
    queueTimer: null,
    healthTimer: null,
    requestController: null,
    sortFolders: [],
    sortProfiles: [],
    sortSession: null,
    sortSessionId: storage.get('sort-session', ''),
    sortLoaded: false,
    sortLoading: false,
    sortBusy: false
  };

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
      if (typeof image === 'string') return { url: image, previewUrl: image, filename: `Image ${index + 1}` };
      return {
        ...image,
        url: image.url || image.image_url || image.src || '',
        previewUrl: image.preview_url || image.thumbnail_url || image.url || '',
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
    source.addEventListener('job', () => loadJobs({ quiet: true }));
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
    state.activeProfile = name;
    storage.set('active-profile', name);
    renderProfileSelectors();
    renderProfiles();
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

  function galleryQuery() {
    return withParams('/api/galleries', {
      url: state.sourceUrl,
      q: state.query,
      page: state.page,
      profile: state.activeProfile,
      show_saved: state.filters.showSaved,
      show_ignored: state.filters.showIgnored
    });
  }

  async function loadGalleries({ quiet = false } = {}) {
    if (state.requestController) state.requestController.abort();
    state.requestController = new AbortController();
    state.loadingGalleries = true;
    renderGallerySkeletons();
    hideNotice();
    try {
      const data = await api(galleryQuery(), { signal: state.requestController.signal });
      state.galleries = apiItems(data, 'galleries').map(normalizeGallery);
      state.total = Number(data?.total ?? state.galleries.length);
      state.page = Number(data?.page ?? state.page ?? 1);
      state.pages = Math.max(1, Number(data?.pages ?? data?.total_pages ?? 1));
      state.sourceUrl = data?.source_url || state.sourceUrl;
      state.nextUrl = data?.next_url || '';
      state.previousUrl = data?.previous_url || '';
      if (state.sourceUrl && state.browseMode === 'url') $('#source-input').value = state.sourceUrl;
      renderGalleries();
      announce(`${state.total} galleries loaded`);
      setServerState(true, $('#server-detail').textContent === 'Unable to connect' ? 'Ready' : $('#server-detail').textContent);
    } catch (error) {
      if (error.name === 'AbortError') return;
      state.galleries = [];
      renderGalleries();
      showNotice(errorMessage(error));
      if (!quiet) toast('Could not load galleries', errorMessage(error), 'error');
    } finally {
      state.loadingGalleries = false;
      $('#gallery-grid').setAttribute('aria-busy', 'false');
      state.requestController = null;
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
    $('#collection-summary').textContent = state.total ? `${formatNumber(state.total)} ${state.total === 1 ? 'gallery' : 'galleries'}${hiddenCount ? ` · ${hiddenCount} hidden` : ''}` : 'No matching galleries';
    renderPagination();
  }

  function renderPagination() {
    const hasPrev = state.page > 1 || Boolean(state.previousUrl);
    const hasNext = state.page < state.pages || Boolean(state.nextUrl);
    const show = hasPrev || hasNext;
    $('#pagination').hidden = !show;
    $('#page-prev').disabled = !hasPrev;
    $('#page-next').disabled = !hasNext;
    $('#page-status').textContent = state.pages > 1 ? `Page ${state.page} of ${state.pages}` : `Page ${state.page}`;
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
    state.previousUrl = '';
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

  async function changePage(direction) {
    if (direction > 0) {
      if (state.nextUrl && state.sourceUrl) {
        state.sourceUrl = state.nextUrl;
        state.page += 1;
      } else if (state.page < state.pages) state.page += 1;
      else return;
    } else {
      if (state.previousUrl && state.sourceUrl) {
        state.sourceUrl = state.previousUrl;
        state.page = Math.max(1, state.page - 1);
      } else if (state.page > 1) state.page -= 1;
      else return;
    }
    await loadGalleries();
    $('#collection-title').scrollIntoView({ behavior: 'smooth', block: 'start' });
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

  async function openGallery(id) {
    const summary = state.galleries.find(item => String(item.id) === String(id));
    if (!summary) return;
    state.gallery = { ...summary, images: [] };
    state.selectedImages = new Set();
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
      renderGallerySummary();
      renderImages();
      renderGalleries();
      $('#gallery-modal-kicker').textContent = displayHost(state.gallery.url);
      $('#gallery-modal-title').textContent = state.gallery.title;
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

  function renderImages() {
    const grid = $('#image-grid');
    grid.replaceChildren();
    const images = state.gallery?.images || [];
    $('#images-empty').hidden = Boolean(images.length);
    images.forEach((image, index) => {
      const label = document.createElement('label');
      label.className = `image-option${state.selectedImages.has(image.url) ? ' is-selected' : ''}`;
      label.classList.toggle('is-downloaded', Boolean(image.downloaded));
      label.dataset.imageUrl = image.url;
      label.title = image.filename;
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = state.selectedImages.has(image.url);
      input.setAttribute('aria-label', `Select ${image.filename}`);
      const placeholder = document.createElement('div');
      placeholder.className = 'image-placeholder';
      placeholder.innerHTML = '<svg><use href="#i-image"></use></svg>';
      const preview = document.createElement('img');
      preview.loading = 'lazy';
      preview.decoding = 'async';
      loadImage(preview, image.previewUrl, image.filename);
      const check = document.createElement('span');
      check.className = 'image-check';
      check.innerHTML = '<svg><use href="#i-check"></use></svg>';
      const number = document.createElement('span');
      number.className = 'image-number';
      number.textContent = String(index + 1).padStart(2, '0');
      label.append(input, placeholder, preview, check, number);
      if (image.downloaded) {
        const saved = document.createElement('span');
        saved.className = 'image-saved';
        saved.innerHTML = '<svg><use href="#i-check"></use></svg> Saved';
        label.append(saved);
      }
      grid.append(label);
    });
    updateSelectionUi();
  }

  function toggleImage(url, checked) {
    if (checked) state.selectedImages.add(url);
    else state.selectedImages.delete(url);
    const option = $$('.image-option', $('#image-grid')).find(item => item.dataset.imageUrl === url);
    option?.classList.toggle('is-selected', checked);
    updateSelectionUi();
  }

  function selectAllImages(selected) {
    state.selectedImages = new Set(selected ? (state.gallery?.images || []).map(image => image.url) : []);
    $$('.image-option', $('#image-grid')).forEach(option => {
      const checked = state.selectedImages.has(option.dataset.imageUrl);
      option.classList.toggle('is-selected', checked);
      const input = $('input', option);
      if (input) input.checked = checked;
    });
    updateSelectionUi();
  }

  function updateSelectionUi() {
    const selected = state.selectedImages.size;
    const total = state.gallery?.images?.length || 0;
    $('#selected-count').textContent = formatNumber(selected);
    const downloaded = state.gallery?.downloadedImages || 0;
    $('#selection-summary').textContent = state.loadingDetail ? 'Scanning the source page…' : `${formatNumber(selected)} of ${formatNumber(total)} selected${downloaded ? ` · ${formatNumber(downloaded)} already saved` : ''}`;
    $('#queue-download').disabled = state.loadingDetail || !selected || !state.activeProfile;
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
      $('#gallery-modal').close();
      renderGalleries();
      toast('Added to queue', `${formatNumber(count)} images will download to “${profile}”.`);
      await loadJobs({ quiet: true });
    } catch (error) {
      toast('Could not start download', errorMessage(error), 'error');
    } finally { setButtonBusy(button, false); }
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
      loadImage($('.job-thumb img', row), job.thumbnailUrl, '');
      $('.job-heading h3', row).textContent = job.title;
      $('.job-heading p', row).textContent = `${job.profile}${job.createdAt ? ` · ${relativeTime(job.createdAt)}` : ''}`;
      const stateLabel = $('.job-state', row);
      const displayStatus = job.status.replaceAll('_', ' ');
      stateLabel.textContent = displayStatus;
      stateLabel.className = `job-state ${job.status === 'completed_with_errors' ? 'partial' : job.status}`;
      $('.job-progress > span', row).style.width = `${job.progress}%`;
      const counts = job.total ? `${formatNumber(job.complete)} / ${formatNumber(job.total)} images` : `${Math.round(job.progress)}%`;
      $('.job-progress-label', row).textContent = counts;
      $('.job-speed', row).textContent = job.speed ? `${formatBytes(job.speed)}/s` : job.bytes ? formatBytes(job.bytes) : '';
      const error = $('.job-error', row);
      error.hidden = !job.error && !(job.status === 'completed_with_errors' && job.failed);
      error.textContent = job.error || (job.failed ? `${formatNumber(job.failed)} images failed` : '');
      $('.job-toggle', row).hidden = true;
      $('.job-remove', row).title = isTerminalJob(job) ? 'Remove' : 'Cancel download';
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
      toast(isTerminalJob(job) ? 'Transfer removed' : 'Download cancelled', job.title, 'info');
    } catch (error) { toast('Could not remove transfer', errorMessage(error), 'error'); }
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
    if (!['discover', 'queue', 'profiles', 'sort', 'settings'].includes(view)) view = 'discover';
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
    if (view === 'queue') loadJobs({ quiet: true });
    if (view === 'profiles') loadProfiles({ quiet: true });
    if (view === 'sort' && !state.sortLoaded) loadSortWorkspace();
    if (view === 'settings' && !Object.keys(state.settings).length) loadSettings();
    announce(`${view[0].toUpperCase()}${view.slice(1)} view`);
  }

  async function refreshCurrent() {
    $('#refresh-button').classList.add('is-spinning');
    if (state.view === 'discover') await loadGalleries();
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
    $('#page-prev').addEventListener('click', () => changePage(-1));
    $('#page-next').addEventListener('click', () => changePage(1));
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
    $('#image-grid').addEventListener('change', event => {
      if (event.target.matches('input[type="checkbox"]')) toggleImage(event.target.closest('.image-option').dataset.imageUrl, event.target.checked);
    });
    $('#modal-ignore').addEventListener('click', () => {
      if (!state.gallery) return;
      const listItem = state.galleries.find(item => String(item.id) === String(state.gallery.id)) || state.gallery;
      toggleIgnore(listItem, $('#modal-ignore'));
    });
    $('#queue-download').addEventListener('click', queueGallery);
    window.addEventListener('scroll', () => $('.topbar').classList.toggle('is-scrolled', window.scrollY > 4), { passive: true });
    window.addEventListener('hashchange', () => setView(location.hash.slice(1), { updateHash: false }));
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) {
        checkHealth();
        loadJobs({ quiet: true });
      }
    });
    document.addEventListener('keydown', handleKeyboard);
  }

  function handleKeyboard(event) {
    const target = event.target;
    const editing = target.matches('input, textarea, select, [contenteditable="true"]');
    const galleryOpen = $('#gallery-modal').open;
    const anyDialogOpen = Boolean($('dialog[open]'));
    const key = event.key.toLowerCase();
    if (state.view === 'sort' && !editing && !anyDialogOpen && !event.ctrlKey && !event.metaKey && !event.altKey && ['z', 'n', 's'].includes(key)) {
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
    } else if (galleryOpen && key === 'a' && !editing) {
      event.preventDefault();
      selectAllImages(true);
    } else if (galleryOpen && event.key === 'Enter' && !editing && !$('#queue-download').disabled) {
      event.preventDefault();
      queueGallery();
    }
  }

  async function init() {
    bindEvents();
    syncFilterControls();
    $$('.density-switch button').forEach(button => button.classList.toggle('is-active', button.dataset.density === state.density));
    renderGallerySkeletons();
    const hashView = location.hash.slice(1);
    setView(['discover', 'queue', 'profiles', 'sort', 'settings'].includes(hashView) ? hashView : 'discover', { updateHash: !hashView });
    connectEvents();
    await Promise.all([checkHealth(), loadSettings(), loadProfiles({ quiet: true }), loadJobs({ quiet: true })]);
    await loadHistory();
    await loadGalleries({ quiet: true });
    window.clearInterval(state.healthTimer);
    state.healthTimer = window.setInterval(() => checkHealth(), 30000);
  }

  init();
})();
