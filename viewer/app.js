"use strict";

const state = {
  manifest: null,
  byId: new Map(),
  shardCache: new Map(),
  currentFeature: null,
  currentData: null,
  searchIndex: [],
  searchResults: [],
  activeSearchResult: -1,
  currentPanel: "activations",
  tweetExplorerLoaded: false,
  tweetResults: [],
  selectedTweet: null,
  tweetSearchController: null,
  tweetFilterOptions: null,
  representativeHandles: new Map(),
  selectedActivationFilters: [],
  tweetStats: null,
  selectedTrendActivation: null,
  tweetTextMode: "nice",
  hideLexical: true,
  activationLabelMode: "broad",
  graphLoaded: false,
  graphInteractionInstalled: false,
  graphResizeObserver: null,
  graph: null,
};

const PANEL_TITLES = {
  activations: "Explore activations",
  tweets: "Explore tweets",
  graph: "Activations graph",
};

const GRAPH_RANGE_SETTINGS = [
  ["graphEdgePower", "graphEdgePowerValue", "graphEdgePowerFA2v2", 1],
  ["graphAttraction", "graphAttractionValue", "graphAttractionFA2", 1],
  ["graphRepulsion", "graphRepulsionValue", "graphRepulsionFA2", 3],
  ["graphGravity", "graphGravityValue", "graphGravityFA2", 2],
  ["graphCommunityCohesion", "graphCommunityCohesionValue", "graphCommunityCohesionFA2", 2],
  ["graphEdgeCutoff", "graphEdgeCutoffValue", "graphEdgeCutoffFA2", 2],
];

function updateGraphSettingOutputs() {
  for (const [controlName, outputName, , decimals] of GRAPH_RANGE_SETTINGS) {
    elements[outputName].value = Number.parseFloat(elements[controlName].value).toFixed(decimals);
  }
}

const hfSignature = new URL(window.location.href).searchParams.get("__sign");

function resourceUrl(path) {
  const url = new URL(path, window.location.href);
  if (hfSignature) url.searchParams.set("__sign", hfSignature);
  return `${url.pathname}${url.search}`;
}

const elements = {
  input: document.querySelector("#feature-input"),
  search: document.querySelector("#feature-search"),
  searchResults: document.querySelector("#search-results"),
  settingsStack: document.querySelector("#settings-stack"),
  displaySettings: document.querySelector("#display-settings"),
  graphSettings: document.querySelector("#graph-settings"),
  niceTweetText: document.querySelector("#nice-tweet-text"),
  hideLexical: document.querySelector("#hide-lexical"),
  broadActivationLabels: document.querySelector("#broad-activation-labels"),
  status: document.querySelector("#status"),
  view: document.querySelector("#feature-view"),
  activationStat: document.querySelector("#activation-stat"),
  coreLabel: document.querySelector("#core-label"),
  coreExplanation: document.querySelector("#core-explanation"),
  coreAccuracy: document.querySelector("#core-accuracy"),
  broadLabel: document.querySelector("#broad-label"),
  broadExplanation: document.querySelector("#broad-explanation"),
  broadAccuracy: document.querySelector("#broad-accuracy"),
  relation: document.querySelector("#relation"),
  modelMeta: document.querySelector("#model-meta"),
  topExamples: document.querySelector("#top-examples"),
  randomExamples: document.querySelector("#random-examples"),
  tweetSearch: document.querySelector("#tweet-search"),
  tweetSearchMeta: document.querySelector("#tweet-search-meta"),
  tweetYearFrom: document.querySelector("#tweet-year-from"),
  tweetYearTo: document.querySelector("#tweet-year-to"),
  tweetParty: document.querySelector("#tweet-party"),
  tweetRepresentative: document.querySelector("#tweet-representative"),
  tweetRepresentatives: document.querySelector("#tweet-representatives"),
  tweetActivationFilter: document.querySelector("#tweet-activation-filter"),
  tweetActivationOptions: document.querySelector("#tweet-activation-options"),
  addActivationFilter: document.querySelector("#add-activation-filter"),
  activationMatchMode: document.querySelector("#activation-match-mode"),
  activationFilterChips: document.querySelector("#activation-filter-chips"),
  clearTweetFilters: document.querySelector("#clear-tweet-filters"),
  tweetStats: document.querySelector("#tweet-stats"),
  tweetDensity: document.querySelector("#tweet-density"),
  tweetDensityPeriod: document.querySelector("#tweet-density-period"),
  activationBars: document.querySelector("#activation-bars"),
  activationTrend: document.querySelector("#activation-trend"),
  activationTrendSelect: document.querySelector("#activation-trend-select"),
  activationTrendPeriod: document.querySelector("#activation-trend-period"),
  statsNote: document.querySelector("#stats-note"),
  tweetExplorerStatus: document.querySelector("#tweet-explorer-status"),
  tweetExplorerView: document.querySelector("#tweet-explorer-view"),
  tweetResults: document.querySelector("#tweet-results"),
  tweetDetail: document.querySelector("#tweet-detail"),
  graphSearch: document.querySelector("#graph-search"),
  graphSearchOptions: document.querySelector("#graph-search-options"),
  graphFind: document.querySelector("#graph-find"),
  graphZoomOut: document.querySelector("#graph-zoom-out"),
  graphZoomIn: document.querySelector("#graph-zoom-in"),
  graphFit: document.querySelector("#graph-fit"),
  graphFitSelection: document.querySelector("#graph-fit-selection"),
  graphReset: document.querySelector("#graph-reset"),
  graphDynamic: document.querySelector("#graph-dynamic"),
  graphEdgeMode: document.querySelector("#graph-edge-mode"),
  graphLinLog: document.querySelector("#graph-linlog"),
  graphEdgePower: document.querySelector("#graph-edge-power"),
  graphEdgePowerValue: document.querySelector("#graph-edge-power-value"),
  graphAttraction: document.querySelector("#graph-attraction"),
  graphAttractionValue: document.querySelector("#graph-attraction-value"),
  graphRepulsion: document.querySelector("#graph-repulsion"),
  graphRepulsionValue: document.querySelector("#graph-repulsion-value"),
  graphGravity: document.querySelector("#graph-gravity"),
  graphGravityValue: document.querySelector("#graph-gravity-value"),
  graphCommunityCohesion: document.querySelector("#graph-community-cohesion"),
  graphCommunityCohesionValue: document.querySelector("#graph-community-cohesion-value"),
  graphEdgeCutoff: document.querySelector("#graph-edge-cutoff"),
  graphEdgeCutoffValue: document.querySelector("#graph-edge-cutoff-value"),
  graphStatus: document.querySelector("#graph-status"),
  graphStage: document.querySelector("#graph-stage"),
  graphCanvas: document.querySelector("#activation-graph-canvas"),
  graphTooltip: document.querySelector("#graph-tooltip"),
  graphDetail: document.querySelector("#graph-detail"),
  tweetTemplate: document.querySelector("#tweet-template"),
  tabs: [...document.querySelectorAll(".workspace-tabs [role='tab']")],
  panels: [...document.querySelectorAll(".workspace-panel")],
};

function showPanel(name, updateUrl = true) {
  if (!elements.tabs.some((tab) => tab.dataset.panel === name)) name = "activations";
  state.currentPanel = name;
  for (const tab of elements.tabs) {
    tab.setAttribute("aria-selected", String(tab.dataset.panel === name));
  }
  for (const panel of elements.panels) {
    panel.hidden = panel.id !== `panel-${name}`;
  }
  elements.graphSettings.hidden = name !== "graph";
  updateSettingsForPanel(name);
  document.title = name === "activations" && state.currentFeature
    ? `Activation ${state.currentFeature} · Congress Tweets Activation Explorer`
    : `${PANEL_TITLES[name]} · Congress Tweets Activation Explorer`;
  if (updateUrl) {
    const url = new URL(window.location.href);
    url.hash = name;
    window.history.pushState({ panel: name, feature: state.currentFeature }, "", url);
  }
  if (name === "tweets") initializeTweetExplorer();
  if (name === "graph") initializeActivationGraph();
  if (state.graph?.forceAtlas) {
    if (name === "graph" && elements.graphDynamic.checked) state.graph.forceAtlas.resume();
    else state.graph.forceAtlas.pause();
  }
}

function updateSettingsForPanel(name) {
  const relevance = {
    "tweet-text": name === "activations" || name === "tweets",
    "hide-lexical": name === "tweets" || name === "graph",
    "activation-labels": name === "tweets" || name === "graph",
  };
  for (const [setting, relevant] of Object.entries(relevance)) {
    const label = elements.displaySettings.querySelector(`[data-setting="${setting}"]`);
    if (!label) continue;
    label.classList.toggle("setting-irrelevant", !relevant);
    const control = label.querySelector("input");
    if (control) control.disabled = !relevant;
  }
}

function normalizeSearch(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase()
    .replace(/[^a-z0-9%]+/g, " ")
    .trim();
}

function makeSearchRecord(feature) {
  const text = [
    feature.core_label,
    feature.broad_label,
    feature.core_explanation,
    feature.broad_explanation,
    feature.relation,
  ].map(normalizeSearch).join(" ");
  return { feature, text };
}

function broadActivationLabel(feature) {
  return feature?.broad_label || feature?.core_label || "Uninterpreted activation";
}

function coreActivationLabel(feature) {
  return feature?.core_label || feature?.broad_label || "Uninterpreted activation";
}

const LEXICAL_PATTERNS = [
  /\b(?:lexical|lexeme|lemma|orthograph\w*|morpholog\w*|morpheme\w*|substring|prefix\w*|suffix\w*|digraph|trigraph|spelling|punctuation|capitalization|uppercase|lowercase|inflection\w*|homographs?|homophones?)\b/i,
  /\bwords?\s+(?:beginning|starting|ending|containing|formed from|derived from|with (?:the )?(?:root|stem|prefix|suffix)|sharing (?:a )?(?:root|stem))\b/i,
  /\b(?:word|verb|noun|adjective|adverb)\s+famil(?:y|ies)\b/i,
  /\b(?:word|verb|noun|adjective|adverb)\s+forms?\b/i,
  /\b(?:forms?|variants?)\s+of\s+(?:the\s+)?(?:word|root|stem|lexeme)\b/i,
  /\b(?:letter|character)\s+(?:pair|sequence|pattern|cluster|combination|presence|occurrence)\b/i,
  /\b(?:initial|final)\s+(?:letter|character|token)s?\b/i,
  /\b(?:[a-z]{1,4}-initial|initial [a-z]{1,4})\b/i,
  /\b(?:capitalized|uppercase|lowercase)\s+(?:words?|tokens?|names?|acronyms?|letters?)\b/i,
  /\b(?:capitalized|uppercase|lowercase)\s+["']?[a-z]{1,4}["']?\s+(?:as\s+an?\s+)?initial\b/i,
  /\b(?:bi|tri|n-?)gram\b/i,
  /\b(?:text|tweets?|tokens?|strings?|proper nouns?|names?|acronyms?)\s+(?:containing|beginning|starting|ending)\b/i,
  /\b(?:presence|occurrences?)\s+of\s+(?:the\s+)?(?:a\s+)?(?:letter|digit|substring|string|sequence|bigram|trigram|word|token)\b/i,
  /\b(?:letter|digit|character)\s+["']?[a-z0-9]{1,4}["']?\s+(?:in|within|anywhere|presence|occurrence)\b/i,
  /\bthe\s+(?:specific\s+|literal\s+)?word\b/i,
  /\b(?:root|stem)[- ](?:based|related)\s+(?:words?|vocabulary|forms?)\b/i,
  /\b(?:vocabulary|wording)\s+(?:patterns?|forms?|related to)\b/i,
  /\b(?:names?|proper nouns?|acronyms?)\s+(?:beginning|starting|ending)\s+with\b/i,
  /\b(?:word|token)[- ]initial\b/i,
];

function isLexicalActivation(feature) {
  if (!feature) return false;
  const text = feature.broad_label || feature.core_label || "";
  return LEXICAL_PATTERNS.some((pattern) => pattern.test(text));
}

function displayActivationLabel(feature) {
  const metadata = state.byId.get(feature?.id) || feature;
  return state.activationLabelMode === "core"
    ? coreActivationLabel(metadata)
    : broadActivationLabel(metadata);
}

function graphNodeVisible(node) {
  if (!node) return false;
  return !state.hideLexical || !state.byId.get(node.id)?.lexical;
}

function graphNodeLabel(node) {
  return displayActivationLabel(node);
}

function rankSearchResults(rawQuery) {
  const query = normalizeSearch(rawQuery);
  if (!query) return [];
  const tokens = [...new Set(query.split(/\s+/).filter(Boolean))];
  const numeric = /^\d+$/.test(query) ? Number.parseInt(query, 10) : null;

  if (numeric !== null && state.byId.has(numeric)) return [state.byId.get(numeric)];

  return state.searchIndex
    .filter((record) => tokens.every((token) => record.text.includes(token)))
    .map((record) => record.feature)
    .sort((a, b) => a.id - b.id)
    .slice(0, 50);
}

function closeSearchResults() {
  elements.searchResults.hidden = true;
  elements.search.setAttribute("aria-expanded", "false");
  elements.search.removeAttribute("aria-activedescendant");
  state.activeSearchResult = -1;
}

function setActiveSearchResult(position) {
  if (!state.searchResults.length) return;
  state.activeSearchResult = (position + state.searchResults.length) % state.searchResults.length;
  const options = [...elements.searchResults.querySelectorAll(".search-result")];
  options.forEach((option, index) => {
    const active = index === state.activeSearchResult;
    option.classList.toggle("active", active);
    option.setAttribute("aria-selected", String(active));
    if (active) {
      elements.search.setAttribute("aria-activedescendant", option.id);
      option.scrollIntoView({ block: "nearest" });
    }
  });
}

function selectSearchResult(featureId) {
  closeSearchResults();
  showFeature(featureId);
}

function renderSearchResults() {
  state.searchResults = rankSearchResults(elements.search.value);
  state.activeSearchResult = -1;
  elements.searchResults.replaceChildren();

  if (!normalizeSearch(elements.search.value)) {
    closeSearchResults();
    return;
  }

  if (!state.searchResults.length) {
    const empty = document.createElement("p");
    empty.className = "search-empty";
    empty.textContent = "No matching activation interpretations.";
    elements.searchResults.appendChild(empty);
  } else {
    for (const feature of state.searchResults) {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "search-result";
      option.id = `search-option-${feature.id}`;
      option.role = "option";
      option.dataset.feature = feature.id;
      option.setAttribute("aria-selected", "false");

      const title = document.createElement("span");
      title.className = "search-result-title";
      title.textContent = broadActivationLabel(feature);
      const meta = document.createElement("span");
      meta.className = "search-result-meta";
      const labels = [
        `Activation ${feature.id}`,
        `${formatPercent(feature.pct_activating)} of tweets`,
      ].filter(Boolean);
      meta.textContent = labels.join(" · ");
      option.append(title, meta);
      option.addEventListener("click", () => selectSearchResult(feature.id));
      elements.searchResults.appendChild(option);
    }
  }

  elements.searchResults.hidden = false;
  elements.search.setAttribute("aria-expanded", "true");
}

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", isError);
  elements.status.hidden = !message;
}

function formatActivation(value) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toPrecision(3);
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 10) return `${value.toFixed(1)}%`;
  if (value >= 1) return `${value.toFixed(2)}%`;
  if (value >= 0.01) return `${value.toFixed(3)}%`;
  return `${value.toFixed(4)}%`;
}

function formatAccuracy(value) {
  if (!Number.isFinite(value)) return "Accuracy unavailable";
  const percent = value * 100;
  return `${percent === 100 ? "100" : percent.toFixed(1)}% accuracy`;
}

function formatDate(value) {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value).slice(0, 10);
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(parsed);
}

function decodeEntities(value) {
  const documentValue = new DOMParser().parseFromString(String(value || ""), "text/html");
  return documentValue.documentElement.textContent || "";
}

function cleanTweetText(value) {
  return decodeEntities(value)
    .replace(/\r\n?/g, "\n")
    .replace(/[ \t]*\n+[ \t]*(@[A-Za-z0-9_]+)[ \t]*\n+[ \t]*/g, " $1 ")
    .replace(/[ \t]*\n+[ \t]*(@[A-Za-z0-9_]+)/g, " $1")
    .replace(/(@[A-Za-z0-9_]+)[ \t]*\n+[ \t]*/g, "$1 ")
    .replace(/[ \t]+([,.;:!?])/g, "$1")
    .replace(/(\[|\()[ \t]+/g, "$1")
    .replace(/[ \t]+(\]|\))/g, "$1")
    .replace(/^[ \t]*(?:\.(?!\.)|,)[ \t]*/, "")
    .replace(/-->/g, "→")
    .replace(/--/g, "—")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function makeInitials(name, handle) {
  const parts = String(name || "")
    .replace(/[^A-Za-z0-9]+/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (parts.length >= 2) return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return String(handle || "?").replace(/^@/, "").slice(0, 2).toUpperCase() || "?";
}

function openTweetTextSearch(query) {
  const alreadyLoaded = state.tweetExplorerLoaded;
  elements.tweetSearch.value = query;
  showPanel("tweets");
  if (alreadyLoaded) searchTweets();
}

function renderTweetText(container, rawText) {
  if (state.tweetTextMode === "raw") {
    container.textContent = rawText || "";
    return;
  }
  const text = cleanTweetText(rawText);
  const tokenPattern = /((?:https?:\/\/|www\.)\S+|[@#][A-Za-z0-9_]+|\$[A-Za-z]{1,6}\b)/gi;
  let cursor = 0;
  for (const match of text.matchAll(tokenPattern)) {
    container.append(document.createTextNode(text.slice(cursor, match.index)));
    const isUrl = /^(?:https?:\/\/|www\.)/i.test(match[0]);
    const isHashtag = match[0].startsWith("#");
    const token = document.createElement(isUrl || isHashtag ? "a" : "span");
    token.className = isUrl ? "social-token url-token" : "social-token";
    token.textContent = match[0];
    if (isUrl) {
      token.href = match[0].startsWith("www.") ? `https://${match[0]}` : match[0];
      token.target = "_blank";
      token.rel = "noopener noreferrer";
    } else if (isHashtag) {
      token.href = "#tweets";
      token.title = `Search tweets containing ${match[0]}`;
      token.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        openTweetTextSearch(match[0]);
      });
    }
    container.append(token);
    cursor = match.index + match[0].length;
  }
  container.append(document.createTextNode(text.slice(cursor)));
}

function createTweetCard(tweet, activation = tweet.activation, { preview = false } = {}) {
  const fragment = elements.tweetTemplate.content.cloneNode(true);
  const card = fragment.querySelector(".tweet-card");
  card.querySelector(".tweet-avatar").textContent = makeInitials(tweet.name, tweet.twitter);
  card.querySelector(".tweet-name").textContent = tweet.name || "Unknown author";
  card.querySelector(".tweet-handle").textContent = tweet.twitter ? `@${tweet.twitter}` : "";
  card.querySelector(".tweet-date").textContent = formatDate(tweet.posted_at);
  card.querySelector(".tweet-date").dateTime = tweet.posted_at || "";
  const badge = card.querySelector(".activation-badge");
  badge.hidden = !Number.isFinite(activation);
  if (!badge.hidden) {
    badge.textContent = formatActivation(activation);
    badge.title = "SAE activation";
  }
  const text = card.querySelector(".tweet-text");
  renderTweetText(text, tweet.text);
  text.classList.toggle("tweet-text-preview", preview);
  return card;
}

function renderTweets(container, tweets) {
  container.replaceChildren();
  for (const tweet of tweets) container.appendChild(createTweetCard(tweet));

  if (!tweets.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No activating tweets are available for this activation.";
    container.appendChild(empty);
  }
}

function setTweetExplorerStatus(message, isError = false) {
  elements.tweetExplorerStatus.textContent = message;
  elements.tweetExplorerStatus.classList.toggle("error", isError);
  elements.tweetExplorerStatus.hidden = !message;
}

function renderTweetResults() {
  elements.tweetResults.replaceChildren();
  for (const tweet of state.tweetResults) {
    const card = createTweetCard(tweet, null, { preview: true });
    card.classList.add("tweet-result-card");
    card.classList.toggle("selected", state.selectedTweet?.tweet?.row_id === tweet.row_id);
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.addEventListener("click", () => selectTweet(tweet.row_id));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectTweet(tweet.row_id);
      }
    });
    elements.tweetResults.appendChild(card);
  }
  if (!state.tweetResults.length) {
    const empty = document.createElement("p");
    empty.className = "tweet-detail-placeholder";
    empty.textContent = "No tweets match this search.";
    elements.tweetResults.appendChild(empty);
  }
}

function renderTweetDetail() {
  elements.tweetDetail.replaceChildren();
  if (!state.selectedTweet) {
    const placeholder = document.createElement("p");
    placeholder.className = "tweet-detail-placeholder";
    placeholder.textContent = "Select a tweet to inspect its strongest activations.";
    elements.tweetDetail.appendChild(placeholder);
    return;
  }

  const { tweet, activations, n_activations: activationCount } = state.selectedTweet;
  const visibleActivations = state.hideLexical
    ? activations.filter((activation) => !state.byId.get(activation.id)?.lexical)
    : activations;
  const card = createTweetCard(tweet, null);
  card.classList.add("tweet-detail-card");

  const heading = document.createElement("div");
  heading.className = "activation-list-heading";
  const title = document.createElement("h3");
  title.textContent = state.selectedTweet.complete ? "All activations" : "Strongest activations";
  const meta = document.createElement("p");
  const hiddenCount = activations.length - visibleActivations.length;
  meta.textContent = state.selectedTweet.complete
    ? `${visibleActivations.length.toLocaleString()} shown${hiddenCount ? ` · ${hiddenCount.toLocaleString()} lexical hidden` : ""}`
    : `${activationCount.toLocaleString()} nonzero activations · strongest ${visibleActivations.length} shown`;
  heading.append(title, meta);

  const list = document.createElement("div");
  list.className = "tweet-activation-list";
  for (const activation of visibleActivations) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tweet-activation-item";
    const index = document.createElement("span");
    index.className = "tweet-activation-index";
    index.textContent = `${activation.id} ·`;
    const label = document.createElement("span");
    label.className = "tweet-activation-label";
    label.textContent = displayActivationLabel(activation);
    const value = document.createElement("span");
    value.className = "tweet-activation-value";
    value.textContent = formatActivation(activation.value);
    button.append(index, label, value);
    button.addEventListener("click", () => {
      showPanel("activations");
      showFeature(activation.id);
    });
    list.appendChild(button);
  }
  elements.tweetDetail.append(card, heading, list);
}

async function selectTweet(rowId) {
  try {
    setTweetExplorerStatus("Loading tweet activations…");
    const response = await fetch(resourceUrl(`api/tweets/${rowId}`));
    if (!response.ok) throw new Error(`Could not load tweet activations (${response.status}).`);
    state.selectedTweet = await response.json();
    renderTweetResults();
    renderTweetDetail();
    setTweetExplorerStatus("");
  } catch (error) {
    setTweetExplorerStatus(error.message || "Could not load tweet activations.", true);
  }
}

function activationFilterValue(feature) {
  return `${feature.id} · ${displayActivationLabel(feature)}`;
}

function resolveActivationFilter(rawValue) {
  const value = String(rawValue || "").trim();
  const indexMatch = value.match(/^(\d+)(?:\s*·|$)/);
  if (indexMatch) return state.byId.get(Number.parseInt(indexMatch[1], 10)) || null;
  const normalized = normalizeSearch(value);
  return state.manifest?.features.find(
    (feature) => normalizeSearch(displayActivationLabel(feature)) === normalized
  ) || null;
}

function renderActivationFilterChips() {
  elements.activationFilterChips.replaceChildren();
  for (const activationId of state.selectedActivationFilters) {
    const feature = state.byId.get(activationId);
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "activation-filter-chip";
    chip.title = `Remove activation ${activationId}`;
    chip.append(document.createTextNode(`${activationId} · ${displayActivationLabel(feature)} `));
    const remove = document.createElement("span");
    remove.setAttribute("aria-hidden", "true");
    remove.textContent = "×";
    chip.appendChild(remove);
    chip.addEventListener("click", () => {
      state.selectedActivationFilters = state.selectedActivationFilters.filter(
        (value) => value !== activationId
      );
      renderActivationFilterChips();
      searchTweets();
    });
    elements.activationFilterChips.appendChild(chip);
  }
  elements.activationMatchMode.disabled = state.selectedActivationFilters.length < 2;
}

function addActivationFilter() {
  const feature = resolveActivationFilter(elements.tweetActivationFilter.value);
  if (!feature) {
    setTweetExplorerStatus("Choose an activation by index or from the interpretation list.", true);
    return;
  }
  if (state.hideLexical && feature.lexical) {
    setTweetExplorerStatus(`Activation ${feature.id} is hidden by the lexical filter.`, true);
    elements.tweetActivationFilter.value = "";
    return;
  }
  if (!state.selectedActivationFilters.includes(feature.id)) {
    state.selectedActivationFilters.push(feature.id);
    state.selectedActivationFilters.sort((a, b) => a - b);
  }
  elements.tweetActivationFilter.value = "";
  renderActivationFilterChips();
  searchTweets();
}

async function searchTweets() {
  if (state.tweetSearchController) state.tweetSearchController.abort();
  const controller = new AbortController();
  state.tweetSearchController = controller;
  const query = elements.tweetSearch.value.trim();
  const representativeValue = elements.tweetRepresentative.value.trim();
  const representative = state.representativeHandles.get(representativeValue) ||
    representativeValue.replace(/^.*\(@?([^()]+)\)$/, "$1").replace(/^@/, "");
  const params = new URLSearchParams({ q: query, limit: "30" });
  if (elements.tweetYearFrom.value) params.set("year_from", elements.tweetYearFrom.value);
  if (elements.tweetYearTo.value) params.set("year_to", elements.tweetYearTo.value);
  if (elements.tweetParty.value) params.set("party", elements.tweetParty.value);
  if (representative) params.set("representative", representative);
  if (state.selectedActivationFilters.length) {
    params.set("activations", state.selectedActivationFilters.join(","));
    params.set("activation_mode", elements.activationMatchMode.value);
  }
  if (state.hideLexical) params.set("hide_lexical", "true");
  try {
    setTweetExplorerStatus("Searching tweets…");
    const response = await fetch(resourceUrl(`api/tweets/search?${params}`), {
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Tweet search failed (${response.status}).`);
    const payload = await response.json();
    state.tweetResults = payload.tweets;
    state.tweetStats = payload.stats;
    if (!payload.tweets.some((tweet) => tweet.row_id === state.selectedTweet?.tweet?.row_id)) {
      state.selectedTweet = null;
    }
    const hasFilters = Boolean(
      elements.tweetYearFrom.value || elements.tweetYearTo.value ||
      elements.tweetParty.value || representative || state.selectedActivationFilters.length
    );
    elements.tweetSearchMeta.textContent = query || hasFilters
      ? `${payload.total.toLocaleString()} matching tweets · showing ${payload.tweets.length}`
      : `${payload.total.toLocaleString()} tweets in the SAE corpus · showing the newest ${payload.tweets.length}`;
    elements.tweetExplorerView.hidden = false;
    renderTweetStats();
    renderTweetResults();
    renderTweetDetail();
    setTweetExplorerStatus("");
    if (payload.tweets.length && !state.selectedTweet) await selectTweet(payload.tweets[0].row_id);
  } catch (error) {
    if (error.name !== "AbortError") {
      setTweetExplorerStatus(error.message || "Tweet search failed.", true);
    }
  }
}

async function initializeTweetExplorer() {
  if (state.tweetExplorerLoaded) return;
  state.tweetExplorerLoaded = true;
  try {
    const response = await fetch(resourceUrl("api/tweets/options"));
    if (!response.ok) throw new Error(`Could not load tweet filters (${response.status}).`);
    state.tweetFilterOptions = await response.json();
    populateTweetFilters();
  } catch (error) {
    setTweetExplorerStatus(error.message || "Could not load tweet filters.", true);
    return;
  }
  await searchTweets();
}

function populateTweetFilters() {
  const options = state.tweetFilterOptions;
  for (const select of [elements.tweetYearFrom, elements.tweetYearTo]) {
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "Any year";
    select.appendChild(all);
    for (let year = options.year_min; year <= options.year_max; year += 1) {
      const option = document.createElement("option");
      option.value = String(year);
      option.textContent = String(year);
      select.appendChild(option);
    }
  }
  for (const person of options.representatives) {
    const label = `${person.name} (@${person.twitter})`;
    state.representativeHandles.set(label, person.twitter);
    const option = document.createElement("option");
    option.value = label;
    elements.tweetRepresentatives.appendChild(option);
  }
  rebuildActivationOptions();
  renderActivationFilterChips();
}

function rebuildActivationOptions() {
  elements.tweetActivationOptions.replaceChildren();
  const activationFragment = document.createDocumentFragment();
  for (const feature of state.manifest.features) {
    if (state.hideLexical && feature.lexical) continue;
    const option = document.createElement("option");
    option.value = activationFilterValue(feature);
    activationFragment.appendChild(option);
  }
  elements.tweetActivationOptions.appendChild(activationFragment);
}

function activationLink(activation, className) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.addEventListener("click", () => {
    showPanel("activations");
    showFeature(activation.id);
  });
  return button;
}

function renderTweetStats() {
  const stats = state.tweetStats;
  elements.tweetDensity.replaceChildren();
  elements.activationBars.replaceChildren();
  elements.activationTrend.replaceChildren();
  if (!stats?.time_bins?.length) {
    elements.tweetStats.hidden = true;
    return;
  }
  elements.tweetStats.hidden = false;
  const cadence = stats.period === "day" ? "Daily" : stats.period === "week" ? "Weekly" : "Monthly";
  elements.tweetDensityPeriod.textContent = `${cadence} matching tweets`;
  elements.activationTrendPeriod.textContent = `${cadence} average activation`;
  renderDensityChart(stats.time_bins, stats.period);
  const visibleActivations = state.hideLexical
    ? stats.top_activations.filter((item) => !state.byId.get(item.id)?.lexical)
    : stats.top_activations;
  const maximum = Math.max(...visibleActivations.map((item) => item.mean), 0.001);
  for (const activation of visibleActivations) {
    const row = activationLink(activation, "activation-bar-row");
    const index = document.createElement("span");
    index.className = "activation-bar-index";
    index.textContent = `${activation.id} ·`;
    const label = document.createElement("span");
    label.className = "activation-bar-label";
    label.textContent = displayActivationLabel(activation);
    label.title = displayActivationLabel(activation);
    const track = document.createElement("span");
    track.className = "activation-bar-track";
    const fill = document.createElement("span");
    fill.className = "activation-bar-fill";
    fill.style.width = `${100 * activation.mean / maximum}%`;
    track.appendChild(fill);
    const value = document.createElement("span");
    value.className = "activation-bar-value";
    value.textContent = formatActivation(activation.mean);
    row.append(index, label, track, value);
    elements.activationBars.appendChild(row);
  }
  const availableIds = new Set(visibleActivations.map((activation) => activation.id));
  if (!availableIds.has(state.selectedTrendActivation)) {
    state.selectedTrendActivation = visibleActivations[0]?.id ?? null;
  }
  elements.activationTrendSelect.replaceChildren();
  for (const activation of visibleActivations) {
    const option = document.createElement("option");
    option.value = activation.id;
    option.textContent = `${activation.id} · ${displayActivationLabel(activation)}`;
    option.selected = activation.id === state.selectedTrendActivation;
    elements.activationTrendSelect.append(option);
  }
  const selected = visibleActivations.find(
    (activation) => activation.id === state.selectedTrendActivation,
  );
  if (selected) renderTrendChart(selected, stats.time_bins, stats.period);
  elements.statsNote.textContent =
    `Values show ${stats.basis}. Empty periods are treated as missing and bridged in trajectories.`;
}

function niceCeiling(value) {
  if (!(value > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  const scaled = value / power;
  const nice = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return nice * power;
}

function activationCeiling(value) {
  if (!(value > 0)) return 0.01;
  return Math.max(0.01, Math.ceil((value - 1e-9) * 100) / 100);
}

function formatAxisValue(value, integer = false) {
  if (integer) return Math.round(value).toLocaleString();
  return value.toFixed(2).replace(/\.0+$/, "").replace(/(\.[0-9])0$/, "$1");
}

function timelineLabel(value, period) {
  const date = new Date(`${value}${period === "month" ? "-01" : ""}T00:00:00Z`);
  if (period === "month") {
    return new Intl.DateTimeFormat("en-US", { month: "short", year: "2-digit", timeZone: "UTC" }).format(date);
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short", day: "numeric", year: "2-digit", timeZone: "UTC",
  }).format(date);
}

function axisPositions(length, maximum = 6) {
  if (length <= maximum) return [...Array(length).keys()];
  return [...new Set([...Array(maximum).keys()].map((index) =>
    Math.round(index * (length - 1) / (maximum - 1))))];
}

function appendChartAxes(svg, {
  width, height, margin, maxY, xLabels, x, yTicks = 4, integerY = false,
}) {
  const namespace = "http://www.w3.org/2000/svg";
  const y = (value) => height - margin.bottom - value / maxY * (height - margin.top - margin.bottom);
  for (let tick = 0; tick <= yTicks; tick += 1) {
    const value = maxY * tick / yTicks;
    const line = document.createElementNS(namespace, "line");
    line.setAttribute("x1", margin.left); line.setAttribute("x2", width - margin.right);
    line.setAttribute("y1", y(value)); line.setAttribute("y2", y(value));
    line.setAttribute("class", "trend-gridline");
    svg.appendChild(line);
    const label = document.createElementNS(namespace, "text");
    label.setAttribute("x", margin.left - 7); label.setAttribute("y", y(value) + 4);
    label.setAttribute("class", "trend-axis-label"); label.setAttribute("text-anchor", "end");
    label.textContent = formatAxisValue(value, integerY);
    svg.appendChild(label);
  }
  for (const position of axisPositions(xLabels.length)) {
    const label = document.createElementNS(namespace, "text");
    label.setAttribute("x", x(position)); label.setAttribute("y", height - 8);
    label.setAttribute("class", "trend-axis-label"); label.setAttribute("text-anchor", "middle");
    label.textContent = xLabels[position];
    svg.appendChild(label);
  }
  return y;
}

function renderDensityChart(bins, period) {
  const namespace = "http://www.w3.org/2000/svg";
  const width = 620;
  const height = 250;
  const margin = { top: 14, right: 18, bottom: 30, left: 52 };
  const maxY = niceCeiling(Math.max(...bins.map((point) => point.count), 1) * 1.05);
  const x = (index) => margin.left + (bins.length === 1 ? 0.5 : index / (bins.length - 1)) *
    (width - margin.left - margin.right);
  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${period} density of matching tweets`);
  const labels = bins.map((point) => timelineLabel(point.start, period));
  const y = appendChartAxes(svg, {
    width, height, margin, maxY, xLabels: labels, x, integerY: true,
  });
  const baseline = height - margin.bottom;
  const area = document.createElementNS(namespace, "path");
  const linePoints = bins.map((point, index) => `${x(index)},${y(point.count)}`);
  area.setAttribute(
    "d",
    `M${x(0)},${baseline} L${linePoints.join(" L")} L${x(bins.length - 1)},${baseline} Z`,
  );
  area.setAttribute("class", "density-area");
  svg.appendChild(area);
  const line = document.createElementNS(namespace, "polyline");
  line.setAttribute("points", linePoints.join(" "));
  line.setAttribute("class", "density-line");
  svg.appendChild(line);
  elements.tweetDensity.appendChild(svg);
}

function renderTrendChart(activation, bins, period) {
  const namespace = "http://www.w3.org/2000/svg";
  const width = 620;
  const height = 250;
  const margin = { top: 14, right: 18, bottom: 30, left: 52 };
  const points = activation.timeline.filter((point) => Number.isFinite(point.mean));
  if (!points.length) return;
  const starts = bins.map((point) => point.start);
  const maxY = activationCeiling(Math.max(...points.map((point) => point.mean), 0.001));
  const x = (index) => margin.left + (starts.length === 1 ? 0.5 : index / (starts.length - 1)) *
    (width - margin.left - margin.right);
  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Average activation by ${period}`);
  const labels = starts.map((start) => timelineLabel(start, period));
  const y = appendChartAxes(svg, {
    width, height, margin, maxY, xLabels: labels, x, yTicks: 5,
  });
  const byTime = new Map(activation.timeline.map((point) => [point.start, point.mean]));
  const polyline = document.createElementNS(namespace, "polyline");
  polyline.setAttribute("points", starts.flatMap((start, position) => {
    const value = byTime.get(start);
    return Number.isFinite(value) ? [`${x(position)},${y(value)}`] : [];
  }).join(" "));
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", "#145c45");
  polyline.setAttribute("class", "trend-line");
  svg.appendChild(polyline);
  elements.activationTrend.append(svg);
}

function setGraphStatus(message, isError = false) {
  elements.graphStatus.textContent = message;
  elements.graphStatus.classList.toggle("error", isError);
  elements.graphStatus.hidden = !message;
}

function graphColor(cluster, alpha = 1) {
  const hue = (cluster * 137.508 + 18) % 360;
  return `hsla(${hue}, 42%, 43%, ${alpha})`;
}

function graphScreenPoint(node) {
  const graph = state.graph;
  return {
    x: node.x * graph.scale + graph.translateX,
    y: node.y * graph.scale + graph.translateY,
  };
}

function graphNodeRadius(node) {
  const graph = state.graph;
  const logFrequency = Math.log1p(node.n);
  const span = Math.max((graph?.frequencyLogMax || 1) - (graph?.frequencyLogMin || 0), 1e-6);
  const normalized = Math.max(0, Math.min(1, (logFrequency - graph.frequencyLogMin) / span));
  const baseRadius = 1.2 + 5.8 * normalized ** 1.4;
  const zoomFactor = Math.max(0.48, Math.min(0.82, 0.45 + graph.scale * 0.018));
  return baseRadius * zoomFactor;
}

function resizeActivationGraph() {
  const graph = state.graph;
  if (!graph) return;
  const rect = elements.graphCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  graph.width = Math.max(1, rect.width);
  graph.height = Math.max(1, rect.height);
  elements.graphCanvas.width = Math.round(graph.width * dpr);
  elements.graphCanvas.height = Math.round(graph.height * dpr);
  graph.context.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawActivationGraph();
}

function setGraphFitTransform(nodes, padding = 34, blend = 1) {
  const graph = state.graph;
  if (!graph || !nodes.length) return;
  const xs = nodes.map((node) => node.x);
  const ys = nodes.map((node) => node.y);
  const minX = Math.min(...xs); const maxX = Math.max(...xs);
  const minY = Math.min(...ys); const maxY = Math.max(...ys);
  const maximumScale = nodes.length === 1 ? 18 : 45;
  const targetScale = Math.min(maximumScale, Math.max(0.5, Math.min(
    (graph.width - padding * 2) / Math.max(maxX - minX, 1),
    (graph.height - padding * 2) / Math.max(maxY - minY, 1),
  )));
  const targetX = graph.width / 2 - (minX + maxX) / 2 * targetScale;
  const targetY = graph.height / 2 - (minY + maxY) / 2 * targetScale;
  graph.scale += (targetScale - graph.scale) * blend;
  graph.translateX += (targetX - graph.translateX) * blend;
  graph.translateY += (targetY - graph.translateY) * blend;
}

function fitGraphNodes(nodes, padding = 34) {
  setGraphFitTransform(nodes, padding, 1);
  drawActivationGraph();
}

function fitActivationGraph() {
  const graph = state.graph;
  if (!graph) return;
  fitGraphNodes(graph.nodes.filter(graphNodeVisible));
}

function fitSelectedGraphNodes() {
  const selected = [...(state.graph?.selectedNodes || [])].filter(graphNodeVisible);
  if (!selected.length) {
    setGraphStatus("Select one or more activations to fit them in the graph.", true);
    return;
  }
  setGraphStatus("");
  fitGraphNodes(selected, 70);
}

function zoomActivationGraph(factor, centerX = null, centerY = null) {
  const graph = state.graph;
  if (!graph) return;
  graph.autoTrackLayout = false;
  const x = centerX ?? graph.width / 2;
  const y = centerY ?? graph.height / 2;
  const worldX = (x - graph.translateX) / graph.scale;
  const worldY = (y - graph.translateY) / graph.scale;
  const nextScale = Math.min(70, Math.max(0.5, graph.scale * factor));
  graph.scale = nextScale;
  graph.translateX = x - worldX * nextScale;
  graph.translateY = y - worldY * nextScale;
  drawActivationGraph();
}

function animateGraphZoom(factor, duration = 190) {
  const graph = state.graph;
  if (!graph) return;
  graph.autoTrackLayout = false;
  if (graph.zoomAnimationFrame) window.cancelAnimationFrame(graph.zoomAnimationFrame);
  const centerX = graph.width / 2;
  const centerY = graph.height / 2;
  const worldX = (centerX - graph.translateX) / graph.scale;
  const worldY = (centerY - graph.translateY) / graph.scale;
  const startScale = graph.scale;
  const targetScale = Math.min(70, Math.max(0.5, startScale * factor));
  const startedAt = performance.now();
  const step = (now) => {
    const progress = Math.min(1, (now - startedAt) / duration);
    const eased = 1 - (1 - progress) ** 3;
    graph.scale = startScale * (targetScale / startScale) ** eased;
    graph.translateX = centerX - worldX * graph.scale;
    graph.translateY = centerY - worldY * graph.scale;
    drawActivationGraph();
    if (progress < 1) graph.zoomAnimationFrame = window.requestAnimationFrame(step);
    else graph.zoomAnimationFrame = null;
  };
  graph.zoomAnimationFrame = window.requestAnimationFrame(step);
}

function resetActivationGraph() {
  const graph = state.graph;
  if (!graph) return;
  graph.labelNodeIds = [];
  for (const node of graph.nodes) {
    node.x = node.originalX;
    node.y = node.originalY;
  }
  graph.forceAtlas?.syncFromNodes();
  fitActivationGraph();
  kickActivationGraph({ track: true });
}

function graphForceSettings() {
  return {
    attraction: Number.parseFloat(elements.graphAttraction.value) || 0,
    repulsion: Number.parseFloat(elements.graphRepulsion.value) || 0,
    gravity: Number.parseFloat(elements.graphGravity.value) || 0,
    communityCohesion: Number.parseFloat(elements.graphCommunityCohesion.value) || 0,
    edgePower: Number.parseFloat(elements.graphEdgePower.value) || 1,
    edgeCutoff: Number.parseFloat(elements.graphEdgeCutoff.value) || 0,
    linLog: elements.graphLinLog.checked,
  };
}

function rebuildForceAtlasRuntime() {
  const graph = state.graph;
  if (!graph || !window.ForceAtlas2Runtime) return;
  graph.forceAtlas?.kill();
  const nodes = graph.nodes.filter(graphNodeVisible);
  const visibleIds = new Set(nodes.map((node) => node.id));
  const edges = graph.edges.filter((edge) =>
    visibleIds.has(edge.source.id) && visibleIds.has(edge.target.id)
  );
  graph.forceAtlas = new window.ForceAtlas2Runtime(nodes, edges, {
    onFrame: () => {
      graph.layoutFrame = (graph.layoutFrame || 0) + 1;
      if (graph.autoTrackLayout && graph.layoutFrame % 4 === 0) {
        setGraphFitTransform(nodes, 34, 0.14);
      }
      drawActivationGraph();
    },
    onState: ({ state: layoutState }) => {
      graph.layoutState = layoutState;
      if (layoutState === "settling") setGraphStatus("Settling activation constellations…");
      else {
        graph.autoTrackLayout = false;
        setGraphStatus("");
        drawActivationGraph();
      }
    },
    onCommunities: ({ count }) => {
      graph.communityCount = count;
      graph.labelNodeIds = [];
      drawActivationGraph();
    },
  });
  graph.forceAtlas.configure(graphForceSettings());
  if (elements.graphDynamic.checked && state.currentPanel === "graph") graph.forceAtlas.reheat();
}

function configureForceAtlasRuntime({ track = true } = {}) {
  const graph = state.graph;
  if (!graph?.forceAtlas) return;
  graph.autoTrackLayout = track;
  graph.forceAtlas.configure(graphForceSettings());
  if (elements.graphDynamic.checked && state.currentPanel === "graph") graph.forceAtlas.reheat();
}

function kickActivationGraph({ track = false } = {}) {
  const graph = state.graph;
  if (!graph?.forceAtlas || !elements.graphDynamic.checked || state.currentPanel !== "graph") return;
  graph.autoTrackLayout = track;
  graph.forceAtlas.reheat();
}

function drawActivationGraph() {
  const graph = state.graph;
  if (!graph) return;
  const ctx = graph.context;
  const edgePower = Number.parseFloat(elements.graphEdgePower.value) || 1;
  const edgeCutoff = Number.parseFloat(elements.graphEdgeCutoff.value) || 0;
  ctx.clearRect(0, 0, graph.width, graph.height);
  ctx.fillStyle = "#fbfcfa";
  ctx.fillRect(0, 0, graph.width, graph.height);

  for (const edge of graph.edges) {
    if (!graphNodeVisible(edge.source) || !graphNodeVisible(edge.target)) continue;
    if (edge.visualWeight < edgeCutoff) continue;
    const source = graphScreenPoint(edge.source);
    const target = graphScreenPoint(edge.target);
    if ((source.x < 0 && target.x < 0) || (source.y < 0 && target.y < 0) ||
        (source.x > graph.width && target.x > graph.width) ||
        (source.y > graph.height && target.y > graph.height)) continue;
    const selected = graph.selectedNodes.size === 1 &&
      (graph.selectedNodes.has(edge.source) || graph.selectedNodes.has(edge.target));
    if (!selected && graph.scale < 7 && edge.visualWeight < 0.62) continue;
    if (!selected && graph.scale >= 7 && graph.scale < 12 && edge.visualWeight < 0.28) continue;
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    const visual = edge.visualWeight ** edgePower;
    ctx.strokeStyle = selected
      ? `rgba(24, 76, 57, ${0.48 + visual * 0.42})`
      : `rgba(73, 96, 86, ${0.006 + visual ** 1.8 * 0.18})`;
    ctx.lineWidth = selected
      ? 1.1 + visual * 2.7
      : 0.12 + visual ** 1.8 * 1.55;
    ctx.stroke();
  }

  const onScreenNodes = [];
  for (const node of graph.nodes) {
    if (!graphNodeVisible(node)) continue;
    const point = graphScreenPoint(node);
    if (point.x < -8 || point.y < -8 || point.x > graph.width + 8 || point.y > graph.height + 8) continue;
    const radius = graphNodeRadius(node);
    const emphasized = graph.selectedNodes.has(node) || node === graph.hovered;
    ctx.beginPath();
    ctx.arc(point.x, point.y, emphasized ? radius + 2.2 : radius, 0, Math.PI * 2);
    ctx.fillStyle = graphColor(node.cluster, emphasized ? 1 : 0.78);
    ctx.fill();
    if (emphasized) {
      ctx.strokeStyle = "#17241e";
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }
    onScreenNodes.push({ node, point });
  }

  const selectedPool = graph.selectedNodes.size
    ? onScreenNodes.filter(({ node }) => graph.selectedNodes.has(node))
    : onScreenNodes;
  const focusX = graph.width / 2;
  const focusY = graph.height / 2;
  const focusRadius = Math.max(0, Math.min(graph.width, graph.height) / 2 - 36);
  const focusPool = selectedPool.filter(({ point }) =>
    Math.hypot(point.x - focusX, point.y - focusY) <= focusRadius
  );
  const poolById = new Map(focusPool.map((item) => [item.node.id, item]));
  const retainedLabels = (graph.labelNodeIds || [])
    .map((id) => poolById.get(id))
    .filter(Boolean)
    .slice(0, 10);
  const retainedIds = new Set(retainedLabels.map(({ node }) => node.id));
  const visibleForLabels = retainedLabels.concat(spatiallyDistributedLabels(
    focusPool.filter(({ node }) => !retainedIds.has(node.id)),
    10 - retainedLabels.length,
    graph.width,
    graph.height,
    retainedLabels,
  ));
  graph.labelNodeIds = visibleForLabels.map(({ node }) => node.id);
  ctx.font = "650 12px Inter, -apple-system, sans-serif";
  ctx.textBaseline = "middle";
  const occupiedLabels = [];
  for (const { node, point } of visibleForLabels) {
    const label = `${node.id} · ${graphNodeLabel(node)}`;
    const maximumLength = 44;
    const clipped = label.length > maximumLength
      ? `${label.slice(0, maximumLength - 1)}…`
      : label;
    const width = ctx.measureText(clipped).width;
    const radius = graphNodeRadius(node);
    let placement = null;
    const rightX = point.x + radius + 5;
    const leftX = point.x - radius - width - 5;
    const placements = [
      [rightX, point.y], [leftX, point.y],
      [rightX, point.y - 17], [leftX, point.y - 17],
      [rightX, point.y + 17], [leftX, point.y + 17],
      [point.x - width / 2, point.y - radius - 11],
      [point.x - width / 2, point.y + radius + 11],
    ];
    const validPlacements = [];
    for (const [x, y] of placements) {
      if (x < 3 || x + width > graph.width - 3 || y < 10 || y > graph.height - 10) continue;
      const bounds = { left: x - 2, right: x + width + 2, top: y - 9, bottom: y + 9 };
      validPlacements.push({ x, y, bounds });
      const overlaps = occupiedLabels.some((other) =>
        bounds.left < other.right && bounds.right > other.left &&
        bounds.top < other.bottom && bounds.bottom > other.top
      );
      if (!overlaps) {
        placement = { x, y, bounds };
        break;
      }
    }
    if (!placement && validPlacements.length) {
      placement = validPlacements.reduce((best, candidate) => {
        const overlap = occupiedLabels.reduce((total, other) => {
          const overlapWidth = Math.max(0, Math.min(candidate.bounds.right, other.right) - Math.max(candidate.bounds.left, other.left));
          const overlapHeight = Math.max(0, Math.min(candidate.bounds.bottom, other.bottom) - Math.max(candidate.bounds.top, other.top));
          return total + overlapWidth * overlapHeight;
        }, 0);
        return !best || overlap < best.overlap ? { ...candidate, overlap } : best;
      }, null);
    }
    if (!placement) continue;
    ctx.fillStyle = "rgba(251, 252, 250, 0.88)";
    ctx.fillRect(placement.x - 2, placement.y - 8, width + 4, 16);
    ctx.fillStyle = "#26332d";
    ctx.fillText(clipped, placement.x, placement.y);
    occupiedLabels.push(placement.bounds);
  }
  if (graph.drag?.type === "marquee") {
    const left = Math.min(graph.drag.startX, graph.drag.x);
    const top = Math.min(graph.drag.startY, graph.drag.y);
    const width = Math.abs(graph.drag.x - graph.drag.startX);
    const height = Math.abs(graph.drag.y - graph.drag.startY);
    ctx.fillStyle = "rgba(31, 111, 85, 0.09)";
    ctx.strokeStyle = "rgba(31, 111, 85, 0.75)";
    ctx.lineWidth = 1;
    ctx.fillRect(left, top, width, height);
    ctx.strokeRect(left + 0.5, top + 0.5, width, height);
  }
}

function spatiallyDistributedLabels(items, limit, width, height, existing = []) {
  if (limit <= 0 || !items.length) return [];
  if (items.length <= limit) return [...items].sort((a, b) => b.node.n - a.node.n || a.node.id - b.node.id);

  const candidates = [...items].sort((a, b) => b.node.n - a.node.n || a.node.id - b.node.id);
  const counts = candidates.map(({ node }) => Math.log1p(Math.max(0, node.n || 0)));
  const minimumCount = Math.min(...counts);
  const countRange = Math.max(1e-6, Math.max(...counts) - minimumCount);
  const diagonal = Math.max(1, Math.hypot(width, height));
  const chosen = [...existing];
  const result = [];
  if (!chosen.length) {
    const first = candidates.shift();
    chosen.push(first);
    result.push(first);
  }

  while (result.length < limit && candidates.length) {
    let bestIndex = 0;
    let bestScore = -Infinity;
    for (let index = 0; index < candidates.length; index += 1) {
      const candidate = candidates[index];
      const distance = Math.min(...chosen.map(({ point }) =>
        Math.hypot(candidate.point.x - point.x, candidate.point.y - point.y)
      )) / diagonal;
      const popularity = (Math.log1p(Math.max(0, candidate.node.n || 0)) - minimumCount) / countRange;
      const score = distance * 0.82 + popularity * 0.18;
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }
    const next = candidates.splice(bestIndex, 1)[0];
    chosen.push(next);
    result.push(next);
  }

  return result;
}

function nearestGraphNode(clientX, clientY) {
  const graph = state.graph;
  const rect = elements.graphCanvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  let nearest = null;
  let nearestDistance = Infinity;
  for (const node of graph.nodes) {
    if (!graphNodeVisible(node)) continue;
    const point = graphScreenPoint(node);
    const distance = Math.hypot(point.x - x, point.y - y);
    if (distance <= Math.max(8, graphNodeRadius(node) + 3) && distance < nearestDistance) {
      nearest = node;
      nearestDistance = distance;
    }
  }
  return { node: nearest, x, y };
}

function renderGraphDetail() {
  const graph = state.graph;
  const node = graph?.selected;
  if (elements.graphFitSelection) {
    elements.graphFitSelection.disabled = !graph || ![...graph.selectedNodes].some(graphNodeVisible);
  }
  elements.graphDetail.replaceChildren();
  if (!node || !graph.selectedNodes.size) {
    const placeholder = document.createElement("p");
    placeholder.className = "graph-detail-placeholder";
    placeholder.textContent = "Select one or more activations to inspect their interpretation and co-occurrences.";
    elements.graphDetail.appendChild(placeholder);
    return;
  }
  const close = document.createElement("button");
  close.type = "button";
  close.className = "graph-detail-close";
  close.setAttribute("aria-label", "Close activation details");
  close.textContent = "×";
  close.addEventListener("click", () => selectGraphNode(null));
  if (graph.selectedNodes.size > 1) {
    const kicker = document.createElement("p");
    kicker.className = "card-kicker";
    kicker.textContent = `${graph.selectedNodes.size.toLocaleString()} activations selected`;
    const title = document.createElement("h3");
    title.textContent = "Selected activations";
    const meta = document.createElement("p");
    meta.className = "graph-detail-meta";
    meta.textContent = "Ordered by number of activating tweets";
    const list = document.createElement("div");
    list.className = "graph-selection-list";
    const selected = [...graph.selectedNodes]
      .filter(graphNodeVisible)
      .sort((a, b) => b.n - a.n || a.id - b.id);
    for (const selectedNode of selected) {
      const button = document.createElement("button");
      button.type = "button";
      const index = document.createElement("code");
      index.textContent = `${selectedNode.id} ·`;
      const identity = document.createElement("span");
      identity.textContent = graphNodeLabel(selectedNode);
      identity.title = graphNodeLabel(selectedNode);
      const occurrence = document.createElement("small");
      occurrence.textContent = `${selectedNode.n.toLocaleString()} tweets`;
      button.append(index, identity, occurrence);
      button.addEventListener("click", () => focusGraphNode(selectedNode));
      list.appendChild(button);
    }
    elements.graphDetail.append(close, kicker, title, meta, list);
    return;
  }
  const kicker = document.createElement("p");
  kicker.className = "card-kicker";
  kicker.textContent = `Activation ${node.id}`;
  const title = document.createElement("h3");
  title.textContent = graphNodeLabel(node);
  const meta = document.createElement("p");
  meta.className = "graph-detail-meta";
  meta.textContent = `${node.n.toLocaleString()} activating tweets`;
  const open = document.createElement("button");
  open.type = "button";
  open.className = "graph-open-activation";
  open.textContent = "Open activation";
  open.addEventListener("click", () => {
    showPanel("activations");
    showFeature(node.id);
  });
  const heading = document.createElement("h4");
  heading.textContent = "Strongest co-occurrences";
  const neighbors = document.createElement("div");
  neighbors.className = "graph-neighbors";
  const relations = graph.adjacency.get(node.id).filter(
    (relation) => graphNodeVisible(relation.node),
  ).slice(0, 10);
  const strongestWeight = relations[0]?.visualWeight || 1;
  for (const relation of relations) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "graph-neighbor";
    const index = document.createElement("code");
    index.textContent = `${relation.node.id} ·`;
    const label = document.createElement("span");
    label.textContent = graphNodeLabel(relation.node);
    label.title = graphNodeLabel(relation.node);
    const count = document.createElement("small");
    count.textContent = `≈${relation.cooccurrence.toLocaleString()} shared tweets`;
    const meter = document.createElement("i");
    meter.style.width = `${Math.max(5, relation.visualWeight / strongestWeight * 100)}%`;
    button.append(index, label, count, meter);
    button.addEventListener("click", () => focusGraphNode(relation.node));
    neighbors.appendChild(button);
  }
  elements.graphDetail.append(close, kicker, title, meta, open, heading, neighbors);
}

function selectGraphNode(node, { toggle = false } = {}) {
  const graph = state.graph;
  setGraphStatus("");
  graph.labelNodeIds = [];
  if (!node) {
    graph.selectedNodes.clear();
    graph.selected = null;
  } else if (toggle) {
    if (graph.selectedNodes.has(node)) graph.selectedNodes.delete(node);
    else graph.selectedNodes.add(node);
    graph.selected = graph.selectedNodes.has(node)
      ? node
      : [...graph.selectedNodes].at(-1) || null;
  } else {
    graph.selectedNodes = new Set([node]);
    graph.selected = node;
  }
  renderGraphDetail();
  drawActivationGraph();
}

function focusGraphNode(node) {
  const graph = state.graph;
  if (!graphNodeVisible(node)) {
    setGraphStatus(`Activation ${node.id} is hidden by the lexical filter.`, true);
    return;
  }
  graph.scale = Math.max(graph.scale, 13);
  graph.translateX = graph.width / 2 - node.x * graph.scale;
  graph.translateY = graph.height / 2 - node.y * graph.scale;
  selectGraphNode(node);
}

function resolveGraphSearch() {
  const feature = resolveActivationFilter(elements.graphSearch.value);
  const node = feature ? state.graph?.nodeById.get(feature.id) : null;
  if (!node || !graphNodeVisible(node)) {
    setGraphStatus(
      node && !graphNodeVisible(node)
        ? `Activation ${feature.id} is hidden by the lexical filter.`
        : feature
        ? `Activation ${feature.id} has no reliable co-occurrence edges and is hidden.`
        : "Choose an activation by index or from the interpretation list.",
      true,
    );
    return;
  }
  setGraphStatus("");
  focusGraphNode(node);
}

function installGraphInteraction() {
  if (state.graphInteractionInstalled) return;
  state.graphInteractionInstalled = true;
  const canvas = elements.graphCanvas;
  canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    canvas.focus({ preventScroll: true });
    const graph = state.graph;
    const hit = nearestGraphNode(event.clientX, event.clientY);
    const additive = event.metaKey || event.ctrlKey;
    canvas.setPointerCapture(event.pointerId);
    if (hit.node) {
      if (additive) selectGraphNode(hit.node, { toggle: true });
      else if (!graph.selectedNodes.has(hit.node)) selectGraphNode(hit.node);
      graph.drag = {
        type: "nodes",
        nodes: graph.selectedNodes.has(hit.node) ? [...graph.selectedNodes] : [],
        nodeSet: graph.selectedNodes.has(hit.node) ? new Set(graph.selectedNodes) : new Set(),
        x: hit.x,
        y: hit.y,
      };
      graph.forceAtlas?.pause();
    } else if (additive || event.shiftKey) {
      graph.drag = {
        type: "marquee",
        startX: hit.x,
        startY: hit.y,
        x: hit.x,
        y: hit.y,
        baseSelection: new Set(graph.selectedNodes),
      };
    } else {
      graph.autoTrackLayout = false;
      graph.drag = {
        type: "pan",
        startX: hit.x,
        startY: hit.y,
        x: hit.x,
        y: hit.y,
      };
    }
  });
  canvas.addEventListener("pointermove", (event) => {
    const graph = state.graph;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (graph.drag) {
      const dx = x - graph.drag.x;
      const dy = y - graph.drag.y;
      if (graph.drag.type === "pan") {
        graph.translateX += dx;
        graph.translateY += dy;
      } else if (graph.drag.type === "marquee") {
        graph.drag.x = x;
        graph.drag.y = y;
      } else {
        const worldDx = dx / graph.scale;
        const worldDy = dy / graph.scale;
        const moving = graph.drag.nodeSet;
        for (const node of graph.drag.nodes) {
          node.x += worldDx;
          node.y += worldDy;
        }
        if (elements.graphDynamic.checked) {
          const followers = new Map();
          for (const node of graph.drag.nodes) {
            for (const relation of graph.adjacency.get(node.id)) {
              if (moving.has(relation.node) || !graphNodeVisible(relation.node)) continue;
              const influence = 0.06 + relation.visualWeight * 0.24;
              followers.set(
                relation.node,
                Math.max(followers.get(relation.node) || 0, influence),
              );
            }
          }
          for (const [node, influence] of followers) {
            node.x += worldDx * influence;
            node.y += worldDy * influence;
          }
        }
      }
      graph.drag.x = x;
      graph.drag.y = y;
      drawActivationGraph();
      return;
    }
    const hit = nearestGraphNode(event.clientX, event.clientY);
    if (graph.hovered !== hit.node) {
      graph.hovered = hit.node;
      drawActivationGraph();
    }
    if (hit.node) {
      elements.graphTooltip.textContent = `${hit.node.id} · ${graphNodeLabel(hit.node)}`;
      elements.graphTooltip.style.left = `${hit.x + 14}px`;
      elements.graphTooltip.style.top = `${hit.y + 10}px`;
      elements.graphTooltip.hidden = false;
      canvas.style.cursor = "grab";
    } else {
      elements.graphTooltip.hidden = true;
      canvas.style.cursor = "move";
    }
  });
  const endDrag = () => {
    if (!state.graph) return;
    const graph = state.graph;
    const drag = graph.drag;
    if (drag?.type === "marquee") {
      const left = Math.min(drag.startX, drag.x);
      const right = Math.max(drag.startX, drag.x);
      const top = Math.min(drag.startY, drag.y);
      const bottom = Math.max(drag.startY, drag.y);
      const selected = new Set(drag.baseSelection);
      if (right - left > 3 && bottom - top > 3) {
        for (const node of graph.nodes) {
          const point = graphScreenPoint(node);
          if (point.x >= left && point.x <= right && point.y >= top && point.y <= bottom) {
            if (graphNodeVisible(node)) selected.add(node);
          }
        }
      }
      graph.selectedNodes = selected;
      graph.selected = [...selected].at(-1) || null;
      graph.labelNodeIds = [];
      setGraphStatus("");
      renderGraphDetail();
    }
    const wasNodeDrag = drag?.type === "nodes";
    const clearSelection = drag?.type === "pan" &&
      Math.hypot(drag.x - drag.startX, drag.y - drag.startY) <= 3 &&
      graph.selectedNodes.size > 0;
    graph.drag = null;
    if (clearSelection) selectGraphNode(null);
    else drawActivationGraph();
    if (wasNodeDrag) {
      graph.forceAtlas?.syncFromNodes(drag.nodes);
      kickActivationGraph();
    }
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      selectGraphNode(null);
    } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "a") {
      event.preventDefault();
      const visible = state.graph.nodes.filter(graphNodeVisible);
      state.graph.selectedNodes = new Set(visible);
      state.graph.selected = visible.at(-1) || null;
      state.graph.labelNodeIds = [];
      setGraphStatus("");
      renderGraphDetail();
      drawActivationGraph();
    }
  });
  canvas.addEventListener("pointerleave", () => {
    elements.graphTooltip.hidden = true;
    if (state.graph && !state.graph.drag) state.graph.hovered = null;
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const graph = state.graph;
    graph.autoTrackLayout = false;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    zoomActivationGraph(Math.exp(-event.deltaY * 0.0036), x, y);
  }, { passive: false });
}

async function initializeActivationGraph() {
  if (state.graphLoaded) return;
  state.graphLoaded = true;
  try {
    setGraphStatus("Loading activation graph…");
    const graphFile = elements.graphEdgeMode.value === "activation"
      ? "data/activation_graph_weighted.json"
      : "data/activation_graph.json";
    const response = await fetch(resourceUrl(graphFile));
    if (!response.ok) throw new Error(`Could not load activation graph (${response.status}).`);
    const payload = await response.json();
    if (payload.schema_version !== 2) throw new Error("Activation graph data needs to be rebuilt.");
    const nodes = payload.nodes.map((node) => ({
      ...node,
      originalX: node.x,
      originalY: node.y,
    }));
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const adjacency = new Map(nodes.map((node) => [node.id, []]));
    const edges = payload.edges.map((edge) => {
      const source = nodeById.get(edge.source);
      const target = nodeById.get(edge.target);
      return { source, target, weight: edge.weight, cooccurrence: edge.cooccurrence };
    });
    const sortedLogWeights = edges
      .map((edge) => Math.log(Math.max(edge.weight, 1e-7)))
      .sort((a, b) => a - b);
    const quantile = (fraction) => sortedLogWeights[
      Math.min(sortedLogWeights.length - 1, Math.floor(sortedLogWeights.length * fraction))
    ];
    const lowWeight = quantile(0.08);
    const highWeight = quantile(0.98);
    for (const edge of edges) {
      const logWeight = Math.log(Math.max(edge.weight, 1e-7));
      edge.visualWeight = Math.max(0.03, Math.min(1, (logWeight - lowWeight) / (highWeight - lowWeight)));
      adjacency.get(edge.source.id).push({
        node: edge.target,
        weight: edge.weight,
        visualWeight: edge.visualWeight,
        cooccurrence: edge.cooccurrence,
      });
      adjacency.get(edge.target.id).push({
        node: edge.source,
        weight: edge.weight,
        visualWeight: edge.visualWeight,
        cooccurrence: edge.cooccurrence,
      });
    }
    for (const relations of adjacency.values()) {
      relations.sort((a, b) => b.weight - a.weight);
    }
    const frequencyLogs = nodes.map((node) => Math.log1p(node.n));
    state.graph = {
      nodes, edges, nodeById, adjacency,
      context: elements.graphCanvas.getContext("2d"),
      width: 1, height: 1, scale: 1, translateX: 0, translateY: 0,
      selected: null, selectedNodes: new Set(), hovered: null, drag: null,
      zoomAnimationFrame: null,
      labelNodeIds: [],
      autoTrackLayout: false, layoutFrame: 0, layoutState: "idle", communityCount: 0,
      frequencyLogMin: Math.min(...frequencyLogs),
      frequencyLogMax: Math.max(...frequencyLogs),
    };
    rebuildGraphSearchOptions();
    elements.graphStage.hidden = false;
    setGraphStatus("");
    resizeActivationGraph();
    fitActivationGraph();
    installGraphInteraction();
    rebuildForceAtlasRuntime();
    state.graphResizeObserver?.disconnect();
    state.graphResizeObserver = new ResizeObserver(resizeActivationGraph);
    state.graphResizeObserver.observe(elements.graphCanvas);
  } catch (error) {
    state.graphLoaded = false;
    setGraphStatus(error.message || "Could not load the activation graph.", true);
  }
}

function rebuildGraphSearchOptions() {
  elements.graphSearchOptions.replaceChildren();
  if (!state.graph) return;
  const options = document.createDocumentFragment();
  for (const node of state.graph.nodes) {
    if (!graphNodeVisible(node)) continue;
    const option = document.createElement("option");
    option.value = `${node.id} · ${graphNodeLabel(node)}`;
    options.appendChild(option);
  }
  elements.graphSearchOptions.appendChild(options);
}

async function loadShard(shardName) {
  if (!state.shardCache.has(shardName)) {
    const promise = fetch(resourceUrl(`data/features/${shardName}.json`)).then((response) => {
    if (!response.ok) throw new Error(`Could not load activation data (${response.status}).`);
      return response.json();
    });
    state.shardCache.set(shardName, promise);
  }
  return state.shardCache.get(shardName);
}

async function showFeature(featureId, updateHistory = true) {
  const indexEntry = state.byId.get(featureId);
  if (!indexEntry) {
    elements.view.hidden = true;
    setStatus(`Activation ${featureId} is not available in this export.`, true);
    return;
  }

  setStatus(`Loading activation ${featureId}…`);
  try {
    const shard = await loadShard(indexEntry.shard);
    const feature = shard.features.find((item) => item.id === featureId);
    if (!feature) throw new Error(`Activation ${featureId} is missing from shard ${indexEntry.shard}.`);

    const interpretation = feature.interpretation;
    elements.input.value = feature.id;
    elements.activationStat.textContent =
      `${feature.stats.n_activating.toLocaleString()} activating tweets · ` +
      `${formatPercent(feature.stats.pct_activating)} of corpus`;

    elements.coreLabel.textContent = interpretation?.core?.label || "No interpretation available";
    elements.coreExplanation.textContent = interpretation?.core?.explanation || "";
    elements.coreAccuracy.textContent = formatAccuracy(interpretation?.scores?.score_core);
    elements.broadLabel.textContent = interpretation?.broad?.label || "No broad interpretation available";
    elements.broadExplanation.textContent = interpretation?.broad?.explanation || "";
    elements.broadAccuracy.textContent = formatAccuracy(interpretation?.scores?.score_broad);
    elements.relation.textContent = interpretation?.relation || "";
    elements.relation.hidden = !interpretation?.relation;
    const modelName = interpretation?.model === "deepseek-v4-flash"
      ? "DeepSeek V4 Flash"
      : interpretation?.model;
    elements.modelMeta.textContent = modelName ? `Interpreted with ${modelName}` : "";
    renderTweets(elements.topExamples, feature.examples.top);
    renderTweets(elements.randomExamples, feature.examples.random);

    state.currentFeature = feature.id;
    state.currentData = feature;
    elements.view.hidden = false;
    setStatus("");

    if (updateHistory) {
      const url = new URL(window.location.href);
      url.searchParams.set("feature", String(feature.id));
      window.history.pushState({ feature: feature.id }, "", url);
    }
    document.title = `Activation ${feature.id} · Congress Tweets Activation Explorer`;
  } catch (error) {
    elements.view.hidden = true;
    setStatus(error.message || "Could not load this activation.", true);
  }
}

async function initialize() {
  try {
    try {
      const savedTweetText = window.localStorage.getItem("tweetTextMode");
      if (["nice", "raw"].includes(savedTweetText)) state.tweetTextMode = savedTweetText;
      const savedHideLexical = window.localStorage.getItem("hideLexicalActivationsV2");
      if (savedHideLexical !== null) state.hideLexical = savedHideLexical === "true";
      const savedLabelMode = window.localStorage.getItem("activationLabelMode");
      if (["broad", "core"].includes(savedLabelMode)) state.activationLabelMode = savedLabelMode;
      const savedGraphDynamics = window.localStorage.getItem("dynamicActivationGraphFA2v5");
      if (savedGraphDynamics !== null) elements.graphDynamic.checked = savedGraphDynamics === "true";
      const savedLinLog = window.localStorage.getItem("graphLinLog");
      if (savedLinLog !== null) elements.graphLinLog.checked = savedLinLog === "true";
      const savedEdgeMode = window.localStorage.getItem("graphEdgeModeV2");
      if (["presence", "activation"].includes(savedEdgeMode)) elements.graphEdgeMode.value = savedEdgeMode;
      for (const [controlName, , storageKey] of GRAPH_RANGE_SETTINGS) {
        const saved = window.localStorage.getItem(storageKey);
        if (saved !== null && Number.isFinite(Number.parseFloat(saved))) elements[controlName].value = saved;
      }
    } catch (_) {
      // Storage can be unavailable in restrictive browser modes; the UI still works.
    }

    const response = await fetch(resourceUrl("data/manifest.json"));
    if (!response.ok) throw new Error(`Could not load viewer manifest (${response.status}).`);
    state.manifest = await response.json();
    for (const feature of state.manifest.features) feature.lexical = isLexicalActivation(feature);
    state.byId = new Map(state.manifest.features.map((feature) => [feature.id, feature]));
    state.searchIndex = state.manifest.features.map(makeSearchRecord);
    elements.niceTweetText.checked = state.tweetTextMode === "nice";
    elements.hideLexical.checked = state.hideLexical;
    elements.broadActivationLabels.checked = state.activationLabelMode === "broad";
    updateGraphSettingOutputs();

    const ids = state.manifest.features.map((feature) => feature.id);
    if (!ids.length) throw new Error("The viewer export contains no activations.");
    elements.input.min = Math.min(...ids);
    elements.input.max = Math.max(...ids);
    const requestedPanel = window.location.hash.replace(/^#/, "");
    showPanel(requestedPanel || "activations", false);

    const requested = Number.parseInt(new URL(window.location.href).searchParams.get("feature"), 10);
    const initial = state.byId.has(requested) ? requested : ids[0];
    await showFeature(initial, false);
  } catch (error) {
    setStatus(error.message || "Could not initialize the viewer.", true);
  }
}

let indexUpdateTimer;
function showFeatureFromIndex() {
  const featureId = Number.parseInt(elements.input.value, 10);
  if (Number.isInteger(featureId)) showFeature(featureId);
}

elements.input.addEventListener("input", () => {
  window.clearTimeout(indexUpdateTimer);
  indexUpdateTimer = window.setTimeout(showFeatureFromIndex, 250);
});
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    window.clearTimeout(indexUpdateTimer);
    showFeatureFromIndex();
  }
});

elements.search.addEventListener("input", renderSearchResults);
elements.search.addEventListener("focus", () => {
  if (elements.search.value) renderSearchResults();
});
elements.search.addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown") {
    event.preventDefault();
    if (elements.searchResults.hidden) renderSearchResults();
    setActiveSearchResult(state.activeSearchResult + 1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    if (elements.searchResults.hidden) renderSearchResults();
    setActiveSearchResult(state.activeSearchResult - 1);
  } else if (event.key === "Enter" && state.activeSearchResult >= 0) {
    event.preventDefault();
    selectSearchResult(state.searchResults[state.activeSearchResult].id);
  } else if (event.key === "Escape") {
    closeSearchResults();
  }
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".feature-search")) closeSearchResults();
});

let tweetSearchTimer;
elements.tweetSearch.addEventListener("input", () => {
  window.clearTimeout(tweetSearchTimer);
  tweetSearchTimer = window.setTimeout(searchTweets, 500);
});

for (const filter of [elements.tweetYearFrom, elements.tweetYearTo, elements.tweetParty]) {
  filter.addEventListener("change", searchTweets);
}
elements.tweetRepresentative.addEventListener("input", () => {
  window.clearTimeout(tweetSearchTimer);
  tweetSearchTimer = window.setTimeout(searchTweets, 500);
});
elements.addActivationFilter.addEventListener("click", addActivationFilter);
elements.tweetActivationFilter.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addActivationFilter();
  }
});
elements.activationMatchMode.addEventListener("change", searchTweets);
elements.activationTrendSelect.addEventListener("change", () => {
  state.selectedTrendActivation = Number.parseInt(elements.activationTrendSelect.value, 10);
  elements.activationTrend.replaceChildren();
  const activation = state.tweetStats?.top_activations.find(
    (item) => item.id === state.selectedTrendActivation,
  );
  if (activation) renderTrendChart(activation, state.tweetStats.time_bins, state.tweetStats.period);
});
elements.clearTweetFilters.addEventListener("click", () => {
  elements.tweetSearch.value = "";
  elements.tweetYearFrom.value = "";
  elements.tweetYearTo.value = "";
  elements.tweetParty.value = "";
  elements.tweetRepresentative.value = "";
  elements.tweetActivationFilter.value = "";
  elements.activationMatchMode.value = "all";
  state.selectedActivationFilters = [];
  renderActivationFilterChips();
  searchTweets();
});
elements.tweetSearch.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    window.clearTimeout(tweetSearchTimer);
    searchTweets();
  }
});
elements.graphFind.addEventListener("click", resolveGraphSearch);
elements.graphSearch.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    resolveGraphSearch();
  }
});
elements.graphFit.addEventListener("click", fitActivationGraph);
elements.graphFitSelection.addEventListener("click", fitSelectedGraphNodes);
elements.graphZoomIn.addEventListener("click", () => animateGraphZoom(1.4));
elements.graphZoomOut.addEventListener("click", () => animateGraphZoom(1 / 1.4));
elements.graphReset.addEventListener("click", resetActivationGraph);
elements.graphEdgeMode.addEventListener("change", async () => {
  try { window.localStorage.setItem("graphEdgeModeV2", elements.graphEdgeMode.value); } catch (_) {}
  state.graph?.forceAtlas?.kill();
  if (state.graph?.zoomAnimationFrame) window.cancelAnimationFrame(state.graph.zoomAnimationFrame);
  state.graph = null;
  state.graphLoaded = false;
  elements.graphStage.hidden = true;
  setGraphStatus("Loading activation graph…");
  await initializeActivationGraph();
});
elements.graphDynamic.addEventListener("change", () => {
  try {
    window.localStorage.setItem("dynamicActivationGraphFA2v5", String(elements.graphDynamic.checked));
  } catch (_) {
    // Keep the in-memory preference when browser storage is unavailable.
  }
  const graph = state.graph;
  if (!graph) return;
  if (!elements.graphDynamic.checked) {
    graph.forceAtlas?.pause();
    graph.autoTrackLayout = false;
    setGraphStatus("");
  } else kickActivationGraph({ track: true });
});
elements.graphLinLog.addEventListener("change", () => {
  try { window.localStorage.setItem("graphLinLog", String(elements.graphLinLog.checked)); } catch (_) {}
  configureForceAtlasRuntime();
});
let graphSettingsTimer;
for (const [controlName, , storageKey] of GRAPH_RANGE_SETTINGS) {
  elements[controlName].addEventListener("input", () => {
    updateGraphSettingOutputs();
    try { window.localStorage.setItem(storageKey, elements[controlName].value); } catch (_) {}
    drawActivationGraph();
    window.clearTimeout(graphSettingsTimer);
    graphSettingsTimer = window.setTimeout(configureForceAtlasRuntime, 120);
  });
}

for (const tab of elements.tabs) {
  tab.addEventListener("click", () => showPanel(tab.dataset.panel));
}

elements.niceTweetText.addEventListener("change", () => {
  state.tweetTextMode = elements.niceTweetText.checked ? "nice" : "raw";
  try { window.localStorage.setItem("tweetTextMode", state.tweetTextMode); } catch (_) {}
  if (state.currentData) {
    renderTweets(elements.topExamples, state.currentData.examples.top);
    renderTweets(elements.randomExamples, state.currentData.examples.random);
  }
  if (state.tweetExplorerLoaded) {
    renderTweetResults();
    renderTweetDetail();
  }
});

elements.broadActivationLabels.addEventListener("change", () => {
  state.activationLabelMode = elements.broadActivationLabels.checked ? "broad" : "core";
  try { window.localStorage.setItem("activationLabelMode", state.activationLabelMode); } catch (_) {}
  if (state.tweetExplorerLoaded) {
    rebuildActivationOptions();
    renderActivationFilterChips();
    renderTweetStats();
    renderTweetDetail();
  }
  rebuildGraphSearchOptions();
  renderGraphDetail();
  drawActivationGraph();
});

elements.hideLexical.addEventListener("change", () => {
  state.hideLexical = elements.hideLexical.checked;
  try { window.localStorage.setItem("hideLexicalActivationsV2", String(state.hideLexical)); } catch (_) {}
  if (state.tweetExplorerLoaded) {
    if (state.hideLexical) {
      state.selectedActivationFilters = state.selectedActivationFilters.filter(
        (id) => !state.byId.get(id)?.lexical,
      );
    }
    rebuildActivationOptions();
    renderActivationFilterChips();
    searchTweets();
  }
  if (state.graph) {
    state.graph.selectedNodes = new Set(
      [...state.graph.selectedNodes].filter(graphNodeVisible),
    );
    state.graph.selected = [...state.graph.selectedNodes].at(-1) || null;
    if (!graphNodeVisible(state.graph.hovered)) state.graph.hovered = null;
    rebuildGraphSearchOptions();
    renderGraphDetail();
    drawActivationGraph();
    rebuildForceAtlasRuntime();
  }
});

window.addEventListener("popstate", (event) => {
  showPanel(window.location.hash.replace(/^#/, "") || "activations", false);
  const requested = event.state?.feature ??
    Number.parseInt(new URL(window.location.href).searchParams.get("feature"), 10);
  if (state.byId.has(requested)) showFeature(requested, false);
});

initialize();
