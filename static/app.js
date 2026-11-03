(() => {
  const $ = (id) => document.getElementById(id);
  const q = $("q"), list = $("suggestions"), results = $("results"), status = $("status");
  let titles = [], matches = [], active = -1, current = null, timer = null;

  // ---------- load all titles once; filter locally ----------
  fetch("/api/titles").then(r => r.json()).then(t => {
    titles = t.map(x => ({...x, lt: x.t.toLowerCase()}));
    const key = new URLSearchParams(location.search).get("book");
    if (key) choose(key);
  });

  const esc = (s) => s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

  function suggest(text) {
    const s = text.trim().toLowerCase();
    if (!s) return [];
    const starts = [], contains = [];
    for (const b of titles) {
      const i = b.lt.indexOf(s);
      if (i === 0) starts.push(b); else if (i > 0) contains.push(b);
      if (starts.length + contains.length > 400) break;
    }
    return starts.concat(contains).slice(0, 8);
  }

  function render() {
    if (!matches.length) { list.hidden = true; q.setAttribute("aria-expanded", "false"); return; }
    const s = q.value.trim().toLowerCase();
    list.innerHTML = matches.map((b, i) => {
      const at = b.lt.indexOf(s);
      const label = esc(b.t.slice(0, at)) + "<mark>" + esc(b.t.slice(at, at + s.length)) + "</mark>" + esc(b.t.slice(at + s.length));
      return `<li role="option" data-k="${esc(b.k)}" class="${i === active ? "active" : ""}">${label}<small>${esc(b.a)}</small></li>`;
    }).join("");
    list.hidden = false; q.setAttribute("aria-expanded", "true");
  }

  q.addEventListener("input", () => { matches = suggest(q.value); active = -1; render(); });
  q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { active = Math.min(active + 1, matches.length - 1); render(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { active = Math.max(active - 1, 0); render(); e.preventDefault(); }
    else if (e.key === "Enter") { const b = matches[active >= 0 ? active : 0]; if (b) choose(b.k); }
    else if (e.key === "Escape") { matches = []; render(); }
  });
  list.addEventListener("mousedown", (e) => {
    e.preventDefault(); // Prevents input #q from losing focus during click
    const li = e.target.closest("li");
    if (li) choose(li.dataset.k);
  });

  q.addEventListener("blur", () => {
    setTimeout(() => {
      matches = [];
      render();
    }, 150);
  });
  
  function choose(key) {
    const b = titles.find(x => x.k === key);
    if (b) q.value = b.t;
    current = key; matches = []; render();
    $("controls").hidden = false;
    history.replaceState(null, "", "?book=" + encodeURIComponent(key));   // shareable URL
    load();
  }

  // ---------- sliders: keep the three weights summing to 100 ----------
  const ids = ["wd", "wg", "wr"];
  function showWeights() {
    const tot = ids.reduce((a, i) => a + +$(i).value, 0) || 1;
    ids.forEach(i => $("o-" + i).textContent = Math.round(100 * $(i).value / tot) + "%");
  }
  function weights() {
    const tot = ids.reduce((a, i) => a + +$(i).value, 0) || 1;
    return ids.map(i => (+$(i).value / tot));
  }
  ids.forEach(i => $(i).addEventListener("input", () => { showWeights(); debounce(); }));
  ["n", "min_rating", "same_author"].forEach(i => $(i).addEventListener("change", load));
  showWeights();

  function debounce() { clearTimeout(timer); timer = setTimeout(load, 120); }

  // ---------- fetch + render cards ----------
  async function load() {
    if (!current) return;
    const [wd, wg, wr] = weights();
    const p = new URLSearchParams({book: current, wd, wg, wr, n: $("n").value,
      min_rating: $("min_rating").value, same_author: $("same_author").checked ? 1 : 0});
    const r = await fetch("/api/recommend?" + p);
    const data = await r.json();
    if (!r.ok) { status.textContent = data.error; results.innerHTML = ""; return; }
    status.textContent = `Books similar to “${data.source.title}” by ${data.source.author}`;
    results.innerHTML = data.results.map(card).join("") || "<p>No matches with these filters.</p>";
  }

  const stars = (r) => "★".repeat(Math.round(r)) + "☆".repeat(5 - Math.round(r));

  function card(b) {
    const cover = b.cover
      ? `<img src="${esc(b.cover)}" alt="" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'cover ph',textContent:'${"📖"}'}))">`
      : `<div class="cover ph">📖</div>`;
    const w = b.why;
    return `<article class="card">
      ${cover}
      <div class="body">
        <h3><a href="?book=${encodeURIComponent(b.key)}" data-k="${esc(b.key)}">${esc(b.title)}</a></h3>
        <div class="author">by ${esc(b.author)}</div>
        <div class="rating" title="${b.rating}">${stars(b.rating)} <span>${b.rating.toFixed(2)}</span></div>
        <div class="match"><div style="width:${Math.max(0, Math.min(100, b.similarity * 100))}%"></div></div>
        <div class="why-bar" title="Why this book?">
          <span class="d" style="flex:${w.description}"></span>
          <span class="g" style="flex:${w.genres}"></span>
          <span class="r" style="flex:${w.rating}"></span>
        </div>
        <div class="why-txt">${w.description}% description · ${w.genres}% genres · ${w.rating}% rating</div>
        <div class="chips">${b.shared_genres.map(g => `<span>${esc(g)}</span>`).join("")}</div>
      </div></article>`;
  }

  // click a recommendation -> pivot to its own recommendations
  results.addEventListener("click", (e) => {
    const a = e.target.closest("a[data-k]");
    if (a) { e.preventDefault(); window.scrollTo({top: 0, behavior: "smooth"}); choose(a.dataset.k); }
  });
})();
