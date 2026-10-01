// The small touches that make the feed feel alive: a live bullying meter while
// you type, likes without a page reload, double-tap to like, toasts and a theme
// switch. All of it is extra. Every page still works with JavaScript turned off.
(function () {
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const root = document.documentElement;
  const HEART = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20s-7.5-4.6-7.5-10.3A4.2 4.2 0 0 1 12 7.2a4.2 4.2 0 0 1 7.5 2.5C19.5 15.4 12 20 12 20z"/></svg>';

  // Theme: follow the system until someone picks one.
  const isDark = () =>
    (root.dataset.theme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";
  document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = isDark() ? "light" : "dark";
      root.dataset.theme = next;
      try {
        localStorage.setItem("theme", next);
      } catch (error) {
        // Private browsing can block storage. The switch still works for this page.
      }
    });
  });

  // Toasts slide away on their own after a few seconds.
  document.querySelectorAll(".toast").forEach((toast, i) => {
    const leave = () => (calm ? toast.remove() : toast.classList.add("leaving"));
    toast.querySelector("button")?.addEventListener("click", leave);
    toast.addEventListener("animationend", (event) => {
      if (event.animationName === "toast-out") toast.remove();
    });
    setTimeout(leave, 4500 + i * 600);
  });

  // Forms that need a second thought, and likes without a reload.
  document.addEventListener("submit", (event) => {
    const form = event.target;
    if (form.dataset.confirm && !window.confirm(form.dataset.confirm)) {
      event.preventDefault();
      return;
    }
    const button = form.querySelector("[data-like]");
    if (button) {
      event.preventDefault();
      toggleLike(form, button);
    }
  });

  async function toggleLike(form, button) {
    if (button.disabled) return;
    button.disabled = true;
    try {
      const response = await fetch(form.action, {
        method: "POST",
        headers: { "X-CSRF-Token": csrf, "X-Requested-With": "fetch" },
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const { liked, likes } = await response.json();
      button.classList.toggle("liked", liked);
      button.setAttribute("aria-pressed", String(liked));
      button.querySelector("[data-like-count]").textContent = likes;
      if (liked && !calm) {
        button.classList.remove("pop");
        void button.offsetWidth; // restart the animation
        button.classList.add("pop");
      }
    } catch (error) {
      form.submit(); // fall back to an ordinary form post
    } finally {
      button.disabled = false;
    }
  }

  // Double-tap (or double-click) a post to like it, like on Instagram.
  document.addEventListener("dblclick", (event) => {
    const body = event.target.closest("[data-double-tap]");
    const button = body?.closest("[data-post]")?.querySelector("[data-like]");
    if (!button) return; // not logged in, so there's nothing to like with
    window.getSelection()?.removeAllRanges(); // a double click selects a word, undo that
    if (!calm) {
      const heart = document.createElement("span");
      heart.className = "burst";
      heart.innerHTML = HEART;
      heart.addEventListener("animationend", () => heart.remove());
      body.appendChild(heart);
    }
    if (!button.classList.contains("liked")) button.form.requestSubmit();
  });

  // The live meter under the composer: score the draft a moment after typing stops.
  document.querySelectorAll("[data-composer]").forEach((form) => {
    const textarea = form.querySelector("textarea");
    const live = form.querySelector("[data-live]");
    const label = form.querySelector("[data-live-label]");
    const count = form.querySelector("[data-count]");
    const threshold = parseFloat(form.dataset.threshold) || 0.5;
    live.style.setProperty("--threshold", threshold);
    let timer = null;
    let latest = 0;

    function show(result) {
      const state = result.is_bullying ? "bad" : result.score >= threshold * 0.6 ? "warn" : "good";
      const target = result.category && result.category !== "other" ? `${result.category}-based bullying` : "bullying";
      live.hidden = false;
      live.dataset.state = state;
      live.style.setProperty("--score", result.score);
      label.textContent = {
        good: "Looks friendly",
        warn: "This might come across as unkind",
        bad: `This reads like ${target} and won't go up`,
      }[state];
    }

    async function score() {
      const ticket = ++latest;
      try {
        const response = await fetch(form.dataset.scoreUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: textarea.value }),
        });
        if (response.ok && ticket === latest) show(await response.json());
      } catch (error) {
        // Offline or busy. The server still checks the post when it's sent.
      }
    }

    function changed() {
      const length = textarea.value.length;
      count.textContent = length ? `${length}/${textarea.maxLength}` : "";
      clearTimeout(timer);
      if (!textarea.value.trim()) {
        latest++;
        live.hidden = true;
        return;
      }
      timer = setTimeout(score, 350);
    }

    textarea.addEventListener("input", changed);
    changed(); // a draft that came back from the server gets a score straight away
  });

  // One-click examples on the model playground.
  document.querySelectorAll("[data-check]").forEach((form) => {
    const textarea = form.querySelector("textarea");
    form.querySelectorAll("[data-example]").forEach((chip) => {
      chip.addEventListener("click", () => {
        textarea.value = chip.dataset.example;
        form.requestSubmit();
      });
    });
  });

  // News thumbnails come from another site. Hide the ones that don't load.
  const hideBroken = (img) => img.remove();
  document.querySelectorAll("img[data-hide-if-broken]").forEach((img) => {
    if (img.complete && img.naturalWidth === 0) hideBroken(img);
    else img.addEventListener("error", () => hideBroken(img));
  });
})();
