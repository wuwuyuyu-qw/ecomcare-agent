/* Local demo sessions: short-lived bearer token in this tab's sessionStorage.
   Only same-origin /api requests receive Authorization. Never use a random
   browser identifier as the customer identity. API authorization is authoritative. */
(function () {
  "use strict";
  const TOKEN_KEY = "mewhelp_access_token";
  const USER_KEY = "mewhelp_authenticated_user";
  const nativeFetch = window.fetch.bind(window);
  function user() {
    try { return JSON.parse(sessionStorage.getItem(USER_KEY) || "null"); }
    catch (_) { return null; }
  }
  function clear() {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(USER_KEY);
  }
  function requireLogin() {
    clear();
    if (location.pathname === "/" || location.pathname === "/static/index.html") {
      window.dispatchEvent(new Event("mewhelp-auth-required"));
    } else {
      location.assign("/?next=" + encodeURIComponent(location.pathname + location.search));
    }
  }
  window.MewAuth = {
    user, clear,
    save(payload) {
      sessionStorage.setItem(TOKEN_KEY, payload.access_token);
      sessionStorage.setItem(USER_KEY, JSON.stringify(payload.user));
    },
  };
  window.fetch = async function (input, init) {
    const url = new URL(input instanceof Request ? input.url : String(input), location.href);
    const api = url.origin === location.origin && url.pathname.startsWith("/api/");
    const opts = { ...(init || {}) };
    if (api && url.pathname !== "/api/auth/login") {
      const headers = new Headers(input instanceof Request ? input.headers : undefined);
      new Headers(opts.headers).forEach((value, key) => headers.set(key, value));
      const token = sessionStorage.getItem(TOKEN_KEY);
      if (token) headers.set("Authorization", "Bearer " + token);
      opts.headers = headers;
    }
    const response = await nativeFetch(input, opts);
    if (api && response.status === 401 && url.pathname !== "/api/auth/login") requireLogin();
    return response;
  };
  if (location.pathname !== "/" && location.pathname !== "/static/index.html") {
    document.addEventListener("DOMContentLoaded", async () => {
      if (!sessionStorage.getItem(TOKEN_KEY)) { requireLogin(); return; }
      try {
        const response = await fetch("/api/auth/me");
        if (!response.ok) return;
        const principal = await response.json();
        sessionStorage.setItem(USER_KEY, JSON.stringify(principal));
        if (principal.role !== "admin") {
          const banner = document.createElement("p");
          banner.textContent = "此页面需要管理员账号。请回到聊天页退出后，用管理员账号登录。";
          banner.style.cssText = "color:#111;background:#fff4da;border:2px solid #222;padding:12px;font-size:15px";
          document.body.prepend(banner);
        }
      } catch (_) { /* Existing page error handling covers unavailable services. */ }
    });
  }
})();
