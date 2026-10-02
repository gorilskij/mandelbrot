// Runs only when no file in web/dist matches (wrangler.toml [assets]): hand
// the request on to the site's Worker, which owns the domain, so a wrong path
// under /mandelbrot/ gets the site's 404 page.
export default {
  fetch: (request) => fetch(request),
};
