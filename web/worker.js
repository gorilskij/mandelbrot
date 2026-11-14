// Runs only when no file in web/dist matches (wrangler.toml [assets]).
// Mandelbrot is one page, with no tree below it: any other path under
// /mandelbrot/ is sent back to the explorer (302, query kept), so nothing
// here ever reaches the site's 404 page.
export default {
  fetch(request) {
    const url = new URL(request.url);
    url.pathname = '/mandelbrot/';
    return Response.redirect(url.toString(), 302);
  },
};
