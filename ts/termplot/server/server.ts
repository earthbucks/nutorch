import { createRequestHandler } from "@react-router/express";
import { type ServerBuild } from "react-router";
import compression from "compression";
import express, {
  type Request,
  type Response,
  type NextFunction,
} from "express";
import morgan from "morgan";

const viteDevServer =
  process.env.NODE_ENV === "production"
    ? undefined
    : await import("vite").then((vite) =>
        vite.createServer({
          server: { middlewareMode: true },
          // server: { middlewareMode: true, watch: { usePolling: true } },
        }),
      );

const app = express();

app.use(compression());

// http://expressjs.com/en/advanced/best-practice-security.html#at-a-minimum-disable-x-powered-by-header
app.disable("x-powered-by");

// handle asset requests
if (viteDevServer) {
  app.use(viteDevServer.middlewares);
} else {
  // Vite fingerprints its assets so we can cache forever.
  app.use(
    "/assets",
    express.static("build/app/client/assets", {
      immutable: true,
      maxAge: "1y",
    }),
  );
}

// Everything else (like favicon.ico) is cached for an hour. You may want to be
// more aggressive with this caching.
app.use(express.static("build/app/client", { maxAge: "1h" }));

if (process.env.NODE_ENV === "development") {
  app.use(morgan("dev")); // Logs with color in the console
} else {
  // app.use(morgan("combined")); // No color, standard format for production
}
