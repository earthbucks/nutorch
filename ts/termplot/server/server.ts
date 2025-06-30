import { createRequestHandler } from "@react-router/express";
import compression from "compression";
import express from "express";
import type { ServerBuild } from "react-router";

const app = express();

app.use(compression());

// http://expressjs.com/en/advanced/best-practice-security.html#at-a-minimum-disable-x-powered-by-header
app.disable("x-powered-by");

// Vite fingerprints its assets so we can cache forever.
app.use(
  "/assets",
  express.static("build/app/client/assets", {
    immutable: true,
    maxAge: "1y",
  }),
);

// Everything else (like favicon.ico) is cached for an hour. You may want to be
// more aggressive with this caching.
app.use(express.static("build/app/client", { maxAge: "1h" }));

async function getAppBuild() {
  try {
    const build = await import("../build/server/index.js");

    return { build: build as unknown as ServerBuild, error: null };
  } catch (error) {
    // Catch error and return null to make express happy and avoid an unrecoverable crash
    console.error("Error creating build:", error);
    return { error: error, build: null as unknown as ServerBuild };
  }
}

// handle SSR requests
app.all(
  "*",
  createRequestHandler({
    build: async () => {
      const { error, build } = await getAppBuild();
      if (error) {
        throw error;
      }
      return build;
    },
  }),
);

const port = process.env.PORT || 3000;

app.listen(port, async () => {
  console.log(`Express server listening at http://localhost:${port}`);
});
