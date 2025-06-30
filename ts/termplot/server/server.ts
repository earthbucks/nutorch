import { createRequestHandler } from "@react-router/express";
import { type ServerBuild } from "react-router";
import compression from "compression";
import express, { type Request, type Response, type NextFunction } from "express";
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
