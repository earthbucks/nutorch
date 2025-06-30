import { createRequestHandler } from "@react-router/express";
import { type ServerBuild } from "react-router";
import compression from "compression";
import express, { type Request, type Response, type NextFunction } from "express";
import morgan from "morgan";
