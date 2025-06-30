import fs from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ansiescapes from "ansi-escapes";
import express from "express";
import puppeteer from "puppeteer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Set up a simple local server to host the plot
const app = express();
app.use(express.static(join(__dirname, "public")));
const server = app.listen(3000, () =>
  console.log("Server running on port 3000"),
);

(async () => {
  try {
    // Launch a headless browser
    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    // Navigate to the local web server hosting the plot
    await page.goto("http://localhost:3000", { waitUntil: "networkidle2" });

    // Take a screenshot
    await page.screenshot({ path: "plot-screenshot.png", fullPage: true });

    // Close the browser and server
    await browser.close();
    server.close();

    // TODO: Display the image in the terminal (focus of this discussion)
    console.log("Screenshot saved as plot-screenshot.png");

    // read in image
    const imageBuffer = await fs.readFile("plot-screenshot.png");

    // Display the image in the terminal
    console.log(ansiescapes.image(imageBuffer, {}));
  } catch (error) {
    console.error("Error:", error);
  }
})();
