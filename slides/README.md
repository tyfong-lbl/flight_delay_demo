# Flight Delay Demo Slides (Slidev)

This directory contains a [Slidev](https://sli.dev/) presentation for the Flight Delay Demo.

## What this is

- Slide source lives in `slides.md` (with a condensed variant in `slides-condensed.md`).
- Static assets are under `public/`.
- The deck uses `@slidev/cli` and a custom `slidev-theme-lbnl` theme.

## Run locally

From this `slides/` directory:

```bash
npm install
npm run dev
```

Then open the local URL shown in the terminal (typically `http://localhost:3030`).

## Build / export

```bash
npm run build
npm run export
```

- `build` generates a production-ready static deck.
- `export` exports a PDF.
