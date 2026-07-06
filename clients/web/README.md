# BoilerAdvisor Web UI

Next.js (App Router) frontend for BoilerAdvisor, styled in Purdue's black & gold.
Talks to the FastAPI backend at `NEXT_PUBLIC_API_BASE_URL` (defaults to
`http://localhost:8000`; overridden in `.env.local`).

## Run

```bash
npm install
npm run dev      # http://localhost:3000
npm run build    # production build (also the quickest full typecheck)
```

## Pages

- `/` — the planner. First visit shows onboarding (profile setup form, catalog-search
  course pickers); afterwards the saved student's newest plan is reloaded from the
  backend (`localStorage` keeps only the student id). Semester cards support move /
  remove / add-course edits (deterministically validated server-side) and every accepted
  change is autosaved via `POST /v1/students/{id}/plans`. The advisor chat asks
  RAG-grounded questions or revises the plan from free-text feedback.
- `/admin` — read-only database browser (table row counts + paged rows) over
  `GET /v1/admin/*`. For editing, run `make adminer` in `catalog_ingestion/`.

## Styling

Tailwind (see `tailwind.config.ts` + `postcss.config.mjs` — both are required for
Tailwind to compile at all) plus a small set of theme primitives in `app/globals.css`
(`.card`, `.btn-gold`, `.btn-ghost`, `.field`, `.kicker`, `.boiler-stripe`) built on
Purdue's palette (Boilermaker Gold `#CFB991` on warm black).

## Known gaps

See the root `TODO.md` — notably: no program/major picker yet (degree is free text until
the program crawl completes), and "Edit profile" creates a new student row because the
backend has no profile-update route.
