# Deployment

The backend (FastAPI + PyTorch model) runs on **Google Cloud Run**. User
feedback (image + correction) is stored in **Supabase** (Postgres + Storage).
The frontend is a static React build on **GitHub Pages** and doesn't need any
of the steps below except the last one.

Both Cloud Run and Supabase have permanent free tiers (not a one-time credit
that expires) and Cloud Run scales to zero, so there's no cost while the app
is idle.

## 1. Supabase (feedback storage)

1. Create a free project at [supabase.com](https://supabase.com).
2. Open the SQL editor and run [backend/supabase_setup.sql](backend/supabase_setup.sql) — this creates the `feedback` table.
3. Go to **Storage** → **New bucket** → name it `feedback-images` (private is fine).
4. Go to **Project Settings → API** and copy:
   - **Project URL** → this is `SUPABASE_URL`
   - **service_role key** (not the `anon` key — this one bypasses row-level security and must never be exposed to the frontend) → this is `SUPABASE_SERVICE_KEY`

## 2. Google Cloud (backend hosting)

1. Create a project at [console.cloud.google.com](https://console.cloud.google.com) and note the **Project ID** and **Project number** (both shown on the Dashboard page).
2. Enable the Cloud Run API, Cloud Build API, Artifact Registry API, and IAM Service Account Credentials API for the project (**APIs & Services → Library**, search + Enable each).
3. Create a Cloud Storage bucket named `rock-classifier-model` (or edit
   `MODEL_GCS_BUCKET` in [.github/workflows/deploy.yml](.github/workflows/deploy.yml)
   to match whatever name you pick) and upload `model_cleaned_best.pth` to it.
   The model is **not** committed to Git (116MB, over GitHub's 100MB limit),
   so this bucket is how Cloud Run gets it — `main.py` downloads it from here
   the first time the container starts.
4. Grant the Cloud Run service's *runtime* service account (by default,
   `PROJECT_NUMBER-compute@developer.gserviceaccount.com` — find it under
   IAM) the `Storage Object Viewer` role on that bucket, so the container is
   allowed to read the model file at startup.

### Authenticating GitHub Actions to GCP — Workload Identity Federation

Many Google Cloud orgs now block downloadable service-account keys
(`iam.disableServiceAccountKeyCreation`) as a security default. Rather than
fight that policy, use **Workload Identity Federation**: GitHub Actions
authenticates directly using its own OIDC token, no key file ever exists.
Run these in **Cloud Shell** (the `>_` icon in the top-right of the Cloud
Console — already authenticated as you, nothing to install):

```bash
PROJECT_ID="your-project-id"        # from step 1
PROJECT_NUMBER="your-project-number" # from step 1
REPO="Fehintiti/Rock-Classifier-App" # exact owner/repo, case-sensitive

# 1. Create the deploy service account (no key — just the identity)
gcloud iam service-accounts create github-deployer \
  --project="$PROJECT_ID" \
  --display-name="GitHub Actions Deployer"

SA_EMAIL="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

# 2. Grant it the roles it needs to build and deploy
for ROLE in roles/run.admin roles/cloudbuild.builds.editor \
            roles/artifactregistry.writer roles/iam.serviceAccountUser \
            roles/storage.admin; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" --role="$ROLE"
done

# 3. Create a Workload Identity Pool + GitHub OIDC provider
gcloud iam workload-identity-pools create "github-pool" \
  --project="$PROJECT_ID" --location="global" \
  --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="$PROJECT_ID" --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == '${REPO}'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# 4. Let only THIS repo impersonate the deploy service account
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${REPO}"

# 5. Print the provider resource name — copy this, it's a GitHub secret below
gcloud iam workload-identity-pools providers describe "github-provider" \
  --project="$PROJECT_ID" --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)"
```

The last command prints something like:
`projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider`
— copy that exact string for the next step.

## 3. GitHub repo secrets

In the repo's Settings → Secrets and variables → Actions, add:

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | your Google Cloud project ID |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | the string printed by step 5 above |
| `GCP_SERVICE_ACCOUNT_EMAIL` | `github-deployer@<your-project-id>.iam.gserviceaccount.com` |
| `SUPABASE_URL` | from step 1 |
| `SUPABASE_SERVICE_KEY` | from step 1 |

## 4. Deploy

Push a change under `backend/` to `main` (or run the workflow manually from
the Actions tab) — [.github/workflows/deploy.yml](.github/workflows/deploy.yml)
builds the Docker image and deploys it to Cloud Run.

Once it finishes, get the live URL:

```
gcloud run services describe rock-classifier --region us-central1 --format='value(status.url)'
```

or read it from the "Deploy to Cloud Run" step's log output in the Actions run.

## 5. Point the frontend at it

Edit `API_BASE_URL` near the top of [frontend/src/App.js](frontend/src/App.js)
to the URL from step 4, then redeploy the frontend:

```
cd frontend
npm run deploy
```

## Notes

- The model file is downloaded from Cloud Storage the first time the
  container starts (see Google Cloud steps 3–4 above), then reused for the container's
  lifetime — it is not re-downloaded per request. Cold starts (see below)
  will be a bit slower on top of the usual model-load time while this
  download happens.
- For **local development**, you can skip Cloud Storage entirely: just place
  `model_cleaned_best.pth` directly in `backend/` next to `main.py` — the
  code only downloads it if it's missing.
- With `--min-instances 0`, the first request after a period of no traffic
  will be slower (several seconds) while Cloud Run cold-starts the container,
  downloads the model (if not already local to that instance), and loads it.
  This is expected and free; raise `--min-instances` to 1 if you want to
  avoid it (that removes the scale-to-zero cost benefit and Cloud Run will
  bill for the always-on instance instead).
